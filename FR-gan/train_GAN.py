import argparse
import os
import torch
from torch import optim, nn
from tqdm import tqdm
import torch.nn.functional as F
from Myutils.dataloder import get_loader
from Myutils.discriminator import Discriminator
from Myutils.generator import SSAE
from Myutils.utils import read_img, input_diversity, input_noise, cal_adv_loss, batch_clamp
from networks.trainmodels import trainmodels
from test_ever import evaluate_LFW, evaluate_LFW2, evaluate_LFW22

if __name__ == '__main__':
    parser = argparse.ArgumentParser('This is a FR-GAN.')
    parser.add_argument('--TARGET_PATH', type=str, default='./dataset/traget/0b8f14c8-4117-4aab-b951-761159d642c5.jpg')
    parser.add_argument('--EPOCHS', type=int, default=40)
    parser.add_argument('--saliency', type=bool, default=True)
    parser.add_argument('--delta', type=float, default=8.0/255)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--RESUME', type=float, default=False)
    args = parser.parse_args()

    # 加载目标图片
    tar = read_img(args.TARGET_PATH)
    # 加载训练模型
    train_models = trainmodels()

    train_loader, _,_= get_loader(4)

    generator = SSAE().to('cuda')
    optimizer = optim.Adam(generator.parameters(), lr=args.lr)
    discriminator = Discriminator(3).to('cuda')
    optimizer_d = optim.Adam(discriminator.parameters(), lr=args.lr)
    mse_loss = nn.MSELoss(reduction='sum')
    from torchsummary import summary

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    t = SSAE().to(device)
    summary(t, (3, 112, 112))
    print(t)
    start_epoch = -1

    if args.RESUME:
        path_checkpoint = "checkpoint/ckpt_best_15.pth"  # 断点路径
        checkpoint = torch.load(path_checkpoint)  # 加载断点
        generator.load_state_dict(checkpoint['generator'])  # 加载模型可学习参数
        discriminator.load_state_dict(checkpoint['discriminator'])  # 加载模型可学习参数
        optimizer.load_state_dict(checkpoint['optimizer_g'])  # 加载优化器参数
        optimizer_d.load_state_dict(checkpoint['optimizer_d'])  # 加载优化器参数
        start_epoch = checkpoint['epoch']  # 设置开始的epoch

    for epoch in range(start_epoch + 1, args.EPOCHS):
        generator.train()
        l=0
        for i, inputs in tqdm(enumerate(train_loader)):

            src = inputs[0].to('cuda')
            perturbations, saliency_map = generator(src)
            if args.saliency:
                perturbations = batch_clamp(args.delta, perturbations) * saliency_map
            else:
                perturbations = batch_clamp(args.delta, perturbations)
            adv = src + perturbations
            adv = torch.clamp(adv, 0,1)
            pred_fake = discriminator(adv)
            fake_out = torch.sigmoid(pred_fake)
            loss_discr = F.mse_loss(fake_out, torch.ones_like(fake_out, device='cuda'))
            targeted_loss = 0
            fake_A_diversity = []
            count = 0
            adv=adv*255
            for ii in range(5):
                fake_A_diversity.append(input_diversity(input_noise(adv)).to('cuda'))
            for model_name in train_models.keys():
                for iii in range(5):
                    target_loss_A = cal_adv_loss(fake_A_diversity[iii], tar, model_name, train_models)
                    targeted_loss += target_loss_A
                    count += 1
            loss_adv = targeted_loss / count
            # loss_adv = targeted_loss/5
            l+=loss_adv
            if args.saliency:
                frobenius_loss = torch.norm(saliency_map, dim=(1, 2)).sum()
                loss =0.00001 * frobenius_loss + loss_adv + 0.06*loss_discr
                if torch.isnan(frobenius_loss):
                    print("there are nans in frobenius loss")
                    break
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # if i%2==1:
            adv=adv/255
            optimizer_d.zero_grad()
            pred_fake = discriminator(adv.detach())
            pred_real = discriminator(src)
            out_fake = torch.sigmoid(pred_fake)
            out_real = torch.sigmoid(pred_real)
            ad_true_loss = F.mse_loss(out_real, torch.ones_like(out_real, device='cuda'))
            ad_fake_loss = F.mse_loss(out_fake, torch.zeros_like(out_fake, device='cuda'))
            loss_d = 0.06*(ad_true_loss + ad_fake_loss)
            loss_d.backward()
            optimizer_d.step()

        SphereFace_FAR01, \
        CosFace_FAR01, \
        FaceNet_VGGFace2_FAR01, \
        MobileFace_FAR01, \
        IR50_PGDArcFace_FAR01, \
        IR50_TradesArcFace_FAR01, \
        total = evaluate_LFW22(tar, generator, args.delta, use_saliency_map=args.saliency)
        print("SphereFace:" + str(SphereFace_FAR01))
        print("CosFace:" + str(CosFace_FAR01))
        print("FaceNet_VGGFace2:" + str(FaceNet_VGGFace2_FAR01))
        print("MobileFace:" + str(MobileFace_FAR01))
        print("IR50_PGDArcFace:" + str(IR50_PGDArcFace_FAR01))
        print("IR50_TradesArcFace:" + str(IR50_TradesArcFace_FAR01))
        print(total)
        print(l)
        print("========================================")
        checkpoint = {
            "generator": generator.state_dict(),
            'optimizer_g': optimizer.state_dict(),
            "epoch": epoch,
            "discriminator": discriminator.state_dict(),
            'optimizer_d': optimizer_d.state_dict(),
        }
        if not os.path.isdir("checkpoint_16_new_3"):
            os.mkdir("checkpoint_16_new_3")
        torch.save(checkpoint, './checkpoint_16_new_3/ckpt_best_%s.pth' % (str(epoch)))
