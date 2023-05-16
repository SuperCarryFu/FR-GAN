import torch
from tqdm import tqdm
import torch.nn.functional as F
from Models.train_models import ir152
from Models.train_models import irse,facenet
from Myutils.dataloder import get_loader
from Myutils.utils import  batch_clamp

def tests_1(ir152=ir152, facenet=facenet):
    ir152 = ir152.IR_152((112, 112))
    ir152.load_state_dict(torch.load('./Models/train_models/ir152.pth'))
    ir152.to('cuda')
    ir152.eval()

    facenet = facenet.InceptionResnetV1(num_classes=8631, device='cuda')
    facenet.load_state_dict(torch.load('./Models/train_models/facenet.pth'))
    facenet.to('cuda')
    facenet.eval()

    MobileFace = irse.MobileFaceNet(512)
    MobileFace.load_state_dict(torch.load('./Models/train_models/mobile_face.pth'))
    MobileFace.to('cuda')
    MobileFace.eval()

    irse50 = irse.Backbone(50, 0.6, 'ir_se')
    irse50.load_state_dict(torch.load('./Models/train_models/irse50.pth'))
    irse50.to('cuda')
    irse50.eval()

    return facenet

def evaluate_LADN(tar, generator, delta, use_saliency_map=False):
    _, test_loader, _ = get_loader(1)
    # 加载测试模型
    test_models = tests_1()
    d_FAR01, d_FAR001, d_FAR0001, a_total = 0, 0, 0, 0
    th_dict = {'ir152': (0.094632, 0.166788, 0.227922), 'irse50': (0.144840, 0.241045, 0.312703),
               'facenet': (0.256587, 0.409131, 0.591191), 'mobile_face': (0.183635, 0.301611, 0.380878)}
    if generator:
        generator.eval()
    with torch.no_grad():
        for i, inputs in tqdm(enumerate(test_loader)):
            src = inputs[0].to('cuda')
            perturbations, saliency_map = generator(src)
            if use_saliency_map:
                adv = src + batch_clamp(delta, perturbations) * saliency_map
            else:
                adv = src + batch_clamp(delta, perturbations)
            adv=torch.clamp(adv,-1,1)
            adv_embbeding = test_models((F.interpolate(adv, size=(160, 160), mode='bilinear')))
            tar_embbeding = test_models((F.interpolate(tar, size=(160, 160), mode='bilinear')))
            cos_simi = torch.cosine_similarity(adv_embbeding, tar_embbeding)
            for cos in cos_simi:
                if cos.item() > th_dict['facenet'][0]:
                    d_FAR01 += 1
                if cos.item() > th_dict['facenet'][1]:
                    d_FAR001 += 1
                if cos.item() > th_dict['facenet'][2]:
                    d_FAR0001 += 1
                a_total += 1

    d1 = 100. * d_FAR01 / a_total
    d2 = 100. * d_FAR001 / a_total
    d3 = 100. * d_FAR0001 / a_total

    return d1, d2, d3, a_total, d_FAR01, d_FAR001, d_FAR0001


def evaluate_CelebA_HQ(tar, generator, delta, use_saliency_map=False):
    _, _, test_loader = get_loader(5)
    # 加载测试模型
    test_models = tests_1()
    d_FAR01, d_FAR001, d_FAR0001, a_total = 0, 0, 0, 0
    th_dict = {'ir152': (0.094632, 0.166788, 0.227922), 'irse50': (0.144840, 0.241045, 0.312703),
               'facenet': (0.256587, 0.409131, 0.591191), 'mobile_face': (0.183635, 0.301611, 0.380878)}
    if generator:
        generator.eval()
    with torch.no_grad():
        for i, inputs in tqdm(enumerate(test_loader)):
            src = inputs[0].to('cuda')
            perturbations, saliency_map = generator(src)
            if use_saliency_map:
                adv = src + batch_clamp(delta, perturbations) * saliency_map
            else:
                adv = src + batch_clamp(delta, perturbations)
            adv=torch.clamp(adv,-1,1)
            adv_embbeding = test_models((F.interpolate(adv, size=(160, 160), mode='bilinear')))
            tar_embbeding = test_models((F.interpolate(tar, size=(160, 160), mode='bilinear')))
            cos_simi = torch.cosine_similarity(adv_embbeding, tar_embbeding)
            for cos in cos_simi:
                if cos.item() > th_dict['facenet'][0]:
                    d_FAR01 += 1
                if cos.item() > th_dict['facenet'][1]:
                    d_FAR001 += 1
                if cos.item() > th_dict['facenet'][2]:
                    d_FAR0001 += 1
                a_total += 1

    d1 = 100. * d_FAR01 / a_total
    d2 = 100. * d_FAR001 / a_total
    d3 = 100. * d_FAR0001 / a_total

    return d1, d2, d3, a_total, d_FAR01, d_FAR001, d_FAR0001
