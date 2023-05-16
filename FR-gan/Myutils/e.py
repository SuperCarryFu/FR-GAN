import torch
from torchvision.utils import save_image
from tqdm import tqdm
import torch.nn.functional as F
from Myutils.dataloder import get_loader
from Myutils.utils import tensor2img, batch_clamp
from networks.trainmodels import testmodels


def evaluate_LADN(tar, generator, delta, use_saliency_map=False):
    _, test_loader = get_loader(1)
    # 加载测试模型
    test_models = testmodels()
    d_FAR01, d_FAR001, d_FAR0001, a_total = 0, 0, 0, 0
    th_dict = {'ir152': (0.094632, 0.166788, 0.227922), 'irse50': (0.144840, 0.241045, 0.312703),
               'facenet': (0.256587, 0.409131, 0.591191), 'mobile_face': (0.183635, 0.301611, 0.380878)}
    if generator:
        generator.eval()
    with torch.no_grad():
        for i, inputs in tqdm(enumerate(test_loader)):
            src = inputs[0].to('cuda')
            perturbations, saliency_map = generator(src)
            perturbations_with_saliency = perturbations * saliency_map
            c = batch_clamp(delta, perturbations) * saliency_map
            if use_saliency_map:
                adv = src + batch_clamp(delta, perturbations) * saliency_map
            else:
                adv = src + batch_clamp(delta, perturbations)
            adv=torch.clamp(adv,0,255)
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

            if i < 5:
                save_image(saliency_map, './log_LADN/cls_mask_%d.jpg' % i)
                tensor2img(adv)
                save_image(adv, './log_LADN/cls_adv_%d.png' % i)
                tensor2img(src)
                save_image(src, './log_LADN/cls_src_%d.png' % i)
                tensor2img(perturbations)
                save_image(perturbations, './log_LADN/cls_pertubations_%d.png' % i)
                tensor2img(c)
                save_image(c, './log_LADN/cls_c_%d.png' % i)
                tensor2img(perturbations_with_saliency)
                save_image(perturbations_with_saliency, './log_LADN/cls_perturbations_with_saliency_%d.png' % i)

            # tensor2img(adv)
            # save_image(adv, './log_LADN/cls_adv_%d.png' % i)


    d1 = 100. * d_FAR01 / a_total
    d2 = 100. * d_FAR001 / a_total
    d3 = 100. * d_FAR0001 / a_total

    return d1, d2, d3, a_total, d_FAR01, d_FAR001, d_FAR0001


# def evaluate_CelebA_HQ(tar, generator, delta, use_saliency_map=False):
#     _, _, test_loader = get_loader(5)
#     # 加载测试模型
#     test_models = tests_1()
#     d_FAR01, d_FAR001, d_FAR0001, a_total = 0, 0, 0, 0
#     th_dict = {'ir152': (0.094632, 0.166788, 0.227922), 'irse50': (0.144840, 0.241045, 0.312703),
#                'facenet': (0.256587, 0.409131, 0.591191), 'mobile_face': (0.183635, 0.301611, 0.380878)}
#     if generator:
#         generator.eval()
#     with torch.no_grad():
#         for i, inputs in tqdm(enumerate(test_loader)):
#             src = inputs[0].to('cuda')
#             perturbations, saliency_map = generator(src)
#             perturbations_with_saliency = perturbations * saliency_map
#             c = batch_clamp(delta, perturbations) * saliency_map
#             if use_saliency_map:
#                 adv = src + batch_clamp(delta, perturbations) * saliency_map
#             else:
#                 adv = src + batch_clamp(delta, perturbations)
#             adv=torch.clamp(adv,-1,1)
#             adv_embbeding = test_models((F.interpolate(adv, size=(160, 160), mode='bilinear')))
#             tar_embbeding = test_models((F.interpolate(tar, size=(160, 160), mode='bilinear')))
#             cos_simi = torch.cosine_similarity(adv_embbeding, tar_embbeding)
#             for cos in cos_simi:
#                 if cos.item() > th_dict['facenet'][0]:
#                     d_FAR01 += 1
#                 if cos.item() > th_dict['facenet'][1]:
#                     d_FAR001 += 1
#                 if cos.item() > th_dict['facenet'][2]:
#                     d_FAR0001 += 1
#                 a_total += 1
#
#             if i < 5:
#                 save_image(saliency_map, './log_CelebA_HQ/cls_mask_%d.jpg' % i)
#                 tensor2img(adv)
#                 save_image(adv, './log_CelebA_HQ/cls_adv_%d.png' % i)
#                 tensor2img(src)
#                 save_image(src, './log_CelebA_HQ/cls_src_%d.png' % i)
#                 tensor2img(perturbations)
#                 save_image(perturbations, './log_CelebA_HQ/cls_pertubations_%d.png' % i)
#                 tensor2img(c)
#                 save_image(c, './log_CelebA_HQ/cls_c_%d.png' % i)
#                 tensor2img(perturbations_with_saliency)
#                 save_image(perturbations_with_saliency, './log_CelebA_HQ/cls_perturbations_with_saliency_%d.png' % i)
#
#             # tensor2img(adv)
#             # save_image(adv, './log_CelebA_HQ/cls_adv_%d.png' % i)
#
#     d1 = 100. * d_FAR01 / a_total
#     d2 = 100. * d_FAR001 / a_total
#     d3 = 100. * d_FAR0001 / a_total
#
    return d1, d2, d3, a_total, d_FAR01, d_FAR001, d_FAR0001
