import torch
import torchvision
from PIL import Image
from torch import nn
from torchvision import transforms
import torch.nn.functional as F

from Myutils.dataloder import get_loader


class CosineLoss(nn.Module):
    def __init__(self, ):
        super(CosineLoss, self).__init__()

    def forward(self, x, y):
        return ((torch.cosine_similarity(x, y, dim=1) + 1)).sum()

def read_img(data_dir):
    src = Image.open(data_dir)
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    img=transform(src)
    img=torch.unsqueeze(img, dim=0).to('cuda')
    img=img*255
    return img

def cos_simi(emb_1, emb_2):
    return torch.mean(torch.sum(torch.mul(emb_2, emb_1), dim=1) / emb_2.norm(dim=1) / emb_1.norm(dim=1))

def cal_adv_loss(source, target, model_name, target_models):
    # input_size = target_models[model_name][0]
    fr_model = target_models[model_name][1]
    # source_resize = F.interpolate(source, size=input_size, mode='bilinear')
    # target_resize = F.interpolate(target, size=input_size, mode='bilinear')
    emb_source = fr_model(source)

    # emb_source = nn.functional.normalize(emb_source, dim=1, p=2)

    emb_target = fr_model(target).detach()

    # emb_target = nn.functional.normalize(emb_target, dim=1, p=2)

    cos_loss = 1 - cos_simi(emb_source, emb_target)
    # cosine_loss = CosineLoss().to('cuda')
    # cos_loss=-cosine_loss(emb_target,emb_source)
    # cos_loss=torch.mean((emb_source - emb_target) ** 2)
    # cos_loss = cos_simi(emb_source, emb_target)
    return cos_loss

def cal_adv_loss1(source, target, model_name, target_models):
    # input_size = target_models[model_name][0]
    fr_model = target_models[model_name][1]
    # source_resize = F.interpolate(source, size=input_size, mode='bilinear')
    # target_resize = F.interpolate(target, size=input_size, mode='bilinear')
    emb_source = fr_model(source)

    # emb_source = nn.functional.normalize(emb_source, dim=1, p=2)

    emb_target = fr_model(target).detach()

    # emb_target = nn.functional.normalize(emb_target, dim=1, p=2)

    cos_loss =cos_simi(emb_source, emb_target)
    # cosine_loss = CosineLoss().to('cuda')
    # cos_loss=-cosine_loss(emb_target,emb_source)
    # cos_loss=torch.mean((emb_source - emb_target) ** 2)
    # cos_loss = cos_simi(emb_source, emb_target)
    return cos_loss

def un_cal_adv_loss(source, target, model_name, target_models):
    input_size = target_models[model_name][0]
    fr_model = target_models[model_name][1]
    source_resize = F.interpolate(source, size=input_size, mode='bilinear')
    target_resize = F.interpolate(target, size=input_size, mode='bilinear')
    emb_source = fr_model(source_resize)
    emb_target = fr_model(target_resize).detach()
    cos_loss = cos_simi(emb_source, emb_target)
    return cos_loss

def input_diversity(x):
    img_size = x.shape[-1]
    img_resize = int(img_size * 0.9)

    if 0.9 < 1:
        img_size = img_resize
        img_resize = x.shape[-1]

    rnd = torch.randint(low=img_size, high=img_resize, size=(1,), dtype=torch.int32)
    rescaled = F.interpolate(x, size=[rnd, rnd], mode='bilinear', align_corners=False).to('cuda')
    h_rem = img_resize - rnd
    w_rem = img_resize - rnd
    pad_top = torch.randint(low=0, high=h_rem.item(), size=(1,), dtype=torch.int32)
    pad_bottom = h_rem - pad_top
    pad_left = torch.randint(low=0, high=w_rem.item(), size=(1,), dtype=torch.int32)
    pad_right = w_rem - pad_left

    padded = F.pad(rescaled, [pad_left.item(), pad_right.item(), pad_top.item(), pad_bottom.item()], value=0) \
        .to('cuda')

    return padded if torch.rand(1) < 0.5 else x

def input_noise(x):
    rnd = torch.rand(1).to('cuda')
    noise = torch.randn_like(x).to('cuda')*0.8 * rnd * (0.1 ** 0.5)

    x_noised = x + noise*10
    x_noised = torch.clamp(x_noised, 0, 255)
    x_noised.to('cuda')
    return x_noised if torch.rand(1) < 0.5 else x

def tensor2img(tensors, mean=[0.5, 0.5, 0.5], std=[0.5,0.5,0.5]):
    for c in range(3):
        tensors.data[:, c, :, :] = tensors.data[:, c, :, :] * std[c] + mean[c]

def clamp(input, min=None, max=None):
    if min is not None and max is not None:
        return torch.clamp(input, min=min, max=max)
    elif min is None and max is None:
        return input
    elif min is None and max is not None:
        return torch.clamp(input, max=max)
    elif min is not None and max is None:
        return torch.clamp(input, min=min)
    else:
        raise ValueError("This is impossible")

def _batch_clamp_tensor_by_vector(vector, batch_tensor):
    """Equivalent to the following
    for ii in range(len(vector)):
        batch_tensor[ii] = clamp(
            batch_tensor[ii], -vector[ii], vector[ii])
    """
    return torch.min(
        torch.max(batch_tensor.transpose(0, -1), -vector), vector
    ).transpose(0, -1).contiguous()

def batch_clamp(float_or_vector, tensor):
    if isinstance(float_or_vector, torch.Tensor):
        assert len(float_or_vector) == len(tensor)
        tensor = _batch_clamp_tensor_by_vector(float_or_vector, tensor)
        return tensor
    elif isinstance(float_or_vector, float):
        tensor = clamp(tensor, -float_or_vector, float_or_vector)
    else:
        raise TypeError("Value has to be float or torch.Tensor")
    return tensor


def tv_loss(input_t):
    temp1 = torch.cat((input_t[:, :, 1:, :], input_t[:, :, -1, :].unsqueeze(2)), 2)
    temp2 = torch.cat((input_t[:, :, :, 1:], input_t[:, :, :, -1].unsqueeze(3)), 3)
    temp = (input_t - temp1) ** 2 + (input_t - temp2) ** 2
    return temp.sum()

if __name__ == '__main__':
    from tqdm import tqdm
    loader,_ = get_loader(batch_size=1)
    for i, inputs in tqdm(enumerate(loader)):
        src = inputs[0].to('cuda')
        src1=src*255
        noise = input_noise(src1)
        torchvision.utils.save_image(noise, 'cc.png',normalize=True)
        print("ccc")

