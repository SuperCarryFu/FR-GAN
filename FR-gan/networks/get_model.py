from networks.MobileFace import MobileFace
from networks.Mobilenet import Mobilenet
from networks.MobilenetV2 import MobilenetV2
from networks.ResNet import resnet
from networks.ShuffleNetV2 import ShuffleNetV2
from networks.ShuffleNet import ShuffleNetV1
from networks.CosFace import CosFace
from networks.SphereFace import SphereFace
from networks.FaceNet import FaceNet
from networks.ArcFace import ArcFace
from networks.IR import IR


def getmodel(face_model, **kwargs):
    """
        select the face model according to its name
        :param face_model: string
        return:
        a model class
    """
    img_shape = (112, 112)
    if face_model == 'MobileFace':
        model = MobileFace(**kwargs)
    elif face_model == 'Mobilenet':
        model = Mobilenet(**kwargs)
    elif face_model == 'Mobilenet-stride1':
        model = Mobilenet(stride=1, **kwargs)
    elif face_model == 'MobilenetV2':
        model = MobilenetV2(**kwargs)
    elif face_model == 'MobilenetV2-stride1':
        model = MobilenetV2(stride=1, **kwargs)
    elif face_model == 'ResNet50':
        model = resnet(depth=50, **kwargs)
    elif face_model == 'ResNet50-casia':
        model = resnet(depth=50, dataset='casia', **kwargs)
    elif face_model == 'ShuffleNet_V1_GDConv':
        model = ShuffleNetV1(pooling='GDConv', **kwargs)
    elif face_model == 'ShuffleNet_V2_GDConv-stride1':
        model = ShuffleNetV2(stride=1, pooling='GDConv', **kwargs)
    elif face_model == 'CosFace':
        model = CosFace(**kwargs)
        img_shape = (112, 96)
    elif face_model == 'SphereFace':
        model = SphereFace(**kwargs)
        img_shape = (112, 96)
    elif face_model == 'FaceNet-VGGFace2':
        model = FaceNet(dataset='vggface2', use_prewhiten=False, **kwargs)
        img_shape = (160, 160)
    elif face_model == 'FaceNet-casia':
        model = FaceNet(dataset='casia-webface', use_prewhiten=False, **kwargs)
        img_shape = (160, 160)
    elif face_model == 'ArcFace':
        model = ArcFace(**kwargs)
    elif face_model == 'IR50-Softmax':
        model = IR(loss='Softmax', **kwargs)
    elif face_model == 'IR50-Softmax-BR':
        model = IR(loss='Softmax', transform='BitReudction', **kwargs)
    elif face_model == 'IR50-Softmax-RP':
        model = IR(loss='Softmax', transform='Randomization', **kwargs)
    elif face_model == 'IR50-Softmax-JPEG':
        model = IR(loss='Softmax', transform='JPEG', **kwargs)
    elif face_model == 'IR50-PGDSoftmax':
        model = IR(loss='PGDSoftmax', **kwargs)
    elif face_model == 'IR50-TradesSoftmax':
        model = IR(loss='TradesSoftmax', **kwargs)
    elif face_model == 'IR50-CosFace':
        model = IR(loss='CosFace', **kwargs)
    elif face_model == 'IR50-TradesCosFace':
        model = IR(loss='TradesCosFace', **kwargs)
    elif face_model == 'IR50-PGDCosFace':
        model = IR(loss='PGDCosFace', **kwargs)
    elif face_model == 'IR50-Am':
        model = IR(loss='Am', **kwargs)
    elif face_model == 'IR50-PGDAm':
        model = IR(loss='PGDAm', **kwargs)
    elif face_model == 'IR50-ArcFace':
        model = IR(loss='ArcFace', **kwargs)
    elif face_model == 'IR50-PGDArcFace':
        model = IR(loss='PGDArcFace', **kwargs)
    elif face_model == 'IR50-TradesArcFace':
        model = IR(loss='TradesArcFace', **kwargs)
    elif face_model == 'IR50-SphereFace':
        model = IR(loss='SphereFace', **kwargs)
    elif face_model == 'IR50-PGDSphereFace':
        model = IR(loss='PGDSphereFace', **kwargs)
    elif face_model == 'CASIA-Softmax':
        model = IR(loss='CASIA-Softmax', **kwargs)
    elif face_model == 'CASIA-CosFace':
        model = IR(loss='CASIA-CosFace', **kwargs)
    elif face_model == 'CASIA-ArcFace':
        model = IR(loss='CASIA-ArcFace', **kwargs)
    elif face_model == 'CASIA-SphereFace':
        model = IR(loss='CASIA-SphereFace', **kwargs)
    elif face_model == 'CASIA-Am':
        model = IR(loss='CASIA-Am', **kwargs)
    elif face_model == 'FaceNet-VGGFace2-BR':
        model = FaceNet(dataset='vggface2', use_prewhiten=False,transform='BitReudction', **kwargs)
    elif face_model == 'FaceNet-VGGFace2-RP':
        model = FaceNet(dataset='vggface2', use_prewhiten=False,transform='Randomization', **kwargs)
    elif face_model == 'FaceNet-VGGFace2-JPEG':
        model = FaceNet(dataset='vggface2', use_prewhiten=False,transform='JPEG', **kwargs)
    else:
        raise NotImplementedError
    return model, img_shape
