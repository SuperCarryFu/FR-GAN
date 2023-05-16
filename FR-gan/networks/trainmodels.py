from networks.get_model import getmodel

def trainmodels():
    ShuffleNet1, _ = getmodel('ShuffleNet_V1_GDConv', device='cuda')
    ShuffleNet2, _ = getmodel('ShuffleNet_V2_GDConv-stride1', device='cuda')
    MobilenetV2, _ = getmodel('MobilenetV2', device='cuda')
    Mobilenet, _ = getmodel('Mobilenet', device='cuda')
    ArcFace, _ = getmodel('ArcFace', device='cuda')
    IR50Softmax, _ = getmodel('IR50-Softmax', device='cuda')
    IR50CosFace, _ = getmodel('IR50-CosFace', device='cuda')
    IR50SphereFace, _ = getmodel('IR50-SphereFace', device='cuda')
    IR50Am, _ = getmodel('IR50-Am', device='cuda')
    train_models = {}
    train_models['ShuffleNet1'] = []
    train_models['ShuffleNet1'].append((112, 112))
    train_models['ShuffleNet1'].append(ShuffleNet1)

    train_models['ShuffleNet2'] = []
    train_models['ShuffleNet2'].append((112, 112))
    train_models['ShuffleNet2'].append(ShuffleNet2)

    train_models['MobilenetV2'] = []
    train_models['MobilenetV2'].append((112, 112))
    train_models['MobilenetV2'].append(MobilenetV2)

    train_models['Mobilenet'] = []
    train_models['Mobilenet'].append((112, 112))
    train_models['Mobilenet'].append(Mobilenet)

    train_models['ArcFace'] = []
    train_models['ArcFace'].append((112, 112))
    train_models['ArcFace'].append(ArcFace)

    train_models['IR50Softmax'] = []
    train_models['IR50Softmax'].append((112, 112))
    train_models['IR50Softmax'].append(IR50Softmax)

    train_models['IR50CosFace'] = []
    train_models['IR50CosFace'].append((112, 112))
    train_models['IR50CosFace'].append(IR50CosFace)

    train_models['IR50SphereFace'] = []
    train_models['IR50SphereFace'].append((112, 112))
    train_models['IR50SphereFace'].append(IR50SphereFace)

    train_models['IR50Am'] = []
    train_models['IR50Am'].append((112, 112))
    train_models['IR50Am'].append(IR50Am)
    return train_models

def testmodels():
    # FaceNet_JPEG, _ = getmodel('FaceNet-VGGFace2-JPEG', device='cuda')
    # FaceNet_RP, _ = getmodel('FaceNet-VGGFace2-RP', device='cuda')
    # FaceNet_BR, _ = getmodel('FaceNet-VGGFace2-BR', device='cuda')
    FaceNet_VGGFace2, _ = getmodel('FaceNet-VGGFace2', device='cuda')
    MobileFace, _ = getmodel('MobileFace', device='cuda')
    IR50_PGDArcFace, _ = getmodel('IR50-PGDArcFace', device='cuda')
    # IR50_PGDCosFace, _ = getmodel('IR50-PGDCosFace', device='cuda')
    # IR50_PGDSoftmax, _ = getmodel('IR50-PGDSoftmax', device='cuda')
    SphereFace, _ = getmodel('SphereFace', device='cuda')
    CosFace, _ = getmodel('CosFace', device='cuda')
    IR50_TradesArcFace, _ = getmodel('IR50-TradesArcFace', device='cuda')
    train_models = {}
    # train_models['FaceNet_JPEG'] = []
    # train_models['FaceNet_JPEG'].append((160, 160))
    # train_models['FaceNet_JPEG'].append(FaceNet_JPEG)

    # train_models['FaceNet_RP'] = []
    # train_models['FaceNet_RP'].append((112, 112))
    # train_models['FaceNet_RP'].append(FaceNet_RP)

    # train_models['FaceNet_BR'] = []
    # train_models['FaceNet_BR'].append((112, 112))
    # train_models['FaceNet_BR'].append(FaceNet_BR)

    train_models['FaceNet_VGGFace2'] = []
    train_models['FaceNet_VGGFace2'].append((112, 112))
    train_models['FaceNet_VGGFace2'].append(FaceNet_VGGFace2)

    train_models['IR50_PGDArcFace'] = []
    train_models['IR50_PGDArcFace'].append((112, 112))
    train_models['IR50_PGDArcFace'].append(IR50_PGDArcFace)

    # train_models['IR50_PGDCosFace'] = []
    # train_models['IR50_PGDCosFace'].append((112, 112))
    # train_models['IR50_PGDCosFace'].append(IR50_PGDCosFace)

    train_models['IR50_TradesArcFace'] = []
    train_models['IR50_TradesArcFace'].append((112, 112))
    train_models['IR50_TradesArcFace'].append(IR50_TradesArcFace)

    train_models['CosFace'] = []
    train_models['CosFace'].append((112, 96))
    train_models['CosFace'].append(CosFace)

    train_models['SphereFace'] = []
    train_models['SphereFace'].append((112, 96))
    train_models['SphereFace'].append(SphereFace)

    train_models['MobileFace'] = []
    train_models['MobileFace'].append((112, 112))
    train_models['MobileFace'].append(MobileFace)
    return train_models

if __name__ == '__main__':
    t= testmodels()
    print("c")
