import torch
import torchvision.models as models
from shape_infer import infer_onnx
from src.csrnet import csrnet
from src.dcgan import _netG as dcgan_netG
from src.cgan import GeneratorCGAN as cgan_netG
from src.GhostNet import ghost_net
from src.scnet import scnet101
from src.hetconv import vgg16bn as hetconv_vgg16bn 
from src.pyconv.build_model import build_model as pyconv_model
from src.pyconv.args_file import parser as pyconv_args
from src.DC_CDN_IJCAI21 import build_CDCN
from src.net48 import Net48
from src.testnet import TestNet
from src.gcn_1703_02719 import GCN
from src.unet import UNet
from src.vit import ViT
from src.simple_vit import SimpleViT
from src.deepvit import DeepViT
from src.nasnet import NASNetALarge
from src.hrnet.models.hrnet import hrnet18, hrnet32, hrnet48 

from src.yolov3.models.yolo import Model as yolov3_model
from src.yolov5.models.yolo import Model as yolov5_model
from src.yolov7.models.yolo import Model as yolov7_model
from src.swin_transformer import SwinTransformer

batch_size = 1


def export_inferred_onnx(name, input_shape, precision=32):
    fn = f"onnx/{name}.onnx"
    dummy_input = torch.randn(*input_shape)
    print(name, '='*20)
    if name == 'csrnet':
        model = csrnet()
    elif name == 'swin_transformer':
        model = SwinTransformer()
    elif name == 'hrnet18':
        model = hrnet18(False)
    elif name == 'hrnet32':
        model = hrnet32(False)
    elif name == 'hrnet48':
        model = hrnet48(False)

    elif name == 'nasnet':
        model = NASNetALarge()
    elif name == 'vit':
        model = ViT(image_size = 256, patch_size = 32, num_classes = 1000, dim = 1024, depth = 6, heads = 16, mlp_dim = 2048, dropout = 0.1, emb_dropout = 0.1)
    elif name == 'simple_vit':
        model = SimpleViT(image_size = 256,
                        patch_size = 32,
                        num_classes = 1000,
                        dim = 1024,
                        depth = 6,
                        heads = 16,
                        mlp_dim = 2048)
    elif name == 'deepvit':
        model = DeepViT(image_size = 256,
                        patch_size = 32,
                        num_classes = 1000,
                        dim = 1024,
                        depth = 6,
                        heads = 16,
                        mlp_dim = 2048,
                        dropout = 0.1,
                        emb_dropout = 0.1)
    elif name == 'yolov3':       
        model = yolov3_model(cfg='src/yolov3/models/yolov3.yaml')
    elif name == 'yolov5':        
        model = yolov5_model(cfg='src/yolov5/models/yolov5s.yaml')
    elif name == 'yolov7':
        from src.yolov7.models.yolo import Model as yolo7_model
        model = yolo7_model(cfg='src/yolov7/cfg/baseline/yolor-csp.yaml')
    elif name == 'dcgan':
        model = dcgan_netG(1)
    elif name == 'cgan':
        z_dim, c_dim = 100, 10
        model = cgan_netG(z_dim=z_dim, c_dim=c_dim)
    elif name == 'ghost':
        model = ghost_net()
    elif name == 'scnet':
        model = scnet101()
    elif name == 'hetconv':
        n_parts=1
        model = hetconv_vgg16bn(64, 64, 128, 128, 256, 256, 256,
                            512, 512, 512, 512, 512, 512, n_parts)
    elif name == 'pyconv': # Pyramid conv
        model = pyconv_model(pyconv_args.parse_args(args=['-a','pyconvresnet']))
    elif name =='dc-cdn':
        model = build_CDCN()
    elif name =='Net48':
        model = Net48()
    elif name =='TestNet':
        model = TestNet()
    elif name =='GCN':
        model = GCN(21)
    elif name =='unet':
        model = UNet(3, 10, [32, 64, 32], 0, 3, group_norm=0)
    else:
        model = getattr(models, name)(pretrained=False)

    if precision == 16:
        """ if torch.cuda.is_available():
            model = model.eval().cuda().half()
            dummy_input = dummy_input.cuda().half()
        else:
            model = model.eval().half()
            dummy_input = dummy_input.half() """
        model = model.eval().half()
        dummy_input = dummy_input.half()
        fn = f"onnx_fp16/{name}.onnx"
    else:
        """ if torch.cuda.is_available():
            model = model.eval().cuda()
            dummy_input = dummy_input.cuda()
        else:
            model = model.eval() """
        model = model.eval() 
    # # Without parameters
    # torch.onnx.export(model, dummy_input, fn,
    #   export_params=False, verbose=False)
    torch.onnx.export(model, dummy_input, fn,
                      export_params=False, verbose=False, opset_version=13)
                    #   export_params=True, verbose=False, opset_version=12)
    infer_onnx(fn)

    # Print model info
    with open(f'info/{name}.txt', 'w') as f:
        print(model, file=f)

input_shape = (1, 3, 224, 224)
export_inferred_onnx('swin_transformer', input_shape)

input_shape = (1, 3, 256, 256)
export_inferred_onnx('yolov3', input_shape)
export_inferred_onnx('yolov5', input_shape)
export_inferred_onnx('yolov7', input_shape)

input_shape = (batch_size, 3, 224, 224)
export_inferred_onnx('hrnet18', input_shape)
export_inferred_onnx('hrnet32', input_shape)
export_inferred_onnx('hrnet48', input_shape)

input_shape = (2, 3, 331, 331)
export_inferred_onnx('nasnet', input_shape)

input_shape = (1, 3, 256, 256)
export_inferred_onnx('vit', input_shape)
export_inferred_onnx('simple_vit', input_shape)
export_inferred_onnx('deepvit', input_shape)

input_shape = (batch_size, 100, 1, 1)
export_inferred_onnx('dcgan', input_shape)

input_shape = (1, 100+10, 1, 1)
export_inferred_onnx('cgan', input_shape, 'cgan.onnx')

input_shape = (batch_size, 3, 224, 224)
for name in ('resnet18', 'alexnet', 'squeezenet1_0', 'vgg16', 'densenet161', 'googlenet', 'shufflenet_v2_x1_0', 'mobilenet_v2', 'resnext50_32x4d', 'wide_resnet50_2', 'mnasnet1_0'):
    export_inferred_onnx(name, input_shape)

input_shape = (batch_size, 3, 299, 299)
for name in ('inception_v3',):
    export_inferred_onnx(name, input_shape)

input_shape = (batch_size, 512, 14, 14)
export_inferred_onnx('csrnet', input_shape)

input_shape = (32,3,32,32)
export_inferred_onnx('ghost', input_shape)

input_shape = (batch_size, 3, 224, 224)
export_inferred_onnx('scnet', input_shape)

input_shape = (batch_size, 3, 14, 14)
export_inferred_onnx('hetconv', input_shape)

input_shape = (batch_size, 3, 224, 224)
export_inferred_onnx('pyconv', input_shape)

input_shape = (batch_size, 3, 224, 224)
export_inferred_onnx('dc-cdn', input_shape)

input_shape = (batch_size, 3, 48, 48)
export_inferred_onnx('Net48', input_shape)

input_shape = (batch_size, 256, 24, 24)
export_inferred_onnx('TestNet', input_shape)

input_shape = (1, 3, 224, 224)
export_inferred_onnx('GCN', input_shape) 

input_shape = (batch_size, 3, 224, 224)
export_inferred_onnx('regnet_x_32gf', input_shape)

input_shape = (batch_size, 3, 224, 224)
export_inferred_onnx('unet', input_shape)
