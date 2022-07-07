import argparse

import torch
import os

import sys

sys.path.insert(0, './')
from models.models import *
from utils.google_utils import attempt_download

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='./yolov4.pt', help='weights path')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='image size')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--cfg', type=str, default='cfg/yolor_csp.cfg', help='config file path')
    parser.add_argument('--fp16', action='store_true', help="half floating point precision")
    parser.add_argument('--export_filename', type=str, default='', help='output filename of onnx file')
    opt = parser.parse_args()
    opt.img_size *= 2 if len(opt.img_size) == 1 else 1  # expand
    print(opt)

    # Input
    img = torch.zeros((opt.batch_size, 3, *opt.img_size), device='cuda')  # image size(1,3,320,192) iDetection

    # Load PyTorch model
    attempt_download(opt.weights)
    d = torch.load(opt.weights, map_location=torch.device('cpu'))
    print(d['model'].keys())
    model = Darknet(cfg=opt.cfg, img_size=opt.img_size, export=True)
    model.load_state_dict(d["model"])

    # model = torch.load(opt.weights, map_location=torch.device('cpu'))['model'].float()
    model.float()
    if opt.fp16:
        model.half()
        img = img.half()
        model.to('cuda')
        img.to('cuda')
    model.eval()
    y = model(img)  # dry run
    

    # TorchScript export
    try:
        print('\nStarting TorchScript export with torch %s...' % torch.__version__)
        f, ext = os.path.splitext(opt.weights)
        opt.weights = opt.export_filename + ext
        f = opt.weights.replace('.pt', '.torchscript.pt')  # filename
        ts = torch.jit.trace(model, img)
        ts.save(f)
        print('TorchScript export success, saved as %s' % f)
    except Exception as e:
        print('TorchScript export failure: %s' % e)

    # ONNX export
    try:
        import onnx

        print('\nStarting ONNX export with onnx %s...' % onnx.__version__)
        f = opt.weights.replace('.pt', '.onnx')  # filename
        model.fuse()  # only for ONNX
        print(f"Running export with input {img.shape}")
        torch.onnx.export(model.half(), img.half(), f, verbose=True, opset_version=12, input_names=['images'],
                          output_names=['output'])#['classes', 'boxes'] if y is None else ['output'])

        # Checks
        onnx_model = onnx.load(f)  # load onnx model
        onnx.checker.check_model(onnx_model)  # check onnx model
        print(onnx.helper.printable_graph(onnx_model.graph))  # print a human readable model
        print('ONNX export success, saved as %s' % f)
    except Exception as e:
        print('ONNX export failure: %s' % e)

    # CoreML export
    try:
        import coremltools as ct

        print('\nStarting CoreML export with coremltools %s...' % ct.__version__)
        # convert model from torchscript and apply pixel scaling as per detect.py
        model = ct.convert(ts, inputs=[ct.ImageType(name='images', shape=img.shape, scale=1 / 255.0, bias=[0, 0, 0])])
        f = opt.weights.replace('.pt', '.mlmodel')  # filename
        model.save(f)
        print('CoreML export success, saved as %s' % f)
    except Exception as e:
        print('CoreML export failure: %s' % e)

    # Finish
    print('\nExport complete. Visualize with https://github.com/lutzroeder/netron.')
