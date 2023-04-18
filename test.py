import os, time, argparse
import numpy as np
from PIL import Image

import torch
from torchvision.utils import save_image as imwrite

from utils import load_checkpoint, tensor2cuda

from model.models import Generator

# 调用GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main():
    # 开关定义
    parser = argparse.ArgumentParser(description="network pytorch")
    # train
    parser.add_argument("--model", type=str, default="./checkpoint/", help='checkpoint')
    parser.add_argument("--model_name", type=str, default='Gmodel_40', help='model name')
    # value
    parser.add_argument("--intest", type=str, default="./input/", help='input syn path')
    parser.add_argument("--outest", type=str, default="./output/", help='output syn path')
    argspar = parser.parse_args()

    print("\nnetwork pytorch")
    for p, v in zip(argspar.__dict__.keys(), argspar.__dict__.values()):
        print('\t{}: {}'.format(p, v))
    print('\n')
    arg = parser.parse_args()

    # train
    print('> Loading Generator...')
    name = arg.model_name
    Gmodel_name = 'Gmodel_'+name+'.tar'
    Dmodel_name = 'Dmodel_'+name+'.tar'
    G_Model, _, _ = load_checkpoint(argspar.model, Generator, Gmodel_name)

    os.makedirs(arg.outest, exist_ok=True)
    test(argspar, G_Model)

def test(argspar, model):
    # init
    norm = lambda x: (x - 0.5) / 0.5
    denorm = lambda x: (x + 1) / 2
    files = os.listdir(argspar.intest)
    time_test = []
    model.eval()
    # test
    for i in range(len(files)):
        haze = np.array(Image.open(argspar.intest + files[i]).convert('RGB')) / 255
        
        with torch.no_grad():
            haze = torch.Tensor(haze.transpose(2, 0, 1)[np.newaxis, :, :, :]).cuda()
            haze = tensor2cuda(haze)

            starttime = time.time()
            haze = norm(haze)
            out, att = model(haze)
            endtime1 = time.time()
            
            out = denorm(out)
            imwrite(out, argspar.outest + files[i], range=(0, 1))
            
            time_test.append(endtime1 - starttime)

            print('The ' + str(i) + ' Time: %.4f s.' % (endtime1 - starttime))
    print('Mean Time: %.4f s.'%(time_test/len(time_test)))


if __name__ == '__main__':
    main()
