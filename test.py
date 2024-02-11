import os, time, argparse
import numpy as np
from PIL import Image
import glob
import torch
from torchvision.utils import save_image as imwrite

from utils import load_checkpoint, tensor2cuda

from model.models import Generator
import cv2

# 调用GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def addTransparency(img, factor=1):
    img = img.convert('RGBA')
    img_blender = Image.new('RGBA', img.size, (0, 0, 0, 0))
    img = Image.blend(img_blender, img, factor)
    return img


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
    Gmodel_name = name + '.tar'
    Dmodel_name = name + '.tar'
    G_Model, _, _ = load_checkpoint(argspar.model, Generator, Gmodel_name, arg)

    os.makedirs(arg.outest, exist_ok=True)
    test(argspar, G_Model)


xishu = 0.75


def test(argspar, model):
    # init
    norm = lambda x: (x - 0.5) / 0.5
    denorm = lambda x: (x + 1) / 2
    files = os.listdir(argspar.intest)
    time_test = []
    model.eval()
    # test
    for i in range(len(files)):
        haze = Image.open(argspar.intest + files[i])
        x = haze.width
        y = haze.height
        haze = haze.resize((int(x * xishu), (int(y * xishu))))
        print(x * xishu, y * xishu)
        haze = np.array(haze.convert('RGB')) / 255

        with torch.no_grad():
            haze = torch.Tensor(haze.transpose(2, 0, 1)[np.newaxis, :, :, :]).cuda()
            haze = tensor2cuda(haze)

            starttime = time.time()
            haze = norm(haze)
            out, att = model(haze)
            endtime1 = time.time()

            out = denorm(out)

            # out = out.resize((int(x), (int(y))))
            # out = cv2.resize(out, (x, y))
            imwrite(out, argspar.outest + files[i], value_range=(0, 1))

            time_test.append(endtime1 - starttime)

            print('The ' + str(i) + ' Time: %.4f s.' % (endtime1 - starttime))

    # print('Mean Time: %.4f s.'%(time_test/len(time_test)))

    path = './output/*.png'
    for i in glob.glob(path):
        im1 = Image.open(i)
        im = Image.open(argspar.intest + files[0])
        x = im1.width
        y = im1.height
        im1 = im1.resize((int(x / xishu), (int(y / xishu))))

        if len(im.split()) == 4:
            im2 = addTransparency(im1)
            im2.save(os.path.join('./output/', os.path.basename(i)))


if __name__ == '__main__':
    main()
