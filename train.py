import os, time, argparse, cv2, h5py, math
from PIL import Image
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16
from torch.utils.data import DataLoader
from torchvision.utils import save_image as imwrite

from data.makedataset import Dataset
from utils import adjust_learning_rate, gen_att, load_checkpoint, tensor_metric, save_checkpoint, tensor2cuda, load_excel

from model.loss import msssim, LossNetwork
from model.models import Generator, Discriminator

# 调用GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# 
def main():
    '''
    定义开关和输出

    '''
    parser = argparse.ArgumentParser(description="network pytorch")
    # train
    parser.add_argument("--epoch", type=int, default=240, help='epoch number')
    parser.add_argument("--start_epoch", type=int, default=0, help='start epoch')
    parser.add_argument("--test_epoch", type=int, default=2, help='test epoch')
    parser.add_argument("--bs", type=str, default=2, help='batchsize')
    parser.add_argument("--lr_up", type=str, default=0.5, help='learning rate up')
    parser.add_argument("--lr", type=str, default=1e-3, help='learning rate')
    
    
    parser.add_argument("--model", type=str, default="./checkpoint/", help='checkpoint')
    parser.add_argument("--model_name", type=str, default='184_21.9692_0.7242', help='model name')
    parser.add_argument("--data", type=str, default="./data/NH-Haze20-21-23.h5", help='data')
    # value
    parser.add_argument("--test", type=str, default="./data/test/", help='input syn path')

    parser.add_argument("--out", type=str, default="./result/", help='output syn path')
    argspar = parser.parse_args()

    print("\nnetwork pytorch")
    for p, v in zip(argspar.__dict__.keys(), argspar.__dict__.values()):
        print('\t{}: {}'.format(p, v))
    print('\n')
    arg = parser.parse_args()
    
    print('> Loading dataset...')
    dataset = DataLoader(dataset=Dataset(argspar.data), num_workers=0, batch_size=argspar.bs, shuffle=True)
    
    print('> Loading Generator...')
    name = arg.model_name
    Gmodel_name = 'Gmodel_'+name+'.tar'
    Dmodel_name = 'Dmodel_'+name+'.tar'
    G_Model, G_optimizer, cur_epoch = load_checkpoint(argspar.model, Generator, Gmodel_name, arg)
    D_Model, D_optimizer,_ = load_checkpoint(argspar.model, Discriminator, Dmodel_name, arg)
    cur_epoch = arg.start_epoch
    
    print('> Start training...')
    start_all = time.clock()
    train(G_Model, G_optimizer, D_Model, D_optimizer, cur_epoch, arg, dataset)
    end_all = time.clock()

    print('Whloe Training Time:' + str(end_all - start_all) + 's.')

def train(G_Model, G_optimizer, D_Model, D_optimizer, cur_epoch, argspar, dataset):
    # loss
    
    ### vgg
    vgg_model = vgg16(pretrained=True)
    vgg_model = vgg_model.features[:16].cuda()
    for param in vgg_model.parameters():
        param.requires_grad = False
    loss_network = LossNetwork(vgg_model)
    loss_network.eval()
    
    ### ms-ssim
    msssim_loss = msssim
    
    ### GAN
    adversarial_loss = nn.BCEWithLogitsLoss().cuda()
    
    metric = [['PSNR', 'SSIM']]
    # train
    First = True
    norm = lambda x: (x - 0.5) / 0.5
    denorm = lambda x: (x+1)/2
    
    for epoch in range(cur_epoch, argspar.epoch):
        G_optimizer = adjust_learning_rate(G_optimizer, epoch, argspar.epoch//6, argspar.lr_up)
        D_optimizer = adjust_learning_rate(D_optimizer, epoch, argspar.epoch//6, argspar.lr_up)
        learnrate = G_optimizer.param_groups[-1]['lr']
        G_Model.eval()
        D_Model.train()

        # 合成图像训练
        for i, data in enumerate(dataset, 0):
            clear, haze = data[:, :3, :, :], data[:, 3:, :, :]
            clear, haze = tensor2cuda(clear), tensor2cuda(haze)
            
            clear, haze = norm(clear), norm(haze)
            
            m_gt = gen_att(haze,clear)
            
            if First:
                mask_loss = 0.1
                First = False
            else:
                mask_loss = float(l_mask) if epoch<argspar.epoch//5 else 0
            
            out, m_g = G_Model(haze, m_gt, mask_loss)
            
            D_Model.zero_grad()
            real_out = D_Model(clear).mean()
            fake_out = D_Model(out).mean()
            
            D_loss = 1 - real_out + fake_out
            D_loss.backward(retain_graph=True)
            
            G_Model.zero_grad()
            clear, out = denorm(clear), denorm(out)
            
            l_pixel = F.smooth_l1_loss(out, clear)
            l_mask = F.smooth_l1_loss(m_g, m_gt)
            perceptual_loss = loss_network(out, clear)
            msssim_loss_ = 1-msssim_loss(out, clear, normalize=True)
            a_loss = 1-fake_out
            
            total_loss =    1*l_pixel +\
                            0.3*l_mask +\
                            0.01*perceptual_loss +\
                            0.5*msssim_loss_ + \
                            0.0005*a_loss

            total_loss.backward()
            G_optimizer.step()
            D_optimizer.step()

            mse = tensor_metric(clear, out, 'MSE', data_range=1)
            psnr = tensor_metric(clear, out, 'PSNR', data_range=1)
            ssim = tensor_metric(clear, out, 'SSIM', data_range=1)
            print("[epoch %d][%d/%d] lr: %f total loss: %.4f m_loss: %.4f a_loss: %.4F MSE: %.4f PSNR: %.4f SSIM: %.4f"\
                  % (epoch + 1, i + 1, len(dataset), learnrate, \
                     total_loss.item(), l_mask, a_loss, mse, psnr, ssim))

        if (epoch + 1) % argspar.test_epoch == 0:
            psnr_t1, ssim_t1 = test(argspar, G_Model, epoch)
            metric.append([psnr_t1, ssim_t1])

            print("[epoch %d] Test images PSNR1: %.4f SSIM1: %.4f" % (epoch + 1, psnr_t1, ssim_t1))
            load_excel(metric)
            # 保存模型
            save_checkpoint({'epoch': epoch + 1, 'state_dict': G_Model.state_dict(),\
                'optimizer': G_optimizer.state_dict()}, argspar.model, 'Gmodel',\
                    epoch + 1, psnr_t1, ssim_t1)
            save_checkpoint({'epoch': epoch + 1, 'state_dict': D_Model.state_dict(),\
                'optimizer': D_optimizer.state_dict()}, argspar.model, 'Dmodel',\
                    epoch + 1, psnr_t1, ssim_t1)

def test(argspar, model, epoch=-1):
    # img_name_list
    files_clear = os.listdir(argspar.test + '/clear/')
    
    # init
    psnr, ssim = 0, 0
    norm = lambda x: (x - 0.5) / 0.5
    denorm = lambda x: (x+1)/2
    
    # test
    for i in range(len(files_clear)):
        # read img
        clear = np.array(Image.open(argspar.test + '/clear/' + files_clear[i])) / 255
        haze = np.array(Image.open(argspar.test + '/hazy/' + files_clear[i])) / 255
        
        model.eval()
        with torch.no_grad():
            haze = torch.Tensor(haze.transpose(2, 0, 1)[np.newaxis, :, :, :])
            clear = torch.Tensor(clear.transpose(2, 0, 1)[np.newaxis, :, :, :])
            clear, haze = tensor2cuda(clear), tensor2cuda(haze)
            
            clear, haze = norm(clear), norm(haze)

            m_gt = gen_att(haze, clear)
            
            starttime = time.time()
            out, m_g = model(haze)
            endtime1 = time.time()
            
            haze, clear, out = denorm(haze), denorm(clear), denorm(out)
            m_gt, m_g = torch.cat((m_gt, m_gt, m_gt), dim=1),torch.cat((m_g, m_g, m_g), dim=1)
            
            imwrite(torch.cat((haze, clear, m_g, m_gt, out), dim=3), argspar.out \
                    + files_clear[i][:-4] + '_' + str(epoch + 1) + '.png', range=(0, 1))
            
            psnr += tensor_metric(clear,out, 'PSNR', data_range=1)
            ssim += tensor_metric(clear,out, 'SSIM', data_range=1)
            print('The %s Time: %.5f s.' % (files_clear[i][:-4], endtime1-starttime))

    return psnr / (len(files_clear)), ssim / (len(files_clear))


if __name__ == '__main__':
    main()
