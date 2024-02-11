import os
import pandas as pd
import numpy as np

import torch
from torch.autograd import Variable

from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import mean_squared_error as compare_mse

def adjust_learning_rate(optimizer, epoch, lr_update_freq, rate):
    if not epoch % lr_update_freq and epoch:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * rate
    return optimizer

def gen_att(haze,clear):
    r = haze[:, 0, :, :].unsqueeze(1)
    g = haze[:, 1, :, :].unsqueeze(1)
    b = haze[:, 2, :, :].unsqueeze(1)
    Y = 0.299 * r + 0.587 * g + 0.144 * b
    r_clear = clear[:, 0, :, :].unsqueeze(1)
    g_clear = clear[:, 1, :, :].unsqueeze(1)
    b_clear = clear[:, 2, :, :].unsqueeze(1)
    Y_clear = 0.299 * r_clear + 0.587 * g_clear + 0.144 * b_clear
    m_g = Y - Y_clear
    m_g_max = torch.max(torch.max(m_g,2).values,2).values.unsqueeze(-1).unsqueeze(-1)+1e-6
    m_g_min = torch.min(torch.min(m_g,2).values,2).values.unsqueeze(-1).unsqueeze(-1)
    m_g_l = (m_g- m_g_min)/(m_g_max-m_g_min)
    # s = haze - clear
    return m_g_l

# 加载模型
def load_checkpoint(checkpoint_dir, Model, name, arg, learnrate=None):
    if os.path.exists(checkpoint_dir + name):
        # 加载存在的模型
        model_info = torch.load(checkpoint_dir + name)
        print('==> loading existing model:', checkpoint_dir + name)
        # 模型名称G
        model = Model()
        # 显卡使用
        device_ids = [0]
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        model = torch.nn.DataParallel(model, device_ids=device_ids).cuda()
        # 将模型参数赋值进net
        model.load_state_dict(model_info['state_dict'])
        optimizer = torch.optim.Adam(model.parameters())
        optimizer.load_state_dict(model_info['optimizer'])
        if learnrate!=None:
            for param_group in optimizer.param_groups:
                param_group['lr'] = learnrate
        cur_epoch = model_info['epoch']

    else:
        # 创建模型
        model = Model()
        # 显卡使用
        device_ids = [0]
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        try:
            optimizer = torch.optim.Adam(model.parameters(), lr=arg.lr)
        except:
            print('Must input learnrate!')
        model = torch.nn.DataParallel(model, device_ids=device_ids).cuda()
        cur_epoch = 0
    return model, optimizer, cur_epoch


def tensor_metric(img, imclean, model, data_range=1):#计算图像PSNR输入为Tensor

    img_cpu = img.data.cpu().numpy().astype(np.float32).transpose(0,2,3,1)
    imgclean = imclean.data.cpu().numpy().astype(np.float32).transpose(0,2,3,1)
    
    SUM = 0
    for i in range(img_cpu.shape[0]):
        if model == 'PSNR':
            SUM += compare_psnr(imgclean[i, :, :, :], img_cpu[i, :, :, :],data_range=data_range)
        elif model == 'MSE':
            SUM += compare_mse(imgclean[i, :, :, :], img_cpu[i, :, :, :])
        elif model == 'SSIM':
            SUM += compare_ssim(imgclean[i, :, :, :], img_cpu[i, :, :, :], data_range=data_range, multichannel = True)
        else:
            print('Model False!')
        
    return SUM/img_cpu.shape[0]

def save_checkpoint(state, checkpoint, name, epoch=0, psnr=0, ssim=0, i = None):#保存学习率
    if i is None:
        torch.save(state, checkpoint + name + '_%d_%.4f_%.4f.tar'%(epoch, psnr, ssim))
    else:
        torch.save(state, checkpoint + name + '_%d_%d_%.4f_%.4f.tar'%(epoch, i, psnr, ssim))

def tensor2cuda(img):
    #图像导入
    with torch.no_grad():
        img = Variable(img.cuda(),requires_grad=True)
    return  img #输出雾气、清晰图像

def load_excel(x):
    data = pd.DataFrame(x)
    os.makedirs('log', exist_ok=True)
    writer = pd.ExcelWriter('./log/Metric.xlsx')		# 写入Excel文件
    data.to_excel(writer, 'PSNR-SSIM', float_format='%.5f')		# ‘page_1’是写入excel的sheet名
    writer.save()
    writer.close()
