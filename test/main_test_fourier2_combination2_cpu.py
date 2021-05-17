# -*- coding: utf-8 -*-

# =============================================================================
#  @article{zhang2017beyond,
#    title={Beyond a {Gaussian} denoiser: Residual learning of deep {CNN} for image denoising},
#    author={Zhang, Kai and Zuo, Wangmeng and Chen, Yunjin and Meng, Deyu and Zhang, Lei},
#    journal={IEEE Transactions on Image Processing},
#    year={2017},
#    volume={26},
#    number={7},
#    pages={3142-3155},
#  }
# by Kai Zhang (08/2018)
# cskaizhang@gmail.com
# https://github.com/cszn
# modified on the code from https://github.com/SaoYan/DnCNN-PyTorch
# =============================================================================

# run this to test the model

import argparse
import os, time, datetime
# import PIL.Image as Image
import numpy as np
import torch.nn as nn
import torch.nn.init as init
import torch
from skimage.measure import compare_psnr, compare_ssim
from skimage.io import imread, imsave
from model import Encoder1, DnCNN, Encoder2, NoBatchNorm, DecomposeNet
# from utils import Decomposition
from torch import rfft, irfft, fft, ifft
import matplotlib.pyplot as plt
from torch_complex.tensor import ComplexTensor

def Decomposition(input, N):

    input = input.cpu().detach().numpy()
    High_freq = np.fft.fft2(input)
    # High_freq = np.fft.fftshift(High_freq)
    Low_freq = np.zeros(High_freq.shape)

    width, height = High_freq.shape[1], High_freq.shape[2]
    # for x in range(N//2, width - N //2):
    #     for y in range(N//2, height - N // 2):


    for x in range(int(width*N)):
        for y in range(int(height*N)):
            Low_freq[:, x, y] = High_freq[:, x, y]
            # High_freq[:,x,y] = 0
    High_freq -= Low_freq

    # plt.figure()
    # plt.imshow(np.abs(np.fft.ifft2(High_freq)[0]), cmap='jet')
    # plt.show()
    #
    # plt.figure()
    # plt.imshow(np.abs(np.fft.ifft2(Low_freq)[0]), cmap='jet')
    # plt.show()
    # pass




    High_output = torch.cat((torch.tensor(np.real(np.fft.ifft2(High_freq))).unsqueeze(1).float(),
                                torch.tensor(np.imag(np.fft.ifft2(High_freq))).unsqueeze(1).float()), dim=1)
    Low_output = torch.cat((torch.tensor(np.real(np.fft.ifft2(Low_freq))).unsqueeze(1).float(),
                              torch.tensor(np.imag(np.fft.ifft2(Low_freq))).unsqueeze(1).float()), dim=1)


    return High_output, Low_output

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--set_dir', default='../data/Test', type=str, help='directory of test dataset')
    parser.add_argument('--set_names', default=['Set12', 'Set68'], help='directory of test dataset')
    parser.add_argument('--sigma', default=25, type=int, help='noise level')
    # parser.add_argument('--model_dir', default=os.path.join('../path'), help='directory of the model')
    # parser.add_argument('--model_name', default='model.pth', type=str, help='the model name')
    parser.add_argument('--model_dir', default=os.path.join('../path_upgrade/dncnn_onechannel'), help='directory of the model')
    parser.add_argument('--model_name', default='model_095.pth', type=str, help='the model name')

    parser.add_argument('--decom_model_dir', default=os.path.join('../path_upgrade/frequency_sep2'), help='directory of the model')
    parser.add_argument('--decom_model_name', default='com_model_129.pth', type=str, help='the model name')
    parser.add_argument('--result_dir', default='results', type=str, help='directory of test dataset')
    parser.add_argument('--save_result', default=1, type=int, help='save the denoised image, 1 or 0')
    return parser.parse_args()


def log(*args, **kwargs):
     print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:"), *args, **kwargs)


def save_result(result, path):
    path = path if path.find('.') != -1 else path+'.png'
    ext = os.path.splitext(path)[-1]
    if ext in ('.txt', '.dlm'):
        np.savetxt(path, result, fmt='%2.4f')
    else:
        imsave(path, np.clip(result, 0, 1))


def show(x, title=None, cbar=False, figsize=None):
    import matplotlib.pyplot as plt
    plt.figure(figsize=figsize)
    plt.imshow(x, interpolation='nearest', cmap='jet')
    if title:
        plt.title(title)
    if cbar:
        plt.colorbar()
    plt.show()



if __name__ == '__main__':

    args = parse_args()

    model = DnCNN()
    decompose_model = DecomposeNet(image_channels=5, n_block=32)

    if not os.path.exists(os.path.join(args.decom_model_dir, args.decom_model_name)):
        decompose_model = torch.load(os.path.join(args.model_dir, 'model.pth'))
        # load weights into new model
        log('load trained model on Train400 dataset by kai')
    else:
        # model = torch.load(os.path.join(args.model_dir, args.model_name))
        model.load_state_dict(torch.load(os.path.join(args.model_dir, args.model_name)))
        decompose_model.load_state_dict(torch.load(os.path.join(args.decom_model_dir, args.decom_model_name)))
        # model = torch.load(os.path.join(args.model_dir, args.model_name))
        log('load trained model')

#    params = model.state_dict()
#    print(params.values())
#    print(params.keys())
#
#    for key, value in params.items():
#        print(key)    # parameter name
#    print(params['dncnn.12.running_mean'])
#    print(model.state_dict())
    model.eval()
    decompose_model.eval()  # evaluation mode
#    model.train()


    if not os.path.exists(args.result_dir):
        os.mkdir(args.result_dir)

    for set_cur in args.set_names:

        if not os.path.exists(os.path.join(args.result_dir, set_cur)):
            os.mkdir(os.path.join(args.result_dir, set_cur))
        psnrs = []
        ssims = []

        for im in os.listdir(os.path.join(args.set_dir, set_cur)):
            if im.endswith(".jpg") or im.endswith(".bmp") or im.endswith(".png"):

                x = np.array(imread(os.path.join(args.set_dir, set_cur, im)), dtype=np.float32)/255.0
                np.random.seed(seed=0)  # for reproducibility
                y = x + np.random.normal(0, args.sigma/255.0, x.shape)  # Add Gaussian noise without clipping
                y = y.astype(np.float32)
                y_ = torch.from_numpy(y).view(1, -1, y.shape[0], y.shape[1])


                start_time = time.time()
                #  0.14
                High_noise, Low_noise = Decomposition(y_.squeeze(0), 0.10)

                # dncnn_data = High_noise[:,0].unsqueeze(0)
                decompose_data = torch.cat([y_,High_noise, Low_noise], dim=1)
                # x = torch.cat([High_origin.unsqueeze(1), Low_origin.unsqueeze(1)], dim=1)
                model = model.cpu()
                decom_output = decompose_model(decompose_data.float()).squeeze(0) # inference


                dncnn_output = model(High_noise[:,0].unsqueeze(1)).squeeze(0)
                # dncnn_high, dncnn_low = Decomposition(dncnn_output, 0.10)

                # x_ = output[0].cpu().detach().numpy().astype(np.float32)

                output = ComplexTensor(dncnn_output[0] + decom_output[3], decom_output[2] + decom_output[4]).abs()
                # output = ComplexTensor(dncnn_high[0,0] + decom_output[3], decom_output[2] + decom_output[4]).abs()
                x_ = output.cpu().detach().numpy().astype(np.float32)

                # x_ = ComplexTensor(output[:, 0] + output[:, 2], Low_noise[:, 0] + Low_noise[:, 1]).abs().squeeze(0)
                # x_ = ComplexTensor(output[:, 0] + output[:, 2], output[:, 1] + output[:, 3]).abs().squeeze(0)
                # x_ = torch.add(output[:,0], output[:,1]).squeeze(0)

                # x_ = x_.cpu().detach().numpy().astype(np.float32)

                # x_ = x_.view(y.shape[0], y.shape[1])
                # x_ = x_.cpu()
                # x_ = x_.detach().numpy().astype(np.float32)
                elapsed_time = time.time() - start_time

                psnr_x_ = compare_psnr(x, x_)
                ssim_x_ = compare_ssim(x, x_)

                print('%10s : %10s : %2.4f second %2.2f PSNR' % (set_cur, im, elapsed_time, psnr_x_))

                if args.save_result:
                    name, ext = os.path.splitext(im)
                    show(np.hstack((y, x_)))  # show the image
                    save_result(x_, path=os.path.join(args.result_dir, set_cur, name+'_dncnn'+ext))  # save the denoised image
                psnrs.append(psnr_x_)
                ssims.append(ssim_x_)
        psnr_avg = np.mean(psnrs)
        ssim_avg = np.mean(ssims)
        psnrs.append(psnr_avg)
        ssims.append(ssim_avg)
        if args.save_result:
            save_result(np.hstack((psnrs, ssims)), path=os.path.join(args.result_dir, set_cur, 'results.txt'))
        log('Datset: {0:10s} \n  PSNR = {1:2.2f}dB, SSIM = {2:1.4f}'.format(set_cur, psnr_avg, ssim_avg))








