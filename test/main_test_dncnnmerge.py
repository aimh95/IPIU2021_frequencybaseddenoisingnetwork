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

import matplotlib.pyplot as plt

def Decomposition(input, end =0.6, start = 0.0):

    input = input.cpu().detach().numpy()
    High_freq = np.fft.fft2(input)
    High_freq = np.fft.fftshift(High_freq)
    Low_freq = np.zeros(High_freq.shape)
    Low_freq = np.fft.fft2(Low_freq)
    Low_freq = np.fft.fftshift(Low_freq)

    # plt.figure()
    # plt.imshow(np.abs(High_freq[0]), cmap='jet')
    # plt.show()

    width, height = High_freq.shape[1], High_freq.shape[2]
    centerx, centery = width//2, height//2
    for x in range(centerx-int(width*end/2), centerx+int(width*end/2)):
        for y in range(centery-int(height*end/2), centery+int(height*end/2)):
            Low_freq[:, x, y] = High_freq[:, x, y]
            High_freq[:,x,y] = 0

    High_output = torch.tensor(np.abs((np.fft.ifft2(High_freq))))
    Low_output = torch.tensor(np.abs((np.fft.ifft2(Low_freq))))


    return High_output, Low_output

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--set_dir', default='../data/Test', type=str, help='directory of test dataset')
    parser.add_argument('--set_names', default=['Set12', 'Set68'], help='directory of test dataset')
    parser.add_argument('--sigma', default=25, type=int, help='noise level')
    parser.add_argument('--low_model_dir', default=os.path.join('../path_upgrade/edge_low_freq_final'), help='directory of the model')
    parser.add_argument('--low_model_name', default='model_070.pth', type=str, help='the model name')
    parser.add_argument('--res_model_dir', default='../path_upgrade/edge_low_merge_final', type=str,
                        help='path of train data')
    parser.add_argument('--res_model_name', default='model_077.pth', type=str, help='the model name')
    parser.add_argument('--model_dir', default=os.path.join('../path'), help='directory of the model')
    parser.add_argument('--model_name', default='model.pth', type=str, help='the model name')

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



    low_model = DnCNN()
    res_model = DnCNN()
    model = DnCNN()


    if not os.path.exists(os.path.join(args.low_model_dir, args.low_model_name)):
        low_model = torch.load(os.path.join(args.low_model_dir, 'model.pth'))
        res_model = torch.load(os.path.join(args.res_model_dir, 'model.pth'))
        model = torch.load(os.path.join(args.model_dir, 'model.pth'))
        # load weights into new model
        log('load trained model on Train400 dataset by kai')
    else:
        low_model.load_state_dict(torch.load(os.path.join(args.low_model_dir, args.low_model_name)))
        res_model.load_state_dict(torch.load(os.path.join(args.res_model_dir, args.res_model_name)))
        # model.load_state_dict(torch.load(os.path.join(args.model_dir, args.model_name)))
        model = torch.load(os.path.join(args.model_dir, args.model_name))
        log('load trained model')

#    params = model.state_dict()
#    print(params.values())
#    print(params.keys())
#
#    for key, value in params.items():
#        print(key)    # parameter name
#    print(params['dncnn.12.running_mean'])
#    print(model.state_dict())

    low_model.eval()  # evaluation mode
    res_model.eval()
    model.eval()

    if torch.cuda.is_available():
        low_model = low_model.cuda()
        res_model = res_model.cuda()
        model = model.cuda()
          # evaluation mode

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
                temp = torch.from_numpy(x)
                np.random.seed(seed=0)  # for reproducibility
                y = x + np.random.normal(0, args.sigma/255.0, x.shape)  # Add Gaussian noise without clipping
                y = y.astype(np.float32)
                y_ = torch.from_numpy(y).view(1, -1, y.shape[0], y.shape[1])

                High_origin, Low_origin = Decomposition(torch.from_numpy(x).unsqueeze(0))
                High_noise, Low_noise = Decomposition(y_.squeeze(0))

                torch.cuda.synchronize()
                start_time = time.time()
                y_ = y_.cuda()
                dncnn = model(y_)

                high_, Low_recon = Decomposition(dncnn.squeeze(0))

                low_ = low_model(Low_noise.unsqueeze(1).float().cuda())  # inference

                y_ = y_ - low_

                plt.figure()
                plt.imshow(y_[0,0].cpu().detach().numpy(), cmap='jet')
                plt.show()
                plt.close()


                plt.figure()
                plt.imshow(x-low_[0,0].cpu().detach().numpy(), cmap='jet')
                plt.show()
                plt.close()

                x_ = dncnn[0,0].cpu() - Low_recon[0,0].cpu() + low_.cpu()

                plt.figure()
                plt.imshow(x, cmap='jet')
                plt.show()
                plt.close()

                plt.figure()
                plt.imshow(low_[0,0].cpu().detach().numpy(), cmap='jet')
                plt.show()
                plt.close()

                plt.figure()
                plt.imshow(x_[0,0].cpu().detach().numpy(), cmap='jet')
                plt.show()
                plt.close()


                torch.cuda.synchronize()
                elapsed_time = time.time() - start_time
                # print('%10s : %10s : %2.4f second' % (set_cur, im, elapsed_time))

                x_ = x_.view(y.shape[0], y.shape[1])
                x_ = x_.cpu()
                x_ = x_.detach().numpy().astype(np.float32)

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







