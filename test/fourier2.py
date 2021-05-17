"""
if cuda:
    batch_x, batch_y = batch_yx[:, 0].cuda(), batch_yx[:, 1].cuda()
    for i in range(len(batch_yx)):
        Edge_orgn[i] = torch.clamp(
            toTensor(toPILImage(batch_x[i].squeeze(0).cpu()).filter(ImageFilter.EDGE_ENHANCE_MORE)), 0, 1)
        Edge_noise[i] = torch.clamp(
            toTensor(toPILImage(batch_y[i].squeeze(0).cpu()).filter(ImageFilter.GaussianBlur)), 0, 1)
    Edge_orgn = Edge_orgn.cuda()

"""





import argparse
import re
import os, glob, datetime, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import data_generator as dg
from data_generator import DenoisingDataset
from model import DecomposeNet
from loss import *
from utils import *
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from torchvision import transforms
from PIL import ImageFilter
from torch import rfft, irfft, fft, ifft
from torch_complex import ComplexTensor

# Params
parser = argparse.ArgumentParser(description='PyTorch DnCNN')
parser.add_argument('--model', default='DnCNN', type=str, help='choose a type of model')
parser.add_argument('--batch_size', default=64, type=int, help='batch size')
parser.add_argument('--train_data', default='../data/Train400', type=str, help='path of train data')
parser.add_argument('--save_dir', default='../path_upgrade/frequency', type=str, help='path of train data')
parser.add_argument('--sigma', default=25, type=int, help='noise level')
parser.add_argument('--epoch', default=180, type=int, help='number of train epoches')
parser.add_argument('--lr', default=1e-4, type=float, help='initial learning rate for Adam')
args = parser.parse_args()

batch_size = args.batch_size
cuda = torch.cuda.is_available()
n_epoch = args.epoch
sigma = args.sigma
save_dir = args.save_dir

toTensor = transforms.ToTensor()
toPILImage = transforms.ToPILImage()

# save_dir = os.path.join('models', args.model+'_' + 'sigma' + str(sigma))

if not os.path.exists(save_dir):
    os.mkdir(save_dir)

def DecompositionFuck(input, N):
    High_freq = rfft(input.unsqueeze(3)[0], signal_ndim=3)
    Low_freq = torch.FloatTensor(High_freq.shape).cuda()

    width, height = High_freq.shape[1], High_freq.shape[2]

    # for x in range(N//2, width - N //2):
    #     for y in range(N//2, height - N // 2):
    for x in range(N):
        for y in range(N):
            Low_freq[:,x,y] = High_freq[:,x,y]
            High_freq[:,x,y] = 0
    High_freq -= Low_freq
    # return High_freq, Low_freq
    return irfft(High_freq, signal_ndim=3).squeeze(3), irfft(Low_freq, signal_ndim=3).squeeze(3)

# def Decomposition(input, N):
#
#     input = input.cpu().detach().numpy()
#     High_freq = np.fft.fft2(input)
#     # High_freq = np.fft.fftshift(High_freq)
#
#     # High_freq = np.fft.fftshift(High_freq)
#     Low_freq = np.zeros(High_freq.shape)
#
#     width, height = High_freq.shape[1], High_freq.shape[2]
#     # for x in range(N//2, width - N //2):
#     #     for y in range(N//2, height - N // 2):
#
#     for x in range(int(width*N)):
#         for y in range(int(height*N)):
#             Low_freq[:, x, y] = High_freq[:, x, y]
#             # High_freq[:,x,y] = 0
#     High_freq -= Low_freq
#
#     High_output = torch.cat((torch.tensor(np.real(np.fft.ifft2(High_freq))).unsqueeze(1).float(),
#                                 torch.tensor(np.imag(np.fft.ifft2(High_freq))).unsqueeze(1).float()), dim=1)
#     Low_output = torch.cat((torch.tensor(np.real(np.fft.ifft2(Low_freq))).unsqueeze(1).float(),
#                               torch.tensor(np.imag(np.fft.ifft2(Low_freq))).unsqueeze(1).float()), dim=1)
#     temp = High_output[:,0]
#     output = ComplexTensor(High_output[:, 0]+Low_output[:, 0], High_output[:, 1]+Low_output[:, 1]).abs()
#     High_output = ComplexTensor(High_output[:, 0], High_output[:, 1]).abs()
#     Low_output = ComplexTensor(Low_output[:, 0], Low_output[:, 1]).abs()
#
#     fig = plt.figure()
#     gs = GridSpec(nrows=1, ncols=2)
#
#     plot1 = fig.add_subplot(gs[0, 0])
#     plot1.imshow(input[0],cmap = 'gray')
#     plot2 = fig.add_subplot(gs[0, 1])
#     plot2.imshow(output[0],cmap = 'gray')
#     plt.show()
#     plt.close()
#
#     return High_output, Low_output
#
#     # for i in range(len(Low_freq)):
#     #     fig = plt.figure()
#     #     gs = GridSpec(nrows=2, ncols=2)
#     #
#     #     plot1 = fig.add_subplot(gs[0, 0])
#     #     plot2 = fig.add_subplot(gs[0, 1])
#     #     plot3 = fig.add_subplot(gs[1, 0])
#     #     plot4 = fig.add_subplot(gs[1, 1])
#     #
#     #     plot1.axis('off')
#     #     plot2.axis('off')
#     #     plot3.axis('off')
#     #     plot4.axis('off')
#     #
#     #     plot1.imshow(np.real(Low_freq[i]), cmap='gray')
#     #     plot2.imshow(np.real(High_freq[i]), cmap='gray')
#     #     plot3.imshow(np.real(High_freq[i])+np.real(Low_freq[i]), cmap='gray')
#     #     plot4.imshow(fuck[i], cmap='gray')
#     #
#     #     plt.show()
#     #     plt.close()
#     #     pass
def Decomposition(input, end =0.125, start = 0.0):

    input = input.cpu().detach().numpy()
    High_freq = np.fft.fft2(input)
    High_freq = np.fft.fftshift(High_freq)
    Low_freq = np.zeros(High_freq.shape)
    Low_freq = np.fft.fft2(Low_freq)
    Low_freq = np.fft.fftshift(Low_freq)

    plt.figure()
    plt.imshow(np.log10(np.abs(High_freq[0])), cmap='gray')
    plt.show()

    width, height = High_freq.shape[1], High_freq.shape[2]
    centerx, centery = width//2, height//2
    for x in range(centerx-int(width*end/2), centerx+int(width*end/2)):
        for y in range(centery-int(height*end/2), centery+int(height*end/2)):
            Low_freq[:, x, y] = High_freq[:, x, y]
            High_freq[:,x,y] = 0

    plt.figure()
    plt.imshow(input[0], cmap='gray')
    plt.show()

    plt.figure()
    plt.imshow(np.abs(np.real(np.fft.ifft2(High_freq)[0])), cmap='gray')
    plt.show()

    plt.figure()
    plt.imshow(np.abs(np.imag(np.fft.ifft2(High_freq)[0])), cmap='gray')
    plt.show()

    plt.figure()
    plt.imshow(np.abs(np.real(np.fft.ifft2(Low_freq)[0])), cmap='gray')
    plt.show()

    plt.figure()
    plt.imshow(np.abs(np.imag(np.fft.ifft2(Low_freq)[0])), cmap='gray')
    plt.show()

    plt.figure()
    plt.imshow(input[0]/3+np.abs(np.real(np.fft.ifft2(Low_freq)[0])*2/3), cmap='gray')
    plt.show()

    High_freq = np.fft.ifftshift(High_freq)
    Low_freq = np.fft.ifftshift(Low_freq)

    High_output = torch.cat((torch.tensor(np.real(np.fft.ifft2(High_freq))).unsqueeze(1).float(),
                                torch.tensor(np.imag(np.fft.ifft2(High_freq))).unsqueeze(1).float()), dim=1)
    Low_output = torch.cat((torch.tensor(np.real(np.fft.ifft2(Low_freq))).unsqueeze(1).float(),
                              torch.tensor(np.imag(np.fft.ifft2(Low_freq))).unsqueeze(1).float()), dim=1)

    return High_output, Low_output


if __name__ == '__main__':
    # model selection
    print('===> Building model')

    initial_epoch = findLastCheckpoint(save_dir=save_dir)  # load the last model in matconvnet style
    if initial_epoch > 0:
        print('resuming by loading epoch %03d' % initial_epoch)
        # model.load_state_dict(torch.load(os.path.join(save_dir, 'model_%03d.pth' % initial_epoch)))
        # model = torch.load(os.path.join(save_dir, 'model_%03d.pth' % initial_epoch))
    # criterion = nn.MSELoss(reduction = 'sum')  # PyTorch 0.4.1
    # criterion = sum_squared_error()
    criterion = nn.MSELoss()
    High_frequency = torch.FloatTensor(args.batch_size, 40, 40, 1, 2)
    Low_frequency = torch.FloatTensor(args.batch_size, 40, 40, 1, 2)

    if cuda:
        # device_ids = [0]
        # model = nn.DataParallel(model, device_ids=device_ids).cuda()
        criterion = criterion.cuda()

    for epoch in range(initial_epoch, n_epoch):

        xs = dg.datagenerator(data_dir=args.train_data)
        xs = xs.astype('float32') / 255.0
        xs = torch.from_numpy(xs.transpose((0, 3, 1, 2)))  # tensor of the clean patches, NXCXHXW

        DDataset = DenoisingDataset(xs, sigma)
        batch_y, batch_x = DDataset[:238336]

        dataset = torch.cat((batch_x, batch_y),dim=1)
        DLoader = torch.utils.data.DataLoader(dataset=dataset, num_workers=0, drop_last=True, batch_size=batch_size, shuffle=True)
        epoch_loss = 0
        start_time = time.time()

        for n_count, batch_yx in enumerate(DLoader):
            if cuda:
                batch_x, batch_y = batch_yx[:,0].cuda(), batch_yx[:,1].cuda()
                temp = batch_x





                High_noise, Low_noise = Decomposition(batch_x, 0.125)
                High_noise, Low_noise = Decomposition(batch_y, 0.125)










                High_origin, Low_origin = Decomposition(batch_x, 6)

            batch_y = torch.cat([High_noise.unsqueeze(1), Low_noise.unsqueeze(1)], dim=1)
            batch_x = torch.cat([High_origin.unsqueeze(1), Low_origin.unsqueeze(1)], dim=1)

            High_noise = High_noise.unsqueeze(1)
            High_origin = High_origin.unsqueeze(1)

            # output = model(batch_y)

            # loss = criterion(torch.add(output[:,0], output[:,1]), torch.add(batch_x[:,0], batch_x[:,1]))

            # loss = criterion(output, batch_x)

            fig = plt.figure()
            gs = GridSpec(nrows=2, ncols=3)

            plot1 = fig.add_subplot(gs[0, 0])
            plot2 = fig.add_subplot(gs[0, 1])
            plot3 = fig.add_subplot(gs[0, 2])
            plot4 = fig.add_subplot(gs[1, 0])
            plot5 = fig.add_subplot(gs[1, 1])
            plot6 = fig.add_subplot(gs[1, 2])

            plot1.axis('off')
            plot2.axis('off')
            plot3.axis('off')
            plot4.axis('off')
            plot5.axis('off')
            plot6.axis('off')

            if n_count < 5:
                plot1.imshow(torch.add(batch_x[:,0], batch_x[:,1])[n_count].cpu(), cmap='gray')
                plot2.imshow(torch.add(batch_y[:,0], batch_y[:,1])[n_count].cpu(), cmap='gray')
                plot3.imshow(temp[n_count].cpu().squeeze(0), cmap='gray')
                plot4.imshow(batch_x[:, 0][n_count].cpu(), cmap='gray')
                plot5.imshow(batch_y[:, 0][n_count].cpu(), cmap='gray')

                # plot3.imshow(torch.add(output[:,0], output[:,1])[0].cpu().detach().numpy(), cmap='gray')
                # plot5.imshow(batch_y[0].cpu().squeeze(0)-output[0].cpu().detach().numpy().squeeze(0), cmap='gray')

                plt.show()


            # epoch_loss += loss.item()
            # loss.backward()
            optimizer.step()
            # if n_count % 10 == 0:
            #     print('%4d %4d / %4d loss = %2.4f' % (
            #     epoch + 1, n_count, xs.size(0) // batch_size, loss.item() / batch_size))
        elapsed_time = time.time() - start_time

        log('epcoh = %4d , loss = %4.4f , time = %4.2f s' % (epoch + 1, epoch_loss / n_count, elapsed_time))
        np.savetxt('train_result.txt', np.hstack((epoch + 1, epoch_loss / n_count, elapsed_time)), fmt='%2.4f')
        # torch.save(model.state_dict(), os.path.join(save_dir, 'model_%03d.pth' % (epoch+1)))
        # torch.save(model.state_dict(), os.path.join(save_dir, 'model_%03d.pth' % (epoch + 1)))
