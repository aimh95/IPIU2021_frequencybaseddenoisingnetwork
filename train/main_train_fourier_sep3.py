
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
from model import DecomposeNet, DecomposeUNet
from loss import *
from utils import *
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from torch_complex.tensor import ComplexTensor


# Params
parser = argparse.ArgumentParser(description='PyTorch DnCNN')
parser.add_argument('--model', default='DnCNN', type=str, help='choose a type of model')
parser.add_argument('--batch_size', default=126, type=int, help='batch size')
parser.add_argument('--train_data', default='../data/Train400', type=str, help='path of train data')
parser.add_argument('--save_dir', default='../path_upgrade/frequency_sep3', type=str, help='path of train data')
parser.add_argument('--sigma', default=25, type=int, help='noise level')
parser.add_argument('--epoch', default=180, type=int, help='number of train epoches')
parser.add_argument('--lr', default=1e-3, type=float, help='initial learning rate for Adam')
args = parser.parse_args()

batch_size = args.batch_size
cuda = torch.cuda.is_available()
n_epoch = args.epoch
sigma = args.sigma
save_dir = args.save_dir

# save_dir = os.path.join('models', args.model+'_' + 'sigma' + str(sigma))

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

    High_output = torch.cat((torch.tensor(np.real(np.fft.ifft2(High_freq))).unsqueeze(1).float(),
                                torch.tensor(np.imag(np.fft.ifft2(High_freq))).unsqueeze(1).float()), dim=1)
    Low_output = torch.cat((torch.tensor(np.real(np.fft.ifft2(Low_freq))).unsqueeze(1).float(),
                              torch.tensor(np.imag(np.fft.ifft2(Low_freq))).unsqueeze(1).float()), dim=1)


    return High_output, Low_output


if not os.path.exists(save_dir):
    os.mkdir(save_dir)

if __name__ == '__main__':
    # model selection
    print('===> Building model')
    model = DecomposeNet(image_channels=4, n_block=32)
    # model_lf = DecomposeNet(image_channels=2, n_block=32)

    initial_epoch = findLastCheckpoint(save_dir=save_dir)  # load the last model in matconvnet style

    if initial_epoch > 0:
        print('resuming by loading epoch %03d' % initial_epoch)
        model.load_state_dict(torch.load(os.path.join(save_dir, 'model_%03d.pth' % initial_epoch)))

        # model_hf = torch.load(os.path.join(save_dir, 'high_model_%03d.pth' % initial_epoch))
        # model_lf = torch.load(os.path.join(save_dir, 'low_model_%03d.pth' % initial_epoch))
    model.train()
    # criterion = nn.MSELoss(reduction = 'sum')  # PyTorch 0.4.1
    # criterion = sum_squared_error()
    criterion = nn.MSELoss()

    if cuda:
        model = model.cuda()
        # device_ids = [0]
        # model = nn.DataParallel(model, device_ids=device_ids).cuda()
        # criterion = criterion.cuda()

    optimizer_hf = optim.Adam(model.parameters(), lr=args.lr)
    scheduler_hf = MultiStepLR(optimizer_hf, milestones=[30, 60, 90], gamma=0.3)  # learning rates

    for epoch in range(initial_epoch, n_epoch):

        scheduler_hf.step(epoch)  # step to the learning rate in this epcoh

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
            optimizer_hf.zero_grad()
            if cuda:
                batch_x, batch_y = batch_yx[:,0].cuda(), batch_yx[:,1].cuda()
                temp = batch_x
                High_noise, Low_noise = Decomposition(batch_y, 0.125)
                High_origin, Low_origin = Decomposition(batch_x, 0.125)

            # batch_y = batch_y.unsqueeze(1)
            # batch_x = batch_x.unsqueeze(1)
            #
            batch_y = torch.cat([High_noise, Low_noise], dim=1)
            batch_x = torch.cat([High_origin, Low_origin], dim=1)

            output = model(batch_y.cuda())

            loss_hf = criterion(output.cpu(), batch_x)



            output_hf = ComplexTensor(output[:,0], output[:,1]).abs()
            output_lf = ComplexTensor(output[:, 2], output[:, 3]).abs()

            final_output = ComplexTensor(output[:,0] + output[:,2], output[:,1] + output[:,3]).abs()

            High_noise = ComplexTensor(batch_y[:, 0], batch_y[:, 1]).abs()
            High_origin = ComplexTensor(batch_x[:, 0], batch_x[:, 1]).abs()

            Low_noise = ComplexTensor(batch_y[:, 2], batch_y[:, 3]).abs()
            Low_origin = ComplexTensor(batch_x[:, 2], batch_x[:, 3]).abs()

            fig = plt.figure()
            gs = GridSpec(nrows=2, ncols=4)

            highfreq1 = fig.add_subplot(gs[0, 0])
            highfreq2 = fig.add_subplot(gs[0, 1])
            highfreq3 = fig.add_subplot(gs[0, 2])
            output1 = fig.add_subplot(gs[0, 3])
            lowfreq1 = fig.add_subplot(gs[1, 0])
            lowfreq2 = fig.add_subplot(gs[1, 1])
            lowfreq3 = fig.add_subplot(gs[1, 2])
            output2 = fig.add_subplot(gs[1, 3])

            highfreq1.axis('off')
            highfreq2.axis('off')
            highfreq3.axis('off')
            output1.axis('off')
            lowfreq1.axis('off')
            lowfreq2.axis('off')
            lowfreq3.axis('off')
            output2.axis('off')

            if n_count % 500 == 0:
                highfreq1.imshow(High_noise[0].cpu(), cmap='jet')
                highfreq2.imshow(High_origin[0].cpu(), cmap='jet')
                highfreq3.imshow(output_hf[0].cpu().detach().numpy(), cmap='jet')
                output1.imshow(temp[0].cpu(), cmap='jet')
                lowfreq1.imshow(Low_noise[0].cpu(), cmap='jet')
                lowfreq2.imshow(Low_origin[0].cpu(), cmap='jet')
                lowfreq3.imshow(output_lf[0].cpu().detach().numpy(), cmap='jet')
                output2.imshow(final_output[0].cpu().detach().numpy(), cmap='jet')
                plt.show()



            loss_hf.backward()
            optimizer_hf.step()

            loss = criterion(final_output, temp)

            epoch_loss += loss.item()

            if n_count % 10 == 0:
                print('%4d %4d / %4d loss = %2.8f loss_hf = %2.8f' % (
                epoch + 1, n_count, xs.size(0) // batch_size, loss.item() / batch_size, loss_hf.item() / batch_size))
        elapsed_time = time.time() - start_time

        log('epcoh = %4d , loss = %4.4f , time = %4.2f s' % (epoch + 1, epoch_loss / n_count, elapsed_time))
        np.savetxt('train_result.txt', np.hstack((epoch + 1, epoch_loss / n_count, elapsed_time)), fmt='%2.4f')
        torch.save(model.state_dict(), os.path.join(save_dir, 'model_%03d.pth' % (epoch+1)))
        # torch.save(model, os.path.join(save_dir, 'model_%03d.pth' % (epoch + 1)))
