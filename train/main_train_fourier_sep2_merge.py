
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
from model import DecomposeNet, ComposeNet
from loss import *
from utils import *
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


# Params
parser = argparse.ArgumentParser(description='PyTorch DnCNN')
parser.add_argument('--model', default='DnCNN', type=str, help='choose a type of model')
parser.add_argument('--batch_size', default=126, type=int, help='batch size')
parser.add_argument('--train_data', default='../data/Train400', type=str, help='path of train data')
parser.add_argument('--save_dir', default='../path_upgrade/frequency_sep2', type=str, help='path of train data')
parser.add_argument('--sigma', default=25, type=int, help='noise level')
parser.add_argument('--epoch', default=180, type=int, help='number of train epoches')
parser.add_argument('--lr', default=1e-4, type=float, help='initial learning rate for Adam')
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
    decompose_model = DecomposeNet(image_channels=5, n_block=32)
    compose_model = ComposeNet(n_block=32)

    initial_epoch = findLastCheckpoint(save_dir=save_dir)  # load the last model in matconvnet style
    initial_epoch = 102
    if initial_epoch > 0:
        print('resuming by loading epoch %03d' % initial_epoch)
        decompose_model.load_state_dict(torch.load(os.path.join(save_dir, 'com_model_%03d.pth' % initial_epoch)))
        # model = torch.load(os.path.join(save_dir, 'model_%03d.pth' % initial_epoch))

    # criterion = nn.MSELoss(reduction = 'sum')  # PyTorch 0.4.1
    # criterion = sum_squared_error()
    criterion = nn.MSELoss()

    if cuda:
        decompose_model = decompose_model.cuda()
        compose_model = compose_model.cuda()
        # device_ids = [0]
        # model = nn.DataParallel(model, device_ids=device_ids).cuda()
        # criterion = criterion.cuda()

    optimizer_decompose = optim.Adam(decompose_model.parameters(), lr=args.lr)
    scheduler_decompose = MultiStepLR(optimizer_decompose, milestones=[30, 60, 90], gamma=0.2)  # learning rates
    optimizer_compose = optim.Adam(compose_model.parameters(), lr=args.lr)
    scheduler_compose = MultiStepLR(optimizer_compose, milestones=[30, 60, 90], gamma=0.2)  # learning rates
    for epoch in range(0, n_epoch):
        decompose_model.eval()
        compose_model.train()

        scheduler_decompose.step(epoch)  # step to the learning rate in this epcoh
        scheduler_compose.step(epoch)  # step to the learning rate in this epcoh

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
                temp2 = batch_y
                High_noise, Low_noise = Decomposition(batch_y, 0.125)
                High_origin, Low_origin = Decomposition(batch_x, 0.125)

            batch_y = batch_y.unsqueeze(1)
            x_ = batch_x.unsqueeze(1)

            y_ = torch.cat([batch_y, High_noise.cuda(), Low_noise.cuda()], dim=1)
            # x_ = torch.cat([batch_x, High_origin.cuda(), Low_origin.cuda()], dim=1)

            from torch_complex.tensor import ComplexTensor
            decompose_output = decompose_model(y_.cuda())

            output_hf = ComplexTensor(decompose_output[:, 1], decompose_output[:, 3]).unsqueeze(1).abs()
            output_lf = ComplexTensor(decompose_output[:, 2], decompose_output[:, 4]).unsqueeze(1).abs()
            final_output = ComplexTensor(decompose_output[:, 0] + decompose_output[:, 2], decompose_output[:, 1] + decompose_output[:, 3]).unsqueeze(1).abs()
            color_output = decompose_output[:,0].unsqueeze(1)

            decompose_output = torch.cat([color_output, final_output], dim=1)

            compose_output = compose_model(decompose_output)

            compose_loss = criterion(compose_output, x_)
            optimizer_decompose.zero_grad()
            compose_loss.backward()
            optimizer_compose.step()
            compose_model.eval()

            # output1 = torch.FloatTensor(decompose_output.cpu())
            # compose_output = compose_model(output1.cuda())
            # compose_loss = criterion(compose_output, batch_x)
            # optimizer_compose.zero_grad()
            # compose_loss.backward()
            # optimizer_compose.step()
            # compose_model.eval()

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

            if n_count % 800 == 0:
                highfreq1.imshow(High_noise[0,0].cpu(), cmap='jet')
                highfreq2.imshow(High_origin[0,0].cpu(), cmap='jet')
                highfreq3.imshow(decompose_output[0,1].cpu().detach().numpy(), cmap='jet')
                output1.imshow(temp[0].cpu(), cmap='jet')
                lowfreq1.imshow(decompose_output[0,0].cpu().detach().numpy(), cmap='jet')
                lowfreq2.imshow(decompose_output[0,1].cpu().detach().numpy(), cmap='jet')
                lowfreq3.imshow(temp2[0].cpu().detach().numpy(), cmap='jet')
                output2.imshow(compose_output[0,0].squeeze(0).cpu().detach().numpy(), cmap='jet')
                plt.show()






            epoch_loss += compose_loss.item()

            if n_count % 10 == 0:
                print('%4d %4d / %4d loss = %2.8f loss_hf = %2.8f ' % (
                epoch + 1, n_count, xs.size(0) // batch_size, compose_loss.item() / batch_size, compose_loss.item() / batch_size))
        elapsed_time = time.time() - start_time

        log('epcoh = %4d , loss = %4.4f , time = %4.2f s' % (epoch + 1, epoch_loss / n_count, elapsed_time))
        np.savetxt('train_result.txt', np.hstack((epoch + 1, epoch_loss / n_count, elapsed_time)), fmt='%2.4f')
        torch.save(compose_model.state_dict(), os.path.join(save_dir, 'decom_model_%03d.pth' % (epoch+1)))
        # torch.save(decompose_model.state_dict(), os.path.join(save_dir, 'com_model_%03d.pth' % (epoch+1)))
        # torch.save(model, os.path.join(save_dir, 'model_%03d.pth' % (epoch + 1)))
