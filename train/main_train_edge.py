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
from model import Encoder2, DnCNN
from loss import *
from utils import *
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from torchvision import transforms
from PIL import ImageFilter

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
    # for x in range(int(width*start), int(width*end)):
    #     for y in range(int(height*start), int(height*end)):
            Low_freq[:, x, y] = High_freq[:, x, y]
            High_freq[:,x,y] = 0

    # for x in range(centerx-int(width*end/2), centerx+int(width*end/2)):
    #     for y in range(centery-int(height/2), centery+int(height/2)):
    #         Low_freq[:, x, y] = High_freq[:, x, y]
    #
    # for x in range(centerx-int(width/2), centerx+int(width/2)):
    #     for y in range(centery-int(height*end/2), centery+int(height*end/2)):
    #         Low_freq[:, x, y] = High_freq[:, x, y]
    #
    # for x in range(centerx - int(width * end / 2), centerx + int(width * end / 2)):
    #     for y in range(centery - int(height / 2), centery + int(height / 2)):
    #         High_freq[:,x,y] = 0
    #
    # for x in range(centerx - int(width / 2), centerx + int(width / 2)):
    #     for y in range(centery - int(height * end / 2), centery + int(height * end / 2)):
    #         High_freq[:,x,y] = 0



    # plt.figure()
    # plt.imshow(np.abs(Low_freq[0]), cmap='jet')
    # plt.show()
    #
    # plt.figure()
    # plt.imshow(np.abs(High_freq[0]), cmap='jet')
    # plt.show()
    #
    # plt.figure()
    # plt.imshow(input[0], cmap='jet')
    # plt.show()

    High_freq = np.fft.ifftshift(High_freq)
    Low_freq = np.fft.ifftshift(Low_freq)

    # plt.figure()
    # plt.imshow(np.abs(np.real(np.fft.ifft2(High_freq)[0])), cmap='jet')
    # plt.show()
    #
    # plt.figure()
    # plt.imshow(np.abs(np.imag(np.fft.ifft2(High_freq)[0])), cmap='jet')
    # plt.show()

    # plt.figure()
    # plt.imshow(np.abs((np.fft.ifft2(High_freq)[0])), cmap='jet')
    # plt.show()

    # plt.figure()
    # plt.imshow(np.abs(np.real(np.fft.ifft2(Low_freq)[0])), cmap='jet')
    # plt.show()
    #
    # plt.figure()
    # plt.imshow(np.abs(np.imag(np.fft.ifft2(Low_freq)[0])), cmap='jet')
    # plt.show()

    # plt.figure()
    # plt.imshow(np.abs((np.fft.ifft2(Low_freq)[0])), cmap='jet')
    # plt.show()

    High_output = torch.tensor(np.abs((np.fft.ifft2(High_freq))))
    Low_output = torch.tensor(np.abs((np.fft.ifft2(Low_freq))))

    # High_output = torch.cat((torch.tensor(np.real(np.fft.ifft2(High_freq))).unsqueeze(1).float(),
    #                             torch.tensor(np.imag(np.fft.ifft2(High_freq))).unsqueeze(1).float()), dim=1)
    # Low_output = torch.cat((torch.tensor(np.real(np.fft.ifft2(Low_freq))).unsqueeze(1).float(),
    #                           torch.tensor(np.imag(np.fft.ifft2(Low_freq))).unsqueeze(1).float()), dim=1)

    # plt.figure()
    # plt.imshow(ComplexTensor(High_output[0,0] + Low_output[0,0], High_output[0,1]+ Low_output[0,1]).abs(), cmap='jet')
    # plt.show()

    return High_output, Low_output


# Params
parser = argparse.ArgumentParser(description='PyTorch DnCNN')
parser.add_argument('--model', default='DnCNN', type=str, help='choose a type of model')
parser.add_argument('--batch_size', default=512, type=int, help='batch size')
parser.add_argument('--train_data', default='../data/Train400', type=str, help='path of train data')
parser.add_argument('--save_dir', default='../path_upgrade/edge_high_freq_final', type=str, help='path of train data')
parser.add_argument('--sigma', default=25, type=int, help='noise level')
parser.add_argument('--epoch', default=180, type=int, help='number of train epoches')
parser.add_argument('--lr', default=1e-3, type=float, help='initial learning rate for Adam')
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

if __name__ == '__main__':
    # model selection
    print('===> Building model')
    model = DnCNN()

    initial_epoch = findLastCheckpoint(save_dir=save_dir)  # load the last model in matconvnet style
    if initial_epoch > 0:
        print('resuming by loading epoch %03d' % initial_epoch)
        # model.load_state_dict(torch.load(os.path.join(save_dir, 'model_%03d.pth' % initial_epoch)))
        # model = torch.load(os.path.join(save_dir, 'model_%03d.pth' % initial_epoch))
    model.train()
    # criterion = nn.MSELoss(reduction = 'sum')  # PyTorch 0.4.1
    # criterion = sum_squared_error()
    criterion = nn.MSELoss()
    Edge_enhance = torch.FloatTensor(args.batch_size, 1, 40, 40)
    if cuda:
        model = model.cuda()
        # device_ids = [0]
        # model = nn.DataParallel(model, device_ids=device_ids).cuda()
        criterion = criterion.cuda()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.2)  # learning rates
    for epoch in range(initial_epoch, n_epoch):

        scheduler.step(epoch)  # step to the learning rate in this epcoh
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
            optimizer.zero_grad()
            if cuda:
                batch_x, batch_y = batch_yx[:,0].cuda(), batch_yx[:,1].cuda()
            batch_y = batch_y.unsqueeze(1)
            batch_x = batch_x.unsqueeze(1)

            High_origin, Low_origin = Decomposition(batch_x.squeeze(1))
            High_noise, Low_noise = Decomposition(batch_y.squeeze(1))

            batch_y = High_noise.unsqueeze(1).cuda()
            batch_x = High_origin.unsqueeze(1).cuda()

            output = model(batch_y.float())

            loss = criterion(output, batch_x.float())

            fig = plt.figure()
            gs = GridSpec(nrows=1, ncols=3)

            original = fig.add_subplot(gs[0, 0])
            noised = fig.add_subplot(gs[0, 1])
            out = fig.add_subplot(gs[0, 2])


            original.axis('off')
            noised.axis('off')
            out.axis('off')

            if n_count == 0:
                original.imshow(batch_x[0].cpu().squeeze(0), cmap='jet')
                noised.imshow(batch_y[0].cpu().squeeze(0), cmap='jet')
                out.imshow(output[0].cpu().detach().numpy().squeeze(0), cmap='jet')
                # plot3.imshow((Edge_enhance)[0].cpu().squeeze(0), cmap='gray')
                plt.show()


            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            if n_count % 10 == 0:
                print('%4d %4d / %4d loss = %2.8f' % (
                epoch + 1, n_count, xs.size(0) // batch_size, loss.item() / batch_size))
        elapsed_time = time.time() - start_time

        log('epcoh = %4d , loss = %4.4f , time = %4.2f s' % (epoch + 1, epoch_loss / n_count, elapsed_time))
        np.savetxt('train_result.txt', np.hstack((epoch + 1, epoch_loss / n_count, elapsed_time)), fmt='%2.4f')
        # torch.save(model.state_dict(), os.path.join(save_dir, 'model_%03d.pth' % (epoch+1)))
        torch.save(model.state_dict(), os.path.join(save_dir, 'model_%03d.pth' % (epoch + 1)))
