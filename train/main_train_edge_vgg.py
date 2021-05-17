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
from model import vgg11
from loss import *
from utils import *
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from torchvision import transforms
from PIL import ImageFilter

# Params
parser = argparse.ArgumentParser(description='PyTorch DnCNN')
parser.add_argument('--model', default='DnCNN', type=str, help='choose a type of model')
parser.add_argument('--batch_size', default=128, type=int, help='batch size')
parser.add_argument('--train_data', default='../data/Train400', type=str, help='path of train data')
parser.add_argument('--save_dir', default='../path_upgrade/vgg', type=str, help='path of train data')
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
    model = vgg11()
    ReLU = nn.ReLU()
    initial_epoch = findLastCheckpoint(save_dir=save_dir)  # load the last model in matconvnet style
    if initial_epoch > 0:
        print('resuming by loading epoch %03d' % initial_epoch)
        # model.load_state_dict(torch.load(os.path.join(save_dir, 'model_%03d.pth' % initial_epoch)))
        # model = torch.load(os.path.join(save_dir, 'model_%03d.pth' % initial_epoch))
    model.train()
    # criterion = nn.MSELoss(reduction = 'sum')  # PyTorch 0.4.1
    # criterion = sum_squared_error()
    # criterion = nn.MSELoss()
    criterion = nn.BCELoss()
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

            edge_prob = torch.FloatTensor(len(batch_yx), 1)

            if cuda:
                batch_x, batch_y = batch_yx[:,0].cuda(), batch_yx[:,1].cuda()
                Edge_enhance = Edge_enhance.cuda()
                edge_prob = edge_prob.cuda()
                for i in range(len(batch_x)):
                    Edge_enhance[i] = torch.clamp(
                        toTensor(toPILImage(batch_x[i].squeeze(0).cpu()).filter(ImageFilter.FIND_EDGES)), 0, 1)
                    fig = plt.figure()
                    gs = GridSpec(nrows=1, ncols=2)
                    if torch.mean(Edge_enhance[i]) > 0.15:
                        edge_prob[i, 0] = 1
                        # print(torch.mean(Edge_enhance[i]), ReLU(torch.mean(Edge_enhance[i])))
                    else:
                        edge_prob[i, 0] = 0

                    # print(edge_prob[i])
            batch_y = batch_y.unsqueeze(1)
            batch_x = batch_x.unsqueeze(1)
            output = model(batch_y)

            criterion = criterion.cpu()
            loss = criterion(output.cpu(), edge_prob.cpu())

            loss.backward()
            optimizer.step()

            if n_count % 10 == 0:
                print('%4d %4d / %4d loss = %2.4f' % (
                    epoch + 1, n_count, xs.size(0) // batch_size, loss.item() / batch_size))

            pass
        torch.save(model.state_dict(), os.path.join(save_dir, 'model_%03d.pth' % (epoch + 1)))








