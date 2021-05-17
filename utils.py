import os, glob, datetime, time
import re
import numpy as np
import torch.nn as nn
from torch import rfft, irfft
import torch

def Decomposition(input, N):

    High_freq = rfft(input.unsqueeze(3), signal_ndim=3)
    Low_freq = torch.FloatTensor(High_freq.shape).cuda()

    width, height = High_freq.shape[1], High_freq.shape[2]

    for x in range(N//2, width - N //2):
        for y in range(N//2, height - N // 2):
    # for x in range(0, N):
    #     for y in range(0, N):
            Low_freq[:,x,y] = High_freq[:,x,y]
    High_freq -= Low_freq
    # return High_freq, Low_freq
    return irfft(High_freq, signal_ndim=3).squeeze(3), irfft(Low_freq, signal_ndim=3).squeeze(3)


activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

class get_histogram(nn.Module):
    def __init__(self):
        super(get_histogram, self).__init__()

    def forward(self, batchsize, input):
        count_r = np.zeros(256)
        count_g = np.zeros(256)
        count_b = np.zeros(256)

        for j in range(batchsize):
            x = np.array(input[j] * 255)
            hist_r = np.histogram(x[0, :, :], 256, range=[0, 255])
            hist_g = np.histogram(x[1, :, :], 256, range=[0, 255])
            hist_b = np.histogram(x[2, :, :], 256, range=[0, 255])
            count_r += hist_r[0]
            count_g += hist_g[0]
            count_b += hist_b[0]

        bins = hist_r[1]
        return hist_r, count_r, count_g, count_b

def log(*args, **kwargs):
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:"), *args, **kwargs)

def findLastCheckpoint(save_dir):
    file_list = glob.glob(os.path.join(save_dir, 'model_*.pth'))
    if file_list:
        epochs_exist = []
        for file_ in file_list:
            result = re.findall(".*model_(.*).pth.*", file_)
            epochs_exist.append(int(result[0]))
        initial_epoch = max(epochs_exist)
    else:
        initial_epoch = 0
    return initial_epoch