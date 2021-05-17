import torch
import torch.nn as nn
import numpy as np

class sum_squared_error():  # PyTorch 0.4.1
    """
    Definition: sum_squared_error = 1/2 * nn.MSELoss(reduction = 'sum')
    The backward is defined as: input-target
    """
    def __init__(self, size_average=None, reduce=None, reduction='sum'):
        super(sum_squared_error, self).__init__(size_average, reduce, reduction)

    def forward(self, input, target):
        # return torch.sum(torch.pow(input-target,2), (0,1,2,3)).div_(2)
        return torch.nn.functional.mse_loss(input, target, size_average=None, reduce=None, reduction='sum').div_(2)


class get_intensity_histogram(nn.Module):
    def __init__(self):
        super(get_intensity_histogram, self).__init__()
    def forward(self, batchsize, input):
        count = torch.zeros(256)
        count = count.cuda()
        for j in range(batchsize):
            hist = torch.histc(input[:,:], bins=256, min=0,max=255).cuda()

            count += hist[0]
        bins = hist[1]
        return hist, count

class intensity_hist_loss(nn.Module):
    def __init__(self):
        super(intensity_hist_loss, self).__init__()

    def forward(self, ref_hist, input_hist):
        loss = torch.mean((ref_hist[1] - input_hist[1])**2)

        return loss

class Intensity_loss(nn.Module):
    def __init__(self):
        super(Intensity_loss, self).__init__()
        self.mse_loss = nn.MSELoss().cuda()
        self.intensity_histo_loss = intensity_hist_loss().cuda()
        self.get_intensity_info = get_intensity_histogram().cuda()



    def forward(self, input_img, ref_img, batchsize, gray_rate = 3e-20):
        ref_intensity = self.get_intensity_info(batchsize, ref_img.detach())
        input_intensity = self.get_intensity_info(batchsize, input_img.detach())
        loss_mse = self.mse_loss(input_img, ref_img)
        loss_intensity = self.intensity_histo_loss(ref_intensity, input_intensity)


        return loss_mse + gray_rate * loss_intensity
