import argparse
import os, time, datetime
# import PIL.Image as Image
import numpy as np
import torch.nn as nn
import torch.nn.init as init
import torch
from skimage.measure import compare_psnr, compare_ssim
from skimage.io import imread, imsave
from model import Encoder1, Encoder2, DnCNN
from utils import get_activation, activation
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--set_dir', default='../data/Test', type=str, help='directory of test dataset')
    parser.add_argument('--set_names', default=['Set12', 'Set68'], help='directory of test dataset')
    parser.add_argument('--sigma', default=25, type=int, help='noise level')
    parser.add_argument('--model_dir', default=os.path.join('../path_upgrade/edge_high_freq_20'), help='directory of the model')
    parser.add_argument('--model_name', default='model_012.pth', type=str, help='the model name')
    parser.add_argument('--result_dir', default='results', type=str, help='directory of test dataset')
    parser.add_argument('--save_result', default=1, type=int, help='save the denoised image, 1 or 0')
    return parser.parse_args()


def log(*args, **kwargs):
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:"), *args, **kwargs)


def save_result(result, path):
    path = path if path.find('.') != -1 else path + '.png'
    ext = os.path.splitext(path)[-1]
    if ext in ('.txt', '.dlm'):
        np.savetxt(path, result, fmt='%2.4f')
    else:
        imsave(path, np.clip(result, 0, 1))


def show(x, title=None, cbar=False, figsize=None):
    import matplotlib.pyplot as plt
    plt.figure(figsize=figsize)
    plt.imshow(x, interpolation='nearest', cmap='gray')
    if title:
        plt.title(title)
    if cbar:
        plt.colorbar()
    plt.show()


if __name__ == '__main__':

    args = parse_args()

    model = DnCNN()
    if not os.path.exists(os.path.join(args.model_dir, args.model_name)):
        model = torch.load(os.path.join(args.model_dir, 'model.pth'))
        # load weights into new model
        log('load trained model on Train400 dataset by kai')
    else:
        # model.load_state_dict(torch.load(os.path.join(args.model_dir, args.model_name)))
        model.load_state_dict(torch.load(os.path.join(args.model_dir, args.model_name)))
        log('load trained model')

    model.eval()  # evaluation mode

    if torch.cuda.is_available():
        model = model.cuda()

    if not os.path.exists(args.result_dir):
        os.mkdir(args.result_dir)

    for set_cur in args.set_names:

        if not os.path.exists(os.path.join(args.result_dir, set_cur)):
            os.mkdir(os.path.join(args.result_dir, set_cur))
        psnrs = []
        ssims = []

        for im in os.listdir(os.path.join(args.set_dir, set_cur)):
            if im.endswith(".jpg") or im.endswith(".bmp") or im.endswith(".png"):
                # model.conv1.register_forward_hook(get_activation('conv1'))
                model.dncnn.register_forward_hook(get_activation('noise_level'))


                x = np.array(imread(os.path.join(args.set_dir, set_cur, im)), dtype=np.float32)/255.0
                np.random.seed(seed=0)  # for reproducibility
                y = x + np.random.normal(0, args.sigma/255.0, x.shape)  # Add Gaussian noise without clipping
                y = y.astype(np.float32)
                y_ = torch.from_numpy(y).view(1, -1, y.shape[0], y.shape[1])

                torch.cuda.synchronize()
                start_time = time.time()
                y_ = y_.cuda()
                x_ = model(y_)  # inference
                noise_level = activation['noise_level'].squeeze()

                fig = plt.figure()
                plt.axis("off")
                plt.imshow(noise_level.cpu().squeeze(0), cmap='gray')
                plt.show()

                # for i in range(len(noise_level)):


                x_ = x_.view(y.shape[0], y.shape[1])
                x_ = x_.cpu()
                x_ = x_.detach().numpy().astype(np.float32)
                torch.cuda.synchronize()
                elapsed_time = time.time() - start_time
                print('%10s : %10s : %2.4f second' % (set_cur, im, elapsed_time))

                psnr_x_ = compare_psnr(x, x_)
                ssim_x_ = compare_ssim(x, x_)
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











"""""""""

if __name__ == '__main__':
    if arg.cuda:
        SRNetwork.cuda()

    for epoch in range(arg.nEpoch):
        SR_Epoch_Loss = 0.0

        for i, (images, labels) in enumerate(trainloader):
            SRNetwork.conv1.register_forward_hook(get_activation('SRconv1'))
            SRNetwork.conv2.register_forward_hook(get_activation('SRconv2'))
            SRNetwork.conv3.register_forward_hook(get_activation('SRconv3'))
            SRNetwork.conv4.register_forward_hook(get_activation('SRconv4'))

            for j in range(arg.batchsize):
                LR_img[j] = scale(images[j])
                LR_edge[j] = torch.clamp(toTensor(toPILImage(normalize(LR_img[j])).filter(ImageFilter.FIND_EDGES)), 0,
                                         1)
                HR_img[j] = normalize(images[j])
                HR_edge[j] = torch.clamp(toTensor(toPILImage(normalize(images[j])).filter(ImageFilter.FIND_EDGES)), 0,
                                         1)

            if arg.cuda:
                HR_edge.cuda()
                HR_img = Variable(HR_img.cuda())
                SR_img = SRNetwork(LR_img.cuda())

            else:
                HR_img = Variable(HR_img)
                SR_img = SRNetwork(LR_img)

            SR_loss = SR_criterion(SR_img, HR_img)
            SR_Epoch_Loss += SR_loss
            SR_optimizer.zero_grad()
            SR_loss.backward()
            SR_optimizer.step()

            SR_activated1 = activation['SRconv1'].squeeze()
            SR_activated2 = activation['SRconv2'].squeeze()
            SR_activated3 = activation['SRconv3'].squeeze()
            SR_activated4 = activation['SRconv4'].squeeze()

            # torch.save(SRNetwork.state_dict..(), os.path.join(arg.outputdir, arg.srdir))

            plt.imshow(np.clip(np.transpose((SR_activated4[0]/2+0.5).cpu(), (1, 2, 0)), 0, 1))

            print(epoch)

            plt.show()
            """""""""