
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import numpy as np
from utils.logger import make_logs
import utils.utils as utility

import math
import random
import os
from collections import defaultdict

# from dataset.dataset import DatasetFromHdf5_train, DatasetFromHdf5_test
from dataset.datasetx2 import DatasetFromHdf5List_train, DatasetFromHdf5List_test
from model.IINet_modified import Net
from tensorboardX import SummaryWriter
import time
from model.model_utils import getNetworkDescription

import warnings

warnings.filterwarnings("ignore")

def get_cur_time():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

def load_state_dict_to_model(model, dict):
    m_dict = model.state_dict()
    tmp_dict = {}
    for k, v in dict.items():
        if k in m_dict.keys():
            if v.shape == m_dict[k].shape:
                tmp_dict[k] = v
    # tmp_dict = {k: v for k, v in dict.items() if k in m_dict.keys()}
    m_dict.update(tmp_dict)
    model.load_state_dict(m_dict)
    return model

# --------------------------------------------------------------------------#
# Training settings
parser = argparse.ArgumentParser(description="PyTorch LFSSR-LFIINet training")

# training settings
parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
parser.add_argument("--step", type=int, default=10000, help="Learning rate decay every n epochs")
parser.add_argument("--allepoch", type=int, default=60000, help="Learning rate decay every n epochs")
parser.add_argument("--reduce", type=float, default=0.5, help="Learning rate decay")
parser.add_argument("--patch_size", type=int, default=64, help="Training patch size")
parser.add_argument("--batch_size", type=int, default=4, help="Training batch size")
parser.add_argument("--resume_path", type=str, default="", help="resume from checkpoint path")
parser.add_argument("--num_cp", type=int, default=100, help="Number of epoches for saving checkpoint")
parser.add_argument("--seed", type=int, default=0, help="Initialization seed")

# dataset
parser.add_argument("--dataset", type=str, default="all", help="Dataset for training")
parser.add_argument("--angular_num", type=int, default=5, help="Size of one angular dim")
parser.add_argument("--trainFile", type=str, default="")
parser.add_argument("--testFile", type=str, default="")
# model
parser.add_argument("--scale", type=int, default=2, help="SR factor")
# for recording
parser.add_argument("--record", action="store_true", help="Record? --record makes it True")
parser.add_argument("--cuda", action="store_true", help="Use cuda? --cuda makes it True")
parser.add_argument("--save_dir", type=str, default="/data/zy/realLFSR/result/")
parser.add_argument("--num_threads", type=int, default=6)
parser.add_argument('--test_patch', action=utility.StoreAsArray, type=int, nargs='+', help="number of patches during testing")

def main():
    global opt, model
    opt = parser.parse_args()

    opt.record = True
    opt.dataroot = "/data/zy/realLFSR/dataset/small_x2_xiao"
    opt.test_patch = [1, 1]

    opt.trainFile = "/data/zy/realLFSR/dataset/small_x2_xiao/train.txt"
    opt.testFile = "/data/zy/realLFSR/dataset/small_x2_xiao/test.txt"
    opt.pretrain =  ''
    # --------------------------------------------------------------------------#
    # Device configuration
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if opt.seed < 0:
        opt.seed = random.randint(1, 10000)
        print("Random seed is: {}".format(opt.seed))
    torch.manual_seed(opt.seed)
    if opt.cuda:
        torch.cuda.manual_seed(opt.seed)
    cudnn.benchmark = True

    opt.save_prefix = 'IINet_x{}'.format(opt.scale)

    print(opt)

    an = opt.angular_num
    # --------------------------------------------------------------------------#
    # Data loader
    print('===> Loading train datasets')
    train_set = DatasetFromHdf5List_train(opt.dataroot, opt.trainFile, opt.scale, opt.patch_size)
    train_loader = DataLoader(dataset=train_set, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_threads)
    print('loaded {} LFIs from {}'.format(len(train_loader), opt.trainFile))
    print('===> Loading test datasets')
    test_set = DatasetFromHdf5List_test(opt.dataroot, opt.testFile, opt.scale)
    test_loader = DataLoader(dataset=test_set, batch_size=1, shuffle=False, num_workers=0)
    print('loaded {} LFIs from {}'.format(len(test_loader), opt.testFile))


    # --------------------------------------------------------------------------#
    # Build model
    print("===> building network")
    # srnet_name = 'net{}x'.format(opt.scale)
    # model = eval(srnet_name)(an, opt.layer_num, opt.mode, opt.fn).to(device)
    model = Net(opt)
    # s, n = getNetworkDescription(model)
    # print("Network structure:")
    # print(s)
    # print("Network parameter number: {}".format(n))
    criterion = nn.L1Loss()
    if opt.cuda:
        criterion = criterion.cuda()
        # model = model.cuda()

    # -------------------------------------------------------------------------#
    # optimizer and loss logger
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=opt.step, gamma=opt.reduce)
    losslogger = defaultdict(list)


    if opt.pretrain:
        checkpoint = torch.load(opt.pretrain)
        model.load_state_dict(checkpoint['model'])
        print('load pretrained')
    else:
        print('No pretrain model, initialize randomly')

    # ------------------------------------------------------------------------#
    # optionally resume from a checkpoint
    if opt.resume_path:
        resume_path = opt.resume_path
        # resume_path = os.path.join('/model/liuyutong/LFSSR-PARALALL/2x6l', 'model_epoch_{}.pth'.format(opt.resume_epoch))
        if os.path.isfile(resume_path):
            print("==> loading checkpoint 'epoch{}'".format(resume_path))
            checkpoint = torch.load(resume_path)
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            opt.resume_epoch = checkpoint['epoch']
            losslogger = checkpoint['losslogger']
        else:
            print("==> no model found at 'epoch{}'".format(opt.resume_epoch))
            opt.resume_epoch = 0
    else:
        opt.resume_epoch = 0

    # ------------------------------------------------------------------------#
    print('==> training')
    if opt.record:
        # save_dir = "/output/"
        make_logs("{}log/{}/".format(opt.save_dir, opt.save_prefix), "train_log.log", "train_err.log")
        writer = SummaryWriter(log_dir="{}logs/{}".format(opt.save_dir, opt.save_prefix),
                               comment="Training curve for LFASR-SAS-2-to-8")
        if not os.path.exists("{}checkpoints/{}/".format(opt.save_dir, opt.save_prefix)):
            os.makedirs("{}checkpoints/{}/".format(opt.save_dir, opt.save_prefix))
    
    model = nn.DataParallel(model).cuda()

    for epoch in range(opt.resume_epoch + 1, opt.allepoch):


        loss = train(epoch, model, scheduler, train_loader, optimizer, losslogger, criterion)

        if epoch % opt.num_cp == 0:
            model_save_path = os.path.join("{}checkpoints/{}/model_epoch_{}.pth".format(opt.save_dir, opt.save_prefix, epoch))
            state = {'epoch': epoch, 'model': model.module.state_dict(), 'optimizer': optimizer.state_dict(),
                     'scheduler': scheduler.state_dict(), 'losslogger': losslogger}
            torch.save(state, model_save_path)
            print("checkpoint saved to {}".format(model_save_path))

        if epoch % opt.num_cp == 0:
            PSNR_TEST = test(epoch, model, test_loader, an)
            if opt.record:
                writer.add_scalar("test/PSNR", PSNR_TEST, epoch)
        
        if opt.record:
            writer.add_scalar("train/recon_loss", loss, epoch)


# ------------------------------------------------------------------------#
# loss
def L1_Charbonnier_loss(X, Y):
    eps = 1e-6
    diff = torch.add(X, -Y)
    error = torch.sqrt(diff * diff + eps)
    loss = torch.sum(error) / torch.numel(error)
    return loss

# -----------------------------------------------------------------------#

def train(epoch, model, scheduler, train_loader, optimizer, losslogger, criterion):
    
    model.train()
    scheduler.step()

    print("{}: Epoch = {}, lr = {}".format(get_cur_time(), epoch, optimizer.param_groups[0]["lr"]))

    loss_count = 0.


    for i, batch in enumerate(train_loader, 1):
        # if opt.cuda:
        lr = batch[1].cuda()
        hr = batch[0].cuda()

        sr = model(lr)

        loss = criterion(sr, hr)

        loss_count += loss.item()

        # backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 5 == 0:
            print("{}: Epoch {}, [{}/{}]: SR loss: {:.10f}".format(get_cur_time(), epoch, i, len(train_loader),
                                                               loss.cpu().data))
        # writer.add_scalar("train/recon_loss_iter", loss.cpu().data, i + (epoch - 1) * len(train_loader))

    average_loss = loss_count / len(train_loader)
    losslogger['epoch'].append(epoch)
    losslogger['loss'].append(average_loss)

    return average_loss


# -------------------------------------------------------------------------#

def compt_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 1.0

    if mse > 1000:
        return -100
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def test(epoch, model, test_loader, an):
    model.eval()
    lf_list = []
    lf_psnr_y_list = []
    # lf_ssim_y_list = []

    with torch.no_grad():
        for k, batch in enumerate(test_loader):
            # print('testing LF {}{}'.format(opt.test_dataset, k))
            # ----------- SR ---------------#
            gt_y, sr_ycbcr, lr_y = batch[0].numpy(), batch[1].numpy(), batch[2]
            # gt_y, lr_ycbcr_2_up, lr_y_2, lr_ycbcr_4_up, lr_y_4
            # start = time.time()
            lr_y = lr_y.cuda()
            if opt.test_patch[0] == 1 and opt.test_patch[1] == 1:
                sr_y = model(lr_y).cpu()
                sr_y = sr_y.numpy()
            else:
                Px = opt.test_patch[0]
                Py = opt.test_patch[1]
                pad_size = 32
                H, W = lr_y.shape[2], lr_y.shape[3]

                srLFPatches = []
                for px in range(Px):
                    for py in range(Py):
                        lr_y_patch = utility.getLFPatch(lr_y, Px, Py, H, W, px, py, pad_size)
                        sr_y_patch = model(lr_y_patch).cpu().numpy()
                        srLFPatches.append(sr_y_patch)

                sr_y = utility.mergeLFPatches(srLFPatches, Px, Py, H, W, opt.scale, pad_size)
            # end = time.time()

            sr_ycbcr[:, :, 0] = sr_y
            # ---------compute average PSNR for this LFI----------#

            view_list = []
            view_psnr_y_list = []
            # view_ssim_y_list = []

            for i in range(an * an):
                cur_psnr = compt_psnr(gt_y[0, i], sr_y[0, i])

                view_list.append(i)
                view_psnr_y_list.append(cur_psnr)
                # view_ssim_y_list.append(cur_ssim)

            lf_list.append(k)
            lf_psnr_y_list.append(np.mean(view_psnr_y_list))
            # lf_ssim_y_list.append(np.mean(view_ssim_y_list))

    PSNR = np.mean(lf_psnr_y_list)
    print("Testing epoch: {}, Mean PSNR: {}".format(epoch, PSNR))
    return PSNR

if __name__ == '__main__':
    main()


