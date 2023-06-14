
from functools import partial
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader

import argparse
import numpy as np
import os
from os.path import join

import math
import copy
# import pandas as pd
import time
from utils.logger import make_logs

import h5py

# from scipy import misc
# from skimage.measure import compare_ssim
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from model.OProjNet1 import Net
import utils.utils as utility

from PIL import Image
import warnings

warnings.filterwarnings("ignore")
# --------------------------------------------------------------------------#
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# ----------------------------------------------------------------------------------#
# Test settings
parser = argparse.ArgumentParser(description="PyTorch LFSSR-SAS testing")
parser.add_argument("--model_dir", type=str,default="/data/zy/realLFSR/result/checkpoints/OProjNet_x4_LZ_P.pth",help="model dir")
parser.add_argument("--scale", type=int, default=2, help="SR factor")
parser.add_argument("--train_dataset", type=str, default="all", help="dataset for training")
parser.add_argument("--test_dataset", type=str, default="LytroZoom", help="dataset for test")
parser.add_argument("--angular_num", type=int, default=5, help="Size of one angular dim")
parser.add_argument("--save_img", type=int, default=1, help="save image or not")
parser.add_argument("--mode", type=str, default="parares", help="SR factor")
parser.add_argument('--test_patch', action=utility.StoreAsArray, type=int, nargs='+', help="number of patches during testing")
parser.add_argument("--channel", type=int, default=32, help="Number of epoches for saving checkpoint")

opt = parser.parse_args()
opt.test_patch = [1, 1]
# opt.test_patch = [2, 2]
print(opt)

class DatasetFromHdf5List_test(data.Dataset):
    def __init__(self, dataroot, list_path, scale):
        super(DatasetFromHdf5List_test, self).__init__()

        self.list_path = list_path

        fd = open(self.list_path, 'r')
        self.h5files = [line.strip('\n') for line in fd.readlines()]
        print("Dataset files include {}".format(self.h5files))

        self.lens = []
        self.img_HRs = []
        self.img_LR_2s = []
        self.img_LR_4s = []

        for h5file in self.h5files:
            hf = h5py.File("{}/{}".format(dataroot, h5file))
            img_HR = hf.get('GT_y')  # [N,ah,aw,h,w]
            img_LR_2 = hf.get('x4_ycbcr')  # [N,ah,aw,h/2,w/2]

            self.lens.append(img_HR.shape[0])
            self.img_HRs.append(img_HR)
            self.img_LR_2s.append(img_LR_2)

        self.scale = scale

    def __getitem__(self, index):
        file_index = 0
        batch_index = 0
        for i in range(len(self.h5files)):
            if index < self.lens[i]:
                file_index = i
                batch_index = index
                break
            else:
                index -= self.lens[i]

        h, w = self.img_HRs[file_index].shape[2], self.img_HRs[file_index].shape[3]

        gt_y = np.array(self.img_HRs)[file_index]
        gt_y = gt_y.reshape(-1, h, w)
        gt_y = torch.from_numpy(gt_y.astype(np.float32) / 255.0)

        lr_ycbcr_2 = np.array(self.img_LR_2s)[file_index]
        lr_ycbcr_2 = torch.from_numpy(lr_ycbcr_2.astype(np.float32) / 255.0)

        lr_y_2 = lr_ycbcr_2[:, :, 0, :, :].clone().view(-1, h, w)

        lr_ycbcr_2_up = lr_ycbcr_2.view(1, -1, h, w)
        lr_ycbcr_2_up = torch.nn.functional.interpolate(lr_ycbcr_2_up, scale_factor=1, mode='bilinear',
                                                      align_corners=False)
        lr_ycbcr_2_up = lr_ycbcr_2_up.view(-1, 3, h, w)

       
        return gt_y, lr_ycbcr_2_up, lr_y_2

    def __len__(self):
        total_len = 0
        for i in range(len(self.h5files)):
            total_len += self.lens[i]

        return total_len



# -----------------------------------------------------------------------------------#
# -------------------------------------------------------------------------------#

model_dir = opt.model_dir

if not os.path.exists(model_dir):
    print('model folder is not found ')


an = opt.angular_num
# ------------------------------------------------------------------------#
# Data loader
print('===> Loading test datasets')
opt.dataroot = "/data/zy/realLFSR/dataset/small_x4_xiao"
opt.testFile = "/data/zy/realLFSR/dataset/small_x4_xiao/test.txt"
test_set = DatasetFromHdf5List_test(opt.dataroot, opt.testFile, opt.scale)

test_loader = DataLoader(dataset=test_set, batch_size=1, shuffle=False)
print('loaded {} LFIs from {}'.format(len(test_loader), opt.dataroot))
# -------------------------------------------------------------------------#
# Build model
print("===> building network")
model = Net(opt).cuda()

# ------------------------------------------------------------------------#

# -------------------------------------------------------------------------#
# test
def ycbcr2rgb(ycbcr):
    m = np.array([[65.481, 128.553, 24.966],
                  [-37.797, -74.203, 112],
                  [112, -93.786, -18.214]])
    shape = ycbcr.shape
    if len(shape) == 3:
        ycbcr = ycbcr.reshape((shape[0] * shape[1], 3))
    rgb = copy.deepcopy(ycbcr)
    rgb[:, 0] -= 16. / 255.
    rgb[:, 1:] -= 128. / 255.
    rgb = np.dot(rgb, np.linalg.inv(m.transpose()) * 255.)
    return rgb.clip(0, 1).reshape(shape).astype(np.float32)

def compt_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 1.0

    if mse > 1000:
        return -100
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def test():
    model.eval()
    save_dir = '/data/zy/realLFSR/code_inference/result_final/OProjNet_LytroZoom_x4/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if opt.save_img:
        save_dir = '/data/zy/realLFSR/code_inference/result_final/OProjNet_LytroZoom_x4/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    # os.makedirs('/output/LR')
    lf_list = []
    lf_psnr_y_list = []
    lf_psnr_y2_list = []
    lf_ssim_y_list = []

    make_logs("{}log/".format(save_dir), "inference_log.log", "inference_err.log")
    with torch.no_grad():
        for k, batch in enumerate(test_loader):
            if k%5==0:
                k = int(k/5)
                print('testing LF {}{}'.format(opt.test_dataset, k))
                # ----------- SR ---------------#
                gt_y, sr_ycbcr, lrr = batch[0].numpy(), batch[1].numpy(), batch[2]

                start = time.time()
                lr_y = lrr.to(device)
                if opt.test_patch[0] == 1 and opt.test_patch[1] == 1:
                    try:
                        sr_y = model(lr_y)
                    except RuntimeError as exception:
                        if "out of memory" in str(exception):
                            print("WARNING: out of memory, clear the cache!")
                            if hasattr(torch.cuda, 'empty_cache'):
                                torch.cuda.empty_cache()
                                sr_y = model(lr_y)
                        else:
                            raise exception
                    sr_y = sr_y.cpu().numpy()
                    # sr_y = sr_list[0].cpu()
                else:
                    Px = opt.test_patch[0]
                    Py = opt.test_patch[1]
                    pad_size = 32
                    # print(lr_y.shape)
                    H, W = lr_y.shape[2], lr_y.shape[3]

                    srLF_patches = []

                    for px in range(Px):
                        for py in range(Py):
                            lr_y_patch = utility.getLFPatch(lr_y, Px, Py, H, W, px, py, pad_size)
                            try:
                                sr_y_patch = model(lr_y_patch)
                            except RuntimeError as exception:
                                if "out of memory" in str(exception):
                                    print("WARNING: out of memory, clear the cache!")
                                    if hasattr(torch.cuda, 'empty_cache'):
                                        torch.cuda.empty_cache()
                                        sr_y_patch = model(lr_y_patch)
                                else:
                                    raise exception
                            sr_y_patch = sr_y_patch.cpu().numpy()
                            srLF_patches.append(sr_y_patch)
                    sr_y = utility.mergeLFPatches(srLF_patches, Px, Py, H, W, opt.scale, pad_size)

                end = time.time()
                # print('running time: ', end - start)

                # sr_y = sr_y.numpy()
                sr_ycbcr[:, :, 0] = sr_y
                # ---------compute average PSNR/SSIM for this LFI----------#

                view_list = []
                view_psnr_y_list = []
                view_psnr_y2_list = []
                view_ssim_y_list = []

                for i in range(an * an):
                    if opt.save_img:
                        img_name = '{}/SR{}_view{}.png'.format(save_dir, k, i)
                        sr_rgb_temp = ycbcr2rgb(np.transpose(sr_ycbcr[0, i], (1, 2, 0)))
                        img = (sr_rgb_temp.clip(0, 1) * 255.0).astype(np.uint8)[:,:,::-1]
                        # misc.toimage(img, cmin=0, cmax=255).save(img_name)
                        img = Image.fromarray(img)
                        img.save(img_name)

                    cur_psnr = compt_psnr(gt_y[0, i], sr_y[0, i])
                    cur_psnr2 = compare_psnr(gt_y[0, i], sr_y[0, i])
                    cur_ssim = compare_ssim((gt_y[0, i] * 255.0).astype(np.uint8), (sr_y[0, i] * 255.0).astype(np.uint8),
                                            gaussian_weights=True, sigma=1.5, use_sample_covariance=False)

                    view_list.append(i)
                    view_psnr_y_list.append(cur_psnr)
                    view_psnr_y2_list.append(cur_psnr2)
                    view_ssim_y_list.append(cur_ssim)

                # dataframe_lfi = pd.DataFrame(
                #     {'View_LFI{}'.format(k): view_list, 'psnr Y': view_psnr_y_list, 'ssim Y': view_ssim_y_list})
                # dataframe_lfi.to_csv(csv_name, index=False, sep=',', mode='a')

                lf_list.append(k)
                lf_psnr_y_list.append(np.mean(view_psnr_y_list))
                lf_psnr_y2_list.append(np.mean(view_psnr_y2_list))
                lf_ssim_y_list.append(np.mean(view_ssim_y_list))

                
                print(
                    'Avg. Y PSNR: {:.3f}; PSNR2: {:.3f}; Avg. Y SSIM: {:.5f}'.format(np.mean(view_psnr_y_list), np.mean(view_psnr_y2_list), np.mean(view_ssim_y_list)))

    # dataframe_lfi = pd.DataFrame({'lfiNo': lf_list, 'psnr Y': lf_psnr_y_list, 'ssim Y': lf_ssim_y_list})
    # dataframe_lfi.to_csv(csv_name, index=False, sep=',', mode='a')
    # dataframe_lfi = pd.DataFrame(
    #     {'summary': ['avg'], 'psnr Y': [np.mean(lf_psnr_y_list)], 'ssim Y': [np.mean(lf_ssim_y_list)]})
    # dataframe_lfi.to_csv(csv_name, index=False, sep=',', mode='a')

    print('Over all {} LFIs on {}: Avg. Y PSNR: {:.3f}, Avg. Y SSIM: {:.5f}'.format(len(test_loader), opt.test_dataset,
                                                                                    np.mean(lf_psnr_y_list),
                                                                                    np.mean(lf_ssim_y_list)))

# ------------------------------------------------------------------------#

# print('===> test epoch {}'.format(opt.epoch))
# resume_path = join(model_dir, "model_epoch_{}.pth".format(opt.epoch))
resume_path = model_dir
checkpoint = torch.load(resume_path)
model.load_state_dict(checkpoint['model'])
print('loaded model {}'.format(resume_path))
test()

