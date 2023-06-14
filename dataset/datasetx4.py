import torch.utils.data as data
import torch
# import tables
import h5py
import numpy as np
import random
from scipy import misc

class DatasetFromHdf5List_train(data.Dataset):
    def __init__(self, dataroot, list_path, scale, patch_size):
        super(DatasetFromHdf5List_train, self).__init__()

        # hf = tables.open_file(file_path,driver="H5FD_CORE")

        # self.img_HR = hf.root.img_HR           # [N,ah,aw,h,w]
        # self.img_LR_2 = hf.root.img_LR_2   # [N,ah,aw,h/2,w/2]
        # self.img_LR_4 = hf.root.img_LR_4   # [N,ah,aw,h/4,w/4]

        # self.img_size = hf.root.img_size #[N,2]
        self.list_path = list_path

        fd = open(self.list_path, 'r')
        self.h5files = [line.strip('\n') for line in fd.readlines()]
        print("Dataset files include {}".format(self.h5files))

        self.lens = []
        self.img_HRs = []
        self.img_LR_2s = []
        self.img_LR_3s = []
        self.img_LR_4s = []
        self.img_LR_8s = []

        for h5file in self.h5files:
            hf = h5py.File("{}/{}".format(dataroot, h5file))
            img_HR = hf.get('GT_y')  # [N,ah,aw,h,w]
            # img_LR_3 = hf.get('x3_y')  # [N,ah,aw,h/4,w/4]
            img_LR_4 = hf.get('x4_ycbcr')[:,:,0,:,:]  # [N,ah,aw,h/4,w/4]

            self.lens.append(img_HR.shape[0])
            self.img_HRs.append(img_HR)
            self.img_LR_4s.append(img_LR_4)

        self.scale = scale
        self.psize = patch_size

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

        # get one item
        H, W = self.img_HRs[file_index].shape[2], self.img_HRs[file_index].shape[3]
        hr = np.array(self.img_HRs)[file_index]  # [ah,aw,h,w]
        lr_4 = np.array(self.img_LR_4s)[file_index]  # [ah,aw,h/4,w/4]

        # crop to patch
        # H, W = self.img_size[index]
        # print(H, W)
        x = random.randrange(0, H - self.psize, 8)
        y = random.randrange(0, W - self.psize, 8)
        hr = hr[:, :, x:x + self.psize, y:y + self.psize]  # [ah,aw,ph,pw]
        lr_4 = lr_4[:, :, x:x + self.psize, y:y + self.psize]  # [ah,aw,ph/4,pw/4]

        # 4D augmentation
        # flip
        if np.random.rand(1) > 0.5:
            hr = np.flip(np.flip(hr, 0), 2)
            lr_4 = np.flip(np.flip(lr_4, 0), 2)
            # lr_8 = np.flip(np.flip(lr_8,0),2)
        if np.random.rand(1) > 0.5:
            hr = np.flip(np.flip(hr, 1), 3)
            lr_4 = np.flip(np.flip(lr_4, 1), 3)
            # lr_8 = np.flip(np.flip(lr_8,1),3)
        # rotate
        r_ang = np.random.randint(1, 5)
        hr = np.rot90(hr, r_ang, (2, 3))
        hr = np.rot90(hr, r_ang, (0, 1))
        lr_4 = np.rot90(lr_4, r_ang, (2, 3))
        lr_4 = np.rot90(lr_4, r_ang, (0, 1))

        # to tensor
        hr = hr.reshape(-1, self.psize, self.psize)  # [an,ph,pw]
        lr_4 = lr_4.reshape(-1, self.psize, self.psize)  # [an,phs,pws]

        hr = torch.from_numpy(hr.astype(np.float32) / 255.0)
        lr_4 = torch.from_numpy(lr_4.astype(np.float32) / 255.0)

        # print(hr.shape)
        return hr, lr_4

    def __len__(self):
        total_len = 0
        for i in range(len(self.h5files)):
            total_len += self.lens[i]

        return total_len




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
        self.img_LR_3s = []
        self.img_LR_4s = []
        self.img_LR_8s = []

        for h5file in self.h5files:
            hf = h5py.File("{}/{}".format(dataroot, h5file))
            img_HR = hf.get('GT_y')  # [N,ah,aw,h,w]
            # img_LR_3 = hf.get('x3_y')  # [N,ah,aw,h/4,w/4]
            img_LR_4 = hf.get('x4_ycbcr')  # [N,ah,aw,h/4,w/4]

            self.lens.append(img_HR.shape[0])
            self.img_HRs.append(img_HR)
            self.img_LR_4s.append(img_LR_4)

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

        lr_ycbcr_4 = np.array(self.img_LR_4s)[file_index]
        lr_ycbcr_4 = torch.from_numpy(lr_ycbcr_4.astype(np.float32) / 255.0)

        lr_y_4 = lr_ycbcr_4[:, :, 0, :, :].clone().view(-1, h, w)

        lr_ycbcr_4_up = lr_ycbcr_4.view(1, -1, h, w)
        lr_ycbcr_4_up = torch.nn.functional.interpolate(lr_ycbcr_4_up, scale_factor=1, mode='bilinear',
                                                      align_corners=False)
        lr_ycbcr_4_up = lr_ycbcr_4_up.view(-1, 3, h, w)

        return gt_y, lr_ycbcr_4_up, lr_y_4

    def __len__(self):
        total_len = 0
        for i in range(len(self.h5files)):
            total_len += self.lens[i]

        return total_len




