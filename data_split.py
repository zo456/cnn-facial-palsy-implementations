import os
import natsort
import numpy as np
import torch
import torch.utils.data as data
from sklearn.model_selection import train_test_split

def data_split(data_dir, split, temporal):

    subdirs = os.listdir(data_dir)
    temp = []
    for subdir in subdirs:
        if subdir.endswith('stroke_arr'):
            temp.append(subdir)

    subdirs = temp

    images0 = []
    images1 = []
    images2 = []
    images3 = []
    images4 = []

    for subdir in subdirs:
        if subdir.startswith('stroke'):
            label = 1
        else:
            label = 0

        img_dict = {}
        if not temporal:
            img_list = os.listdir(data_dir + os.sep + subdir)
        else: 
            img_list = natsort.natsorted(os.listdir(data_dir + os.sep + subdir))

        for item in img_list:
            patient_id = item.split('_')
            if patient_id[0] not in img_dict.keys():
                img_dict[patient_id[0]] = []
            img_dict[patient_id[0]].append(data_dir + os.sep + subdir + os.sep + item)
        set_, set0 = train_test_split(list(img_dict.keys()), test_size=0.2, shuffle=True, random_state=3546)
        set_, set1 = train_test_split(set_, test_size=0.25, shuffle=True, random_state=45)
        set_, set2 = train_test_split(set_, test_size=0.33, shuffle=True, random_state=45)
        set3, set4 = train_test_split(set_, test_size=0.5, shuffle=True, random_state=45)

        if not temporal:
            for patient in set0:
                for k in range(0, len(img_dict[patient])):
                    images0.append((img_dict[patient][k], label))
            for patient in set1:
                for k in range(0, len(img_dict[patient])):
                    images1.append((img_dict[patient][k], label))
            for patient in set2:
                for k in range(0, len(img_dict[patient])):
                    images2.append((img_dict[patient][k], label))
            for patient in set3:
                for k in range(0, len(img_dict[patient])):
                    images3.append((img_dict[patient][k], label))
            for patient in set4:
                for k in range(0, len(img_dict[patient])):
                    images4.append((img_dict[patient][k], label))
        else:
            for patient in set0:
                for k in range(0, len(img_dict[patient]) - 3):
                    images0.append(([img_dict[patient][k], img_dict[patient][k + 1], img_dict[patient][k + 2], img_dict[patient][k + 3]], label))
            for patient in set1:
                for k in range(0, len(img_dict[patient]) - 3):
                    images1.append(([img_dict[patient][k], img_dict[patient][k + 1], img_dict[patient][k + 2], img_dict[patient][k + 3]], label))
            for patient in set2:
                for k in range(0, len(img_dict[patient]) - 3):
                    images2.append(([img_dict[patient][k], img_dict[patient][k + 1], img_dict[patient][k + 2], img_dict[patient][k + 3]], label))
            for patient in set3:
                for k in range(0, len(img_dict[patient]) - 3):
                    images3.append(([img_dict[patient][k], img_dict[patient][k + 1], img_dict[patient][k + 2], img_dict[patient][k + 3]], label))
            for patient in set4:
                for k in range(0, len(img_dict[patient]) - 3):
                    images4.append(([img_dict[patient][k], img_dict[patient][k + 1], img_dict[patient][k + 2], img_dict[patient][k + 3]], label))
            
        if split == 0:
            images_train = images0 + images1 + images2
            images_val = images3
            images_test = images4
        if split == 1:
            images_train = images1 + images2 + images3
            images_val = images4
            images_test = images0
        if split == 2:
            images_train = images2 + images3 + images4
            images_val = images0
            images_test = images1
        if split == 3:
            images_train = images3 + images4 + images0
            images_val = images1
            images_test = images2
        if split == 4:
            images_train = images4 + images0 + images1
            images_val = images2
            images_test = images3
        
    return images_train, images_val, images_test

def data_no_split(data_dir, temporal):
    subdirs = os.listdir(data_dir)
    temp = []
    for subdir in subdirs:
        if subdir.endswith('stroke_arr'):
            temp.append(subdir)

    subdirs = temp

    images = []

    for subdir in subdirs:
        if subdir.startswith('stroke'):
            label = 1
        else:
            label = 0

        img_dict = {}
        if not temporal:
            img_list = os.listdir(data_dir + os.sep + subdir)
        else:
            img_list = natsort.natsorted(os.listdir(data_dir + os.sep + subdir))
        
        for item in img_list:
            patient_id = item.split('_')
            if patient_id[0] not in img_dict.keys():
                img_dict[patient_id[0]] = []
            img_dict[patient_id[0]].append(data_dir + os.sep + subdir + os.sep + item)

        if not temporal:
            for patient in list(img_dict.keys()):
                for k in range(0, len(img_dict[patient])):
                    images.append((img_dict[patient][k], label))
        else:
            for patient in list(img_dict.keys()):
                for k in range(0, len(img_dict[patient]) - 3):
                    images.append(([img_dict[patient][k], img_dict[patient][k + 1], img_dict[patient][k + 2], img_dict[patient][k + 3]], label))

    return images

class ImageDataset(data.Dataset):
    def __init__(self, data_dir, split=0, mode='train', transform=None, temporal=False):
        #super().__init__(data_dir, mode, transform)
        self.train_list, self.val_list, self.test_list = data_split(data_dir, split, temporal)
        self.transform = transform
        self.mode = mode
        self.temporal = temporal
        
    def __getitem__(self, idx):
        if self.mode == 'train':
            if not self.temporal:
                path, target = self.train_list[idx]
                image = np.load(path)
                if self.transform is not None:
                    frame = self.transform(image)
                    image_set = torch.cat([frame], dim=0)
                return image_set, target
            else:
                path, target = self.train_list[idx]
                images = [np.load(path[0]), np.load(path[1]), np.load(path[2]), np.load(path[3])]
                if self.transform is not None:
                    frame0 = self.transform(images[0])
                    frame1 = self.transform(images[1])
                    frame2 = self.transform(images[2])
                    frame3 = self.transform(images[3])
                    image_set = torch.cat([frame0, frame1, frame2, frame3], dim=0)
                return image_set, target
        if self.mode == 'val':
            if not self.temporal:
                path, target = self.val_list[idx]
                image = np.load(path)
                if self.transform is not None:
                    frame = self.transform(image)
                    image_set = torch.cat([frame], dim=0)
                return image_set, target
            else:
                path, target = self.val_list[idx]
                images = [np.load(path[0]), np.load(path[1]), np.load(path[2]), np.load(path[3])]
                if self.transform is not None:
                    frame0 = self.transform(images[0])
                    frame1 = self.transform(images[1])
                    frame2 = self.transform(images[2])
                    frame3 = self.transform(images[3])
                    image_set = torch.cat([frame0, frame1, frame2, frame3], dim=0)
                return image_set, target
        if self.mode == 'test':
            if not self.temporal:
                path, target = self.test_list[idx]
                image = np.load(path)
                if self.transform is not None:
                    frame = self.transform(image)
                    image_set = torch.cat([frame], dim=0)
                return image_set, target
            else:
                path, target = self.test_list[idx]
                images = [np.load(path[0]), np.load(path[1]), np.load(path[2]), np.load(path[3])]
                if self.transform is not None:
                    frame0 = self.transform(images[0])
                    frame1 = self.transform(images[1])
                    frame2 = self.transform(images[2])
                    frame3 = self.transform(images[3])
                    image_set = torch.cat([frame0, frame1, frame2, frame3], dim=0)
                return image_set, target
    
    def __len__(self):
        if self.mode == 'train':
            return len(self.train_list)
        elif self.mode == 'val':
            return len(self.val_list)
        elif self.mode == 'test':
            return len(self.test_list)

class CrossDataset(data.Dataset):
    def __init__(self, data_dir, transform=None, temporal=False):
        self.test_list = data_no_split(data_dir, temporal)
        self.transform = transform
        self.temporal = temporal

    def __getitem__(self, idx):
        if not self.temporal:
            path, target = self.test_list[idx]
            image = np.load(path)
            if self.transform is not None:
                frame = self.transform(image)
                images = torch.cat([frame], dim=0)
            return images, target
        else:
            path, target = self.test_list[idx]
            images = [np.load(path[0]), np.load(path[1]), np.load(path[2]), np.load(path[3])]
            if self.transform is not None:
                frame0 = self.transform(images[0])
                frame1 = self.transform(images[1])
                frame2 = self.transform(images[2])
                frame3 = self.transform(images[3])
                image_set = torch.cat([frame0, frame1, frame2, frame3], dim=0)
            return image_set, target

    def __len__(self):
        return len(self.test_list)      
        
        
        
        