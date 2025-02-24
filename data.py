import os
from PIL import Image
import random
import math
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

CLASS_NAMES = ['APTOS', 'RSNA', 'LAG', 'His', 'BrainTumor']


class FSAD_Dataset_train(Dataset):
    def __init__(self,
                 dataset_path='',
                 class_name='RSNA',
                 is_train=True,
                 resize=256,
                 shot=2,
                 batch=32
                 ):
        assert class_name in CLASS_NAMES, 'class_name: {}, should be in {}'.format(class_name, CLASS_NAMES)
        self.dataset_path = dataset_path
        self.class_name = class_name
        self.is_train = is_train
        self.resize = resize
        self.shot = shot
        self.batch = batch
        # load dataset
        self.query_dir, self.support_dir = self.load_dataset_folder()
        # set transforms
        self.transform_x = transforms.Compose([
            transforms.Resize((resize, resize), Image.LANCZOS),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __getitem__(self, idx):
        query_list, support_list = self.query_dir[idx], self.support_dir[idx]
        query_img = None
        support_sub_img = None
        support_img = None

        for i in range(len(query_list)):
            image = Image.open(query_list[i]).convert('RGB')
            image = self.transform_x(image)
            image = image.unsqueeze(dim=0)
            if query_img is None:
                query_img = image
            else:
                query_img = torch.cat([query_img, image], dim=0)
            for k in range(self.shot):
                image = Image.open(support_list[i][k]).convert('RGB')
                image = self.transform_x(image)
                image = image.unsqueeze(dim=0)
                if support_sub_img is None:
                    support_sub_img = image
                else:
                    support_sub_img = torch.cat([support_sub_img, image], dim=0)

            support_sub_img = support_sub_img.unsqueeze(dim=0)
            if support_img is None:
                support_img = support_sub_img
            else:
                support_img = torch.cat([support_img, support_sub_img], dim=0)

            support_sub_img = None

        mask = torch.zeros([self.batch, self.resize, self.resize])
        return query_img, support_img, mask

    def __len__(self):
        return len(self.query_dir)

    def shuffle_dataset(self):
        phase = 'train' if self.is_train else 'test'

        data_img = {}

        for class_name_one in CLASS_NAMES:
            if class_name_one != self.class_name: # other class use for training
                data_img[class_name_one] = []
                img_dir = os.path.join(self.dataset_path, class_name_one, phase, 'good')
                img_types = sorted(os.listdir(img_dir))
                for img_type in img_types:
                    img_type_dir = os.path.join(img_dir, img_type)
                    data_img[class_name_one].append(img_type_dir)
                random.shuffle(data_img[class_name_one])

        query_dir, support_dir = [], []
        for class_name_one in data_img.keys():
            for image_index in range(0, len(data_img[class_name_one]), self.batch):
                query_sub_dir = []
                support_sub_dir = []
                for batch_count in range(0, self.batch):
                    if image_index + batch_count >= len(data_img[class_name_one]):
                        break
                    image_dir_one = data_img[class_name_one][image_index + batch_count]
                    support_dir_one = []
                    query_sub_dir.append(image_dir_one)
                    for k in range(self.shot):
                        random_choose = random.randint(0, (len(data_img[class_name_one]) - 1))
                        while data_img[class_name_one][random_choose] == image_dir_one:
                            random_choose = random.randint(0, (len(data_img[class_name_one]) - 1))
                        support_dir_one.append(data_img[class_name_one][random_choose])
                    support_sub_dir.append(support_dir_one)

                query_dir.append(query_sub_dir)
                support_dir.append(support_sub_dir)

        assert len(query_dir) == len(support_dir), 'number of query_dir and support_dir should be same'
        self.query_dir = query_dir
        self.support_dir = support_dir

    def load_dataset_folder(self):
        phase = 'train' if self.is_train else 'test'

        data_img = {}
        for class_name_one in CLASS_NAMES:
            if class_name_one != self.class_name:
                data_img[class_name_one] = []
                img_dir = os.path.join(self.dataset_path, class_name_one, phase, 'good')
                img_types = sorted(os.listdir(img_dir))
                for img_type in img_types:
                    img_type_dir = os.path.join(img_dir, img_type)
                    data_img[class_name_one].append(img_type_dir)
                random.shuffle(data_img[class_name_one])

        query_dir, support_dir = [], []
        for class_name_one in data_img.keys():

            for image_index in range(0, len(data_img[class_name_one]), self.batch):
                query_sub_dir = []
                support_sub_dir = []

                for batch_count in range(0, self.batch):
                    if image_index + batch_count >= len(data_img[class_name_one]):
                        break
                    image_dir_one = data_img[class_name_one][image_index + batch_count]
                    support_dir_one = []
                    query_sub_dir.append(image_dir_one)
                    for k in range(self.shot):
                        random_choose = random.randint(0, (len(data_img[class_name_one]) - 1))
                        while data_img[class_name_one][random_choose] == image_dir_one:
                            random_choose = random.randint(0, (len(data_img[class_name_one]) - 1))
                        support_dir_one.append(data_img[class_name_one][random_choose])
                    support_sub_dir.append(support_dir_one)

                query_dir.append(query_sub_dir)
                support_dir.append(support_sub_dir)

        assert len(query_dir) == len(support_dir), 'number of query_dir and support_dir should be same'
        return query_dir, support_dir


class FSAD_Dataset_test(Dataset):
    def __init__(self,
                 dataset_path='',
                 class_name='RSNA',
                 is_train=True,
                 resize=256,
                 shot=2
                 ):
        assert class_name in CLASS_NAMES, 'class_name: {}, should be in {}'.format(class_name, CLASS_NAMES)
        self.dataset_path = dataset_path
        self.class_name = class_name
        self.is_train = is_train
        self.resize = resize
        self.shot = shot
        # load dataset
        self.query_dir = self.load_dataset_folder()
        # set transforms
        self.transform_x = transforms.Compose([
            transforms.Resize((resize, resize), Image.LANCZOS),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.transform_mask = transforms.Compose(
            [transforms.Resize((resize, resize), Image.LANCZOS),
             transforms.ToTensor()])

    def __getitem__(self, idx):
        query_one = self.query_dir[idx]
        query_img = Image.open(query_one).convert('RGB')
        query_img = self.transform_x(query_img)

        if 'good' in query_one and 'ungood' not in query_one: # good/ungood with no mask label
            y = 0
        else:
            y = 1
        return query_img, y

    def __len__(self):
        return len(self.query_dir)

    def load_dataset_folder(self):
        phase = 'train' if self.is_train else 'test'

        data_img = {}
        img_dir = os.path.join(self.dataset_path, self.class_name, phase)
        img_types = sorted(os.listdir(img_dir))
        for img_type in img_types:
            data_img[img_type] = []
            img_type_dir = os.path.join(img_dir, img_type)
            img_num = sorted(os.listdir(img_type_dir))
            for img_one in img_num:
                img_dir_one = os.path.join(img_type_dir, img_one)
                data_img[img_type].append(img_dir_one)
        img_dir_train = os.path.join(self.dataset_path, self.class_name, 'train', 'good')
        img_num = sorted(os.listdir(img_dir_train))

        data_train = []
        for img_one in img_num:
            img_dir_one = os.path.join(img_dir_train, img_one)
            data_train.append(img_dir_one)

        query_dir = []
        for img_type in data_img.keys():
            for image_dir_one in data_img[img_type]:
                query_dir.append(image_dir_one)

        return query_dir