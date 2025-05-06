# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import os
import random
import shutil
import torch
import torch.nn as nn
from data import FSAD_Dataset_train, FSAD_Dataset_test
from torchvision.datasets import ImageFolder
import numpy as np
import random
import os
from torch.utils.data import DataLoader
from resnet import resnet18, resnet34, resnet50, wide_resnet50_2
from de_resnet import de_resnet18, de_resnet34, de_wide_resnet50_2, de_resnet50
import torch.backends.cudnn as cudnn
import argparse

from test import evaluation
from torch.nn import functional as F
from learn_to_weight import Learn_to_weight

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def loss_fucntion(a, b):
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    for item in range(len(a)):
        loss += torch.mean(1 - cos_loss(a[item].view(a[item].shape[0], -1), b[item].view(b[item].shape[0], -1)))
    return loss

def train(_class_, shot):
    epochs = 70  # 70
    learning_rate = 0.0001  # 0.0001
    batch_size = 16  # 16 or 32 or bigger
    image_size = 128  # 128

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    root_path = '/home/medfsad/'
    train_dataset = FSAD_Dataset_train(root_path, _class_, is_train=True, resize=image_size,
                                       shot=shot, batch=1)
    print(len(train_dataset))
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=8)
    test_dataset = FSAD_Dataset_test(root_path, _class_, is_train=False, resize=image_size,
                                     shot=shot)
    print(len(test_dataset))
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=8)

    encoder, _ = wide_resnet50_2(pretrained=True)

    decoder = de_wide_resnet50_2(pretrained=False)
    
    learning_weight = Learn_to_weight()
    
    # ckp_path = f'./checkpoints/testbatch1_{shot}_{_class_}.pth'
    # checkpoint = torch.load(ckp_path)
    # decoder.load_state_dict(checkpoint['decoder'])
    # learning_weight.load_state_dict(checkpoint['weight'])
    
    if torch.cuda.device_count() > 1:
        encoder = nn.DataParallel(encoder)
        decoder = nn.DataParallel(decoder)
        
    encoder = encoder.to(device)
    encoder.eval()
    decoder = decoder.to(device)
    learning_weight = learning_weight.to(device)
    
    optimizer = torch.optim.Adam(list(decoder.parameters()) , lr=learning_rate, betas=(0.5, 0.999))

    # img_roc_auc = evaluation(encoder, decoder, test_dataloader, device, _class_, shot, batch_size, image_size, learning_weight)
    # print('img_roc_auc:{:.4f}--class:{}--shot:{}'.format(img_roc_auc, _class_, shot))
    
    best_img_roc_auc = 0.0
    
    for epoch in range(epochs):
        decoder.train()
        learning_weight.train()
        
        loss_list = []
        for query_img, support_img_list, _ in train_dataloader:
            query_img = query_img.permute(1, 0, 2, 3, 4)
            query_img = query_img.squeeze(0).to(device)

            e4, f3 = encoder(query_img)
            query_outputs = decoder(e4)

            loss2 = loss_fucntion(f3, query_outputs)

            support_img_list = support_img_list.permute(1, 2, 0, 3, 4, 5)
            support_img = support_img_list.squeeze(0).to(device)

            K, B, C, H, W = support_img.shape

            support_outputs_list = [torch.zeros(K, B, 256, 32, 32).to(device), torch.zeros(K, B, 512, 16, 16).to(device), torch.zeros(K, B, 1024, 8, 8).to(device)]

            for k, sup_img in enumerate(support_img):
                Se4, Sf3 = encoder(sup_img)
                support_outputs = decoder(Se4)
                for i in range(3):
                    support_outputs_list[i][k, :, :, :, :] = support_outputs[i]

            support_outputs_weighted = learning_weight(query_outputs, support_outputs_list)

            loss1 = loss_fucntion(query_outputs, support_outputs_weighted)

            loss = loss1 + 0.1 * loss2
            # loss = loss1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
            # scheduler.step()
        print('epoch [{}/{}], loss:{:.8f}'.format(epoch + 1, epochs, np.mean(loss_list)))

        img_roc_auc = evaluation(encoder, decoder, test_dataloader, device, _class_, shot, batch_size, image_size, learning_weight)
        
        if img_roc_auc > best_img_roc_auc:
            best_img_roc_auc = img_roc_auc
            ckp_path = f'./checkpoints/best_{shot}_{_class_}.pth'
            torch.save({'decoder': decoder.state_dict(), 'weight': learning_weight.state_dict()}, ckp_path)

        print('img_roc_auc:{:.4f}----best_img_roc_auc:{:.4f}--class:{}---shot:{}'.format(img_roc_auc, best_img_roc_auc, _class_, shot))
        


if __name__ == '__main__':

    setup_seed(111)
    item_list = ['His', 'LAG', 'APTOS', 'RSNA', 'BrainTumor']
    shot = [2, 4, 8]
    for i in item_list:
        for s in shot:
            train(i, s)
