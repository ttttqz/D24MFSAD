import random
import torch
from PIL import Image
from torchvision.datasets import ImageFolder
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader

from resnet import resnet18, resnet34, resnet50, wide_resnet50_2
from de_resnet import de_resnet18, de_resnet50, de_wide_resnet50_2
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score
import cv2
from scipy.ndimage import gaussian_filter
import os
import shutil
from matplotlib import pyplot as plt
from data import FSAD_Dataset_train, FSAD_Dataset_test
from learn_to_weight import Learn_to_weight


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def cal_anomaly_map(fs_list, ft_list, out_size=224, amap_mode='mul'):
    if amap_mode == 'mul':
        anomaly_map = np.ones([fs_list[0].shape[0], out_size, out_size])
    else:
        anomaly_map = np.zeros([fs_list[0].shape[0], out_size, out_size])
    a_map_list = []
    for i in range(len(ft_list)):
        fs = fs_list[i]
        ft = ft_list[i]
        # fs_norm = F.normalize(fs, p=2)
        # ft_norm = F.normalize(ft, p=2)
        a_map = 1 - F.cosine_similarity(fs, ft)
        a_map = torch.unsqueeze(a_map, dim=1)
        a_map = F.interpolate(a_map, size=out_size, mode='bilinear', align_corners=True)
        a_map = a_map[:, 0, :, :].to('cpu').detach().numpy()
        a_map_list.append(a_map)
        if amap_mode == 'mul':
            anomaly_map *= a_map
        else:
            anomaly_map += a_map
    return anomaly_map, a_map_list


def show_cam_on_image(img, anomaly_map):
    # if anomaly_map.shape != img.shape:
    #    anomaly_map = cv2.applyColorMap(np.uint8(anomaly_map), cv2.COLORMAP_JET)
    cam = np.float32(anomaly_map) / 255 + np.float32(img) / 255
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

def min_max_norm(image):
    a_min, a_max = image.min(), image.max()
    return (image - a_min) / (a_max - a_min)

def cvt2heatmap(gray):
    heatmap = cv2.applyColorMap(np.uint8(gray), cv2.COLORMAP_JET)
    return heatmap

def load_and_resize(image_path, size):
    image = Image.open(image_path).convert('RGB')
    image = image.resize((size, size), Image.LANCZOS)
    return image

def evaluation(encoder, decoder, dataloader, device, _class_, shot, batch, size, learning_weight):

    decoder.eval()
    learning_weight.eval()
    score = []
    gt_list = []

    # loading support set
    train_folder = f'/home/medfsad/med-kshot/{_class_}/train/shot{shot}/good{shot}/' # 支撑集路径
    image_paths = [os.path.join(train_folder, f) for f in os.listdir(train_folder) if
                   f.endswith(('.png', '.jpg'))]
    images = [load_and_resize(img_path, size) for img_path in image_paths]
    tensor_images = torch.stack([torch.FloatTensor(np.array(img)) for img in images])
    tensor_images = tensor_images.permute(0, 3, 1, 2)
    tensor_images_normalized = tensor_images / 255.0
    support_img = tensor_images_normalized.to(device)
    support_img = support_img.repeat(batch, 1, 1, 1, 1)
    support_img = support_img.permute(1, 0, 2, 3, 4)
    
    K, B, C, H, W = support_img.shape
    with torch.no_grad():
        for query_img, label in dataloader:
            query_img = query_img.to(device)
            e4, f3 = encoder(query_img)
            query_outputs = decoder(e4)
            support_outputs_list = [torch.zeros(K, B, 256, 32, 32).to(device), torch.zeros(K, B, 512, 16, 16).to(device), torch.zeros(K, B, 1024, 8, 8).to(device)]

            for k, sup_img in enumerate(support_img):
                Se4, Sf3 = encoder(sup_img)
                support_outputs = decoder(Se4)
                for i in range(3):
                    support_outputs_list[i][k, :, :, :, :] = support_outputs[i]

            support_outputs_weighted = learning_weight(query_outputs, support_outputs_list)
            anomaly_map, _ = cal_anomaly_map(query_outputs, support_outputs_weighted, query_img.shape[-1], amap_mode='a')

            anomaly_map = gaussian_filter(anomaly_map, sigma=4)
            
            for anomaly_map_one in anomaly_map:
                score.append(anomaly_map_one)

            # # use for visual
            # for query_img_one in query_img:
            #     img.append(query_img_one)
            
            for i in label:
                gt_list.append(i)
            
        scores = np.asarray(score)
        
        # Normalization
        max_anomaly_score = scores.max()
        min_anomaly_score = scores.min()
        scores = (scores - min_anomaly_score) / (max_anomaly_score - min_anomaly_score)
        
        # # visual
        # for img, label in zip(img, scores):
        #     ano_map = label
        #     ano_map = cvt2heatmap(ano_map * 255)
        #     img = cv2.cvtColor(img.permute(1, 2, 0).cpu().numpy() * 255, cv2.COLOR_BGR2RGB)
        #     img = np.uint8(min_max_norm(img) * 255)
        #     if not os.path.exists('./results_vis/'+_class_):
        #        os.makedirs('./results_vis/'+_class_)
        #     cv2.imwrite('./results_vis/'+_class_+'/'+str(count)+'_'+'org.png',img)
        #     ano_map = show_cam_on_image(img, ano_map)
        #     cv2.imwrite('./results_vis/'+_class_+'/'+str(count)+'_'+'ad.png', ano_map)
        #     count += 1
        
        # calculate image-level ROC AUC score
        img_scores = scores.reshape(scores.shape[0], -1).max(axis=1)
        gt_list = np.asarray(gt_list)
        img_roc_auc = roc_auc_score(gt_list, img_scores)

    return img_roc_auc


def test(_class_, shot):
    batch_size = 16  # 16
    image_size = 128  # 128

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    root_path = '/home/medfsad/'

    test_dataset = FSAD_Dataset_test(root_path, _class_, is_train=False, resize=image_size,
                                     shot=shot)
    print(len(test_dataset))
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True,
                                                  num_workers=8)

    encoder, _ = wide_resnet50_2(pretrained=True)
    decoder = de_wide_resnet50_2(pretrained=False)

    learning_weight = Learn_to_weight()

    ckp_path = f'./checkpoints/best_{shot}_{_class_}.pth'
    checkpoint = torch.load(ckp_path)
    decoder.load_state_dict(checkpoint['decoder'])
    learning_weight.load_state_dict(checkpoint['weight'])

    if torch.cuda.device_count() > 1:
        encoder = nn.DataParallel(encoder)
        decoder = nn.DataParallel(decoder)

    encoder = encoder.to(device)
    encoder.eval()
    decoder = decoder.to(device)
    learning_weight = learning_weight.to(device)

    img_roc_auc = evaluation(encoder, decoder, test_dataloader, device, _class_, shot, batch_size, image_size, learning_weight)
    print('img_roc_auc:{:.4f}--class:{}--shot:{}'.format(img_roc_auc, _class_, shot))


if __name__ == '__main__':

    setup_seed(111)
    item_list = ['His', 'LAG', 'APTOS', 'RSNA', 'BrainTumor']
    shot = [2, 4, 8]
    for i in item_list:
        for s in shot:
            test(i, s)
