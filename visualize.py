import os
import torch
import torchvision as tv
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import argparse
import skimage.data
import skimage.io
import skimage.transform
import numpy as np
import matplotlib.pyplot as plt
import torchvision.models as models
from PIL import Image
import cv2
from nets.unet import Unet


class FeatureExtractor(nn.Module):
    def __init__(self, submodule, extracted_layers):
        super(FeatureExtractor, self).__init__()
        self.submodule = submodule
        self.extracted_layers = extracted_layers

    def forward(self, x):
        outputs = {}
        # for name, module in self.submodule._modules.items():
        #     if "fc" in name:
        #         x = x.view(x.size(0), -1)
        #
        #     x = module(x)
        #     print(name)
        #     if self.extracted_layers is None or name in self.extracted_layers and 'fc' not in name:
        #         outputs[name] = x

        ################修改成自己的网络，直接在network.py中return你想输出的层

        feat1, feat2, feat3, feat4, feat5, up4, up3, up2, up1, final = self.submodule(x)

        outputs["feat1"] = feat1
        outputs["feat2"] = feat2
        outputs["feat3"] = feat3
        outputs["feat4"] = feat4
        outputs["feat5"] = feat5
        outputs["up4"] = up4
        outputs["up3"] = up3
        outputs["up2"] = up2
        outputs["up1"] = up1
        outputs["final"] = final


        # return outputs
        return outputs


def get_picture(pic_name, transform):
    img = skimage.io.imread(pic_name)
    img = skimage.transform.resize(img, (224, 224))
    img = np.asarray(img, dtype=np.float32)
    return transform(img)


def make_dirs(path):
    if os.path.exists(path) is False:
        os.makedirs(path)


def get_feature():
    therd_size = 256  # 有些图太小，会放大到这个尺寸
    pic_dir = './2007_000027.jpg'  # 往网络里输入一张图片
    resize_img = cv2.imread(pic_dir)
    resize_img = cv2.resize(resize_img, (therd_size, therd_size), interpolation=cv2.INTER_NEAREST)
    transform = transforms.ToTensor()
    img = get_picture(pic_dir, transform)
    #device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    # 插入维度
    img = img.unsqueeze(0)

    img = img.to(device)

    model = Unet(num_classes = 21, pretrained = False, backbone = 'resnet50')
    model.load_state_dict(torch.load('./unet_resnet_voc.pth', map_location=torch.device('cpu')))
    net = model.to(device)
    exact_list = None
    # exact_list = ['conv1_block',""]
    dst = './features'  # 保存的路径

    myexactor = FeatureExtractor(net, exact_list)
    outs = myexactor(img)
    for k, v in outs.items():
        features = v[0]
        iter_range = features.shape[0]
        for i in range(iter_range):
            # plt.imshow(x[0].data.numpy()[0,i,:,:],cmap='jet')
            if 'fc' in k:
                continue

            feature = features.data.cpu().numpy()
            feature_img = feature[i, :, :]
            feature_img = np.asarray(feature_img * 255, dtype=np.uint8)


            dst_path = os.path.join(dst, k)
            if not os.path.exists(dst_path + './feature/'):
                os.makedirs(dst_path + './feature/')

            feature = Image.fromarray(feature_img)
            feature.save(dst_path + './feature/' + str(i) + '_feature.png')#保存特征图
            feature = cv2.imread(dst_path + './feature/' + str(i) + '_feature.png')
            feature = cv2.resize(feature, (therd_size, therd_size), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(dst_path + './feature/' + str(i) + '_feature.png', feature)#resize特征图

            make_dirs(dst_path)
            feature_img = cv2.applyColorMap(feature_img, cv2.COLORMAP_JET)
            if feature_img.shape[0] < therd_size:
                if not os.path.exists(dst_path + './featmap/'):
                    os.makedirs(dst_path + './featmap/')
                tmp_file = os.path.join(dst_path + './featmap/', str(i) + '_' + str(therd_size) + '.png')
                tmp_img = feature_img.copy()
                tmp_img = cv2.resize(tmp_img, (therd_size, therd_size), interpolation=cv2.INTER_NEAREST)

                cv2.imwrite(tmp_file, tmp_img)#resize热力图
                #img_heatmap_activations = cv2.addWeighted(tmp_img, 0.5, resize_img, 0.5, 0)
                #cv2.imwrite(dst_path + str(i) + '_merge' + '.png', img_heatmap_activations)#图像与热力图结合

            #dst_file = os.path.join(dst_path, str(i) + '.png')
            #cv2.imwrite(dst_file, feature_img)#保存热力图


if __name__ == '__main__':
    get_feature()
