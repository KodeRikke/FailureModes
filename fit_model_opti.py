
# This will cause an error after about 8 hours, but it is faster sooo...

path = "./"

# libraries
import re
from tkinter import Y
import math
import os
import copy
import cv2
import heapq
import matplotlib.image as plt
import numpy as np
import time
from torch.autograd import Variable
from PIL import Image
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets

model_dir = path + 'pretrained_models/'

# functions
def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def log(line, file):
  with open(path + file, 'a+') as log:
      content = log.read()
      log.write(content + line + str("\n"))

def list_of_distances(X, Y):
    return torch.sum((torch.unsqueeze(X, dim=2) - torch.unsqueeze(Y.t(), dim=0)) ** 2, dim=1)

def preprocess(x):
    mean, std, y = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), torch.zeros_like(x)
    for i in range(3):
        y[:, i, :, :] = (x[:, i, :, :] - mean[i]) / std[i]
    return y

def undo_preprocess(x):
    mean, std, y = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), torch.zeros_like(x)
    for i in range(3):
        y[:, i, :, :] = x[:, i, :, :] * std[i] + mean[i]
    return y

def find_high_activation_crop(activation_map, percentile=95):
    threshold = np.percentile(activation_map, percentile)
    mask = np.ones(activation_map.shape)
    mask[activation_map < threshold] = 0
    lower_y, upper_y, lower_x, upper_x = 0, 0, 0, 0
    for i in range(mask.shape[0]):
        if np.amax(mask[i]) > 0.5:
            lower_y = i
            break
    for i in reversed(range(mask.shape[0])):
        if np.amax(mask[i]) > 0.5:
            upper_y = i
            break
    for j in range(mask.shape[1]):
        if np.amax(mask[:,j]) > 0.5:
            lower_x = j
            break
    for j in reversed(range(mask.shape[1])):
        if np.amax(mask[:,j]) > 0.5:
            upper_x = j
            break
    return lower_y, upper_y+1, lower_x, upper_x+1

def save_prototype_original_img_with_bbox(fname, epoch, index, bbox_height_start, bbox_height_end, bbox_width_start, bbox_width_end):
    p_img_bgr = cv2.imread(os.path.join(load_img_dir, 'epoch-'+str(epoch), 'prototype-img-original'+str(index)+'.png'))
    cv2.rectangle(p_img_bgr, (bbox_width_start, bbox_height_start), (bbox_width_end-1, bbox_height_end-1), (0, 255, 255), thickness=2)
    p_img_rgb = np.float32(p_img_bgr[...,::-1]) / 255
    plt.imsave(fname, p_img_rgb)

# find nearest
def imsave_with_bbox(fname, img_rgb, bbox_height_start, bbox_height_end,
                     bbox_width_start, bbox_width_end, color=(0, 255, 255)):
    img_bgr_uint8 = cv2.cvtColor(np.uint8(255*img_rgb), cv2.COLOR_RGB2BGR)
    cv2.rectangle(img_bgr_uint8, (bbox_width_start, bbox_height_start), (bbox_width_end-1, bbox_height_end-1),
                  color, thickness=2)
    img_rgb_uint8 = img_bgr_uint8[...,::-1]
    img_rgb_float = np.float32(img_rgb_uint8) / 255
    plt.imsave(fname, img_rgb_float)

class ImagePatch:
    def __init__(self, patch, label, distance, original_img=None, act_pattern=None, patch_indices=None):
        self.patch = patch
        self.label = label
        self.negative_distance = -distance
        self.original_img = original_img
        self.act_pattern = act_pattern
        self.patch_indices = patch_indices
    def __lt__(self, other):
        return self.negative_distance < other.negative_distance

class ImagePatchInfo:
    def __init__(self, label, distance):
        self.label = label
        self.negative_distance = -distance
    def __lt__(self, other):
        return self.negative_distance < other.negative_distance

def find_k_nearest_patches_to_prototypes(dataloader, prototype_network_parallel, k=5, full_save=False, # save all the images
                                         root_dir_for_saving_images='./nearest', prototype_activation_function_in_numpy=None):
    prototype_network_parallel.eval()

    log('find nearest patches', "analysislog.txt")
    start = time.time()
    n_prototypes = prototype_network_parallel.module.num_prototypes
    
    prototype_shape = prototype_network_parallel.module.prototype_shape
    max_dist = prototype_shape[1] * prototype_shape[2] * prototype_shape[3]

    protoL_rf_info = prototype_network_parallel.module.proto_layer_rf_info

    heaps = []
    for _ in range(n_prototypes):
        heaps.append([])

    for idx, (search_batch_input, search_y) in enumerate(dataloader):
        print('batch {}'.format(idx))
        search_batch = search_batch_input
        with torch.no_grad():
            search_batch = search_batch.cuda()
            proto_dist_torch = prototype_network_parallel.module.push_forward(search_batch)

        proto_dist = np.copy(proto_dist_torch.detach().cpu().numpy())

        for img_idx, distance_map in enumerate(proto_dist):
            for j in range(n_prototypes):
                closest_patch_distance_to_prototype_j = np.amin(distance_map[j])

                if full_save:
                    closest_patch_indices_in_distance_map_j = \
                        list(np.unravel_index(np.argmin(distance_map[j],axis=None),
                                              distance_map[j].shape))
                    closest_patch_indices_in_distance_map_j = [0] + closest_patch_indices_in_distance_map_j
                    closest_patch_indices_in_img = \
                        compute_rf_prototype(search_batch.size(2),
                                             closest_patch_indices_in_distance_map_j,
                                             protoL_rf_info)
                    closest_patch = \
                        search_batch_input[img_idx, :,
                                           closest_patch_indices_in_img[1]:closest_patch_indices_in_img[2],
                                           closest_patch_indices_in_img[3]:closest_patch_indices_in_img[4]]
                    closest_patch = closest_patch.numpy()
                    closest_patch = np.transpose(closest_patch, (1, 2, 0))

                    original_img = search_batch_input[img_idx].numpy()
                    original_img = np.transpose(original_img, (1, 2, 0))

                    if prototype_network_parallel.module.prototype_activation_function == 'log':
                        act_pattern = np.log((distance_map[j] + 1)/(distance_map[j] + prototype_network_parallel.module.epsilon))
                    elif prototype_network_parallel.module.prototype_activation_function == 'linear':
                        act_pattern = max_dist - distance_map[j]
                    else:
                        act_pattern = prototype_activation_function_in_numpy(distance_map[j])

                    patch_indices = closest_patch_indices_in_img[1:5]

                    closest_patch = ImagePatch(patch=closest_patch,
                                               label=search_y[img_idx],
                                               distance=closest_patch_distance_to_prototype_j,
                                               original_img=original_img,
                                               act_pattern=act_pattern,
                                               patch_indices=patch_indices)
                else:
                    closest_patch = ImagePatchInfo(label=search_y[img_idx],
                                                   distance=closest_patch_distance_to_prototype_j)

                if len(heaps[j]) < k:
                    heapq.heappush(heaps[j], closest_patch)
                else:
                    heapq.heappushpop(heaps[j], closest_patch)

    for j in range(n_prototypes):
        heaps[j].sort()
        heaps[j] = heaps[j][::-1]

        if full_save:

            dir_for_saving_images = os.path.join(root_dir_for_saving_images,
                                                 str(j))
            makedir(dir_for_saving_images)

            labels = []

            for i, patch in enumerate(heaps[j]):
                # save the activation pattern of the original image where the patch comes from
                np.save(os.path.join(dir_for_saving_images,
                                     'nearest-' + str(i+1) + '_act.npy'),
                        patch.act_pattern)
                
                # save the original image where the patch comes from
                plt.imsave(fname=os.path.join(dir_for_saving_images,
                                              'nearest-' + str(i+1) + '_original.png'),
                           arr=patch.original_img,
                           vmin=0.0,
                           vmax=1.0)
                
                # overlay (upsampled) activation on original image and save the result
                img_size = patch.original_img.shape[0]
                upsampled_act_pattern = cv2.resize(patch.act_pattern,
                                                   dsize=(img_size, img_size),
                                                   interpolation=cv2.INTER_CUBIC)
                rescaled_act_pattern = upsampled_act_pattern - np.amin(upsampled_act_pattern)
                rescaled_act_pattern = rescaled_act_pattern / np.amax(rescaled_act_pattern)
                heatmap = cv2.applyColorMap(np.uint8(255*rescaled_act_pattern), cv2.COLORMAP_JET)
                heatmap = np.float32(heatmap) / 255
                heatmap = heatmap[...,::-1]
                overlayed_original_img = 0.5 * patch.original_img + 0.3 * heatmap
                plt.imsave(fname=os.path.join(dir_for_saving_images,
                                              'nearest-' + str(i+1) + '_original_with_heatmap.png'),
                           arr=overlayed_original_img,
                           vmin=0.0,
                           vmax=1.0)
                
                # if different from original image, save the patch (i.e. receptive field)
                if patch.patch.shape[0] != img_size or patch.patch.shape[1] != img_size:
                    np.save(os.path.join(dir_for_saving_images,
                                         'nearest-' + str(i+1) + '_receptive_field_indices.npy'),
                            patch.patch_indices)
                    plt.imsave(fname=os.path.join(dir_for_saving_images,
                                              'nearest-' + str(i+1) + '_receptive_field.png'),
                               arr=patch.patch,
                               vmin=0.0,
                               vmax=1.0)
                    # save the receptive field patch with heatmap
                    overlayed_patch = overlayed_original_img[patch.patch_indices[0]:patch.patch_indices[1],
                                                             patch.patch_indices[2]:patch.patch_indices[3], :]
                    plt.imsave(fname=os.path.join(dir_for_saving_images,
                                              'nearest-' + str(i+1) + '_receptive_field_with_heatmap.png'),
                               arr=overlayed_patch,
                               vmin=0.0,
                               vmax=1.0)
                    
                # save the highly activated patch    
                high_act_patch_indices = find_high_activation_crop(upsampled_act_pattern)
                high_act_patch = patch.original_img[high_act_patch_indices[0]:high_act_patch_indices[1],
                                                    high_act_patch_indices[2]:high_act_patch_indices[3], :]
                np.save(os.path.join(dir_for_saving_images,
                                     'nearest-' + str(i+1) + '_high_act_patch_indices.npy'),
                        high_act_patch_indices)
                plt.imsave(fname=os.path.join(dir_for_saving_images,
                                              'nearest-' + str(i+1) + '_high_act_patch.png'),
                           arr=high_act_patch,
                           vmin=0.0,
                           vmax=1.0)
                # save the original image with bounding box showing high activation patch
                imsave_with_bbox(fname=os.path.join(dir_for_saving_images,
                                       'nearest-' + str(i+1) + '_high_act_patch_in_original_img.png'),
                                 img_rgb=patch.original_img,
                                 bbox_height_start=high_act_patch_indices[0],
                                 bbox_height_end=high_act_patch_indices[1],
                                 bbox_width_start=high_act_patch_indices[2],
                                 bbox_width_end=high_act_patch_indices[3], color=(0, 255, 255))
            
            labels = np.array([patch.label for patch in heaps[j]])
            np.save(os.path.join(dir_for_saving_images, 'class_id.npy'),
                    labels)


    labels_all_prototype = np.array([[patch.label for patch in heaps[j]] for j in range(n_prototypes)])

    if full_save:
        np.save(os.path.join(root_dir_for_saving_images, 'full_class_id.npy'),
                labels_all_prototype)

    end = time.time()
    log('\tfind nearest patches time: \t{0}'.format(end - start), "analysislog.txt")

    return labels_all_prototype


# receptive field
def compute_rf_protoL_at_spatial_location(img_size, height_index, width_index, protoL_rf_info):
    j, r, start = protoL_rf_info[1], protoL_rf_info[2], protoL_rf_info[3]
    center_h, center_w = start + (height_index*j), start + (width_index*j)
    return [max(int(center_h - (r/2)), 0), min(int(center_h + (r/2)), img_size),
            max(int(center_w - (r/2)), 0), min(int(center_w + (r/2)), img_size)]

def compute_rf_prototype(img_size, prototype_patch_index, protoL_rf_info):
    img_index, height_index, width_index = prototype_patch_index[0], prototype_patch_index[1], prototype_patch_index[2]
    rf_indices = compute_rf_protoL_at_spatial_location(img_size, height_index, width_index, protoL_rf_info)
    return [img_index, rf_indices[0], rf_indices[1],
            rf_indices[2], rf_indices[3]]


def compute_layer_rf_info(layer_filter_size, layer_stride, layer_padding, previous_layer_rf_info):
    n_in = previous_layer_rf_info[0] # input size
    j_in = previous_layer_rf_info[1] # receptive field jump of input layer
    r_in = previous_layer_rf_info[2] # receptive field size of input layer
    start_in = previous_layer_rf_info[3] # center of receptive field of input layer

    if layer_padding == 'SAME':
        n_out = math.ceil(float(n_in) / float(layer_stride))
        if (n_in % layer_stride == 0):
            pad = max(layer_filter_size - layer_stride, 0)
        else:
            pad = max(layer_filter_size - (n_in % layer_stride), 0)
    elif layer_padding == 'VALID':
        pad = 0
        n_out = math.ceil(float(n_in - layer_filter_size + 1) / float(layer_stride))
    else:
        pad = layer_padding * 2
        n_out = math.floor((n_in - layer_filter_size + pad)/layer_stride) + 1

    pL = math.floor(pad/2)

    j_out = j_in * layer_stride
    r_out = r_in + (layer_filter_size - 1)*j_in
    start_out = start_in + ((layer_filter_size - 1)/2 - pL)*j_in
    return [n_out, j_out, r_out, start_out]

def compute_proto_layer_rf_info_v2(img_size, layer_filter_sizes, layer_strides, layer_paddings, prototype_kernel_size):
    rf_info = [img_size, 1, 1, 0.5]
    for i in range(len(layer_filter_sizes)):
        filter_size, stride_size, padding_size = layer_filter_sizes[i], layer_strides[i], layer_paddings[i]
        rf_info = compute_layer_rf_info(layer_filter_size=filter_size, layer_stride=stride_size,
                                        layer_padding=padding_size, previous_layer_rf_info=rf_info)
    proto_layer_rf_info = compute_layer_rf_info(layer_filter_size=prototype_kernel_size, layer_stride=1,
                                                layer_padding='VALID', previous_layer_rf_info=rf_info)
    return proto_layer_rf_info

# resnet
model_urls = {
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth'
}

cfg = {'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
       'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
       'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
       'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],}

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    # class attribute
    expansion = 1
    num_layers = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        # only conv with possibly not 1 stride
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        # if stride is not 1 then self.downsample cannot be None
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        # the residual connection
        out += identity
        out = self.relu(out)

        return out

    def block_conv_info(self):
        block_kernel_sizes = [3, 3]
        block_strides = [self.stride, 1]
        block_paddings = [1, 1]

        return block_kernel_sizes, block_strides, block_paddings

class ResNet_features(nn.Module):
    '''
    the convolutional layers of ResNet
    the average pooling and final fully convolutional layer is removed
    '''
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False):
        super(ResNet_features, self).__init__()

        self.inplanes = 64

        # the first convolutional layer before the structured sequence of blocks
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # comes from the first conv and the following max pool
        self.kernel_sizes = [7, 3]
        self.strides = [2, 2]
        self.paddings = [3, 1]

        # the following layers, each layer is a sequence of blocks
        self.block = block
        self.layers = layers
        self.layer1 = self._make_layer(block=block, planes=64, num_blocks=self.layers[0])
        self.layer2 = self._make_layer(block=block, planes=128, num_blocks=self.layers[1], stride=2)
        self.layer3 = self._make_layer(block=block, planes=256, num_blocks=self.layers[2], stride=2)
        self.layer4 = self._make_layer(block=block, planes=512, num_blocks=self.layers[3], stride=2)

        # initialize the parameters
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, num_blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        # only the first block has downsample that is possibly not None
        layers.append(block(self.inplanes, planes, stride, downsample))

        self.inplanes = planes * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.inplanes, planes))

        # keep track of every block's conv size, stride size, and padding size
        for each_block in layers:
            block_kernel_sizes, block_strides, block_paddings = each_block.block_conv_info()
            self.kernel_sizes.extend(block_kernel_sizes)
            self.strides.extend(block_strides)
            self.paddings.extend(block_paddings)

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

    def conv_info(self):
        return self.kernel_sizes, self.strides, self.paddings

    def num_layers(self):
        '''
        the number of conv layers in the network, not counting the number
        of bypass layers
        '''

        return (self.block.num_layers * self.layers[0]
              + self.block.num_layers * self.layers[1]
              + self.block.num_layers * self.layers[2]
              + self.block.num_layers * self.layers[3]
              + 1)

def resnet34_features(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_features(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        my_dict = model_zoo.load_url(model_urls['resnet34'], model_dir=model_dir)
        my_dict.pop('fc.weight')
        my_dict.pop('fc.bias')
        model.load_state_dict(my_dict, strict=False)
    return model

base_architecture_to_features = {'resnet34': resnet34_features,}

# model fugle
class PPNet(nn.Module):

    def __init__(self, features, img_size, prototype_shape,
                 proto_layer_rf_info, num_classes, init_weights=True,
                 prototype_activation_function='log',
                 add_on_layers_type='bottleneck'):

        super(PPNet, self).__init__()
        self.img_size = img_size
        self.prototype_shape = prototype_shape
        self.num_prototypes = prototype_shape[0]
        self.num_classes = num_classes
        self.epsilon = 1e-4
        self.prototype_activation_function = prototype_activation_function
        self.prototype_class_identity = torch.zeros(self.num_prototypes, self.num_classes)

        num_prototypes_per_class = self.num_prototypes // self.num_classes
        for j in range(self.num_prototypes):
            self.prototype_class_identity[j, j // num_prototypes_per_class] = 1

        self.proto_layer_rf_info = proto_layer_rf_info
        self.features = features

        features_name = str(self.features).upper()
        if features_name.startswith('VGG') or features_name.startswith('RES'):
            first_add_on_layer_in_channels = [i for i in features.modules() if isinstance(i, nn.Conv2d)][-1].out_channels
        elif features_name.startswith('DENSE'):
            first_add_on_layer_in_channels = [i for i in features.modules() if isinstance(i, nn.BatchNorm2d)][-1].num_features
        else:
            raise Exception('other base base_architecture NOT implemented')

        if add_on_layers_type == 'bottleneck':
            add_on_layers, current_in_channels = [], first_add_on_layer_in_channels
            while (current_in_channels > self.prototype_shape[1]) or (len(add_on_layers) == 0):
                current_out_channels = max(self.prototype_shape[1], (current_in_channels // 2))
                add_on_layers.append(nn.Conv2d(in_channels=current_in_channels,
                                               out_channels=current_out_channels, kernel_size=1))
                add_on_layers.append(nn.ReLU())
                add_on_layers.append(nn.Conv2d(in_channels=current_out_channels,
                                               out_channels=current_out_channels,
                                               kernel_size=1))
                if current_out_channels > self.prototype_shape[1]:
                    add_on_layers.append(nn.ReLU())
                else:
                    add_on_layers.append(nn.Sigmoid())
                current_in_channels = current_in_channels // 2
            self.add_on_layers = nn.Sequential(*add_on_layers)
        else:
            self.add_on_layers = nn.Sequential(
                nn.Conv2d(in_channels=first_add_on_layer_in_channels, out_channels=self.prototype_shape[1], kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=self.prototype_shape[1], out_channels=self.prototype_shape[1], kernel_size=1),
                nn.Sigmoid()
                )
        self.prototype_vectors = nn.Parameter(torch.rand(self.prototype_shape), requires_grad=True)
        self.ones = nn.Parameter(torch.ones(self.prototype_shape), requires_grad=False)
        self.last_layer = nn.Linear(self.num_prototypes, self.num_classes, bias=False)
        if init_weights:
            self._initialize_weights()

    def conv_features(self, x):
        return self.add_on_layers(self.features(x))

    @staticmethod
    def _weighted_l2_convolution(input, filter, weights):
        # input of shape N * c * h * w
        # filter of shape P * c * h1 * w1
        # weight of shape P * c * h1 * w1
        input_patch_weighted_norm2 = F.conv2d(input=input ** 2, weight=weights)
        filter_weighted_norm2_reshape = torch.sum(weights * filter ** 2, dim=(1, 2, 3)).view(-1, 1, 1)
        weighted_inner_product = F.conv2d(input=input, weight=filter * weights)
        intermediate_result = - 2 * weighted_inner_product + filter_weighted_norm2_reshape
        distances = F.relu(input_patch_weighted_norm2 + intermediate_result)
        return distances

    def _l2_convolution(self, x):
        x2_patch_sum = F.conv2d(input=x ** 2, weight=self.ones)
        xp = F.conv2d(input=x, weight=self.prototype_vectors)
        intermediate_result = - 2 * xp + torch.sum(self.prototype_vectors ** 2, dim=(1, 2, 3)).view(-1, 1, 1)
        distances = F.relu(x2_patch_sum + intermediate_result)
        return distances

    def prototype_distances(self, x):
        return self._l2_convolution(self.conv_features(x))

    def distance_2_similarity(self, distances):
        return torch.log((distances + 1) / (distances + self.epsilon))

    def forward(self, x):
        distances = self.prototype_distances(x)
        min_distances = -F.max_pool2d(-distances, kernel_size=(distances.size()[2], distances.size()[3]))
        min_distances = min_distances.view(-1, self.num_prototypes)
        prototype_activations = self.distance_2_similarity(min_distances)
        logits = self.last_layer(prototype_activations)
        return logits, min_distances

    def push_forward(self, x):
        conv_output = self.conv_features(x)
        distances = self._l2_convolution(conv_output)
        return conv_output, distances

    def prune_prototypes(self, prototypes_to_prune):
        prototypes_to_keep = list(set(range(self.num_prototypes)) - set(prototypes_to_prune))
        self.prototype_vectors = nn.Parameter(self.prototype_vectors.data[prototypes_to_keep, ...], requires_grad=True)
        self.prototype_shape = list(self.prototype_vectors.size())
        self.num_prototypes = self.prototype_shape[0]
        self.last_layer.in_features = self.num_prototypes
        self.last_layer.out_features = self.num_classes
        self.last_layer.weight.data = self.last_layer.weight.data[:, prototypes_to_keep]
        self.ones = nn.Parameter(self.ones.data[prototypes_to_keep, ...], requires_grad=False)
        self.prototype_class_identity = self.prototype_class_identity[prototypes_to_keep,:]

    def _initialize_weights(self):
        for m in self.add_on_layers.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

def _train_or_test(model, dataloader, optimizer=None, use_l1_mask=True,
                   coefs=None):
    is_train = optimizer is not None
    start = time.time()
    n_examples, n_correct, n_batches, total_cross_entropy = 0, 0, 0, 0
    total_cluster_cost, total_separation_cost, total_avg_separation_cost = 0, 0, 0
    
    scaler = torch.cuda.amp.GradScaler()
    torch.cuda.empty_cache()

    for _, (image, label) in enumerate(dataloader):
        input, target = image.cuda(), label.cuda()

        grad_req = torch.enable_grad() if is_train else torch.no_grad()
        with grad_req:
            output, min_distances = model(input)

            # compute loss
            cross_entropy = torch.nn.functional.cross_entropy(output, target)
            max_dist = (model.module.prototype_shape[1] * model.module.prototype_shape[2] * model.module.prototype_shape[3])
            prototypes_of_correct_class = torch.t(model.module.prototype_class_identity[:,label]).cuda()
            inverted_distances, _ = torch.max((max_dist - min_distances) * prototypes_of_correct_class, dim=1)
            cluster_cost = torch.mean(max_dist - inverted_distances)

            # calculate separation cost
            prototypes_of_wrong_class = 1 - prototypes_of_correct_class
            inverted_distances_to_nontarget_prototypes, _ = \
                torch.max((max_dist - min_distances) * prototypes_of_wrong_class, dim=1)
            separation_cost = torch.mean(max_dist - inverted_distances_to_nontarget_prototypes)

            # calculate avg cluster cost
            avg_separation_cost = \
                torch.sum(min_distances * prototypes_of_wrong_class, dim=1) / torch.sum(prototypes_of_wrong_class, dim=1)
            avg_separation_cost = torch.mean(avg_separation_cost)
                
            if use_l1_mask:
                l1_mask = 1 - torch.t(model.module.prototype_class_identity).cuda()
                l1 = (model.module.last_layer.weight * l1_mask).norm(p=1)
            else:
                l1 = model.module.last_layer.weight.norm(p=1) 

            # evaluation statistics
            _, predicted = torch.max(output.data, 1)
            n_examples += target.size(0)
            n_correct += (predicted == target).sum().item()
            n_batches += 1
            total_cross_entropy += cross_entropy.item()
            total_cluster_cost += cluster_cost.item()
            total_separation_cost += separation_cost.item()
            total_avg_separation_cost += avg_separation_cost.item()

        # compute gradient and do SGD step
        if is_train:
            del prototypes_of_correct_class
            del l1_mask
            with torch.cuda.amp.autocast():
                if coefs is not None:
                    loss = (coefs['crs_ent'] * cross_entropy + coefs['clst'] * cluster_cost
                            + coefs['sep'] * separation_cost + coefs['l1'] * l1)
                else:
                    loss = cross_entropy + 0.8 * cluster_cost - 0.08 * separation_cost + 1e-4 * l1
            optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            
            
    end = time.time()

    log('\ttime: \t{0}'.format(end -  start), "trainlog.txt")
    log('\tcross ent: \t{0}'.format(total_cross_entropy / n_batches), "trainlog.txt")
    log('\tcluster: \t{0}'.format(total_cluster_cost / n_batches), "trainlog.txt")
    log('\tseparation:\t{0}'.format(total_separation_cost / n_batches), "trainlog.txt")
    log('\tavg separation:\t{0}'.format(total_avg_separation_cost / n_batches), "trainlog.txt")
    log('\taccu: \t\t{0}%'.format(n_correct / n_examples * 100), "trainlog.txt")
    log('\tl1: \t\t{0}'.format(model.module.last_layer.weight.norm(p=1).item()), "trainlog.txt")
    p = model.module.prototype_vectors.view(model.module.num_prototypes, -1).cpu()
    with torch.no_grad():
        p_avg_pair_dist = torch.mean(list_of_distances(p, p))
    log('\tp dist pair: \t{0}'.format(p_avg_pair_dist.item()), "trainlog.txt")

    return n_correct / n_examples

def train(model, dataloader, optimizer, coefs=None):
    log('\ttrain', "trainlog.txt")
    model.train()
    return _train_or_test(model=model, dataloader=dataloader, optimizer=optimizer, coefs=coefs)
  
def test(model, dataloader):
    log('\ttest', "trainlog.txt")
    model.eval()
    return _train_or_test(model=model, dataloader=dataloader, optimizer=None)

def last_only(model):
    for p in model.module.features.parameters():
        p.requires_grad = False
    for p in model.module.add_on_layers.parameters():
        p.requires_grad = False
    model.module.prototype_vectors.requires_grad = False
    for p in model.module.last_layer.parameters():
        p.requires_grad = True
    log('\tlast layer', "trainlog.txt")

def warm_only(model):
    for p in model.module.features.parameters():
        p.requires_grad = False
    for p in model.module.add_on_layers.parameters():
        p.requires_grad = True
    model.module.prototype_vectors.requires_grad = True
    for p in model.module.last_layer.parameters():
        p.requires_grad = True
    log('\twarm', "trainlog.txt")

def joint(model):
    for p in model.module.features.parameters():
        p.requires_grad = True
    for p in model.module.add_on_layers.parameters():
        p.requires_grad = True
    model.module.prototype_vectors.requires_grad = True
    for p in model.module.last_layer.parameters():
        p.requires_grad = True
    log('\tjoint', "trainlog.txt")

def push_prototypes(dataloader, # unn
                    prototype_network_parallel, # nn
                    preprocess_input_function=None, # normalize?
                    prototype_layer_stride=1,
                    root_dir_for_saving_prototypes=None,
                    epoch_number=None,
                    prototype_img_filename_prefix=None,
                    prototype_self_act_filename_prefix=None,
                    proto_bound_boxes_filename_prefix=None,
                    save_prototype_class_identity=True,
                    prototype_activation_function_in_numpy=None):

    prototype_network_parallel.eval()
    log('\tpush', "trainlog.txt")

    start = time.time()
    prototype_shape = prototype_network_parallel.module.prototype_shape
    n_prototypes = prototype_network_parallel.module.num_prototypes
    global_min_proto_dist = np.full(n_prototypes, np.inf)
    global_min_fmap_patches = np.zeros([n_prototypes, prototype_shape[1],prototype_shape[2],prototype_shape[3]])

    if save_prototype_class_identity:
        proto_rf_boxes = np.full(shape=[n_prototypes, 6], fill_value=-1)
        proto_bound_boxes = np.full(shape=[n_prototypes, 6], fill_value=-1)
    else:
        proto_rf_boxes = np.full(shape=[n_prototypes, 5], fill_value=-1)
        proto_bound_boxes = np.full(shape=[n_prototypes, 5], fill_value=-1)

    if root_dir_for_saving_prototypes != None:
        if epoch_number != None:
            proto_epoch_dir = os.path.join(root_dir_for_saving_prototypes,
                                           'epoch-'+str(epoch_number))
            makedir(proto_epoch_dir)
        else:
            proto_epoch_dir = root_dir_for_saving_prototypes
    else:
        proto_epoch_dir = None

    search_batch_size = dataloader.batch_size
    num_classes = prototype_network_parallel.module.num_classes

    for push_iter, (search_batch_input, search_y) in enumerate(dataloader):
        start_index_of_search_batch = push_iter * search_batch_size
        update_prototypes_on_batch(search_batch_input,
                                   start_index_of_search_batch,
                                   prototype_network_parallel,
                                   global_min_proto_dist,
                                   global_min_fmap_patches,
                                   proto_rf_boxes,
                                   proto_bound_boxes,
                                   search_y=search_y,
                                   num_classes=num_classes,
                                   preprocess_input_function=preprocess_input_function,
                                   prototype_layer_stride=prototype_layer_stride,
                                   dir_for_saving_prototypes=proto_epoch_dir,
                                   prototype_img_filename_prefix=prototype_img_filename_prefix,
                                   prototype_self_act_filename_prefix=prototype_self_act_filename_prefix,
                                   prototype_activation_function_in_numpy=prototype_activation_function_in_numpy)

    if proto_epoch_dir != None and proto_bound_boxes_filename_prefix != None:
        np.save(os.path.join(proto_epoch_dir, proto_bound_boxes_filename_prefix + '-receptive_field' + str(epoch_number) + '.npy'),
                proto_rf_boxes)
        np.save(os.path.join(proto_epoch_dir, proto_bound_boxes_filename_prefix + str(epoch_number) + '.npy'),
                proto_bound_boxes)

    log('\tExecuting push ...', "trainlog.txt")
    prototype_update = np.reshape(global_min_fmap_patches,
                                  tuple(prototype_shape))
    prototype_network_parallel.module.prototype_vectors.data.copy_(torch.tensor(prototype_update, dtype=torch.float32).cuda())
    end = time.time()
    log('\tpush time: \t{0}'.format(end -  start), "trainlog.txt")

def update_prototypes_on_batch(search_batch_input,
                               start_index_of_search_batch,
                               prototype_network_parallel,
                               global_min_proto_dist, # this will be updated
                               global_min_fmap_patches, # this will be updated
                               proto_rf_boxes, # this will be updated
                               proto_bound_boxes, # this will be updated
                               search_y=None, # required if class_specific == True
                               num_classes=None, # required if class_specific == True
                               preprocess_input_function=None,
                               prototype_layer_stride=1,
                               dir_for_saving_prototypes=None,
                               prototype_img_filename_prefix=None,
                               prototype_self_act_filename_prefix=None,
                               prototype_activation_function_in_numpy=None):

    prototype_network_parallel.eval()

    with torch.no_grad():
        search_batch = search_batch_input.cuda()
        protoL_input_torch, proto_dist_torch = prototype_network_parallel.module.push_forward(search_batch)

    protoL_input_ = np.copy(protoL_input_torch.detach().cpu().numpy())
    proto_dist_ = np.copy(proto_dist_torch.detach().cpu().numpy())

    del protoL_input_torch, proto_dist_torch

    class_to_img_index_dict = {key: [] for key in range(num_classes)}
        # img_y is the image's integer label
    for img_index, img_y in enumerate(search_y):
        img_label = img_y.item()
        class_to_img_index_dict[img_label].append(img_index)

    prototype_shape = prototype_network_parallel.module.prototype_shape
    n_prototypes, proto_h, proto_w = prototype_shape[0], prototype_shape[2], prototype_shape[3]

    for j in range(n_prototypes):
        target_class = torch.argmax(prototype_network_parallel.module.prototype_class_identity[j]).item()
        if len(class_to_img_index_dict[target_class]) == 0:
            continue
        proto_dist_j = proto_dist_[class_to_img_index_dict[target_class]][:,j,:,:]

        batch_min_proto_dist_j = np.amin(proto_dist_j)
        if batch_min_proto_dist_j < global_min_proto_dist[j]:
            batch_argmin_proto_dist_j = \
                list(np.unravel_index(np.argmin(proto_dist_j, axis=None),
                                      proto_dist_j.shape))
            batch_argmin_proto_dist_j[0] = class_to_img_index_dict[target_class][batch_argmin_proto_dist_j[0]]

            # retrieve the corresponding feature map patch
            img_index_in_batch = batch_argmin_proto_dist_j[0]
            fmap_height_start_index = batch_argmin_proto_dist_j[1] * prototype_layer_stride
            fmap_height_end_index = fmap_height_start_index + proto_h
            fmap_width_start_index = batch_argmin_proto_dist_j[2] * prototype_layer_stride
            fmap_width_end_index = fmap_width_start_index + proto_w

            batch_min_fmap_patch_j = protoL_input_[img_index_in_batch,
                                                   :,
                                                   fmap_height_start_index:fmap_height_end_index,
                                                   fmap_width_start_index:fmap_width_end_index]

            global_min_proto_dist[j] = batch_min_proto_dist_j
            global_min_fmap_patches[j] = batch_min_fmap_patch_j
            
            # get the receptive field boundary of the image patch
            # that generates the representation
            protoL_rf_info = prototype_network_parallel.module.proto_layer_rf_info
            rf_prototype_j = compute_rf_prototype(search_batch.size(2), batch_argmin_proto_dist_j, protoL_rf_info)
            
            # get the whole image
            original_img_j = search_batch_input[rf_prototype_j[0]]
            original_img_j = original_img_j.numpy()
            original_img_j = np.transpose(original_img_j, (1, 2, 0))
            original_img_size = original_img_j.shape[0]
            
            # crop out the receptive field
            rf_img_j = original_img_j[rf_prototype_j[1]:rf_prototype_j[2],
                                      rf_prototype_j[3]:rf_prototype_j[4],]
            
            # save the prototype receptive field information
            proto_rf_boxes[j, 0] = rf_prototype_j[0] + start_index_of_search_batch
            proto_rf_boxes[j, 1] = rf_prototype_j[1]
            proto_rf_boxes[j, 2] = rf_prototype_j[2]
            proto_rf_boxes[j, 3] = rf_prototype_j[3]
            proto_rf_boxes[j, 4] = rf_prototype_j[4]
            if proto_rf_boxes.shape[1] == 6 and search_y is not None:
                proto_rf_boxes[j, 5] = search_y[rf_prototype_j[0]].item()

            # find the highly activated region of the original image
            proto_dist_img_j = proto_dist_[img_index_in_batch, j, :, :]
            proto_act_img_j = np.log((proto_dist_img_j + 1) / (proto_dist_img_j + prototype_network_parallel.module.epsilon))
            upsampled_act_img_j = cv2.resize(proto_act_img_j, dsize=(original_img_size, original_img_size), interpolation=cv2.INTER_CUBIC)
            proto_bound_j = find_high_activation_crop(upsampled_act_img_j)
            # crop out the image patch with high activation as prototype image
            proto_img_j = original_img_j[proto_bound_j[0]:proto_bound_j[1], proto_bound_j[2]:proto_bound_j[3], :]

            # save the prototype boundary (rectangular boundary of highly activated region)
            proto_bound_boxes[j, 0] = proto_rf_boxes[j, 0]
            proto_bound_boxes[j, 1] = proto_bound_j[0]
            proto_bound_boxes[j, 2] = proto_bound_j[1]
            proto_bound_boxes[j, 3] = proto_bound_j[2]
            proto_bound_boxes[j, 4] = proto_bound_j[3]
            if proto_bound_boxes.shape[1] == 6 and search_y is not None:
                proto_bound_boxes[j, 5] = search_y[rf_prototype_j[0]].item()

            if dir_for_saving_prototypes is not None:
                if prototype_self_act_filename_prefix is not None:
                    # save the numpy array of the prototype self activation
                    np.save(os.path.join(dir_for_saving_prototypes,
                                         prototype_self_act_filename_prefix + str(j) + '.npy'),
                            proto_act_img_j)
                if prototype_img_filename_prefix is not None:
                    # save the whole image containing the prototype as png
                    plt.imsave(os.path.join(dir_for_saving_prototypes, prototype_img_filename_prefix + '-original' + str(j) + '.png'),
                               original_img_j, vmin=0.0, vmax=1.0)
                    # overlay (upsampled) self activation on original image and save the result
                    rescaled_act_img_j = upsampled_act_img_j - np.amin(upsampled_act_img_j)
                    rescaled_act_img_j = rescaled_act_img_j / np.amax(rescaled_act_img_j)
                    heatmap = cv2.applyColorMap(np.uint8(255*rescaled_act_img_j), cv2.COLORMAP_JET)
                    heatmap = np.float32(heatmap) / 255
                    heatmap = heatmap[...,::-1]
                    overlayed_original_img_j = 0.5 * original_img_j + 0.3 * heatmap
                    plt.imsave(os.path.join(dir_for_saving_prototypes, prototype_img_filename_prefix + '-original_with_self_act' + str(j) + '.png'),
                               overlayed_original_img_j, vmin=0.0, vmax=1.0)
                    
                    # if different from the original (whole) image, save the prototype receptive field as png
                    if rf_img_j.shape[0] != original_img_size or rf_img_j.shape[1] != original_img_size:
                        plt.imsave(os.path.join(dir_for_saving_prototypes, prototype_img_filename_prefix + '-receptive_field' + str(j) + '.png'),
                                   rf_img_j, vmin=0.0, vmax=1.0)
                        overlayed_rf_img_j = overlayed_original_img_j[rf_prototype_j[1]:rf_prototype_j[2], rf_prototype_j[3]:rf_prototype_j[4]]
                        plt.imsave(os.path.join(dir_for_saving_prototypes,prototype_img_filename_prefix + '-receptive_field_with_self_act' + str(j) + '.png'),
                                   overlayed_rf_img_j, vmin=0.0, vmax=1.0)
                    
                    # save the prototype image (highly activated region of the whole image)
                    plt.imsave(os.path.join(dir_for_saving_prototypes, prototype_img_filename_prefix + str(j) + '.png'),
                               proto_img_j, vmin=0.0, vmax=1.0)            
    del class_to_img_index_dict

def fit(model, modelmulti, epochs, warm_epochs):
    log('start training', "trainlog.txt")
    for epoch in range(epochs):
        model.train()
        modelmulti.train()
        log('epoch: \t{0}'.format(epoch + epoch_reached), "trainlog.txt")
        
        if epoch + epoch_reached < warm_epochs:
            warm_only(model=modelmulti)
            train(model=modelmulti, dataloader=train_loader, optimizer=warm_optimizer, coefs=coefs)
        else:
            joint(model=modelmulti)
            train(model=modelmulti, dataloader=train_loader, optimizer=joint_optimizer, coefs=coefs)
            joint_lr_scheduler.step()
        model.eval()
        modelmulti.eval()
        accu = test(model=modelmulti, dataloader=test_loader)
        torch.save({
            'epoch': epoch + epoch_reached,
            'model_state_dict': model.state_dict(),
            'joint_optimizer_state_dict': joint_optimizer.state_dict(),
            'last_layer_optimizer_state_dict': last_layer_optimizer.state_dict(),
            'warm_optimizer_state_dict' : warm_optimizer.state_dict()
            }, os.path.join(model_dir, (str(epoch + epoch_reached) + 'nopush' + '{0:.4f}.pth').format(accu)))

#         if epoch >= push_start and epoch in push_epochs:
#             push_prototypes(
#                 train_push_loader, # pytorch dataloader unnorm
#                 prototype_network_parallel=modelmulti,
#                 preprocess_input_function = preprocess, # norma?
#                 prototype_layer_stride=1,
#                 root_dir_for_saving_prototypes = model_dir + '/img/',
#                 epoch_number = epoch + epoch_reached, # if not provided, prototypes saved previously will be overwritten
#                 prototype_img_filename_prefix = 'prototype-img',
#                 prototype_self_act_filename_prefix = 'prototype-self-act',
#                 proto_bound_boxes_filename_prefix = 'bb',
#                 save_prototype_class_identity=True)
#             last_only(model=modelmulti)
#             for i in range(20):
#                 log('iteration: \t{0}'.format(i), "trainlog.txt")
#                 _ = train(model=modelmulti, dataloader=train_loader, optimizer=last_layer_optimizer, coefs=coefs)
#                 accu = test(model=modelmulti, dataloader=test_loader)
#             torch.save({
#                 'epoch': epoch + epoch_reached,
#                 'model_state_dict': model.state_dict(),
#                 'joint_optimizer_state_dict': joint_optimizer.state_dict(),
#                 'last_layer_optimizer_state_dict': last_layer_optimizer.state_dict(),
#                 'warm_optimizer_state_dict' : warm_optimizer.state_dict()
#                 }, os.path.join(model_dir, (str(epoch + epoch_reached) + 'push' + '{0:.4f}.pth').format(accu)))

def initialize_model(model_name = ""):
    features = base_architecture_to_features['resnet34'](pretrained = True)
    layer_filter_sizes, layer_strides, layer_paddings = features.conv_info()

    ppnet = PPNet(features=features,
                  img_size = img_size,
                  prototype_shape = (2000, 512, 1, 1),
                  num_classes = 200,
                  init_weights = True,
                  prototype_activation_function = 'log',
                  add_on_layers_type = 'bottleneck',
                  proto_layer_rf_info = compute_proto_layer_rf_info_v2(img_size = 224,
                                                                      layer_filter_sizes = layer_filter_sizes,
                                                                      layer_strides = layer_strides,
                                                                      layer_paddings = layer_paddings,
                                                                      prototype_kernel_size = 1))
    if model_name != "":
        checkpoint = torch.load(model_dir + model_name)
        ppnet.load_state_dict(checkpoint['model_state_dict'])

    ppnet = ppnet.cuda()
    ppnet_multi = torch.nn.DataParallel(ppnet)
    return ppnet, ppnet_multi

################################################################################################################################

cuda = torch.device('cuda')  if torch.cuda.is_available() else "cpu"
print("Using : ", cuda)


img_size = 224
prototype_shape = (2000, 128, 1, 1)
num_classes = 200

train_batch_size = 40
test_batch_size = 40
train_push_batch_size = 40

# sets
train_dataset = datasets.ImageFolder(
        path + 'datasets/cub200_cropped/train_cropped_augmented/', 
        transforms.Compose([transforms.Resize(size=(img_size, img_size)), transforms.ToTensor(), 
                            transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225))]))
train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=4, pin_memory=True) # 4 workers?

train_push_dataset = datasets.ImageFolder(
        path + 'datasets/cub200_cropped/train_cropped/',
        transforms.Compose([transforms.Resize(size=(img_size, img_size)),transforms.ToTensor()]))
train_push_loader = torch.utils.data.DataLoader(
        train_push_dataset, batch_size=train_push_batch_size, shuffle=False, num_workers=4, pin_memory=True) # 4 workers?

test_dataset = datasets.ImageFolder(
        path + 'datasets/cub200_cropped/test_cropped/',
        transforms.Compose([transforms.Resize(size=(img_size, img_size)),transforms.ToTensor(),
                            transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225)),]))
test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=4, pin_memory=True) # 4 workers?

model_name = ""
if model_name != "":
    checkpoint = torch.load(model_dir + model_name)
ppnet, ppnet_multi = initialize_model(model_name = model_name)

torch.backends.cudnn.benchmark = True

# optimizers 
joint_optimizer = torch.optim.Adam([{'params': ppnet.features.parameters(), 'lr': 1e-4, 'weight_decay': 1e-3}, 
                                        {'params': ppnet.add_on_layers.parameters(), 'lr': 3e-3, 'weight_decay': 1e-3},
                                        {'params': ppnet.prototype_vectors, 'lr': 3e-3},])
joint_lr_scheduler = torch.optim.lr_scheduler.StepLR(joint_optimizer, step_size = 5, gamma=0.1)
last_layer_optimizer = torch.optim.Adam([{'params': ppnet.last_layer.parameters(), 'lr': 1e-4}])
warm_optimizer = torch.optim.Adam([{'params': ppnet.add_on_layers.parameters(), 'lr': 3e-3, 'weight_decay': 1e-3},
                                {'params': ppnet.prototype_vectors, 'lr': 3e-3},])

epochs = 1000
warm_epochs = 5
epoch_reached = 0
push_start = 1000
push_epochs = [i for i in range(epochs) if i % 10 == 0]
coefs = {'crs_ent': 1, 'clst': 0.8, 'sep': -0.08, 'l1': 1e-4,}

if model_name != "":
    epoch_reached = checkpoint['epoch'] + 1
    print(epoch_reached)
    joint_optimizer.load_state_dict(checkpoint['joint_optimizer_state_dict'])
    last_layer_optimizer.load_state_dict(checkpoint['last_layer_optimizer_state_dict'])
    warm_optimizer.load_state_dict(checkpoint['warm_optimizer_state_dict'])
    

# run fitting
fit(ppnet, ppnet_multi, epochs = epochs, warm_epochs = warm_epochs)
