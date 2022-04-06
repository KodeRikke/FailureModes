import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F

from resnet_features import resnet18_features, resnet34_features, resnet50_features, resnet101_features, resnet152_features
from densenet_features import densenet121_features, densenet161_features, densenet169_features, densenet201_features
from vgg_features import vgg11_features, vgg11_bn_features, vgg13_features, vgg13_bn_features, vgg16_features, vgg16_bn_features,\
                         vgg19_features, vgg19_bn_features


from receptive_field import compute_proto_layer_rf_info_v2

base_architecture_to_features = {'resnet18': resnet18_features,
                                 'resnet34': resnet34_features,
                                 'resnet50': resnet50_features,
                                 'resnet101': resnet101_features,
                                 'resnet152': resnet152_features,
                                 'densenet121': densenet121_features,
                                 'densenet161': densenet161_features,
                                 'densenet169': densenet169_features,
                                 'densenet201': densenet201_features,
                                 'vgg11': vgg11_features,
                                 'vgg11_bn': vgg11_bn_features,
                                 'vgg13': vgg13_features,
                                 'vgg13_bn': vgg13_bn_features,
                                 'vgg16': vgg16_features,
                                 'vgg16_bn': vgg16_bn_features,
                                 'vgg19': vgg19_features,
                                 'vgg19_bn': vgg19_bn_features}
                                 
class PPNet(nn.Module):

    def __init__(self, features, img_size, prototype_shape,
                 proto_layer_rf_info, num_classes, init_weights=True,
                 prototype_activation_function='log',
                 add_on_layers_type='bottleneck'):

        super(PPNet, self).__init__()
        self.img_size = img_size
        self.prototype_shape = prototype_shape # 2000, 512, 1, 1
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

    def conv_features(self, x): #z
        return self.add_on_layers(self.features(x))

    @staticmethod
    def _weighted_l2_convolution(input, filter, weights):
        # input of shape N * c * h * w torch.Size([40, 512, 7, 7])
        # filter of shape P * c * h1 * w1
        # weight of shape P * c * h1 * w1
        input_patch_weighted_norm2 = F.conv2d(input=input ** 2, weight=weights)
        filter_weighted_norm2_reshape = torch.sum(weights * filter ** 2, dim=(1, 2, 3)).view(-1, 1, 1)
        weighted_inner_product = F.conv2d(input=input, weight=filter * weights)
        intermediate_result = - 2 * weighted_inner_product + filter_weighted_norm2_reshape
        distances = F.relu(input_patch_weighted_norm2 + intermediate_result)
        return distances

    def _l2_convolution(self, x): # x = z (torch.Size([40, 512, 7, 7]))
        x2_patch_sum = F.conv2d(input=x ** 2, weight=self.ones)  # torch.Size([40, 2000, 7, 7])
        xp = F.conv2d(input=x, weight=self.prototype_vectors) # torch.Size([40, 2000, 7, 7])
        intermediate_result = - 2 * xp + torch.sum(self.prototype_vectors ** 2, dim=(1, 2, 3)).view(-1, 1, 1)
        distances = F.relu(x2_patch_sum + intermediate_result)
        return distances # torch.Size([40, 2000, 7, 7])

    def prototype_distances(self, x): # dist(z, prototypes)
        return self._l2_convolution(self.conv_features(x)) 

    def distance_2_similarity(self, distances):
        return torch.log((distances + 1) / (distances + self.epsilon))

    def forward(self, x):
        distances = self.prototype_distances(x)
        min_distances = -F.max_pool2d(-distances, kernel_size=(distances.size()[2], distances.size()[3]))
        # print(min_distances.size()) torch.Size([40, 2000, 1, 1])
        min_distances = min_distances.view(-1, self.num_prototypes)
        # print(min_distances.size()) torch.Size([40, 2000])
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
