import torch
import numpy as np

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

############################################
############################################
# MODEL INITIALIZATION do regex plaeaseesese
epoch_reached = 0 # remember it will start from 0, so it should be one + the number in the model name
ppnet, ppnet_multi = initialize_model("")

torch.backends.cudnn.benchmark = True
############################################
############################################
############################################

# optimizers 
joint_optimizer = torch.optim.Adam([{'params': ppnet.features.parameters(), 'lr': 1e-4, 'weight_decay': 1e-3}, 
                                        {'params': ppnet.add_on_layers.parameters(), 'lr': 3e-3, 'weight_decay': 1e-3},
                                        {'params': ppnet.prototype_vectors, 'lr': 3e-3},])
joint_lr_scheduler = torch.optim.lr_scheduler.StepLR(joint_optimizer, step_size = 5, gamma=0.1)
warm_optimizer = torch.optim.Adam([{'params': ppnet.add_on_layers.parameters(), 'lr': 3e-3, 'weight_decay': 1e-3},
                                {'params': ppnet.prototype_vectors, 'lr': 3e-3},])

last_layer_optimizer = torch.optim.Adam([{'params': ppnet.last_layer.parameters(), 'lr': 1e-4}])

epochs = 1000
warm_epochs = 0
push_start = 1000
push_epochs = [i for i in range(epochs) if i % 10 == 0]
coefs = {'crs_ent': 1, 'clst': 0.8, 'sep': -0.08, 'l1': 1e-4,}

# Original iaiabl
####################################################################################################################################
"""
import torch
import numpy as np

base_architecture = 'vgg16'
img_size = 224
prototype_shape = (15, 512, 1, 1)
num_classes = 3

prototype_activation_function = "log"
prototype_activation_function_in_numpy = prototype_activation_function

class_specific = True

add_on_layers_type = 'regular'

experiment_run = '1218_fa='
data_path = '/usr/xtmp/mammo/Lo1136i_with_fa/'
train_dir = data_path + 'train_augmented_5000/'
test_dir = data_path + 'validation/'
train_push_dir = '/usr/xtmp/mammo/Lo1136i_finer/by_margin/train/'

train_batch_size = 75
test_batch_size = 100
train_push_batch_size = 75

joint_optimizer_lrs = {'features': 2e-4,
                       'add_on_layers': 3e-3,
                       'prototype_vectors': 3e-3}
joint_lr_step_size = 5

warm_optimizer_lrs = {'add_on_layers': 2e-3,
                      'prototype_vectors': 3e-3}

last_layer_optimizer_lr = 1e-3

coefs = {
    'crs_ent': 1,
    'clst': 0.8,
    'sep': -0.08,
    'l1': 1e-4,
    'fine': 0.001,
}

num_train_epochs = 130
num_warm_epochs = 10

push_start = 10
push_epochs = [i for i in range(num_train_epochs) if i % 10 == 0]

"""