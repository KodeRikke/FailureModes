# libraries
import os
import cv2
import matplotlib.image as plt
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch

from settings import path, model_dir, num_experiments, seeds, epochs, warm_epochs, \
                     epoch_reached, push_start, last_layer_iterations, push_epochs, \
                     coefs, train_batch_size, test_batch_size, train_push_batch_size, \
                     img_size, num_prototypes, num_classes, model_names, save_name, \
                     base_architectures

from helpers import log, set_seed, seed_worker

def save_prototype_original_img_with_bbox(fname, epoch, index, bbox_height_start, bbox_height_end, bbox_width_start, bbox_width_end):
    p_img_bgr = cv2.imread(os.path.join(load_img_dir, 'epoch-'+str(epoch), 'prototype-img-original'+str(index)+'.png'))
    cv2.rectangle(p_img_bgr, (bbox_width_start, bbox_height_start), (bbox_width_end-1, bbox_height_end-1), (0, 255, 255), thickness=2)
    p_img_rgb = np.float32(p_img_bgr[...,::-1]) / 255
    plt.imsave(fname, p_img_rgb)

from train_and_test import train, test, last_only, warm_only, joint

# model fugle
from ppnet import initialize_model

def fit(model, modelmulti, save_name, epochs, warm_epochs, epoch_reached, last_layer_iterations):
    log('start training', trainlog)
    for epoch in range(epoch_reached, epochs):
        model.train()
        modelmulti.train()
        log('epoch: \t{0}'.format(epoch), trainlog)
        
        if epoch < warm_epochs:
            warm_only(model=modelmulti, trainlog=trainlog)
            train(model=modelmulti, dataloader=train_loader, trainlog=trainlog, optimizer=warm_optimizer, coefs=coefs)
        else:
            joint(model=modelmulti, trainlog=trainlog)
            train(model=modelmulti, dataloader=train_loader, trainlog=trainlog, optimizer=joint_optimizer, coefs=coefs)
            joint_lr_scheduler.step()
        model.eval()
        modelmulti.eval()
        accu = test(model=modelmulti, dataloader=test_loader, trainlog=trainlog)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'joint_optimizer_state_dict': joint_optimizer.state_dict(),
            'joint_lr_scheduler_state_dict': joint_lr_scheduler.state_dict(),
            'last_layer_optimizer_state_dict': last_layer_optimizer.state_dict(),
            'warm_optimizer_state_dict' : warm_optimizer.state_dict()
            }, os.path.join(model_dir, (save_name + "E" + str(epoch) + 'nopush' + '{0:.4f}.pth').format(accu)))

        if epoch >= push_start and epoch in push_epochs:
            last_only(model=modelmulti, trainlog=trainlog)
            for i in range(last_layer_iterations):
                log('iteration: \t{0}'.format(i), trainlog)
                _ = train(model=modelmulti, dataloader=train_loader, trainlog=trainlog, optimizer=last_layer_optimizer, coefs=coefs)
                accu = test(model=modelmulti, dataloader=test_loader, trainlog=trainlog)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'joint_optimizer_state_dict': joint_optimizer.state_dict(),
                    'joint_lr_scheduler_state_dict': joint_lr_scheduler.state_dict(),
                    'last_layer_optimizer_state_dict': last_layer_optimizer.state_dict(),
                    'warm_optimizer_state_dict' : warm_optimizer.state_dict()
                    }, os.path.join(model_dir, (save_name + "E" + str(epoch) + 'I' + str(i) + 'push' + '{0:.4f}.pth').format(accu)))

if len(base_architectures) != len(seeds) != len(list(range(num_experiments))):
    raise Exception("base_architectures, experiments and seeds must be of equal length!")

# cuda
cuda = torch.device('cuda') if torch.cuda.is_available() else "cpu"
print("Using : ", cuda)

for i, seed, base_architecture, model_name in zip(list(range(num_experiments)), seeds, base_architectures, model_names):

    # save
    save_name = "C"+str(num_classes)+"P"+str(num_prototypes)+"S"+str(seed)+base_architecture

    # prototype shapes
    prototype_shape = (num_prototypes * num_classes, 128, 1, 1) 
    if base_architecture == "resnet34": # 256 for resnet34 and 128 for densenet121, vgg19
        prototype_shape = (num_prototypes * num_classes, 256, 1, 1) 
        
    # log names
    trainlog = "trainlog" + save_name + ".txt"
    analysislog = "analysislog" + save_name + ".txt"
    pushlog = "pushlog" + save_name + ".txt"
    
    # reproducibility
    g = set_seed(seed)
    
    # sets
    train_dataset = datasets.ImageFolder(
            path + 'datasets/cub200_cropped/train_cropped_augmented/', 
            transforms.Compose([transforms.Resize(size=(img_size, img_size)), transforms.ToTensor(), 
                                transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225))]))
    train_push_dataset = datasets.ImageFolder(
            path + 'datasets/cub200_cropped/train_cropped/',
            transforms.Compose([transforms.Resize(size=(img_size, img_size)),transforms.ToTensor()]))
    test_dataset = datasets.ImageFolder(
            path + 'datasets/cub200_cropped/test_cropped/',
            transforms.Compose([transforms.Resize(size=(img_size, img_size)),transforms.ToTensor(),
                                transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225)),]))

    # dataloaders
    train_loader = torch.utils.data.DataLoader(train_dataset, 
                                               batch_size=train_batch_size, 
                                               shuffle=True, 
                                               num_workers=4, 
                                               pin_memory=True,
                                               worker_init_fn=seed_worker,
                                               generator=g,)

    train_push_loader = torch.utils.data.DataLoader(train_push_dataset, 
                                                    batch_size=train_push_batch_size, 
                                                    shuffle=False, 
                                                    num_workers=4, 
                                                    pin_memory=True,
                                                    worker_init_fn=seed_worker,
                                                    generator=g)

    test_loader = torch.utils.data.DataLoader(test_dataset, 
                                              batch_size=test_batch_size, 
                                              shuffle=False, 
                                              num_workers=4, 
                                              pin_memory=True,
                                              worker_init_fn=seed_worker,
                                              generator=g)

    # initialize model
    if model_name != "":
        model_name = save_name + "_" + model_name + ".pth" #100C10P1337densenet1216nopush0.3628 # 
        checkpoint = torch.load(model_dir + model_name)
    ppnet, ppnet_multi = initialize_model(base_architecture, prototype_shape, num_classes, model_name = model_name)

#     torch.backends.cudnn.benchmark = True

    # optimizers 
    joint_optimizer = torch.optim.Adam([{'params': ppnet.features.parameters(), 'lr': 1e-4, 'weight_decay': 1e-3}, 
                                            {'params': ppnet.add_on_layers.parameters(), 'lr': 3e-3, 'weight_decay': 1e-3},
                                            {'params': ppnet.prototype_vectors, 'lr': 3e-3},])
    joint_lr_scheduler = torch.optim.lr_scheduler.StepLR(joint_optimizer, step_size = 5, gamma=0.1)
    last_layer_optimizer = torch.optim.Adam([{'params': ppnet.last_layer.parameters(), 'lr': 1e-4}])
    warm_optimizer = torch.optim.Adam([{'params': ppnet.add_on_layers.parameters(), 'lr': 3e-3, 'weight_decay': 1e-3},
                                    {'params': ppnet.prototype_vectors, 'lr': 3e-3},])

    # load model
    if model_name != "":
        epoch_reached = checkpoint['epoch'] + 1 # next epoch
        joint_optimizer.load_state_dict(checkpoint['joint_optimizer_state_dict'])
        joint_lr_scheduler.load_state_dict(checkpoint['joint_lr_scheduler_state_dict'])
        last_layer_optimizer.load_state_dict(checkpoint['last_layer_optimizer_state_dict'])
        warm_optimizer.load_state_dict(checkpoint['warm_optimizer_state_dict'])

    # run fitting
    fit(model=ppnet, 
        modelmulti=ppnet_multi, 
        save_name=save_name+"_", 
        epochs=epochs, 
        warm_epochs=warm_epochs, 
        epoch_reached=epoch_reached, 
        last_layer_iterations=last_layer_iterations)

    # push prototypes
#     push_prototypes(
#         train_push_loader, # pytorch dataloader unnorm
#         prototype_network_parallel=ppnet_multi,
#         preprocess_input_function = preprocess, # norma?
#         prototype_layer_stride=1,
#         root_dir_for_saving_prototypes = model_dir + '/img/',
#         save_name = save_name,
#         prototype_img_filename_prefix = 'prototype-img',
#         prototype_self_act_filename_prefix = 'prototype-self-act',
#         proto_bound_boxes_filename_prefix = 'bb',
#         save_prototype_class_identity=True)
