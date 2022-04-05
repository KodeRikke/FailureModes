import torch

from model import initialize_model
from train_and_test import fit
from settings import model_dir, epochs, warm_epochs, epoch_reached

model_name = "10push0.5516.pth"
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

if model_name != "":
    epoch_reached = checkpoint['epoch'] + 1 # next epoch
    joint_optimizer.load_state_dict(checkpoint['joint_optimizer_state_dict'])
    joint_lr_scheduler.load_state_dict(checkpoint['joint_lr_scheduler_state_dict'])
    last_layer_optimizer.load_state_dict(checkpoint['last_layer_optimizer_state_dict'])
    warm_optimizer.load_state_dict(checkpoint['warm_optimizer_state_dict'])

fit(ppnet, ppnet_multi, epochs = epochs, warm_epochs = warm_epochs, epoch_reached = epoch_reached)