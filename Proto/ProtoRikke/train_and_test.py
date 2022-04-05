import time
import torch

from helpers import list_of_distances, log


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

def fit(model, modelmulti, epochs, warm_epochs, epoch_reached):
    log('start training', "trainlog.txt")
    for epoch in range(epoch_reached, epochs):
        model.train()
        modelmulti.train()
        log('epoch: \t{0}'.format(epoch), "trainlog.txt")
        
        if epoch < warm_epochs:
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
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'joint_optimizer_state_dict': joint_optimizer.state_dict(),
            'joint_lr_scheduler_state_dict': joint_lr_scheduler.state_dict(),
            'last_layer_optimizer_state_dict': last_layer_optimizer.state_dict(),
            'warm_optimizer_state_dict' : warm_optimizer.state_dict()
            }, os.path.join(model_dir, (str(epoch) + 'nopush' + '{0:.4f}.pth').format(accu)))

        if epoch >= push_start and epoch in push_epochs:
            push_prototypes(
                train_push_loader, # pytorch dataloader unnorm
                prototype_network_parallel=modelmulti,
                preprocess_input_function = preprocess, # norma?
                prototype_layer_stride=1,
                root_dir_for_saving_prototypes = model_dir + '/img/',
                epoch_number = 10, #epoch, # if not provided, prototypes saved previously will be overwritten
                prototype_img_filename_prefix = 'prototype-img',
                prototype_self_act_filename_prefix = 'prototype-self-act',
                proto_bound_boxes_filename_prefix = 'bb',
                save_prototype_class_identity=True)
            last_only(model=modelmulti)
            for i in range(100):
                log('iteration: \t{0}'.format(i), "trainlog.txt")
                _ = train(model=modelmulti, dataloader=train_loader, optimizer=last_layer_optimizer, coefs=coefs)
                accu = test(model=modelmulti, dataloader=test_loader)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'joint_optimizer_state_dict': joint_optimizer.state_dict(),
                    'joint_lr_scheduler_state_dict': joint_lr_scheduler.state_dict(),
                    'last_layer_optimizer_state_dict': last_layer_optimizer.state_dict(),
                    'warm_optimizer_state_dict' : warm_optimizer.state_dict()
                    }, os.path.join(model_dir, (str(epoch) + 'push' + '{0:.4f}.pth').format(accu)))

fit(ppnet, ppnet_multi, epochs = epochs, warm_epochs = warm_epochs, epoch_reached = epoch_reached)