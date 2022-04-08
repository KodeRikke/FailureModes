import os
import time
import torch

from helpers import list_of_distances, log

def _train_or_test(model, dataloader, trainlog, optimizer=None, use_l1_mask=True, coefs=None):
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

    log('\ttime: \t{0}'.format(end -  start), trainlog)
    log('\tcross ent: \t{0}'.format(total_cross_entropy / n_batches), trainlog)
    log('\tcluster: \t{0}'.format(total_cluster_cost / n_batches), trainlog)
    log('\tseparation:\t{0}'.format(total_separation_cost / n_batches), trainlog)
    log('\tavg separation:\t{0}'.format(total_avg_separation_cost / n_batches), trainlog)
    log('\taccu: \t\t{0}%'.format(n_correct / n_examples * 100), trainlog)
    log('\tl1: \t\t{0}'.format(model.module.last_layer.weight.norm(p=1).item()), trainlog)
    p = model.module.prototype_vectors.view(model.module.num_prototypes, -1).cpu()
    with torch.no_grad():
        p_avg_pair_dist = torch.mean(list_of_distances(p, p))
    log('\tp dist pair: \t{0}'.format(p_avg_pair_dist.item()), trainlog)

    return n_correct / n_examples

def train(model, dataloader, trainlog, optimizer, coefs=None):
    log('\ttrain', trainlog)
    model.train()
    return _train_or_test(model=model, dataloader=dataloader, trainlog=trainlog, optimizer=optimizer, coefs=coefs)
  
def test(model, dataloader, trainlog):
    log('\ttest', trainlog)
    model.eval()
    return _train_or_test(model=model, dataloader=dataloader, trainlog=trainlog, optimizer=None)

def last_only(model, trainlog):
    for p in model.module.features.parameters():
        p.requires_grad = False
    for p in model.module.add_on_layers.parameters():
        p.requires_grad = False
    model.module.prototype_vectors.requires_grad = False
    for p in model.module.last_layer.parameters():
        p.requires_grad = True
    log('\tlast layer', trainlog)

def warm_only(model, trainlog):
    for p in model.module.features.parameters():
        p.requires_grad = False
    for p in model.module.add_on_layers.parameters():
        p.requires_grad = True
    model.module.prototype_vectors.requires_grad = True
    for p in model.module.last_layer.parameters():
        p.requires_grad = True
    log('\twarm', trainlog)

def joint(model, trainlog):
    for p in model.module.features.parameters():
        p.requires_grad = True
    for p in model.module.add_on_layers.parameters():
        p.requires_grad = True
    model.module.prototype_vectors.requires_grad = True
    for p in model.module.last_layer.parameters():
        p.requires_grad = True
    log('\tjoint', trainlog)
