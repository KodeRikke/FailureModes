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
        log('epoch: \t{0}'.format(epoch + epoch_reached), "trainlog.txt")

        if epoch < warm_epochs:
            warm_only(model=modelmulti)
            train(model=modelmulti, dataloader=train_loader, optimizer=warm_optimizer, coefs=coefs)
        else:
            joint(model=modelmulti)
            train(model=modelmulti, dataloader=train_loader, optimizer=joint_optimizer, coefs=coefs)
            joint_lr_scheduler.step()
        accu = test(model=modelmulti, dataloader=test_loader)

        torch.save(obj=model, f = os.path.join(model_dir, (str(epoch + epoch_reached) + 'nopush' + '{0:.4f}.pth').format(accu)))

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
#             accu = test(model=modelmulti, dataloader=test_loader)
#             torch.save(obj=model, f = os.path.join(model_dir, (str(epoch) + 'push' + '{0:.4f}.pth').format(accu)))

#             last_only(model=modelmulti)
#             for i in range(20):
#                 log('iteration: \t{0}'.format(i), "trainlog.txt")
#                 _ = train(model=modelmulti, dataloader=train_loader, optimizer=last_layer_optimizer, coefs=coefs)
#                 accu = test(model=modelmulti, dataloader=test_loader)
#                 torch.save(obj=model, f = os.path.join(model_dir, (str(epoch) + '_' + str(i) + 'push').format(accu)))

def initialize_model(model_name = ""):
    features = base_architecture_to_features['vgg19_bn'](pretrained = True)
    layer_filter_sizes, layer_strides, layer_paddings = features.conv_info()

    if model_name == "":
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
    else:
        ppnet = torch.load(path + "pretrained_models/" + model_name)

    ppnet = ppnet.cuda()
    ppnet_multi = torch.nn.DataParallel(ppnet)#, device_ids=[0, 1, 2, 3])
    return ppnet, ppnet_multi