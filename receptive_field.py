import math

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