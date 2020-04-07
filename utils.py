import numpy as np
from fixed import fixed

def parse_input_text_file(filename, shape, n_frac):
    '''
    parse input text files (4-D tensor)
    '''
    images = shape[0]
    maps = shape[1]
    map_x = shape[2]
    map_y = shape[3]
    map_x_div_4 = map_x / 4
    map_y_div_4 = map_y / 4
    rows_per_map = map_x_div_4 * map_y_div_4
    rows_per_image = rows_per_map * maps
    rows_total = rows_per_image * images
    input = np.zeros(shape, dtype='float32')
    with open(filename, 'r') as f:
        lines = f.read().split('\n')
        if lines[-1] == '':
            lines.remove('')
        assert len(lines) == rows_total, "Specified shape (%d, %d, %d, %d) does not match with the file size %d" % (shape[0], shape[1], shape[2], shape[3], len(lines))
        for iImage in range(images):
            for iMap in range(maps):
                for iMap_x_div_4 in range(map_x_div_4):
                    for iMap_y_div_4 in range(map_y_div_4):
                        line = lines[iImage * rows_per_image + iMap * rows_per_map + iMap_x_div_4 * map_y_div_4 + iMap_y_div_4]
                        for x in range(4):
                            for y in range(4):
                                temp_str = line[(15-x*4-y)*4:(15-x*4-y)*4+4]
                                temp_num = int(temp_str, 16)
                                if (temp_num >= 2**15):
                                    temp_num -= 2**16
                                input[iImage, iMap, iMap_x_div_4 * 4 + x, iMap_y_div_4 * 4 + y] = temp_num / (2. ** n_frac)
        return input

def parse_fc_act_lg_text_file(filename, shape, n_frac):
    '''
    parse fully-connected activations/local gradients from text files (2-D tensor)
    '''
    images = shape[0]
    neurons = shape[1]
    neurons_div_16 = int(np.ceil(neurons / 16.))
    fc = np.zeros(shape, dtype='float32')
    rows_total = neurons_div_16 * images
    with open(filename, 'r') as f:
        lines = f.read().split('\n')
        if lines[-1] == '':
            lines.remove('')
        assert len(lines) == rows_total, "Specified shape (%d, %d) does not match with the file size %d" % (shape[0], shape[1], len(lines))
        for iNeurons_div_16 in range(neurons_div_16):
            for iImage in range(images):
                line = lines[iNeurons_div_16 * images + iImage]
                for n in range(16):
                    if (iNeurons_div_16 * 16 + n < neurons):
                        temp_str = line[(15-((n+iImage)%16))*4:(15-((n+iImage)%16))*4+4]
                        temp_num = int(temp_str, 16)
                        if (temp_num >= 2**15):
                            temp_num -= 2**16
                        fc[iImage, iNeurons_div_16*16+n] = temp_num / (2. ** n_frac)
    return fc

def parse_weight_text_file(filename, shape, n_frac):
    '''
    parse weight text files (could be convolutional (4D) or fully-connected (2D)
    '''
    if len(shape) > 2:
        n_maps_in = shape[1]
        n_maps_out = shape[0]
        kernel_x = shape[2]
        kernel_y = shape[3]
    else:
        n_maps_in = shape[0]
        n_maps_out = shape[1]
        kernel_x = 1
        kernel_y = 1
    weight = np.zeros(shape, dtype='float32')
    n_maps_out_div_16 = int(np.ceil(n_maps_out / 16.))
    rows_per_kernel = kernel_x * kernel_y
    rows_per_16_out_maps = n_maps_in * rows_per_kernel
    rows_total = n_maps_out_div_16 * n_maps_in * kernel_x * kernel_y
    with open(filename, 'r') as f:
        lines = f.read().split('\n')
        if lines[-1] == '':
            lines.remove('')
        assert len(lines) == rows_total, "Specified shape (%d, %d, %d, %d) does not match with the file size %d" % (n_maps_in, n_maps_out, kernel_x, kernel_y, len(lines))
        for iMap_out_div_16 in range(n_maps_out_div_16):
            for iMap_in in range(n_maps_in):
                for iK_x in range(kernel_x):
                    for iK_y in range(kernel_y):
                        line = lines[iMap_out_div_16 * rows_per_16_out_maps + iMap_in * rows_per_kernel + iK_x * kernel_y + iK_y]
                        for o in range(16):
                            if (iMap_out_div_16 * 16 + o < n_maps_out):
                                temp_str = line[(15-((iMap_in+o)%16))*4:(15-((iMap_in+o)%16))*4+4]
                                temp_num = int(temp_str, 16)
                                if (temp_num >= 2**15):
                                    temp_num -= 2**16
                                if len(shape) > 2:
                                    weight[iMap_out_div_16*16+o, iMap_in, iK_x, iK_y] = temp_num / (2. ** n_frac)
                                else:
                                    weight[iMap_in, iMap_out_div_16*16+o] = temp_num / (2. ** n_frac)
        return weight

def parse_weight_momentum_text_file(filename, shape, n_frac):
    '''
    parse weight text files (could be convolutional (4D) or fully-connected (2D)
    '''
    if len(shape) > 2:
        n_maps_in = shape[1]
        n_maps_out = shape[0]
        kernel_x = shape[2]
        kernel_y = shape[3]
    else:
        n_maps_in = shape[0]
        n_maps_out = shape[1]
        kernel_x = 1
        kernel_y = 1
    weight = np.zeros(shape, dtype='float32')
    n_maps_out_div_16 = int(np.ceil(n_maps_out / 16.))
    rows_per_kernel = kernel_x * kernel_y
    rows_per_16_out_maps = n_maps_in * rows_per_kernel
    rows_total = n_maps_out_div_16 * n_maps_in * kernel_x * kernel_y
    with open(filename, 'r') as f:
        lines = f.read().split('\n')
        if lines[-1] == '':
            lines.remove('')
        assert len(lines) == rows_total, "Specified shape (%d, %d, %d, %d) does not match with the file size %d" % (n_maps_in, n_maps_out, kernel_x, kernel_y, len(lines))
        for iMap_out_div_16 in range(n_maps_out_div_16):
            for iMap_in in range(n_maps_in):
                for iK_x in range(kernel_x):
                    for iK_y in range(kernel_y):
                        line = lines[iMap_out_div_16 * rows_per_16_out_maps + iMap_in * rows_per_kernel + iK_x * kernel_y + iK_y]
                        for o in range(16):
                            if (iMap_out_div_16 * 16 + o < n_maps_out):
                                temp_str = line[(15-o)*4:(15-o)*4+4]
                                temp_num = int(temp_str, 16)
                                if (temp_num >= 2**15):
                                    temp_num -= 2**16
                                if len(shape) > 2:
                                    weight[iMap_out_div_16*16+o, iMap_in, iK_x, iK_y] = temp_num / (2. ** n_frac)
                                else:
                                    weight[iMap_in, iMap_out_div_16*16+o] = temp_num / (2. ** n_frac)
        return weight

def parse_label_text_file(filename):
    '''
    parse labels from text file
    '''
    with open(filename, 'r') as f:
        lines = f.read().split('\n')
        if lines[-1] == '':
            lines.remove('')
        num_labels = len(lines)
        label = np.zeros((num_labels,), dtype='int64')
        for i in range(num_labels):
            label[i] = int(lines[i], 10)
    return label

def dump_weights_to_file(cnn, dirname='./data'):
    params = cnn.get_params()
    for key in params.keys():
        if key.endswith('W'):
            filename = dirname + '/' + key + '_initial.txt'
            print("Dumping initial values of %s to %s" % (key, filename))
            with open(filename, 'w+') as file:
                data = params[key]
                print("  quantized to 16-bit precision with %d integer bits (including the sign bit)" % (data.WL - data.FL))
                data_int = data.value.cpu().numpy()
                if len(data.shape) > 2: # conv layer
                    num_filters = data.shape[0]
                    num_channels = data.shape[1]
                    kernel_size = data.shape[2]
                    num_filters_div_16 = int((num_filters + 15) / 16)
                    for i in range(num_filters_div_16):
                        for j in range(num_channels):
                            for k in range(kernel_size):
                                for l in range(kernel_size):
                                    str = ''
                                    for m in range(16):
                                        if i*16+(15-m-j)%16 < num_filters:
                                            str += ('%04X' % (2**16+data_int[i*16+(15-m-j)%16][j][k][l]))[-4:]
                                        else:
                                            str += '0000'
                                    file.write('%s\n' % str)
                else: # fc layer
                    num_units = data.shape[1]
                    num_inputs = data.shape[0]
                    num_units_div_16 = int((num_units + 15) / 16)
                    for i in range(num_units_div_16):
                        for j in range(num_inputs):
                            str = ''
                            for m in range(16):
                                if i*16+(15-m-j)%16 < num_units:
                                    str += ('%04X' % (2**16+data_int[j][i*16+(15-m-j)%16]))[-4:]
                                else:
                                    str += '0000'
                            file.write('%s\n' % str)

def dump_Inputs_and_Lables_to_file_one_batch(train_X_mb, train_y_mb, group_size=10, dirname='./data'):
    batch_size = train_X_mb.shape[0]
    num_channels = train_X_mb.shape[1]
    map_x_size = train_X_mb.shape[2]
    map_y_size = train_X_mb.shape[3]
    num_groups = batch_size / group_size
    train_X_mb_int = train_X_mb.cpu().numpy()
    train_y_mb_int = train_y_mb.cpu().numpy()
    for group_index in range(num_groups):
        with open(dirname + "/Input_%d_initial.txt" % group_index, 'w+') as file:
            for image_index in range(group_size):
                for channel_index in range(num_channels):
                    for map_x_index in range(int(map_x_size/4)):
                        for map_y_index in range(int(map_y_size/4)):
                            str = ''
                            for xx in range(4):
                                for yy in range(4):
                                    str += ("%04X" % (2**16+train_X_mb_int[group_index*group_size+image_index][channel_index][map_x_index*4+3-xx][map_y_index*4+3-yy]))[-4:]
                            file.write("%s\n" % str)
        with open(dirname + "/Label_%d_initial.txt" % group_index, 'w+') as file:
            for image_index in range(group_size):
                str = ''
                str += "%d" % int(train_y_mb[group_index*group_size+image_index])
                file.write("%s\n" % str)



if __name__ == "__main__":
    A = parse_input_text_file('data_for_verilog_verification_MNIST/Input_0_initial.txt', (10, 1, 28, 28), 15)
    conv_0_W = parse_weight_text_file('data_for_verilog_verification_MNIST/conv_0_W_measured.txt', (8, 1, 5, 5), 14)
    conv_1_W = parse_weight_text_file('data_for_verilog_verification_MNIST/conv_1_W_measured.txt', (16, 8, 5, 5), 15)
    fc_W = parse_weight_text_file('data_for_verilog_verification_MNIST/fc_W_measured.txt', (256, 10), 15)
    import scipy.io as sio
    dict = {}
    dict['conv_0_W'] = conv_0_W
    dict['conv_1_W'] = conv_1_W
    dict['fc_W'] = fc_W
    sio.savemat('data_for_verilog_verification_MNIST/params_measured.mat', dict)
