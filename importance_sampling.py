from models import CNN
import torch
import numpy as np
import argparse
import scipy.io as sio
import time
from ast import literal_eval as bool
from fixed import *
import datetime
import sys
import os
import logging
import re


dtype = torch.int64
device = torch.device("cuda")
np.random.seed(1234)
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

if os.path.exists('./result/log_importance_sampling.txt'):
    os.remove('./result/log_importance_sampling.txt')
print("File Removed!")

log_format = '%(asctime)s   %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M%p')
fh = logging.FileHandler(os.path.join('./result', 'log_importance_sampling.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CNN for MNIST",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-dp', '--dropout_prob', dest='dropout_prob', default=0.8,
                        type=float, help="Initial learning rate.")
    parser.add_argument('-fm', '--filter_mult', dest='filter_mult', default=1,
                        type=float, help="Initial learning rate.")
    parser.add_argument('-ls', '--LR_start', dest='LR_start', default=0.015,
                        type=float, help="Initial learning rate.")
    parser.add_argument('-lf', '--LR_finish', dest='LR_finish', default=0.006,
                        type=float, help="Ending learning rate.")
    parser.add_argument('-mo', '--momentum', dest='momentum', default=0.5,
                        type=float, help="Momentum factor for learning")
    parser.add_argument('-bs', '--batch_size', dest='batch_size',
                        default=40, type=int, help="Batch size for training")
    parser.add_argument('-gs', '--group_size', dest='group_size',
                        default=20, type=int, help="Group size that can be fit in on-chip memory")
    parser.add_argument('-ne', '--num_epochs', dest='num_epochs',
                        default=50, type=int, help="Number of epochs for training")
    parser.add_argument('-fl', '--flip_lr', dest='flip_lr', default=False,
                        type=bool, help="Data augmentation by flipping images horizonatally")
    parser.add_argument('-vb', '--verbose', dest='verbose', default=False,
                        type=bool, help="Display range information if true")
    parser.add_argument('-task_division', type=str, default='9,1')

    args = parser.parse_args()
    dropout_prob = args.dropout_prob
    logging.info(" - - - - - importance_sampling.py  - - - - - - - ")
    logging.info("\n\n\nargs = %s", args)

    # logging.info  sys.argv[1:]

    LR_decay = (args.LR_finish / args.LR_start) ** (1. / (args.num_epochs - 1))
    cnn = CNN()

    # fraction + integer
    FL_A_input = 15  # precision 16bit
    FL_W_conv_0 = 15  # fractional 1 bit + . + 15 bits
    FL_W_conv_1 = 15
    FL_W_conv_2 = 15
    FL_W_conv_3 = 16  # 0 bit + . + 16bits
    FL_W_conv_4 = 16
    FL_W_conv_5 = 16
    FL_W_fc = 16
    FL_WM_conv_0 = 19  # momentum 32bits
    FL_WM_conv_1 = 19
    FL_WM_conv_2 = 19
    FL_WM_conv_3 = 20
    FL_WM_conv_4 = 20
    FL_WM_conv_5 = 20
    FL_WM_fc = 20
    FL_A_conv_0 = FL_A_input + FL_W_conv_0 - 16
    FL_A_conv_1 = FL_A_conv_0 + FL_W_conv_1 - 16
    FL_A_conv_2 = FL_A_conv_1 + FL_W_conv_2 - 16
    FL_A_conv_3 = FL_A_conv_2 + FL_W_conv_3 - 16
    FL_A_conv_4 = FL_A_conv_3 + FL_W_conv_4 - 16
    FL_A_conv_5 = FL_A_conv_4 + FL_W_conv_5 - 16
    FL_A_fc = FL_A_conv_5 + FL_W_fc - 16
    FL_D_fc = FL_A_fc
    FL_D_conv_5 = FL_D_fc + FL_W_fc - 16
    FL_D_conv_4 = FL_D_conv_5 + FL_W_conv_5 - 16
    FL_D_conv_3 = FL_D_conv_4 + FL_W_conv_4 - 16
    FL_D_conv_2 = FL_D_conv_3 + FL_W_conv_3 - 16
    FL_D_conv_1 = FL_D_conv_2 + FL_W_conv_2 - 16
    FL_D_conv_0 = FL_D_conv_1 + FL_W_conv_1 - 16
    FL_D_input = FL_D_conv_0 + FL_W_conv_0 - 16
    FL_WG_conv_0 = FL_A_input + FL_D_conv_0 - 16
    FL_WG_conv_1 = FL_A_conv_0 + FL_D_conv_1 - 16
    FL_WG_conv_2 = FL_A_conv_1 + FL_D_conv_2 - 16
    FL_WG_conv_3 = FL_A_conv_2 + FL_D_conv_3 - 16
    FL_WG_conv_4 = FL_A_conv_3 + FL_D_conv_4 - 16
    FL_WG_conv_5 = FL_A_conv_4 + FL_D_conv_5 - 16
    FL_WG_fc = FL_A_conv_5 + FL_D_fc - 16

    FL_L_WG_conv_0 = 16 + FL_WM_conv_0 - FL_WG_conv_0
    FL_L_WG_conv_1 = 16 + FL_WM_conv_1 - FL_WG_conv_1
    FL_L_WG_conv_2 = 16 + FL_WM_conv_2 - FL_WG_conv_2
    FL_L_WG_conv_3 = 16 + FL_WM_conv_3 - FL_WG_conv_3
    FL_L_WG_conv_4 = 16 + FL_WM_conv_4 - FL_WG_conv_4
    FL_L_WG_conv_5 = 16 + FL_WM_conv_5 - FL_WG_conv_5
    FL_L_WG_fc = 16 + FL_WM_fc - FL_WG_fc
    FL_L_WU_conv_0 = 16 + FL_W_conv_0 - FL_WM_conv_0
    FL_L_WU_conv_1 = 16 + FL_W_conv_1 - FL_WM_conv_1
    FL_L_WU_conv_2 = 16 + FL_W_conv_2 - FL_WM_conv_2
    FL_L_WU_conv_3 = 16 + FL_W_conv_3 - FL_WM_conv_3
    FL_L_WU_conv_4 = 16 + FL_W_conv_4 - FL_WM_conv_4
    FL_L_WU_conv_5 = 16 + FL_W_conv_5 - FL_WM_conv_5
    FL_L_WU_fc = 16 + FL_W_fc - FL_WM_fc

    FL_M_WU_conv_0 = 16
    FL_M_WU_conv_1 = 16
    FL_M_WU_conv_2 = 16
    FL_M_WU_conv_3 = 16
    FL_M_WU_conv_4 = 16
    FL_M_WU_conv_5 = 16
    FL_M_WU_fc = 16

    LR_conv_0_upperbound = args.batch_size * 2 ** (FL_WG_conv_0 - FL_W_conv_0 - 2) * 0.9
    LR_conv_1_upperbound = args.batch_size * 2 ** (FL_WG_conv_1 - FL_W_conv_1 - 2) * 0.9
    LR_conv_2_upperbound = args.batch_size * 2 ** (FL_WG_conv_2 - FL_W_conv_2 - 2) * 0.9
    LR_conv_3_upperbound = args.batch_size * 2 ** (FL_WG_conv_3 - FL_W_conv_3 - 2) * 0.9
    LR_conv_4_upperbound = args.batch_size * 2 ** (FL_WG_conv_4 - FL_W_conv_4 - 2) * 0.9
    LR_conv_5_upperbound = args.batch_size * 2 ** (FL_WG_conv_5 - FL_W_conv_5 - 2) * 0.9
    LR_fc_upperbound = args.batch_size * 2 ** (FL_WG_fc - FL_W_fc - 2) * 0.9
    if args.LR_start > LR_fc_upperbound:
        args.LR_start = LR_fc_upperbound
        args.LR_finish = LR_fc_upperbound
    # scale_conv_0 = 2**((FL_L_WG_conv_0-FL_L_WU_conv_0)/2) * (args.LR_start / args.batch_size) ** 0.5
    # scale_conv_1 = 2**((FL_L_WG_conv_1-FL_L_WU_conv_1)/2) * (args.LR_start / args.batch_size) ** 0.5
    # scale_conv_2 = 2**((FL_L_WG_conv_2-FL_L_WU_conv_2)/2) * (args.LR_start / args.batch_size) ** 0.5
    # scale_conv_3 = 2**((FL_L_WG_conv_3-FL_L_WU_conv_3)/2) * (args.LR_start / args.batch_size) ** 0.5
    # scale_fc = 2**((FL_L_WG_fc-FL_L_WU_fc)/2) * (args.LR_start / args.batch_size) ** 0.5
    scale_conv_0 = 1.05 * 2. ** (FL_L_WG_conv_0 - 15) * args.LR_start / args.batch_size
    scale_conv_1 = 1.05 * 2. ** (FL_L_WG_conv_1 - 15) * args.LR_start / args.batch_size
    scale_conv_2 = 1.05 * 2. ** (FL_L_WG_conv_2 - 15) * args.LR_start / args.batch_size
    scale_conv_3 = 1.05 * 2. ** (FL_L_WG_conv_3 - 15) * args.LR_start / args.batch_size
    scale_conv_4 = 1.05 * 2. ** (FL_L_WG_conv_4 - 15) * args.LR_start / args.batch_size
    scale_conv_5 = 1.05 * 2. ** (FL_L_WG_conv_5 - 15) * args.LR_start / args.batch_size
    scale_fc = 1.05 * 2. ** (FL_L_WG_fc - 15) * args.LR_start / args.batch_size
    scale = 2
    global_vals = dict(globals().items())
    # for name in sorted(global_vals.iterkeys()):
    # if name.startswith('FL') or name.startswith('scale') or name.startswith('LR'):
    # logging.info("%s: %s" % (name, global_vals[name]))
    # Load and preprocess CIFAR10 dataset
    data = sio.loadmat('../order_cifar10.mat')
    train_X = data['train_X']
    valid_X = data['valid_X']
    test_X = data['test_X']
    train_y = np.argmax(data['train_y'], axis=1)
    valid_y = np.argmax(data['valid_y'], axis=1)
    test_y = np.argmax(data['test_y'], axis=1)

    # cloud dataset
    task_list = list(range(10))
    task_division = list(map(int, args.task_division.split(",")))
    cloud_list = task_list[0: task_division[0]]
    logging.info('cloud list %s\n' % (cloud_list))

    num_image_train = 45000 / 10 * task_division[0]
    num_image_test = 10000 / 10 * task_division[0]
    num_image_valid = 5000 / 10 * task_division[0]
    #
    train_X = train_X[0:num_image_train]
    valid_X = valid_X[0:num_image_valid]
    test_X = test_X[0: num_image_test]
    train_y = train_y[0:num_image_train]
    valid_y = valid_y[0:num_image_valid]
    test_y = test_y[0: num_image_test]

    train_X = fixed(train_X, 16, FL_A_input)
    valid_X = fixed(valid_X, 16, FL_A_input)
    test_X = fixed(test_X, 16, FL_A_input)
    train_y = torch.from_numpy(train_y).to(torch.int64)
    valid_y = torch.from_numpy(valid_y).to(torch.int64)
    test_y = torch.from_numpy(test_y).to(torch.int64)

    logging.info('train_X.shape %s' % str(train_X.shape))
    logging.info('test_X.shape %s ' % str(test_X.shape))
    logging.info('valid_X.shape %s' % str(valid_X.shape))
    logging.info('train_y.shape %s' % str(train_y.shape))
    logging.info('test_y.shape %s ' % str(test_y.shape))
    logging.info('valid_y.shape %s' % str(valid_y.shape))
    # Build CNN for CIFAR10
    cnn.append_layer('Conv_fixed',
                     name='conv_0',
                     input_map_size=(32, 32),
                     num_filters=16 * args.filter_mult,
                     num_channels=3,
                     kernel_size=3,
                     pads=1,
                     FL_AO=FL_A_conv_0,
                     FL_DI=FL_D_input,
                     FL_W=FL_W_conv_0,
                     FL_WM=FL_WM_conv_0,
                     FL_WG=FL_WG_conv_0,
                     FL_L_WG=FL_L_WG_conv_0,
                     FL_L_WU=FL_L_WU_conv_0,
                     FL_M_WU=FL_M_WU_conv_0,
                     scale=scale_conv_0,
                     dropout_prob=dropout_prob
                     )
    cnn.append_layer('Conv_fixed',
                     name='conv_1',
                     input_map_size=(32, 32),
                     num_filters=16 * args.filter_mult,
                     num_channels=16 * args.filter_mult,
                     kernel_size=3,
                     pads=1,
                     FL_AO=FL_A_conv_1,
                     FL_DI=FL_D_conv_0,
                     FL_W=FL_W_conv_1,
                     FL_WM=FL_WM_conv_1,
                     FL_WG=FL_WG_conv_1,
                     FL_L_WG=FL_L_WG_conv_1,
                     FL_L_WU=FL_L_WU_conv_1,
                     FL_M_WU=FL_M_WU_conv_1,
                     scale=scale_conv_1,
                     dropout_prob=dropout_prob
                     )
    cnn.append_layer('MaxPooling',
                     name='pool_0',
                     input_map_size=(32, 32),
                     num_channels=16 * args.filter_mult)
    cnn.append_layer('Conv_fixed',
                     name='conv_2',
                     input_map_size=(16, 16),
                     num_filters=32 * args.filter_mult,
                     num_channels=16 * args.filter_mult,
                     kernel_size=3,
                     pads=1,
                     FL_AO=FL_A_conv_2,
                     FL_DI=FL_D_conv_1,
                     FL_W=FL_W_conv_2,
                     FL_WM=FL_WM_conv_2,
                     FL_WG=FL_WG_conv_2,
                     FL_L_WG=FL_L_WG_conv_2,
                     FL_L_WU=FL_L_WU_conv_2,
                     FL_M_WU=FL_M_WU_conv_2,
                     scale=scale_conv_2,
                     dropout_prob=dropout_prob
                     )
    cnn.append_layer('Conv_fixed',
                     name='conv_3',
                     input_map_size=(16, 16),
                     num_filters=32 * args.filter_mult,
                     num_channels=32 * args.filter_mult,
                     kernel_size=3,
                     pads=1,
                     FL_AO=FL_A_conv_3,
                     FL_DI=FL_D_conv_2,
                     FL_W=FL_W_conv_3,
                     FL_WM=FL_WM_conv_3,
                     FL_WG=FL_WG_conv_3,
                     FL_L_WG=FL_L_WG_conv_3,
                     FL_L_WU=FL_L_WU_conv_3,
                     FL_M_WU=FL_M_WU_conv_3,
                     scale=scale_conv_3,
                     dropout_prob=dropout_prob
                     )
    cnn.append_layer('MaxPooling',
                     name='pool_1',
                     input_map_size=(16, 16),
                     num_channels=32 * args.filter_mult)
    cnn.append_layer('Conv_fixed',
                     name='conv_4',
                     input_map_size=(8, 8),
                     num_filters=64 * args.filter_mult,
                     num_channels=32 * args.filter_mult,
                     kernel_size=3,
                     pads=1,
                     FL_AO=FL_A_conv_4,
                     FL_DI=FL_D_conv_3,
                     FL_W=FL_W_conv_4,
                     FL_WM=FL_WM_conv_4,
                     FL_WG=FL_WG_conv_4,
                     FL_L_WG=FL_L_WG_conv_4,
                     FL_L_WU=FL_L_WU_conv_4,
                     FL_M_WU=FL_M_WU_conv_4,
                     scale=scale_conv_4,
                     dropout_prob=dropout_prob
                     )
    cnn.append_layer('Conv_fixed',
                     name='conv_5',
                     input_map_size=(8, 8),
                     num_filters=64 * args.filter_mult,
                     num_channels=64 * args.filter_mult,
                     kernel_size=3,
                     pads=1,
                     FL_AO=FL_A_conv_5,
                     FL_DI=FL_D_conv_4,
                     FL_W=FL_W_conv_5,
                     FL_WM=FL_WM_conv_5,
                     FL_WG=FL_WG_conv_5,
                     FL_L_WG=FL_L_WG_conv_5,
                     FL_L_WU=FL_L_WU_conv_5,
                     FL_M_WU=FL_M_WU_conv_5,
                     scale=scale_conv_5,
                     dropout_prob=dropout_prob
                     )
    cnn.append_layer('MaxPooling',
                     name='pool_2',
                     input_map_size=(8, 8),
                     num_channels=64 * args.filter_mult)
    cnn.append_layer('Flatten',
                     name='flatten',
                     input_map_size=(4, 4),
                     num_channels=64 * args.filter_mult)
    cnn.append_layer('FC_fixed',
                     name='fc',
                     input_dim=1024 * args.filter_mult,
                     num_units=10,
                     relu=False,
                     FL_AO=FL_A_fc,
                     FL_DI=FL_D_conv_5,
                     FL_W=FL_W_fc,
                     FL_WM=FL_WM_fc,
                     FL_WG=FL_WG_fc,
                     FL_L_WG=FL_L_WG_fc,
                     FL_L_WU=FL_L_WU_fc,
                     FL_M_WU=FL_M_WU_fc,
                     scale=scale_fc
                     )
    cnn.append_layer('SquareHingeLoss',
                     name='Loss',
                     num_classes=10)

    # Testing
    logging.info ('loading trained weights...')
    cnn.load_params_mat('./result/result_{}classes/Best_epoch_CIFAR10_W.mat'.format(task_division[0]))
    # cnn.load_params_mat('./result/test_W.mat')
    # print('test_W')
    batch_size_valid = 40
    num_batches_valid = int(num_image_valid / batch_size_valid)
    valid_error = 0.
    valid_loss = 0.

    for j in range(num_batches_valid):  # testing
        predictions, valid_loss_batch = cnn.feed_forward(fixed(valid_X[j * batch_size_valid:(j + 1) * batch_size_valid],
                                                               16, FL_A_input),
                                                         valid_y[j * batch_size_valid:(j + 1) * batch_size_valid],
                                                         train_or_test=0)
        valid_error += torch.sum(predictions.cpu() != valid_y[j * batch_size_valid:(j + 1) * batch_size_valid]).numpy()
        valid_loss += valid_loss_batch
    valid_error /= num_image_valid
    valid_loss /= num_batches_valid
    valid_acc = (100 - (valid_error * 100))
    logging.info("    valid accuracy: %.2f%%" % valid_acc)
    logging.info("    valid loss: %.4f" % valid_loss)

    threshold = task_division[0] / 10.0
    print('\n Generating mask for top %.3f  params' % threshold)
    W = sio.loadmat('./result/result_{}classes/Best_epoch_CIFAR10_W.mat'.format(task_division[0]))  # file Shreyas needs

    # W_loadback = sio.savemat('./result/test_W.mat', W)
    mask = dict()
    # print(type(W['conv_2_W'][0,0,0,0])) ## should be int16
    for key, value in W.items():
        if re.search('conv_', key):
            # print(key, value.shape)
            # print(value[0:5, :, :, :])
            num_channel = value.shape[0]
            metrics = np.sum(abs(value), axis = (1,2,3))
            # print(metrics)
            num_freeze = int(round(num_channel * threshold))
            arg_max = np.argsort(metrics) # Returns the indices sort an array. small->big
            # print(arg_max)
            arg_max_rev = arg_max[::-1][:num_freeze]  # big - > small
            thre = metrics[arg_max_rev[-1]]  # min metrics
            mask[key] = np.ones(value.shape)
            mask[key][arg_max_rev.tolist(), :, :, :] = 0.0  ## mask = 0 means these pixels to be frozen; mask = 1 means these pixels will be updated in the future
            print('mask generated for layer %s, shape: %s ' % (key, mask[key].shape))
            # print(mask[0:5, :, :, :])
            # break
        elif re.search('fc_', key):
            # print(key, value.shape)
            # print(value[0:5,:])
            num_channel = value.shape[0]
            metrics = np.sum(abs(value), axis = (1))
            # print(metrics)
            num_freeze = int(round(num_channel * threshold))
            arg_max = np.argsort(metrics) # Returns the indices sort an array. small->big
            # print(arg_max)
            arg_max_rev = arg_max[::-1][:num_freeze]  # big - > small
            thre = metrics[arg_max_rev[-1]]  # min metrics

            mask[key] = np.ones(value.shape)
            mask[key][arg_max_rev.tolist(), :] = 0.0  ## mask = 0 means these pixels to be frozen; mask = 1 means these pixels will be updated in the future
            print('mask generated for layer %s, shape: %s ' % (key, mask[key].shape))
            # print(mask[key][0:5, :])
            # break

    sio.savemat('./result/result_{}classes/mask_CIFAR10_TaskDivision_{}.mat'.format(task_division[0],args.task_division), mask)
    print('\nmask saved in ./result/result_{}classes/mask_CIFAR10_TaskDivision_{}.mat'.format(task_division[0],args.task_division))
    #
    # mask_test = sio.loadmat('./result/mask_CIFAR10_TaskDivision_{}.mat'.format(args.task_division))
    # for key, value in mask_test.items():
    #     if re.search('W', key):
    #         print(key, np.sum(value, axis = (0,1,2,3)))
