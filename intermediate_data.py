



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
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

if os.path.exists('./result/log_incremental_learning.txt'):
    os.remove('./result/log_incremental_learning.txt')
print("File Removed!")

log_format = '%(asctime)s   %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M%p')
fh = logging.FileHandler(os.path.join('./result', 'log_incremental_learning.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)



def process_dataset(train_X, test_X, valid_X, train_y, test_y, valid_y):

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
    logging.info('valid_y.shape %s\n' % str(valid_y.shape))

    return train_X, test_X, valid_X, train_y, test_y, valid_y

def valid_single_image(image_idx, X, y):
    valid_error = 0.
    # print(X[image_idx:image_idx+1, ...].shape)
    predictions, valid_loss_batch = cnn.feed_forward(fixed(X[image_idx:image_idx+1, ...], 16, FL_A_input), y[image_idx:image_idx+1], train_or_test=0, record = True)
    valid_error += torch.sum(predictions.cpu() != y[image_idx:image_idx+1]).numpy()
    valid_acc = 1 - valid_error
    logging.info('Testing one single image, idx {}'.format(image_idx))
    return valid_acc

def valid(num_image, X, y):
    batch_size_valid = 40
    num_batches = int(num_image / batch_size_valid)
    valid_error = 0.
    valid_loss = 0.

    for j in range(num_batches):  # testing
        predictions, valid_loss_batch = cnn.feed_forward(fixed(X[j * batch_size_valid:(j + 1) * batch_size_valid],
                                                               16, FL_A_input),
                                                         y[j * batch_size_valid:(j + 1) * batch_size_valid],
                                                         train_or_test=0, record = False)
        valid_error += torch.sum(predictions.cpu() !=y[j * batch_size_valid:(j + 1) * batch_size_valid]).numpy()
        valid_loss += valid_loss_batch
    valid_error /= num_image
    valid_loss /= num_batches
    valid_acc = (100 - (valid_error * 100))
    return valid_acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CNN for MNIST",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-dp', '--dropout_prob', dest='dropout_prob', default=0.8,
                        type=float, help="Initial learning rate.")
    parser.add_argument('-fm', '--filter_mult', dest='filter_mult', default=1,
                        type=float, help="Initial learning rate.")
    parser.add_argument('-ls', '--LR_start', dest='LR_start', default=0.0015,
                        type=float, help="Initial learning rate.")
    parser.add_argument('-lf', '--LR_finish', dest='LR_finish', default=0.0006,
                        type=float, help="Ending learning rate.")
    parser.add_argument('-mo', '--momentum', dest='momentum', default=0.5,
                        type=float, help="Momentum factor for learning")
    parser.add_argument('-bs', '--batch_size', dest='batch_size',
                        default=40, type=int, help="Batch size for training")
    parser.add_argument('-gs', '--group_size', dest='group_size',
                        default=20, type=int, help="Group size that can be fit in on-chip memory")
    parser.add_argument('-ne', '--num_epochs', dest='num_epochs',
                        default= 2, type=int, help="Number of epochs for training")
    parser.add_argument('-fl', '--flip_lr', dest='flip_lr', default=False,
                        type=bool, help="Data augmentation by flipping images horizonatally")
    parser.add_argument('-vb', '--verbose', dest='verbose', default=False,
                        type=bool, help="Display range information if true")
    parser.add_argument('-task_division', type=str, default='9,1')

    args = parser.parse_args()
    dropout_prob = args.dropout_prob
    logging.info("\n\n\nargs = %s", args)
    logging.info(" - - - - - incremental_learning.py  - - - - - - - ")

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

    # # cloud dataset
    task_list = list(range(10))
    task_division = list(map(int, args.task_division.split(",")))
    cloud_list = task_list[0: task_division[0]]
    logging.info('\ncloud list %s' % (cloud_list))

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

    data = sio.loadmat('../order_cifar10.mat')
    train_X = data['train_X']
    valid_X = data['valid_X']
    test_X = data['test_X']
    train_y = np.argmax(data['train_y'], axis=1)
    valid_y = np.argmax(data['valid_y'], axis=1)
    test_y = np.argmax(data['test_y'], axis=1)


    edge_list = task_list[task_division[0] : (task_division[0] + task_division[1])]
    logging.info('\n......Edge list %s' % (edge_list))

    cloud_image_train = 45000 / 10 * task_division[0]
    cloud_image_test = 10000 / 10 * task_division[0]
    cloud_image_valid = 5000 / 10 * task_division[0]

    edge_image_train = 45000 / 10 * task_division[1]
    edge_image_test = 10000 / 10 * task_division[1]
    edge_image_valid = 5000 / 10 * task_division[1] #500

    # cloud data
    train_cloud_x = train_X[0:cloud_image_train]
    valid_cloud_x = valid_X[0:cloud_image_valid]
    test_cloud_x = test_X[0: cloud_image_test]
    train_cloud_y = train_y[0:cloud_image_train]
    valid_cloud_y = valid_y[0:cloud_image_valid]
    test_cloud_y = test_y[0: cloud_image_test]

    # edge data
    train_edge_x = train_X[cloud_image_train : (cloud_image_train + edge_image_train)]
    valid_edge_x = valid_X[cloud_image_valid : (cloud_image_valid + edge_image_valid)]
    test_edge_x = test_X[cloud_image_test : (cloud_image_train + edge_image_train)]
    train_edge_y = train_y[cloud_image_train : (cloud_image_train + edge_image_train)]
    valid_edge_y = valid_y[cloud_image_valid: (cloud_image_train + edge_image_train)]
    test_edge_y = test_y[cloud_image_test : (cloud_image_train + edge_image_train)]

    # full data
    train_full_x = train_X[0:45000]
    valid_full_x = valid_X[0:5000]
    test_full_x = test_X[0: 10000]
    train_full_y = train_y[0:45000]
    valid_full_y = valid_y[0:5000]
    test_full_y = test_y[0: 10000]

    train_cloud_x, test_cloud_x, valid_cloud_x, train_cloud_y, test_cloud_y, valid_cloud_y = process_dataset(train_cloud_x, test_cloud_x, valid_cloud_x, train_cloud_y, test_cloud_y, valid_cloud_y)
    train_edge_x,  test_edge_x,  valid_edge_x,  train_edge_y,  test_edge_y,  valid_edge_y =  process_dataset(train_edge_x,  test_edge_x,  valid_edge_x,  train_edge_y,  test_edge_y,  valid_edge_y)
    train_full_x,  test_full_x,  valid_full_x,  train_full_y,  test_full_y,  valid_full_y =  process_dataset(train_full_x,  test_full_x,  valid_full_x,  train_full_y,  test_full_y,  valid_full_y)

    # Training
    logging.info("dropout %f" % (args.dropout_prob))
    logging.info("filter mult %f" % (args.filter_mult))
    logging.info("batch size %f" % (args.batch_size))
    logging.info("group size %f" % (args.group_size))
    logging.info("LR start %f" % (args.LR_start))
    logging.info("LR finish %f" % (args.LR_finish))

    currentDT = datetime.datetime.now()
    logging.info(str(currentDT))
    batch_size = args.batch_size
    group_size = args.group_size
    num_batches = int(edge_image_train / batch_size)
    num_groups = int(batch_size / group_size)
    Learning_Rate = args.LR_start
    best_valid_acc = 0.0

    logging.info("\n\n--------checking mask--------------------")

    cnn.load_params_mat('./result/result_{}classes/incremental_1class_Best_epoch_CIFAR10_W'.format(task_division[0]))

    mask = sio.loadmat('./result/result_{}classes/mask_CIFAR10_TaskDivision_{}.mat'.format(task_division[0], args.task_division))
    for key, value in mask.items():
        if re.search('conv', key):
            logging.info('layer {}, sum of mask {} out of shape{}'.format(key, np.sum(value, axis = (0,1,2,3)), value.shape))
        if re.search('fc', key):
            logging.info('layer {}, sum of mask {} out of shape {}'.format(key, np.sum(value, axis = (0,1)), value.shape))


    #
    cloud_acc = valid(cloud_image_valid, valid_cloud_x, valid_cloud_y)
    edge_acc = valid(edge_image_valid, valid_edge_x, valid_edge_y)
    full_acc = valid(5000, valid_full_x, valid_full_y)
    logging.info("\n\n-------------Test after loading pre-trained model---------------")
    logging.info("On cloud dataset {}, valid accuracy: {:.2f}%".format(cloud_list, cloud_acc))
    logging.info("On edge dataset  {},                         valid accuracy: {:.2f}%".format(edge_list, edge_acc))
    logging.info("On full dataset {}, valid accuracy: {:.2f}%".format(task_list, full_acc))


    cloud_acc = valid_single_image(2, valid_cloud_x, valid_cloud_y)
    edge_acc = valid_single_image(0, valid_edge_x, valid_edge_y)
    full_acc = valid_single_image(2, valid_full_x, valid_full_y)
    logging.info("\n\n-------------Test single image after loading pre-trained model---------------")
    logging.info("On cloud dataset {}, valid single image: {:.2f}".format(cloud_list, cloud_acc))
    logging.info("On edge dataset  {},                         valid single image : {:.2f}".format(edge_list, edge_acc))
    logging.info("On full dataset {}, valid single image : {:.2f}".format(task_list, full_acc))


    content = sio.loadmat('./result/intermediate_output_post_activation.mat')
    for key, value in content.items():
        if re.search('conv', key) or re.search('fc', key):
            print(key, value.shape)
