# Copyright 2018    Shihui Yin    Arizona State University

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
# WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
# MERCHANTABLITY OR NON-INFRINGEMENT.
# See the Apache 2 License for the specific language governing permissions and
# limitations under the License.

# Description: Train a LeNet-like CNN for CIFAR10
# Created on 04/10/2018
# Modified on 05/16/2018, for CIFAR10 training
# Modified on 08/12/2018, scale initialized weights to normalize activations
# Modified on 08/13/2018, for MNIST training
# Modified on 09/26/2018, more hardware-accurate modeling
# Modified on 10/25/2018, faster execution by using performing feed_forward and
#                         feed_backward for the whole mini-batch, not group by 
#                         group
# Modified on 10/29/2018, add data augmentation
from models import CNN
import torch
import numpy as np
import argparse
import scipy.io as sio
import time
from ast import literal_eval as bool
from fixed import *
dtype = torch.int64
device = torch.device("cuda")
np.random.seed(1234)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CNN for MNIST",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-ls', '--LR_start', dest='LR_start', default=0.002, 
        type=float, help="Initial learning rate.")
    parser.add_argument('-lf', '--LR_finish', dest='LR_finish', default=0.002, 
        type=float, help="Ending learning rate.")
    parser.add_argument('-mo', '--momentum', dest='momentum', default=0.5, 
        type=float, help="Momentum factor for learning")
    parser.add_argument('-bs', '--batch_size', dest='batch_size', 
        default=40, type=int, help="Batch size for training")
    parser.add_argument('-gs', '--group_size', dest='group_size',
        default=4, type=int, help="Group size that can be fit in on-chip memory")
    parser.add_argument('-ne', '--num_epochs', dest='num_epochs', 
        default=50, type=int, help="Number of epochs for training")
    parser.add_argument('-da', '--data_augmentation', dest='data_augmentation', default=True,
        type=bool, help="Data augmentation by flipping and cropping")
    parser.add_argument('-vb', '--verbose', dest='verbose', default=False,
        type=bool, help="Display range information if true")
    args = parser.parse_args()
    LR_decay = (args.LR_finish / args.LR_start) ** (1./(args.num_epochs-1))
    cnn = CNN()
    
    FL_A_input = 15
    FL_W_conv_0 = 15
    FL_W_conv_1 = 15
    FL_W_conv_2 = 15
    FL_W_conv_3 = 16
    FL_W_conv_4 = 16
    FL_W_conv_5 = 16
    FL_W_fc = 16
    FL_WM_conv_0 = 19
    FL_WM_conv_1 = 19
    FL_WM_conv_2 = 19
    FL_WM_conv_3 = 20
    FL_WM_conv_4 = 20
    FL_WM_conv_5 = 20
    FL_WM_fc = 20 
    FL_A_conv_0 = FL_A_input+FL_W_conv_0-16
    FL_A_conv_1 = FL_A_conv_0+FL_W_conv_1-16
    FL_A_conv_2 = FL_A_conv_1+FL_W_conv_2-16
    FL_A_conv_3 = FL_A_conv_2+FL_W_conv_3-16
    FL_A_conv_4 = FL_A_conv_3+FL_W_conv_4-16
    FL_A_conv_5 = FL_A_conv_4+FL_W_conv_5-16
    FL_A_fc = FL_A_conv_5+FL_W_fc-16
    FL_D_fc = FL_A_fc
    FL_D_conv_5 = FL_D_fc+FL_W_fc-16
    FL_D_conv_4 = FL_D_conv_5+FL_W_conv_5-16
    FL_D_conv_3 = FL_D_conv_4+FL_W_conv_4-16
    FL_D_conv_2 = FL_D_conv_3+FL_W_conv_3-16
    FL_D_conv_1 = FL_D_conv_2+FL_W_conv_2-16
    FL_D_conv_0 = FL_D_conv_1+FL_W_conv_1-16
    FL_D_input = FL_D_conv_0+FL_W_conv_0-16
    FL_WG_conv_0 = FL_A_input+FL_D_conv_0-16
    FL_WG_conv_1 = FL_A_conv_0+FL_D_conv_1-16
    FL_WG_conv_2 = FL_A_conv_1+FL_D_conv_2-16
    FL_WG_conv_3 = FL_A_conv_2+FL_D_conv_3-16
    FL_WG_conv_4 = FL_A_conv_3+FL_D_conv_4-16
    FL_WG_conv_5 = FL_A_conv_4+FL_D_conv_5-16
    FL_WG_fc = FL_A_conv_5+FL_D_fc-16
    
    FL_L_WG_conv_0 = 16+FL_WM_conv_0-FL_WG_conv_0
    FL_L_WG_conv_1 = 16+FL_WM_conv_1-FL_WG_conv_1
    FL_L_WG_conv_2 = 16+FL_WM_conv_2-FL_WG_conv_2
    FL_L_WG_conv_3 = 16+FL_WM_conv_3-FL_WG_conv_3
    FL_L_WG_conv_4 = 16+FL_WM_conv_4-FL_WG_conv_4
    FL_L_WG_conv_5 = 16+FL_WM_conv_5-FL_WG_conv_5
    FL_L_WG_fc = 16+FL_WM_fc-FL_WG_fc
    FL_L_WU_conv_0 = 16+FL_W_conv_0-FL_WM_conv_0
    FL_L_WU_conv_1 = 16+FL_W_conv_1-FL_WM_conv_1
    FL_L_WU_conv_2 = 16+FL_W_conv_2-FL_WM_conv_2
    FL_L_WU_conv_3 = 16+FL_W_conv_3-FL_WM_conv_3
    FL_L_WU_conv_4 = 16+FL_W_conv_4-FL_WM_conv_4
    FL_L_WU_conv_5 = 16+FL_W_conv_5-FL_WM_conv_5
    FL_L_WU_fc = 16+FL_W_fc-FL_WM_fc
    
    FL_M_WU_conv_0 = 16
    FL_M_WU_conv_1 = 16
    FL_M_WU_conv_2 = 16
    FL_M_WU_conv_3 = 16
    FL_M_WU_conv_4 = 16
    FL_M_WU_conv_5 = 16
    FL_M_WU_fc = 16
    scale = 2
     
    
    # Load and preprocess CIFAR10 dataset
    data = sio.loadmat('../CIFAR10.mat')
    train_X = data['train_X']
    valid_X = data['valid_X']
    test_X = data['test_X']
    train_y = np.argmax(data['train_y'], axis=1)
    valid_y = np.argmax(data['valid_y'], axis=1)
    test_y = np.argmax(data['test_y'], axis=1)
   
    train_X = fixed(train_X, 16, FL_A_input)
    valid_X = fixed(valid_X, 16, FL_A_input)
    test_X = fixed(test_X, 16, FL_A_input)    
    train_y = torch.from_numpy(train_y).to(torch.int64)
    valid_y = torch.from_numpy(valid_y).to(torch.int64)
    test_y = torch.from_numpy(test_y).to(torch.int64)
    # import pdb; pdb.set_trace()
    # Build CNN for CIFAR10
    cnn.append_layer('Conv_fixed', 
                name='conv_0',
                input_map_size=(32,32), 
                num_filters=16, 
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
                scale=scale
                )
    cnn.append_layer('Conv_fixed', 
                name='conv_1',
                input_map_size=(32,32), 
                num_filters=16, 
                num_channels=16, 
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
                scale=scale
                )
    cnn.append_layer('MaxPooling',
                name='pool_0',
                input_map_size=(32,32),
                num_channels=16)
    cnn.append_layer('Conv_fixed', 
                name='conv_2',
                input_map_size=(16,16), 
                num_filters=32, 
                num_channels=16, 
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
                scale=scale
                )
    cnn.append_layer('Conv_fixed', 
                name='conv_3',
                input_map_size=(16,16), 
                num_filters=32, 
                num_channels=32, 
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
                scale=scale
                )
    cnn.append_layer('MaxPooling',
                name='pool_1',
                input_map_size=(16,16),
                num_channels=32)
    cnn.append_layer('Conv_fixed', 
                name='conv_4',
                input_map_size=(8,8), 
                num_filters=64, 
                num_channels=32, 
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
                scale=scale
                )
    cnn.append_layer('Conv_fixed', 
                name='conv_5',
                input_map_size=(8,8), 
                num_filters=64, 
                num_channels=64, 
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
                scale=scale
                )
    cnn.append_layer('MaxPooling',
                name='pool_2',
                input_map_size=(8,8),
                num_channels=64)
    cnn.append_layer('Flatten',
                name='flatten',
                input_map_size=(4,4),
                num_channels=64)
    cnn.append_layer('FC_fixed',
                name='fc',
                input_dim=1024,
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
                scale=scale
                )
    cnn.append_layer('SquareHingeLoss',
                name='Loss',
                num_classes=10)
    # Training
    print("Training")
    batch_size = args.batch_size
    group_size = args.group_size
    num_batches = int(45000 / batch_size)
    num_groups = int(batch_size / group_size)
    Learning_Rate = args.LR_start
    for i in range(args.num_epochs):
        # shuffle training dataset
        start_time = time.time()
        IX = np.random.permutation(np.arange(45000))
        train_X_shuffled = train_X[IX,:]
        train_y_shuffled = train_y[IX]
        wrong_predictions = 0
        train_loss = 0
        for j in range(num_batches):
            #print("Epoch %d (%d/%d)" % (i+1, j+1, num_batches))
            train_X_mb = train_X_shuffled[j*batch_size:(j+1)*batch_size]
            if args.data_augmentation:
                IX_flipped = np.random.choice(batch_size, int(batch_size/2))
                train_X_mb[IX_flipped] = torch.flip(train_X_mb[IX_flipped], [2])
                train_X_mb_padded = torch.zeros(batch_size, 3, 40, 40)
                train_X_mb_padded[:,:,4:36,4:36] = train_X_mb
                off_x = np.random.randint(9, size=batch_size)
                off_y = np.random.randint(9, size=batch_size)
                for k in range(batch_size):
                    train_X_mb[k] = train_X_mb_padded[k,:,off_x[k]:off_x[k]+32,
                        off_y[k]:off_y[k]+32]

            train_X_mb = fixed(train_X_mb, 16, FL_A_input)
            train_y_mb = train_y_shuffled[j*batch_size:(j+1)*batch_size]
            predictions, loss = cnn.feed_forward(train_X_mb, train_y_mb)
            cnn.feed_backward()
            cnn.weight_gradient(num_groups)
	    cnn.apply_weight_gradients(Learning_Rate, args.momentum, batch_size, True)
            wrong_predictions += torch.sum(predictions.cpu() != train_y_mb).numpy()
            train_loss += loss
            #print("Loss: %.4f" % cnn.loss)

        elapsed_time = time.time() - start_time
        print("Epoch %d takes %.2f seconds" % (i, elapsed_time))
        train_error = wrong_predictions / 45000.
        train_loss /= (num_batches*num_groups)
       
        batch_size_valid = 40
        num_batches_valid = int(5000./batch_size_valid)
        valid_error = 0.
        valid_loss = 0.
        for j in range(num_batches_valid):
            predictions, valid_loss_batch = cnn.feed_forward(fixed(valid_X[j*batch_size_valid:(j+1)*batch_size_valid], 
                 16, FL_A_input), valid_y[j*batch_size_valid:(j+1)*batch_size_valid])
            valid_error += torch.sum(predictions.cpu() != valid_y[j*batch_size_valid:(j+1)*batch_size_valid]).numpy()
            valid_loss += valid_loss_batch
        valid_error /= 5000.
        valid_loss /= num_batches_valid
        best_acc = 0.0
        best_epoch = 0.0
        if (100-(valid_error * 100)) > best_acc:
            best_acc = (100-(valid_error * 100))
            best_epoch = i+1
            cnn.save_params_mat('best_epoch_CIFAR10_W.mat')
        print("Epoch %d: " % (i+1))
        print("    Learning_Rate: %.3e" % Learning_Rate)
        print("    train accuracy: %.2f%%" % (100-(train_error * 100)))
        print("    train loss: %.4f" % train_loss)
        print("    valid accuracy: %.2f%%" % (100-(valid_error * 100)))
        print("    valid loss: %.4f" % valid_loss)
        print("    best accuracy: %.2f" % best_acc)
        Learning_Rate *= LR_decay
