import scipy.io as sio
import re

W_pre = sio.loadmat('./result/result_9classes/Best_epoch_CIFAR10_W.mat')  #
W_post = sio.loadmat('./result/result_9classes/incremental_1class_Best_epoch_CIFAR10_W.mat')  #
mask_idx = sio.loadmat(  './result/result_9classes/active_index_CIFAR10_TaskDivision_9,1.mat')

for key, value in W_pre.items():
    if re.search('conv_', key):
        print('\n{}, active idx {}'.format(key, mask_idx[key]))
        print('----pre-----')
        print(W_pre[key][mask_idx[key][0][0]:mask_idx[key][0][0]+3, 0, :, :])
        print('----post-----')
        print(W_post[key][mask_idx[key][0][0]:mask_idx[key][0][0]+3, 0, :, :])
    elif re.search('fc_', key):
        print('\n{}, active idx {}'.format(key, mask_idx[key]))
        print('----pre-----')
        print(W_pre[key][mask_idx[key][0][0]:mask_idx[key][0][0]+3,  :])
        print('----post-----')
        print(W_post[key][mask_idx[key][0][0]:mask_idx[key][0][0]+3, :])