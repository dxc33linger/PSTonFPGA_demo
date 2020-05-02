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

# Description: CNN model
# Created on 07/14/2018
# Modofied on 09/26/2018

from layers import *
import scipy.io as sio
import torch
import re


class CNN(object):
    def __init__(self):
        print("Building CNN model")
        self.layers = []
        self.num_layers = 0
        
    def append_layer(self, type, *args, **kwargs):
        # if type == 'Conv':
        #     self.layers.append(Conv(*args, **kwargs))
        # if type == 'FC':
        #     self.layers.append(FC(*args, **kwargs))
        if type == 'Conv_fixed':
            self.layers.append(Conv_fixed(*args, **kwargs))
        if type == 'FC_fixed':
            self.layers.append(FC_fixed(*args, **kwargs))
        if type == 'MaxPooling':
            self.layers.append(MaxPooling(*args, **kwargs))
        if type == 'Flatten':
            self.layers.append(Flatten(*args, **kwargs))
        if type == 'SquareHingeLoss':
            self.layers.append(SquareHingeLoss(*args, **kwargs))
        self.num_layers += 1
        
    def feed_forward(self, input, labels=None, train_or_test=1, record = False):
        self.logits = input
        if record: output_dict = dict()

        for i in range(self.num_layers - 1):
            self.logits = self.layers[i].feed_forward(self.logits,train_or_test)
            if record:
                output_dict[self.layers[i].name] = self.layers[i].output.value.cpu().numpy()
                print(self.layers[i].name)
        if record:
            sio.savemat('./result/intermediate_output_post_activation.mat', output_dict)
        if labels is None:
            self.predictions = self.layers[-1].feed_forward(self.logits)
            return self.predictions
        else:
            self.predictions, self.loss = \
                self.layers[-1].feed_forward(self.logits, labels)

            return self.predictions, self.loss
    
    def feed_backward(self):
        self.input_gradients = self.layers[-1].feed_backward()
        for i in range(self.num_layers - 2):
            self.input_gradients = \
         self.layers[self.num_layers-2-i].feed_backward(self.input_gradients)
        self.layers[0].feed_backward(self.input_gradients, skip=True)
    def weight_gradient(self, groups=0):
        for i in range(self.num_layers):
            self.input_gradients = \
         self.layers[self.num_layers-1-i].weight_gradient(groups)
         
    def apply_weight_gradients(self, learning_rate=1.0, momentum=0.5, batch_size=100,
           last_group=False, mask=None):
        if mask:
            for i in range(self.num_layers):
                layer_mask = None
                layer_name = self.layers[self.num_layers-1-i].name + '_W'
                if re.search('conv', layer_name) or re.search('fc', layer_name):
                    layer_mask = mask[layer_name]
                    self.input_gradients = self.layers[self.num_layers-1-i].apply_weight_gradients_mask(learning_rate,
                                                                        momentum, batch_size, last_group, layer_mask)
                else: # non-conv non-FC layers
                    self.input_gradients = self.layers[self.num_layers - 1 - i].apply_weight_gradients(learning_rate,
                                                                                    momentum, batch_size, last_group)
        else: #regular training
            for i in range(self.num_layers):
                self.input_gradients = self.layers[self.num_layers-1-i].apply_weight_gradients(learning_rate,
                                                                        momentum, batch_size, last_group)
    def get_params(self):
        params = dict()
        for i in range(self.num_layers):
            if self.layers[i].type == 'Conv' or self.layers[i].type == 'FC':
                params[self.layers[i].name + '_W'] = self.layers[i].W
                params[self.layers[i].name + '_Wm'] = self.layers[i].W_momentum
        return params
    def save_params_mat(self, path):
        weights = dict()
        for i in range(self.num_layers):
            if self.layers[i].type == 'Conv' or self.layers[i].type == 'FC':
                weights[self.layers[i].name + '_W'] = self.layers[i].W.value.cpu().numpy().astype('int16')
        sio.savemat(path, weights)
    def load_params_mat(self, path):
        weights = sio.loadmat(path)
        for i in range(self.num_layers):
            if self.layers[i].type == 'Conv' or self.layers[i].type == 'FC':
                self.layers[i].W.value = torch.from_numpy(weights[self.layers[i].name + '_W']).to(self.layers[i].W.device).to(torch.int64)
    def get_activations(self):
        activations = dict()
        for i in range(self.num_layers):
            if self.layers[i].type == 'Conv' or self.layers[i].type == 'FC' or self.layers[i].type == 'Flatten' or self.layers[i].type == 'MaxPooling':
                activations[self.layers[i].name + '_act'] = self.layers[i].output.cpu().numpy()
        return activations
    def get_local_gradients(self):
        local_gradients = dict()
        for i in range(self.num_layers):
            if self.layers[i].type == 'Conv' or self.layers[i].type == 'FC' or self.layers[i].type == 'Flatten' or self.layers[i].type == 'MaxPooling':
                local_gradients[self.layers[i].name + '_lg'] = self.layers[i].local_gradients.cpu().numpy()
        return local_gradients      
