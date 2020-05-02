import torch
from fixed import *
import numpy as np
import scipy.io as sio
from op import *

class Conv_fixed(object):
    def __init__(self, input_map_size, num_filters, num_channels, kernel_size,
                 pads, FL_W, FL_WM, FL_WG, FL_AO, FL_DI, FL_L_WG,
                 FL_L_WU, FL_M_WU, scale, name, dropout_prob):
        '''
        Implement a simple 2D convolutional layer, stride size = 1, weights 
        are initialized uniformly, biases is initialized as zeros.
        Convolution implemented as matmul inspired by 
        https://github.com/wiseodd/hipsternet
        FL_W: fraction length of weights
        FL_WM: fraction length of weight momentum
        FL_AI: fraction length of input activations
        FL_AO: fraction length of output activations
        FL_DI: fraction length of previous-layer local gradients
        FL_DO: fraction length of local gradients
        FL_L_WG: fraction length of learning rate used in weight gradient scaling
        FL_L_WU: fraction length of learning rate used in weight update
        FL_M_WU: fraction length of momentum factor used in weight update
        scale: scaling factor of learning rate used in weight gradient scaling
               and weight update to make full use of 16-b representation
        '''
        self.input_map_size = input_map_size
        self.num_filters = num_filters
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.pads = pads
        self.dropout_prob = dropout_prob
        self.output_map_size = (int(input_map_size[0] + 2 * pads + 1 - \
            kernel_size), int(input_map_size[1] + 2 * pads + 1 - kernel_size))
        fan_in = num_channels * kernel_size * kernel_size
        fan_out = num_filters * kernel_size * kernel_size
        weight_std = np.sqrt(2. / fan_in)
        
        
        # data = sio.loadmat('Best_epoch_CIFAR10_W_fl.mat')
        # wt = data[name+'_W']
        
        wt = np.random.normal(loc=0.0, scale=weight_std, size=(num_filters, num_channels,kernel_size, kernel_size))
        self.W = fixed(wt, 16, FL_W)
        # self.W = wt
                    
                    
        # self.W= fixed(np.random.uniform(
                  # low = -np.sqrt(6. / (num_channels*input_map_size[0]*input_map_size[0] + num_filters*input_map_size[0]*input_map_size[0])), 
                  # high = np.sqrt(6. / (num_channels*input_map_size[0]*input_map_size[0] + num_filters*input_map_size[0]*input_map_size[0])),
                  # size=(num_filters, num_channels,kernel_size, kernel_size)
                  # ), 16, FL_W)
                    
        self.W_momentum = zeros((num_filters, num_channels,
                          kernel_size, kernel_size), 16, FL_WM)
        self.type = 'Conv'
        self.FL_W = FL_W
        self.FL_WM = FL_WM
        self.FL_WG = FL_WG
        self.FL_AO = FL_AO
        self.FL_DI = FL_DI
        self.FL_L_WG = FL_L_WG
        self.FL_L_WU = FL_L_WU
        self.FL_M_WU = FL_M_WU
        self.scale = scale
        self.name = name

    def feed_forward(self, input,train_or_test):
        '''
        Perform feed forward pass for input (very naive implementation for easy
        use as a reference for RTL implementation)
        input shape: (batch_size, num_channels, input_map_size[0], 
                      input_map_size[1])
        Note: here weights are not flipped before convolution unlike numpy and 
        other implementations.
        '''
        # t_start = time.time()
        self.num_images = input.shape[0]
        # Pad zeros
        if self.pads > 0:
            self.input_padded = zeros((self.num_images, self.num_channels,
                                     input.shape[2] + 2 * self.pads,
                                     input.shape[3] + 2 * self.pads),
                                     WL=input.WL, FL=input.FL)
            self.input_padded[:,:,self.pads : self.pads + input.shape[2], \
                        self.pads : self.pads + input.shape[3]] = input.value
        else:
            self.input_padded = input.copy()
        # Convolution
        self.convolved = conv2D_fixed(self.input_padded, self.W, self.FL_AO)
        #bias = self.b.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        # import pdb;pdb.set_trace()

        
        if(train_or_test==0): # for testing
            p_r = fixed (self.dropout_prob,self.convolved.WL,self.convolved.FL)
            self.drop_convolved = (self.convolved * p_r).round(self.convolved.WL,self.convolved.FL,True) # or self.convolved*self.dropout_prob
        else:
            self.drop_convolved, self.dropout_derivatives = self.convolved.dropout(self.dropout_prob)
            self.dropout_derivatives = self.dropout_derivatives.to(torch.int64)
            
        self.biased = self.drop_convolved
        self.activation_derivatives = (self.biased.value >= 0).to(torch.int64)
        self.output = self.biased
        self.output.value = torch.where(self.output.value > 0, 
                      self.output.value, torch.zeros_like(self.output.value))
        # time_elased = time.time() - t_start
        # print("%s feed forward: %.3f sec" % (self.name, time_elased))
        return self.output

    def feed_backward(self, output_gradients, skip=False):
        # t_start = time.time()
        self.local_gradients = fixed(output_gradients.value * \
                          self.activation_derivatives, 
                     output_gradients.WL, output_gradients.FL)
                     
        self.local_gradients = fixed(self.local_gradients.value * \
                          self.dropout_derivatives, 
                     output_gradients.WL, output_gradients.FL)
        if skip:
            return None

        pads_out = int(self.kernel_size - 1 - self.pads)
        if pads_out > 0:
            local_gradients_padded = zeros((self.num_images,
                                      self.num_filters,
                                      self.output_map_size[0] + 2*pads_out,
                                      self.output_map_size[1] + 2*pads_out),
                                      self.local_gradients.WL,
                                      self.local_gradients.FL)
            local_gradients_padded[:,:,pads_out : pads_out + \
                self.output_map_size[0],pads_out : pads_out + \
                self.output_map_size[1]] = self.local_gradients.value
        else:
            local_gradients_padded = self.local_gradients.copy()
        W_transposed = self.W.copy().transpose(0,1)
        flip_index = torch.arange(start=self.kernel_size-1,
                     end=-1, step=-1, dtype=torch.long, device=self.W.device)
        W_transposed.value = torch.index_select(W_transposed.value, 2, 
                                                flip_index)
        W_transposed.value = torch.index_select(W_transposed.value, 3, 
                                                flip_index)
        self.input_gradients = conv2D_fixed(local_gradients_padded,
            W_transposed, self.FL_DI)
        # time_elased = time.time() - t_start
        # print("%s feed backward: %.3f sec" % (self.name, time_elased))
        return self.input_gradients

    def weight_gradient(self, groups=0):
        # t_start = time.time()
        if groups > 0:
            group_size = int(self.num_images / groups)
            input_list = [fixed(self.input_padded[i*group_size:(i+1)*group_size], 
                self.input_padded.WL, self.input_padded.FL) for i in range(groups)]
            local_gradients_list = [ \
                fixed(self.local_gradients[i*group_size:(i+1)*group_size],
                self.local_gradients.WL, self.local_gradients.FL) \
                for i in range(groups)]
            self.weight_gradients = [conv2D_fixed(input_list[i].transpose(0,1),
                local_gradients_list[i].transpose(0,1), self.FL_WG).transpose(0,1) \
                for i in range(groups)]
        else:
            input_padded_tranposed = self.input_padded.copy().transpose(0,1)
            local_gradients_tranposed = self.local_gradients.copy().transpose(0,1)
            self.weight_gradients = conv2D_fixed(input_padded_tranposed,
                                        local_gradients_tranposed, self.FL_WG)
            self.weight_gradients.transpose(0,1)

    def apply_weight_gradients(self, learning_rate, momentum, 
                            batch_size, last_group):# add mask within this function to enable segmented training
        # ## mask = 0 means these pixels to be frozen; mask = 1 means these pixels will be updated in the future
        learning_rate_scaled = fixed(learning_rate / self.scale / batch_size, 
                                     16, self.FL_L_WG)
        
        if self.num_images == batch_size:
            num_groups = len(self.weight_gradients)
            for i in range(num_groups):
                scaled_WG = (self.weight_gradients[i] * \
                    learning_rate_scaled).round(16, self.FL_WM) # scaled weight gradient
                self.W_momentum += scaled_WG
            last_group = True
        else:
            scaled_WG = (self.weight_gradients * learning_rate_scaled).round(16, 
                     self.FL_WM)
            self.W_momentum += scaled_WG  # momentum updating
        if last_group:
            scale_fp = fixed(self.scale, 16, self.FL_L_WU)
            scaled_WM = (scale_fp * self.W_momentum).round(16, self.FL_W) # momentum
            
            nonzero_grad = np.count_nonzero(scaled_WM.value.cpu().numpy())
            total_params = np.size(scaled_WM.value.cpu().numpy())
            zero_grad = total_params - nonzero_grad
            wtgrad_sparsity = zero_grad*100.0/total_params
            if (wtgrad_sparsity > 95.0):
                print('Warning..!%s has almost zero wt  gradients'%(self.name))

            self.W -= scaled_WM
            # print('NO mask applied')
            momentum_fp = fixed(momentum, 16, self.FL_M_WU)
            self.W_momentum = (self.W_momentum * momentum_fp).round(16, 
                              self.FL_WM)


    def apply_weight_gradients_mask(self, learning_rate, momentum,
                               batch_size, last_group,
                               layer_mask):  # add mask within this function to enable segmented training
        # ## mask = 0 means these pixels to be frozen; mask = 1 means these pixels will be updated in the future
        learning_rate_scaled = fixed(learning_rate / self.scale / batch_size,
                                     16, self.FL_L_WG)

        if self.num_images == batch_size:
            num_groups = len(self.weight_gradients)
            for i in range(num_groups):
                scaled_WG = (self.weight_gradients[i] * learning_rate_scaled).round(16, self.FL_WM)  # scaled weight gradient
                self.W_momentum += scaled_WG
            last_group = True
        else:
            scaled_WG = (self.weight_gradients * learning_rate_scaled).round(16,
                                                                             self.FL_WM)
            self.W_momentum += scaled_WG  # momentum updating
        if last_group:
            scale_fp = fixed(self.scale, 16, self.FL_L_WU)
            scaled_WM = (scale_fp * self.W_momentum).round(16, self.FL_W)  # momentum

            nonzero_grad = np.count_nonzero(scaled_WM.value.cpu().numpy())
            total_params = np.size(scaled_WM.value.cpu().numpy())
            zero_grad = total_params - nonzero_grad
            wtgrad_sparsity = zero_grad * 100.0 / total_params
            if (wtgrad_sparsity > 95.0):
                print('Warning..!%s has almost zero wt  gradients' % (self.name))

            print(layer_mask[-5:-1, 0, :, :])

            layer_mask = fixed(layer_mask, 16, self.FL_L_WU)
            self.W -= np.multiply(scaled_WM, layer_mask)
            print(mask.shape, self.W.shape)
            print(self.W[-5:-1, 0, :, :])
            print('\nMask applied\n')  # weight update function


            momentum_fp = fixed(momentum, 16, self.FL_M_WU)
            self.W_momentum = (self.W_momentum * momentum_fp).round(16, self.FL_WM)



class FC_fixed(object):
    def __init__(self, name, input_dim, num_units, FL_W, FL_WG,
                    FL_AO, FL_DI, FL_WM, FL_L_WG, FL_L_WU, FL_M_WU,
                    scale, relu=True, dropout_prob=1):
        '''
        Implement a simple fully connected layer, weights are initialized 
        uniformly, biases is initialized as zeros.
        '''
        self.input_dim = input_dim
        self.num_units = num_units
        weight_bound = np.sqrt(6. / (input_dim + num_units))
        
        # data = sio.loadmat('Best_epoch_CIFAR10_W_fl.mat')
        # wt = data[name+'_W']
        # print ('initializing weights...')
        wt = np.random.uniform(low=-weight_bound,high=weight_bound,size=(input_dim, num_units))
        
        self.W = fixed(wt, 16, FL_W)
        
        self.W_momentum = zeros((input_dim, num_units), 16, FL_WM)
        self.type = 'FC'
        self.FL_W = FL_W
        self.FL_WG = FL_WG
        self.FL_AO = FL_AO
        self.FL_DI = FL_DI
        self.FL_WM = FL_WM
        self.FL_L_WG = FL_L_WG
        self.FL_L_WU = FL_L_WU
        self.FL_M_WU = FL_M_WU
        self.scale = scale
        self.relu = relu
        self.name = name
        self.dropout_prob = dropout_prob
         
    def feed_forward(self, input, train_or_test):
        '''
        Perform feed forward pass for input (very naive implementation for easy
        use as a reference for RTL implementation)
        input shape: (batch_size, input_dim)
        '''
        # t_start = time.time()
        self.input = input
        self.num_images = input.shape[0]
        self.multiplied = matmul_fixed(input, self.W, self.FL_AO)
        self.biased = self.multiplied
        self.output = self.biased
        # if (train_or_test==1):
            # self.drop_output, self.dropout_derivatives = self.output.dropout(self.dropout_prob)
        # else:
            # p_r = fixed (self.dropout_prob,self.output.WL,self.output.FL)
            # self.drop_output = (self.output * p_r).round(self.output.WL,self.output.FL,True)
        self.drop_output = self.output
        if self.relu:
            self.activation_derivatives = (self.drop_output >= 0)
            self.drop_output.value = torch.where(self.drop_output.value > 0, 
                   self.drop_output.value, torch.zeros_like(self.drop_output.value))
        # time_elased = time.time() - t_start
        # print("%s feed forward: %.3f sec" % (self.name, time_elased))
        return self.drop_output
    
    def feed_backward(self, output_gradients):
        # t_start = time.time()
        # self.local_gradients = fixed(output_gradients.value * \
                 # self.dropout_derivatives, output_gradients.WL,
                 # output_gradients.FL)
        if self.relu:
            self.local_gradients = fixed(output_gradients.value * \
                 self.activation_derivatives, output_gradients.WL,
                 output_gradients.FL)
        else:
            self.local_gradients = output_gradients
        self.input_gradients = matmul_fixed(self.local_gradients,
                    self.W.copy().transpose(0,1), self.FL_DI)
        # time_elased = time.time() - t_start
        # print("%s feed backward: %.3f sec" % (self.name, time_elased))
        return self.input_gradients
 
    def weight_gradient(self, groups=0):
        # t_start = time.time()
        if groups > 0:
            group_size = int(self.num_images / groups)
            input_list = [fixed(self.input[i*group_size:(i+1)*group_size], 
                self.input.WL, self.input.FL) for i in range(groups)]
            local_gradients_list = [ \
                fixed(self.local_gradients[i*group_size:(i+1)*group_size],
                self.local_gradients.WL, self.local_gradients.FL) \
                for i in range(groups)]
            self.weight_gradients = [matmul_fixed(input_list[i].transpose(0,1),
                local_gradients_list[i], self.FL_WG) for i in range(groups)]
        else:
            self.weight_gradients = matmul_fixed(self.input.copy().transpose(0,1),
                         self.local_gradients, self.FL_WG)

    def apply_weight_gradients(self, learning_rate, momentum,
                            batch_size, last_group, layer_mask = None):
        learning_rate_scaled = fixed(learning_rate / self.scale / batch_size,
                                     16, self.FL_L_WG)
        if batch_size == self.num_images:
            num_groups = len(self.weight_gradients)
            scaled_WG = [(self.weight_gradients[i] * learning_rate_scaled).round(16,
                         self.FL_WM) for i in range(num_groups)]
            for i in range(num_groups):
                self.W_momentum += scaled_WG[i]
            last_group = True
        else:
            scaled_WG = (self.weight_gradients * learning_rate_scaled).round(16,
                         self.FL_WM)
            self.W_momentum += scaled_WG
        if last_group:
            scale_fp = fixed(self.scale, 16, self.FL_L_WU)
            scaled_WM = (scale_fp * self.W_momentum).round(16, self.FL_W)
            
            nonzero_grad = np.count_nonzero(scaled_WM.value.cpu().numpy())
            total_params = np.size(scaled_WM.value.cpu().numpy())
            zero_grad = total_params - nonzero_grad
            wtgrad_sparsity = zero_grad*100.0/total_params
            
            if (wtgrad_sparsity > 95.0):
                print('WARNING..!%s has almost zero wt  gradients'%(self.name))

            self.W -= scaled_WM
            # print('NO mask applied')
            momentum_fp = fixed(momentum, 16, self.self.FL_M_WU)
            self.W_momentum = (self.W_momentum * momentum_fp).round(16,
                              self.FL_WM)


    def apply_weight_gradients_mask(self, learning_rate, momentum,
                               batch_size, last_group, layer_mask):
        learning_rate_scaled = fixed(learning_rate / self.scale / batch_size,
                                     16, self.FL_L_WG)
        if batch_size == self.num_images:
            num_groups = len(self.weight_gradients)
            scaled_WG = [(self.weight_gradients[i] * learning_rate_scaled).round(16,
                                                                                 self.FL_WM) for i in range(num_groups)]
            for i in range(num_groups):
                self.W_momentum += scaled_WG[i]
            last_group = True
        else:
            scaled_WG = (self.weight_gradients * learning_rate_scaled).round(16,
                                                                             self.FL_WM)
            self.W_momentum += scaled_WG
        if last_group:
            scale_fp = fixed(self.scale, 16, self.FL_L_WU)
            scaled_WM = (scale_fp * self.W_momentum).round(16, self.FL_W)

            nonzero_grad = np.count_nonzero(scaled_WM.value.cpu().numpy())
            total_params = np.size(scaled_WM.value.cpu().numpy())
            zero_grad = total_params - nonzero_grad
            wtgrad_sparsity = zero_grad * 100.0 / total_params

            if (wtgrad_sparsity > 95.0):
                print('WARNING..!%s has almost zero wt  gradients' % (self.name))

            layer_mask = fixed(layer_mask, 16, self.self.FL_M_WU)

            self.W -= np.multiply(scaled_WM, layer_mask)
            print(mask.shape, self.W.shape)
            print(self.W[-5:-1, 0, :, :])
            print('Mask applied')  # weight update function

            momentum_fp = fixed(momentum, 16, self.FL_M_WU)
            self.W_momentum = (self.W_momentum * momentum_fp).round(16,
                                                                    self.FL_WM)
class Flatten(object):
    def __init__(self, name, input_map_size, num_channels):
        self.input_map_size = input_map_size
        self.num_channels = num_channels
        self.num_units = input_map_size[0] * input_map_size[1] * num_channels
        self.type = 'Flatten'
        self.name = name

    def feed_forward(self, input,train_or_test):
        self.input = input
        self.num_images = input.shape[0]
        self.output = input.copy().reshape(self.num_images, self.num_units)
        return self.output

    def feed_backward(self, output_gradients):
        self.local_gradients = output_gradients
        self.input_gradients = output_gradients.copy().reshape( \
             self.num_images, self.num_channels, self.input_map_size[0], \
             self.input_map_size[1])
        return self.input_gradients

    def weight_gradient(self, groups=0):
        pass

    def apply_weight_gradients(self, learning_rate=1.0, momentum=0.5, 
                               batch_size=100, last_group=False):
        pass

class MaxPooling(object):
    def __init__(self, name, input_map_size, num_channels):
        '''
        Here only 2x2 max pooling is implemented, may not be the fastest,
        but try to mimic hardware implementation
        '''
        self.input_map_size = np.int32(input_map_size)
        self.num_channels = int(num_channels)
        self.output_map_size = (self.input_map_size/2).astype('int32')
        self.type = 'MaxPooling'
        self.name = name

    def feed_forward(self, input,train_or_test):
        self.input = input
        self.num_images = input.shape[0]
        self.num_values = int(self.num_images * self.num_channels * \
                self.output_map_size[0] * self.output_map_size[1])
        num_values = self.num_values
        input_00 = torch.reshape(self.input[:,:,0::2,0::2], (num_values,))
        input_01 = torch.reshape(self.input[:,:,0::2,1::2], (num_values,))
        input_10 = torch.reshape(self.input[:,:,1::2,0::2], (num_values,))
        input_11 = torch.reshape(self.input[:,:,1::2,1::2], (num_values,))
        input_reshaped = torch.stack((input_00, input_01, input_10, \
                        input_11), dim=0)
        max_pooling_index_reshaped = torch.argmax(input_reshaped, dim=0)
        self.max_pooling_index = torch.reshape(max_pooling_index_reshaped,
          (self.num_images, self.num_channels, self.output_map_size[0],
          self.output_map_size[1]))
        self.max_pooling_index_reshaped = max_pooling_index_reshaped
        output_reshaped, index = torch.max(input_reshaped, dim=0)
        self.output = fixed(torch.reshape(output_reshaped, (self.num_images,
          self.num_channels, self.output_map_size[0],
          self.output_map_size[1])), input.WL, input.FL)
        return self.output

    def feed_backward(self, output_gradients):
        self.local_gradients = output_gradients
        input_gradients_reshaped = torch.zeros(4, self.num_values, \
                             dtype=torch.int64, device=self.input.device)
        input_gradients_reshaped[self.max_pooling_index_reshaped, \
            np.arange(self.num_values)] = \
            self.local_gradients.copy().reshape(self.num_values).value
        self.input_gradients = zeros(self.input.shape, output_gradients.WL,
                                     output_gradients.FL)
        self.input_gradients[:,:,0::2,0::2] = \
            torch.reshape(input_gradients_reshaped[0,:], (self.num_images,
            self.num_channels, self.output_map_size[0],\
             self.output_map_size[1]))
        self.input_gradients[:,:,0::2,1::2] = \
            torch.reshape(input_gradients_reshaped[1,:], (self.num_images,
            self.num_channels, self.output_map_size[0],\
             self.output_map_size[1]))
        self.input_gradients[:,:,1::2,0::2] = \
            torch.reshape(input_gradients_reshaped[2,:], (self.num_images,
            self.num_channels, self.output_map_size[0],\
             self.output_map_size[1]))
        self.input_gradients[:,:,1::2,1::2] = \
            torch.reshape(input_gradients_reshaped[3,:], (self.num_images,
            self.num_channels, self.output_map_size[0],\
             self.output_map_size[1]))
        return self.input_gradients

    def weight_gradient(self, groups=0):
        pass

    def apply_weight_gradients(self, learning_rate=1.0, momentum=0.5, 
                                    batch_size=100, last_group=False):
        pass

class SquareHingeLoss(object):
    def __init__(self, name, num_classes):
        self.num_classes = int(num_classes)
        self.type = 'Loss'
        self.name = name

    def feed_forward(self, logits, labels=None):
        '''
        Make predictions, evaluate loss if labels provided
        logits: (num_images, num_classes), float32
        labels: (num_images,), int32
        '''
        self.num_images = logits.shape[0]
        self.logits = logits
        self.predictions = torch.argmax(logits.value, dim=1)
        if labels is None:
            return self.predictions
        self.labels = fixed((2*(torch.eye(self.num_classes,
            dtype=torch.int64).index_select(0, 
            labels.type(torch.long)))-1)*(2**logits.FL), 
            logits.WL, logits.FL)
        rough_loss = (self.logits.get_real_torch() - 
               self.labels.get_real_torch()) ** 2 / 2.
        self.conditions = self.logits.get_real_torch() * \
               self.labels.get_real_torch()
        rough_loss = torch.where(self.conditions > 1.,
            torch.zeros_like(rough_loss), rough_loss)
        self.loss = torch.mean(rough_loss)

        return self.predictions, self.loss

    def feed_backward(self):
        '''
        Evaluate gradient of loss with respect to logits
        '''
        rough_gradients = (self.logits - self.labels)
        rough_gradients.value = torch.where(self.conditions >= 1.,
                          torch.zeros_like(rough_gradients.value),
                            rough_gradients.value)
        self.input_gradients = rough_gradients
        return self.input_gradients

    def weight_gradient(self, groups=0):
        pass

    def apply_weight_gradients(self, learning_rate=1.0, momentum=0.5,
                               batch_size=100, last_group=False):
        pass
