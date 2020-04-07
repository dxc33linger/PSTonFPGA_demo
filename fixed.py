import torch
import numpy as np

class fixed(object):
    '''
    signed fixed-point object (hardware emulation)  
    '''
    def __init__(self, value_np, WL=16, FL=8, device=torch.device("cuda:0")):
        '''
        Construction function
        value_np: value in numpy format
        WL: word length
        FL: fraction length
        device: torch device
        '''
        IL = WL - FL
        self.WL = WL
        self.FL = FL
        self.IL = IL
        self.device = device
        if isinstance(value_np, torch.Tensor) and value_np.dtype == torch.int64:
            self.value = value_np.to(device)
        else:
            upper_bound = 2**(WL-1) - 1
            lower_bound = -2**(WL-1)
            value_int = np.clip(np.round(np.atleast_1d(value_np) * 2.**FL), 
                                lower_bound, upper_bound)
            self.value = torch.from_numpy(value_int).to(device).to(torch.int64)
        self.shape = self.value.shape

    def round(self, WL_new, FL_new, clip=False):
        '''
        Mimic hardware rounding
        '''
        DISC_L = self.FL - FL_new
        value_new = torch.__rshift__(self.value, DISC_L)
        carry_in = torch.__rshift__(torch.__and__(self.value, 2**(DISC_L - 1)),
                                     DISC_L - 1)
        value_new = value_new + carry_in
        if clip:
            value_new.clamp_(-2**(WL_new-1), 2**(WL_new-1)-1)
        self.value = torch.__rshift__(torch.__lshift__(torch.__and__(value_new, 
                                     (2**WL_new - 1)), 64-WL_new), 64-WL_new)
        self.FL = FL_new
        self.WL = WL_new
        return self
  
    def clip(self, clip=False):
        if clip:
            self.value.clamp(-2**(self.WL-1), 2**(self.WL-1)-1)
        else:
            self.value = torch.__rshift__(torch.__lshift__(torch.__and__(self.value, 
                         (2**self.WL - 1)), 64-self.WL), 64-self.WL)
        return self

    def zeros_like(self):
        value = torch.zeros(self.shape, dtype=torch.int64)
        return fixed(value, self.WL, self.FL)

    def copy(self):
        return fixed(self.value, self.WL, self.FL)

    def reshape(self, *shape):
        self.value = self.value.reshape(shape)
        self.shape = self.value.shape
        return self
 
    def permute(self, *dims):
        self.value = self.value.permute(dims)
        self.shape = self.value.shape
        return self
    
    def transpose(self, dim0, dim1):
        self.value = self.value.transpose(dim0, dim1)
        self.shape = self.value.shape
        return self    
        
    def dropout(self, prob):
        # import pdb;pdb.set_trace()
        # mask = torch.distributions.Binomail(1,probs=prob).sample(self.value.shape)
        mask = torch.distributions.Binomial(1,probs=prob).sample(self.value.shape)
        # import pdb;pdb.set_trace()
        mask = mask.type(torch.cuda.LongTensor)
        self.value = self.value * mask
        return self, mask
 
    def __getitem__(self, indices):
        return self.value[indices]

    def __setitem__(self, indices, values):
        self.value[indices] = values

    def __mul__(self, other):
        prod_value = self.value * other.value
        return fixed(prod_value, self.WL+other.WL, self.FL+other.FL)

    def __add__(self, other):
        assert self.FL == other.FL, "fraction bits should match!"
        sum_value = self.value + other.value
        if self.WL == other.WL:
            WL_sum = self.WL + 1
        else:
            WL_sum = max(self.WL, other.WL)
        return fixed(sum_value, WL_sum, self.FL)
    
    def __sub__(self, other):
        assert self.FL == other.FL, "fraction bits should match!"
        sum_value = self.value - other.value
        return fixed(sum_value, self.WL, self.FL).clip()
    
    def __iadd__(self, other):
        assert self.FL == other.FL, "fraction bits should match!"
        sum_value = self.value + other.value
        return fixed(sum_value, self.WL, self.FL).clip() 

    def __isub__(self, other):
        assert self.FL == other.FL, "fraction bits should match!"
        sum_value = self.value - other.value
        return fixed(sum_value, self.WL, self.FL).clip()

    def __repr__(self):
        return "Fixed-point array:\n" + str(self.value) + ", \n%d bits with %d fraction bits" % (self.WL, self.FL)

    def get_real_number(self):
        value_np = np.squeeze(self.value.cpu().numpy() / 2.**self.FL)
        return value_np
    def get_real_torch(self):
        value_torch = self.value.to(torch.float64) / 2.**self.FL
        return value_torch

def zeros(shape, WL=16, FL=16):
    return fixed(torch.zeros(shape, dtype=torch.int64), WL, FL)
