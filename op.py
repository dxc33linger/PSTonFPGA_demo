from fixed import *
import torch
import itertools

def conv2D_fixed(I, W, FL_O, WL_O=16):  # activation
    '''
    Mimic the 16-b MAC array hardware implementation
    I: Q(16-FL_I, FL_I)
    W: Q(16-FL_W, FL_W)
    P: Q(32-FL_I, FL_W, FL_I+FL_W-16)
    O(before rounding): Q(38-FL_I-FL_W, FL_I+FL_W-16)
    O(after rounding): Q(16-FL_O, FL_O)
    '''
    NB, NI, XI, YI = I.shape
    NO, NI, XW, YW = W.shape
    XO = XI + 1 - XW
    YO = YI + 1 - YW
    #FL_P = I.FL+W.FL-16
    FL_P = I.FL+W.FL
    IV = I.value
    WV = W.value
    IV = IV.transpose(0,1)
    I_reshaped = torch.empty(XW*YW, NI, NB, XO, YO, dtype=torch.float32, device=I.device)
    for i, (xw,yw) in enumerate(itertools.product(xrange(XW),xrange(YW))):
        I_reshaped[i,:,:,:,:] = IV[:,:,xw:xw+XO,yw:yw+YO]
    I_reshaped = I_reshaped.transpose(0,1).reshape(NI*XW*YW,NB*XO*YO)
    W_reshaped = WV.reshape(NO, NI*XW*YW).t()
    #P = fixed(torch.bmm(I_reshaped.unsqueeze(2).to(torch.float64), W_reshaped.unsqueeze(1).to(torch.float64)).to(torch.int64), I.WL+W.WL, I.FL+W.FL).round(16, FL_P)
    P = fixed(torch.bmm(I_reshaped.unsqueeze(2).to(torch.float64), W_reshaped.unsqueeze(1).to(torch.float64)).to(torch.int64), I.WL+W.WL, I.FL+W.FL)
    #O = fixed(torch.sum(P.value,0), 22, FL_P).round(WL_O, FL_O, True)
    O = fixed(torch.sum(P.value,0), 32, FL_P).round(WL_O, FL_O, True)
    
    return O.reshape(NB, XO, YO, NO).permute(0,3,1,2)

def matmul_fixed(I, W, FL_O, WL_O=16):
    '''
    Mimic the 16-b MAC array hardware implementation
    I: Q(16-FL_I, FL_I)
    W: Q(16-FL_W, FL_W)
    P: Q(32-FL_I, FL_W, FL_I+FL_W-16)
    O(before rounding): Q(38-FL_I-FL_W, FL_I+FL_W-16)
    O(after rounding): Q(16-FL_O, FL_O)
    '''
    NB, NI = I.shape
    NI, NO = W.shape
    #FL_P = I.FL+W.FL-16
    FL_P = I.FL+W.FL
    IV = I.value.to(torch.float64)
    WV = W.value.to(torch.float64)
    #O_pre = zeros((NB, NO), 22, FL_P)
    O_pre = zeros((NB, NO), 32, FL_P)
    for ni in range(NI):
        # P = fixed(torch.ger(IV[:,ni], WV[ni,:]).to(torch.int64), I.WL+W.WL, I.FL+W.FL).round(16, FL_P)
        P = fixed(torch.ger(IV[:,ni], WV[ni,:]).to(torch.int64), I.WL+W.WL, I.FL+W.FL)
        O_pre += P
    O = O_pre.round(WL_O, FL_O, True)
    return O
