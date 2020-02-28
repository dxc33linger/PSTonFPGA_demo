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

# Description: Generate instructions
# Created on 08/31/2018
# Modified on 12/28/2018, fix upsampling bugs

import numpy as np

# Training parameters

BATCH_SIZE      =   4
KERNEL_SIZE     =   3
PAD_ZEROS       =   1

MAP_N_input     =   3
MAP_X_input     =   32
MAP_Y_input     =   32
MAP_N_conv_0    =   16
MAP_X_conv_0    =   32
MAP_Y_conv_0    =   32
LOFF_W_conv_0   =   0
LOFF_A_conv_0   =   0
LOFF_P_conv_0   =   0
LOFF_B_conv_0   =   0
MAP_N_conv_1    =   16
MAP_X_conv_1    =   32
MAP_Y_conv_1    =   32
LOFF_W_conv_1   =   27
LOFF_A_conv_1   =   256
LOFF_P_conv_1   =   0
LOFF_B_conv_1   =   1
MAP_N_pool_0    =   16
MAP_X_pool_0    =   16
MAP_Y_pool_0    =   16
MAP_N_conv_2    =   32
MAP_X_conv_2    =   16
MAP_Y_conv_2    =   16
LOFF_W_conv_2   =   171
LOFF_A_conv_2   =   512
LOFF_P_conv_2   =   256
LOFF_B_conv_2   =   2
MAP_N_conv_3    =   32
MAP_X_conv_3    =   16
MAP_Y_conv_3    =   16
LOFF_W_conv_3   =   459
LOFF_A_conv_3   =   640
LOFF_P_conv_3   =   256
LOFF_B_conv_3   =   4
MAP_N_pool_1    =   32
MAP_X_pool_1    =   8
MAP_Y_pool_1    =   8
MAP_N_conv_4    =   64
MAP_X_conv_4    =   8
MAP_Y_conv_4    =   8
LOFF_W_conv_4   =   1035
LOFF_A_conv_4   =   768
LOFF_P_conv_4   =   384
LOFF_B_conv_4   =   6
MAP_N_conv_5    =   64
MAP_X_conv_5    =   8
MAP_Y_conv_5    =   8
LOFF_W_conv_5   =   2187
LOFF_A_conv_5   =   832
LOFF_P_conv_5   =   384
LOFF_B_conv_5   =   10
MAP_N_flatten   =   1024
MAP_X_flatten   =   1
MAP_Y_flatten   =   1
MAP_N_fc        =   10
MAP_X_fc        =   1
MAP_Y_fc        =   1
LOFF_W_fc       =   4491
LOFF_A_fc       =   896
LOFF_P_fc       =   384
LOFF_B_fc       =   14
LOFF_ACT_input  =   0
LOFF_ACT_conv_0 =   768
LOFF_LGD_conv_0 =   768
LOFF_LGD_conv_1 =   8960
LOFF_ACT_pool_0 =   4864
LOFF_LGD_pool_0 =   4864
LOFF_ACT_conv_2 =   5888
LOFF_LGD_conv_2 =   5888
LOFF_LGD_conv_3 =   9984
LOFF_ACT_pool_1 =   7936
LOFF_LGD_pool_1 =   7936
LOFF_ACT_conv_4 =   8448
LOFF_LGD_conv_4 =   8448
LOFF_LGD_conv_5 =   10496
LOFF_LGD_pool_2 =   9472
LOFF_ACT_flatten =   9472
LOFF_LGD_fc     =   9728
XPOSED_LGD      =   5515
ADDR_WIDTHI     =   14
ADDR_WIDTHW     =   14
ADDR_WIDTHB     =   7
ADDR_WIDTHA     =   10
ADDR_WIDTHP     =   10
DATA_WIDTH      =   16 

lr              = 0.004
momentum        = 0.499985
batchsize       = 40

# precision control parameters
#  Bit_Sel
Sel_A_conv_0 = 0
Sel_A_conv_1 = 0
Sel_A_conv_2 = 0
Sel_A_conv_3 = 0
Sel_A_conv_4 = 0
Sel_A_conv_5 = 0
Sel_A_fc     = 0
Sel_D_conv_0 = 0
Sel_D_conv_1 = 0
Sel_D_conv_2 = 0
Sel_D_conv_3 = 0
Sel_D_conv_4 = 0
Sel_D_conv_5 = 0
Sel_D_fc     = 0
Sel_W_conv_0 = 0
Sel_W_conv_1 = 0
Sel_W_conv_2 = 0
Sel_W_conv_3 = 0
Sel_W_conv_4 = 0
Sel_W_conv_5 = 0
Sel_W_fc     = 0

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

#  LearningRate during weight update
# FL_L_WG_conv_0 = 26
# FL_L_WG_conv_1 = 26
# FL_L_WG_conv_2 = 26
# FL_L_WG_conv_3 = 28
# FL_L_WG_conv_4 = 28
# FL_L_WG_conv_5 = 28
# FL_L_WG_fc     = 28
#  LearningRate during bias update
FL_L_BG_conv_0 = 24
FL_L_BG_conv_1 = 22
FL_L_BG_conv_2 = 20
FL_L_BG_conv_3 = 20
FL_L_BG_conv_4 = 20
FL_L_BG_conv_5 = 20
FL_L_BG_fc     = 20
#  LearningRate during actual weight update
# FL_L_WU_conv_0 = 12
# FL_L_WU_conv_1 = 12
# FL_L_WU_conv_2 = 12
# FL_L_WU_conv_3 = 12
# FL_L_WU_conv_4 = 12
# FL_L_WU_conv_5 = 12
# FL_L_WU_fc     = 12
#  LearningRate during actual bias update
# FL_L_BU_conv_0 = 12
# FL_L_BU_conv_1 = 12
# FL_L_BU_conv_2 = 12
# FL_L_BU_conv_3 = 12
# FL_L_BU_conv_4 = 12
# FL_L_BU_conv_5 = 12
# FL_L_BU_fc     = 12
#  Momentum during actual weight update
# FL_M_WU_conv_0 = 16
# FL_M_WU_conv_1 = 16
# FL_M_WU_conv_2 = 16
# FL_M_WU_conv_3 = 16
# FL_M_WU_conv_4 = 16
# FL_M_WU_conv_5 = 16
# FL_M_WU_fc     = 16
#  Momentum during actual bias update
# FL_M_BU_conv_0 = 16
# FL_M_BU_conv_1 = 16
# FL_M_BU_conv_2 = 16
# FL_M_BU_conv_3 = 16
# FL_M_BU_conv_4 = 16
# FL_M_BU_conv_5 = 16
# FL_M_BU_fc     = 16
#  Final output
# FL_A_fc        = 12

#  Scaling factor
# scale             = 1

# Mode constants

FORWARD     =   list('00')
BACKWARD    =   list('01')
WEIGHT_UPD  =   list('10')
CONV        =   '0'
FC          =   '1'

# Instruction structure
end_of_instructions = 0
auto_index = 1
ena_bypass = 2
ena_input_read = 3
ena_use_weight = 4
ena_use_LG_xposed = 5
ena_transpose = 6
ena_input_saving = 7
ena_xposed_LG_saving = 8
ena_weight_update = 9
ena_apply_wu = 10
ena_pooling = 11
ena_upsampling = 12
ena_bias_add = 13
ena_bias_update = 14
ena_apply_bu = 15
ena_gen_AD = 16
ena_mask_AD = 17
ena_output_LG = 18
ena_relu = 19
ena_reshape = 20
layer_type = 21
Mode = range(23,21,-1)
Bit_Sel = range(25,23,-1)
Config_batch_size = range(34,25,-1)
Config_in_map_n = range(46,34,-1)
Config_out_map_n = range(58,46,-1)
Config_in_map_x = range(66,58,-1)
Config_in_map_y = range(74,66,-1)
Config_kernel_size = range(78,74,-1)
Pad_zeros = range(82,78,-1)
Image_Index_from_instruction = range(91,82,-1)
Feat_Map_Index_div_16_from_instruction = range(99,91,-1)
Input_Feat_Map_Index_from_instruction = range(111,99,-1)
Xoffset_div_4_from_instruction = range(117,111,-1)
Yoffset_div_4_from_instruction = range(123,117,-1)
LearningRate = range(123+DATA_WIDTH,124-1,-1)
Momentum = range(123+DATA_WIDTH*2,124+DATA_WIDTH-1,-1)
Constant_plus_one = range(123+DATA_WIDTH*3,124+DATA_WIDTH*2-1,-1)
Layer_Addr_Offset_input_read = range(123+DATA_WIDTH*3+ADDR_WIDTHI,124+DATA_WIDTH*3-1,-1)
Layer_Addr_Offset_input_write = range(123+DATA_WIDTH*3+ADDR_WIDTHI*2,124+DATA_WIDTH*3+ADDR_WIDTHI-1,-1)
Layer_Addr_Offset_weight_read = range(123+DATA_WIDTH*3+ADDR_WIDTHI*2+ADDR_WIDTHW,124+DATA_WIDTH*3+ADDR_WIDTHI*2-1,-1)
Layer_Addr_Offset_weight_write = range(123+DATA_WIDTH*3+ADDR_WIDTHI*2+ADDR_WIDTHW*2,124+DATA_WIDTH*3+ADDR_WIDTHI*2+ADDR_WIDTHW-1,-1)
Layer_Addr_Offset_bias = range(123+DATA_WIDTH*3+ADDR_WIDTHI*2+ADDR_WIDTHW*2+ADDR_WIDTHB,124+DATA_WIDTH*3+ADDR_WIDTHI*2+ADDR_WIDTHW*2-1,-1)
Layer_Addr_Offset_AD = range(123+DATA_WIDTH*3+ADDR_WIDTHI*2+ADDR_WIDTHW*2+ADDR_WIDTHB+ADDR_WIDTHA,124+DATA_WIDTH*3+ADDR_WIDTHI*2+ADDR_WIDTHW*2+ADDR_WIDTHB-1,-1)
Layer_Addr_Offset_PI = range(123+DATA_WIDTH*3+ADDR_WIDTHI*2+ADDR_WIDTHW*2+ADDR_WIDTHB+ADDR_WIDTHA+ADDR_WIDTHP,124+DATA_WIDTH*3+ADDR_WIDTHI*2+ADDR_WIDTHW*2+ADDR_WIDTHB+ADDR_WIDTHA-1,-1)


Instruction_list = []

print("Instruction %02d: Layer: input->conv_0, forward" % len(Instruction_list))
Instr = np.array(list('0' * 256))
Instr[auto_index] = '1' 
Instr[ena_input_read] = '1' 
Instr[ena_use_weight] = '1'
Instr[ena_input_saving] = '1'
Instr[ena_gen_AD] = '1'
Instr[ena_relu] = '1'
Instr[layer_type] = CONV
Instr[Mode] = FORWARD
Instr[Bit_Sel] = list("{0:02b}".format(Sel_A_conv_0))
Instr[Config_batch_size] = list("{0:09b}".format(BATCH_SIZE))
Instr[Config_in_map_n] = list("{0:012b}".format(MAP_N_input))
Instr[Config_out_map_n] = list("{0:012b}".format(MAP_N_conv_0))
Instr[Config_in_map_x] = list("{0:08b}".format(MAP_X_input))
Instr[Config_in_map_y] = list("{0:08b}".format(MAP_Y_input))
Instr[Config_kernel_size] = list("{0:04b}".format(KERNEL_SIZE))
Instr[Pad_zeros] = list("{0:04b}".format(PAD_ZEROS))
Instr[Layer_Addr_Offset_input_read] = list("{0:014b}".format(LOFF_ACT_input))
Instr[Layer_Addr_Offset_input_write] = list("{0:014b}".format(LOFF_ACT_conv_0))
Instr[Layer_Addr_Offset_weight_read] = list("{0:014b}".format(LOFF_W_conv_0))
Instr[Layer_Addr_Offset_bias] = list("{0:07b}".format(LOFF_B_conv_0))
Instr[Layer_Addr_Offset_AD] = list("{0:010b}".format(LOFF_A_conv_0))
Instr[Layer_Addr_Offset_PI] = list("{0:010b}".format(LOFF_P_conv_0))
Instr = np.flipud(Instr).reshape((64,4))
output = ''
for i in range(64):
    output += "{0:01x}".format(int("".join(Instr[i]),2))
Instruction_list.append(output)

print("Instruction %02d: Layer: conv_0->conv_1->pool_0, forward" % len(Instruction_list))
Instr = np.array(list('0' * 256))
Instr[auto_index] = '1' 
Instr[ena_input_read] = '1' 
Instr[ena_use_weight] = '1'
Instr[ena_input_saving] = '1'
Instr[ena_gen_AD] = '1'
Instr[ena_relu] = '1'
Instr[ena_pooling] = '1'
Instr[layer_type] = CONV
Instr[Mode] = FORWARD
Instr[Bit_Sel] = list("{0:02b}".format(Sel_A_conv_1))
Instr[Config_batch_size] = list("{0:09b}".format(BATCH_SIZE))
Instr[Config_in_map_n] = list("{0:012b}".format(MAP_N_conv_0))
Instr[Config_out_map_n] = list("{0:012b}".format(MAP_N_conv_1))
Instr[Config_in_map_x] = list("{0:08b}".format(MAP_X_conv_0))
Instr[Config_in_map_y] = list("{0:08b}".format(MAP_Y_conv_0))
Instr[Config_kernel_size] = list("{0:04b}".format(KERNEL_SIZE))
Instr[Pad_zeros] = list("{0:04b}".format(PAD_ZEROS))
Instr[Layer_Addr_Offset_input_read] = list("{0:014b}".format(LOFF_ACT_conv_0))
Instr[Layer_Addr_Offset_input_write] = list("{0:014b}".format(LOFF_ACT_pool_0))
Instr[Layer_Addr_Offset_weight_read] = list("{0:014b}".format(LOFF_W_conv_1))
Instr[Layer_Addr_Offset_bias] = list("{0:07b}".format(LOFF_B_conv_1))
Instr[Layer_Addr_Offset_AD] = list("{0:010b}".format(LOFF_A_conv_1))
Instr[Layer_Addr_Offset_PI] = list("{0:010b}".format(LOFF_P_conv_1))
Instr = np.flipud(Instr).reshape((64,4))
output = ''
for i in range(64):
    output += "{0:01x}".format(int("".join(Instr[i]),2))
Instruction_list.append(output)

print("Instruction %02d: Layer: pool_0->conv_2, forward" % len(Instruction_list))
Instr = np.array(list('0' * 256))
Instr[auto_index] = '1' 
Instr[ena_input_read] = '1' 
Instr[ena_use_weight] = '1'
Instr[ena_input_saving] = '1'
Instr[ena_gen_AD] = '1'
Instr[ena_relu] = '1'
Instr[layer_type] = CONV
Instr[Mode] = FORWARD
Instr[Bit_Sel] = list("{0:02b}".format(Sel_A_conv_2))
Instr[Config_batch_size] = list("{0:09b}".format(BATCH_SIZE))
Instr[Config_in_map_n] = list("{0:012b}".format(MAP_N_pool_0))
Instr[Config_out_map_n] = list("{0:012b}".format(MAP_N_conv_2))
Instr[Config_in_map_x] = list("{0:08b}".format(MAP_X_pool_0))
Instr[Config_in_map_y] = list("{0:08b}".format(MAP_Y_pool_0))
Instr[Config_kernel_size] = list("{0:04b}".format(KERNEL_SIZE))
Instr[Pad_zeros] = list("{0:04b}".format(PAD_ZEROS))
Instr[Layer_Addr_Offset_input_read] = list("{0:014b}".format(LOFF_ACT_pool_0))
Instr[Layer_Addr_Offset_input_write] = list("{0:014b}".format(LOFF_ACT_conv_2))
Instr[Layer_Addr_Offset_weight_read] = list("{0:014b}".format(LOFF_W_conv_2))
Instr[Layer_Addr_Offset_bias] = list("{0:07b}".format(LOFF_B_conv_2))
Instr[Layer_Addr_Offset_AD] = list("{0:010b}".format(LOFF_A_conv_2))
Instr[Layer_Addr_Offset_PI] = list("{0:010b}".format(LOFF_P_conv_2))
Instr = np.flipud(Instr).reshape((64,4))
output = ''
for i in range(64):
    output += "{0:01x}".format(int("".join(Instr[i]),2))
Instruction_list.append(output)

print("Instruction %02d: Layer: conv_2->conv_3->pool_1, forward" % len(Instruction_list))
Instr = np.array(list('0' * 256))
Instr[auto_index] = '1' 
Instr[ena_input_read] = '1' 
Instr[ena_use_weight] = '1'
Instr[ena_input_saving] = '1'
Instr[ena_gen_AD] = '1'
Instr[ena_relu] = '1'
Instr[ena_pooling] = '1'
Instr[layer_type] = CONV
Instr[Mode] = FORWARD
Instr[Bit_Sel] = list("{0:02b}".format(Sel_A_conv_3))
Instr[Config_batch_size] = list("{0:09b}".format(BATCH_SIZE))
Instr[Config_in_map_n] = list("{0:012b}".format(MAP_N_conv_2))
Instr[Config_out_map_n] = list("{0:012b}".format(MAP_N_conv_3))
Instr[Config_in_map_x] = list("{0:08b}".format(MAP_X_conv_2))
Instr[Config_in_map_y] = list("{0:08b}".format(MAP_Y_conv_2))
Instr[Config_kernel_size] = list("{0:04b}".format(KERNEL_SIZE))
Instr[Pad_zeros] = list("{0:04b}".format(PAD_ZEROS))
Instr[Layer_Addr_Offset_input_read] = list("{0:014b}".format(LOFF_ACT_conv_2))
Instr[Layer_Addr_Offset_input_write] = list("{0:014b}".format(LOFF_ACT_pool_1))
Instr[Layer_Addr_Offset_weight_read] = list("{0:014b}".format(LOFF_W_conv_3))
Instr[Layer_Addr_Offset_bias] = list("{0:07b}".format(LOFF_B_conv_3))
Instr[Layer_Addr_Offset_AD] = list("{0:010b}".format(LOFF_A_conv_3))
Instr[Layer_Addr_Offset_PI] = list("{0:010b}".format(LOFF_P_conv_3))
Instr = np.flipud(Instr).reshape((64,4))
output = ''
for i in range(64):
    output += "{0:01x}".format(int("".join(Instr[i]),2))
Instruction_list.append(output)

print("Instruction %02d: Layer: pool_1->conv_4, forward" % len(Instruction_list))
Instr = np.array(list('0' * 256))
Instr[auto_index] = '1' 
Instr[ena_input_read] = '1' 
Instr[ena_use_weight] = '1'
Instr[ena_input_saving] = '1'
Instr[ena_gen_AD] = '1'
Instr[ena_relu] = '1'
Instr[layer_type] = CONV
Instr[Mode] = FORWARD
Instr[Bit_Sel] = list("{0:02b}".format(Sel_A_conv_4))
Instr[Config_batch_size] = list("{0:09b}".format(BATCH_SIZE))
Instr[Config_in_map_n] = list("{0:012b}".format(MAP_N_pool_1))
Instr[Config_out_map_n] = list("{0:012b}".format(MAP_N_conv_4))
Instr[Config_in_map_x] = list("{0:08b}".format(MAP_X_pool_1))
Instr[Config_in_map_y] = list("{0:08b}".format(MAP_Y_pool_1))
Instr[Config_kernel_size] = list("{0:04b}".format(KERNEL_SIZE))
Instr[Pad_zeros] = list("{0:04b}".format(PAD_ZEROS))
Instr[Layer_Addr_Offset_input_read] = list("{0:014b}".format(LOFF_ACT_pool_1))
Instr[Layer_Addr_Offset_input_write] = list("{0:014b}".format(LOFF_ACT_conv_4))
Instr[Layer_Addr_Offset_weight_read] = list("{0:014b}".format(LOFF_W_conv_4))
Instr[Layer_Addr_Offset_bias] = list("{0:07b}".format(LOFF_B_conv_4))
Instr[Layer_Addr_Offset_AD] = list("{0:010b}".format(LOFF_A_conv_4))
Instr[Layer_Addr_Offset_PI] = list("{0:010b}".format(LOFF_P_conv_4))
Instr = np.flipud(Instr).reshape((64,4))
output = ''
for i in range(64):
    output += "{0:01x}".format(int("".join(Instr[i]),2))
Instruction_list.append(output)

print("Instruction %02d: Layer: conv_4->conv_5->pool_2->flatten, forward" % len(Instruction_list))
Instr = np.array(list('0' * 256))
Instr[auto_index] = '1' 
Instr[ena_input_read] = '1' 
Instr[ena_use_weight] = '1'
Instr[ena_input_saving] = '1'
Instr[ena_gen_AD] = '1'
Instr[ena_relu] = '1'
Instr[ena_pooling] = '1'
Instr[ena_reshape] = '1'
Instr[layer_type] = CONV
Instr[Mode] = FORWARD
Instr[Bit_Sel] = list("{0:02b}".format(Sel_A_conv_5))
Instr[Config_batch_size] = list("{0:09b}".format(BATCH_SIZE))
Instr[Config_in_map_n] = list("{0:012b}".format(MAP_N_conv_4))
Instr[Config_out_map_n] = list("{0:012b}".format(MAP_N_conv_5))
Instr[Config_in_map_x] = list("{0:08b}".format(MAP_X_conv_4))
Instr[Config_in_map_y] = list("{0:08b}".format(MAP_Y_conv_4))
Instr[Config_kernel_size] = list("{0:04b}".format(KERNEL_SIZE))
Instr[Pad_zeros] = list("{0:04b}".format(PAD_ZEROS))
Instr[Layer_Addr_Offset_input_read] = list("{0:014b}".format(LOFF_ACT_conv_4))
Instr[Layer_Addr_Offset_input_write] = list("{0:014b}".format(LOFF_ACT_flatten))
Instr[Layer_Addr_Offset_weight_read] = list("{0:014b}".format(LOFF_W_conv_5))
Instr[Layer_Addr_Offset_bias] = list("{0:07b}".format(LOFF_B_conv_5))
Instr[Layer_Addr_Offset_AD] = list("{0:010b}".format(LOFF_A_conv_5))
Instr[Layer_Addr_Offset_PI] = list("{0:010b}".format(LOFF_P_conv_5))
Instr = np.flipud(Instr).reshape((64,4))
output = ''
for i in range(64):
    output += "{0:01x}".format(int("".join(Instr[i]),2))
Instruction_list.append(output)

print("Instruction %02d: Layer: flatten->fc,   forward,  evaluate local gradients" % len(Instruction_list))
Instr = np.array(list('0' * 256))
Instr[auto_index] = '1' 
Instr[ena_input_read] = '1' 
Instr[ena_use_weight] = '1'
Instr[ena_transpose] = '1'
Instr[ena_input_saving] = '1'
Instr[ena_xposed_LG_saving] = '1'
Instr[ena_output_LG] = '1'
Instr[layer_type] = FC
Instr[Mode] = FORWARD
Instr[Bit_Sel] = list("{0:02b}".format(Sel_A_fc))
Instr[Config_batch_size] = list("{0:09b}".format(BATCH_SIZE))
Instr[Config_in_map_n] = list("{0:012b}".format(MAP_N_flatten))
Instr[Config_out_map_n] = list("{0:012b}".format(MAP_N_fc))
Instr[Config_in_map_x] = list("{0:08b}".format(MAP_X_flatten))
Instr[Config_in_map_y] = list("{0:08b}".format(MAP_Y_flatten))
Instr[Config_kernel_size] = list("{0:04b}".format(KERNEL_SIZE))
Instr[Pad_zeros] = list("{0:04b}".format(PAD_ZEROS))
Instr[Constant_plus_one] = list("{0:016b}".format(1<<FL_A_fc))
Instr[Layer_Addr_Offset_input_read] = list("{0:014b}".format(LOFF_ACT_flatten))
Instr[Layer_Addr_Offset_input_write] = list("{0:014b}".format(LOFF_LGD_fc)) 
Instr[Layer_Addr_Offset_weight_read] = list("{0:014b}".format(LOFF_W_fc))
Instr[Layer_Addr_Offset_weight_write] = list("{0:014b}".format(XPOSED_LGD))
Instr[Layer_Addr_Offset_bias] = list("{0:07b}".format(LOFF_B_fc))
Instr[Layer_Addr_Offset_AD] = list("{0:010b}".format(LOFF_A_fc))
Instr = np.flipud(Instr).reshape((64,4))
output = ''
for i in range(64):
    output += "{0:01x}".format(int("".join(Instr[i]),2))
Instruction_list.append(output)

print("Instruction %02d: Layer: fc, weight update" % len(Instruction_list))
Instr = np.array(list('0' * 256))
Instr[auto_index] = '1' 
Instr[ena_input_read] = '1' 
Instr[ena_use_LG_xposed] = '1'
Instr[ena_transpose] = '1'
Instr[ena_weight_update] = '1'
Instr[layer_type] = FC
Instr[Mode] = WEIGHT_UPD
Instr[Bit_Sel] = list("{0:02b}".format(Sel_W_fc))
Instr[Config_batch_size] = list("{0:09b}".format(BATCH_SIZE))
Instr[Config_in_map_n] = list("{0:012b}".format(MAP_N_flatten))
Instr[Config_out_map_n] = list("{0:012b}".format(MAP_N_fc))
Instr[Config_in_map_x] = list("{0:08b}".format(MAP_X_flatten))
Instr[Config_in_map_y] = list("{0:08b}".format(MAP_Y_flatten))
Instr[Config_kernel_size] = list("{0:04b}".format(KERNEL_SIZE))
Instr[Pad_zeros] = list("{0:04b}".format(PAD_ZEROS))
Instr[LearningRate] = list("{0:016b}".format(int(round(lr / batchsize / scale * (1<<FL_L_WG_fc)))))
Instr[Layer_Addr_Offset_input_read] = list("{0:014b}".format(LOFF_ACT_flatten))
Instr[Layer_Addr_Offset_weight_read] = list("{0:014b}".format(XPOSED_LGD))
Instr[Layer_Addr_Offset_weight_write] = list("{0:014b}".format(LOFF_W_fc))
Instr = np.flipud(Instr).reshape((64,4))
output = ''
for i in range(64):
    output += "{0:01x}".format(int("".join(Instr[i]),2))
Instruction_list.append(output)

print("Instruction %02d: Layer: fc->flatten->pool_2,   backward" % len(Instruction_list))
Instr = np.array(list('0' * 256))
Instr[auto_index] = '1'
Instr[ena_input_read] = '1'
Instr[ena_use_weight] = '1'
Instr[ena_transpose] = '1'
Instr[ena_input_saving] = '1'
Instr[ena_reshape] = '1'
Instr[layer_type] = FC
Instr[Mode] = BACKWARD
Instr[Bit_Sel] = list("{0:02b}".format(Sel_D_fc))
Instr[Config_batch_size] = list("{0:09b}".format(BATCH_SIZE))
Instr[Config_in_map_n] = list("{0:012b}".format(MAP_N_fc))
Instr[Config_out_map_n] = list("{0:012b}".format(MAP_N_flatten))
Instr[Config_in_map_x] = list("{0:08b}".format(MAP_X_fc))
Instr[Config_in_map_y] = list("{0:08b}".format(MAP_Y_fc))
Instr[Config_kernel_size] = list("{0:04b}".format(KERNEL_SIZE))
Instr[Pad_zeros] = list("{0:04b}".format(PAD_ZEROS))
Instr[LearningRate] = list("{0:016b}".format(12345))
Instr[Momentum] = list("{0:016b}".format(23456))
Instr[Layer_Addr_Offset_input_read] = list("{0:014b}".format(LOFF_LGD_fc))
Instr[Layer_Addr_Offset_input_write] = list("{0:014b}".format(LOFF_LGD_pool_2))
Instr[Layer_Addr_Offset_weight_read] = list("{0:014b}".format(LOFF_W_fc))
Instr = np.flipud(Instr).reshape((64,4))
output = ''
for i in range(64):
    output += "{0:01x}".format(int("".join(Instr[i]),2))
Instruction_list.append(output)

print("Instruction %02d: End of instruction" % len(Instruction_list))
Instr = np.array(list('0' * 256))
Instr[end_of_instructions] = '1'
Instr = np.flipud(Instr).reshape((64,4))
output = ''
for i in range(64):
    output += "{0:01x}".format(int("".join(Instr[i]),2))
Instruction_list.append(output)

print('\n'.join(Instruction_list))
Instr_file_name = "Instr_part_0.txt"
with open(Instr_file_name, 'w') as f:
    f.write('\n'.join(Instruction_list))
    
Instruction_list = []

print("Instruction %02d: Layer: conv_5, upsampling (externally) and mask AD,    bypass MAC operation" % len(Instruction_list))
Instr = np.array(list('0' * 256))
Instr[auto_index] = '1'
Instr[ena_bypass] = '1'
Instr[ena_input_read] = '1'
Instr[ena_transpose] = '1'
Instr[ena_input_saving] = '1'
Instr[ena_xposed_LG_saving] = '1'
Instr[ena_mask_AD] = '1'
Instr[layer_type] = CONV
Instr[Mode] = BACKWARD
Instr[Config_batch_size] = list("{0:09b}".format(BATCH_SIZE))
Instr[Config_in_map_n] = list("{0:012b}".format(MAP_N_conv_5))
Instr[Config_out_map_n] = list("{0:012b}".format(MAP_N_conv_5))
Instr[Config_in_map_x] = list("{0:08b}".format(MAP_X_conv_5))
Instr[Config_in_map_y] = list("{0:08b}".format(MAP_Y_conv_5))
Instr[Config_kernel_size] = list("{0:04b}".format(1))
Instr[Pad_zeros] = list("{0:04b}".format(0))
Instr[LearningRate] = list("{0:016b}".format(int(round(lr / batchsize / scale * (1<<FL_L_BG_conv_5)))))
Instr[Layer_Addr_Offset_input_read] = list("{0:014b}".format(LOFF_LGD_pool_2))
Instr[Layer_Addr_Offset_input_write] = list("{0:014b}".format(LOFF_LGD_conv_5))
Instr[Layer_Addr_Offset_weight_write] = list("{0:014b}".format(XPOSED_LGD))
Instr[Layer_Addr_Offset_bias] = list("{0:07b}".format(LOFF_B_conv_5))
Instr[Layer_Addr_Offset_AD] = list("{0:010b}".format(LOFF_A_conv_5))
Instr[Layer_Addr_Offset_PI] = list("{0:010b}".format(LOFF_P_conv_5))
Instr = np.flipud(Instr).reshape((64,4))
output = ''
for i in range(64):
    output += "{0:01x}".format(int("".join(Instr[i]),2))
Instruction_list.append(output)

print("Instruction %02d: Layer: conv_5, weight update" % len(Instruction_list))
Instr = np.array(list('0' * 256))
Instr[auto_index] = '1'
Instr[ena_input_read] = '1'
Instr[ena_use_LG_xposed] = '1'
Instr[ena_transpose] = '1'
Instr[ena_weight_update] = '1'
Instr[layer_type] = CONV
Instr[Mode] = WEIGHT_UPD
Instr[Bit_Sel] = list("{0:02b}".format(Sel_W_conv_5))
Instr[Config_batch_size] = list("{0:09b}".format(BATCH_SIZE))
Instr[Config_in_map_n] = list("{0:012b}".format(MAP_N_conv_4))
Instr[Config_out_map_n] = list("{0:012b}".format(MAP_N_conv_5))
Instr[Config_in_map_x] = list("{0:08b}".format(MAP_X_conv_4))
Instr[Config_in_map_y] = list("{0:08b}".format(MAP_Y_conv_4))
Instr[Config_kernel_size] = list("{0:04b}".format(KERNEL_SIZE))
Instr[Pad_zeros] = list("{0:04b}".format(PAD_ZEROS))
Instr[LearningRate] = list("{0:016b}".format(int(round(lr / batchsize / scale * (1<<FL_L_WG_conv_5)))))
Instr[Layer_Addr_Offset_input_read] = list("{0:014b}".format(LOFF_ACT_conv_4))
Instr[Layer_Addr_Offset_weight_read] = list("{0:014b}".format(XPOSED_LGD))
Instr[Layer_Addr_Offset_weight_write] = list("{0:014b}".format(LOFF_W_conv_5))
Instr = np.flipud(Instr).reshape((64,4))
output = ''
for i in range(64):
    output += "{0:01x}".format(int("".join(Instr[i]),2))
Instruction_list.append(output)

print("Instruction %02d: Layer: conv_5->conv_4, backward" % len(Instruction_list))
Instr = np.array(list('0' * 256))
Instr[auto_index] = '1'
Instr[ena_input_read] = '1'
Instr[ena_use_weight] = '1'
Instr[ena_transpose] = '1'
Instr[ena_input_saving] = '1'
Instr[ena_xposed_LG_saving] = '1'
Instr[ena_bias_update] = '1'
Instr[ena_mask_AD] = '1'
Instr[layer_type] = CONV
Instr[Mode] = BACKWARD
Instr[Bit_Sel] = list("{0:02b}".format(Sel_D_conv_5))
Instr[Config_batch_size] = list("{0:09b}".format(BATCH_SIZE))
Instr[Config_in_map_n] = list("{0:012b}".format(MAP_N_conv_5))
Instr[Config_out_map_n] = list("{0:012b}".format(MAP_N_conv_4))
Instr[Config_in_map_x] = list("{0:08b}".format(MAP_X_conv_5))
Instr[Config_in_map_y] = list("{0:08b}".format(MAP_Y_conv_5))
Instr[Config_kernel_size] = list("{0:04b}".format(KERNEL_SIZE))
Instr[Pad_zeros] = list("{0:04b}".format(PAD_ZEROS))
Instr[LearningRate] = list("{0:016b}".format(int(round(lr / batchsize / scale * (1<<FL_L_BG_conv_4)))))
Instr[Layer_Addr_Offset_input_read] = list("{0:014b}".format(LOFF_LGD_conv_5))
Instr[Layer_Addr_Offset_input_write] = list("{0:014b}".format(LOFF_LGD_conv_4))
Instr[Layer_Addr_Offset_weight_read] = list("{0:014b}".format(LOFF_W_conv_5))
Instr[Layer_Addr_Offset_weight_write] = list("{0:014b}".format(XPOSED_LGD))
Instr[Layer_Addr_Offset_bias] = list("{0:07b}".format(LOFF_B_conv_4))
Instr[Layer_Addr_Offset_AD] = list("{0:010b}".format(LOFF_A_conv_4))
Instr[Layer_Addr_Offset_PI] = list("{0:010b}".format(LOFF_P_conv_4))
Instr = np.flipud(Instr).reshape((64,4))
output = ''
for i in range(64):
    output += "{0:01x}".format(int("".join(Instr[i]),2))
Instruction_list.append(output)

print("Instruction %02d: Layer: conv_4, weight update" % len(Instruction_list))
Instr = np.array(list('0' * 256))
Instr[auto_index] = '1'
Instr[ena_input_read] = '1'
Instr[ena_use_LG_xposed] = '1'
Instr[ena_transpose] = '1'
Instr[ena_weight_update] = '1'
Instr[layer_type] = CONV
Instr[Mode] = WEIGHT_UPD
Instr[Bit_Sel] = list("{0:02b}".format(Sel_W_conv_4))
Instr[Config_batch_size] = list("{0:09b}".format(BATCH_SIZE))
Instr[Config_in_map_n] = list("{0:012b}".format(MAP_N_pool_1))
Instr[Config_out_map_n] = list("{0:012b}".format(MAP_N_conv_4))
Instr[Config_in_map_x] = list("{0:08b}".format(MAP_X_pool_1))
Instr[Config_in_map_y] = list("{0:08b}".format(MAP_Y_pool_1))
Instr[Config_kernel_size] = list("{0:04b}".format(KERNEL_SIZE))
Instr[Pad_zeros] = list("{0:04b}".format(PAD_ZEROS))
Instr[LearningRate] = list("{0:016b}".format(int(round(lr / batchsize / scale * (1<<FL_L_WG_conv_4)))))
Instr[Layer_Addr_Offset_input_read] = list("{0:014b}".format(LOFF_ACT_pool_1))
Instr[Layer_Addr_Offset_weight_read] = list("{0:014b}".format(XPOSED_LGD))
Instr[Layer_Addr_Offset_weight_write] = list("{0:014b}".format(LOFF_W_conv_4))
Instr = np.flipud(Instr).reshape((64,4))
output = ''
for i in range(64):
    output += "{0:01x}".format(int("".join(Instr[i]),2))
Instruction_list.append(output)

print("Instruction %02d: Layer: conv_4->pool_1, conv backward" % len(Instruction_list))
Instr = np.array(list('0' * 256))
Instr[auto_index] = '1'
Instr[ena_input_read] = '1'
Instr[ena_use_weight] = '1'
Instr[ena_input_saving] = '1'
Instr[layer_type] = CONV
Instr[Mode] = BACKWARD
Instr[Bit_Sel] = list("{0:02b}".format(Sel_D_conv_4))
Instr[Config_batch_size] = list("{0:09b}".format(BATCH_SIZE))
Instr[Config_in_map_n] = list("{0:012b}".format(MAP_N_conv_4))
Instr[Config_out_map_n] = list("{0:012b}".format(MAP_N_pool_1))
Instr[Config_in_map_x] = list("{0:08b}".format(MAP_X_conv_4))
Instr[Config_in_map_y] = list("{0:08b}".format(MAP_Y_conv_4))
Instr[Config_kernel_size] = list("{0:04b}".format(KERNEL_SIZE))
Instr[Pad_zeros] = list("{0:04b}".format(PAD_ZEROS))
Instr[LearningRate] = list("{0:016b}".format(int(round(lr / batchsize / scale * (1<<FL_L_BG_conv_3)))))
Instr[Layer_Addr_Offset_input_read] = list("{0:014b}".format(LOFF_LGD_conv_4))
Instr[Layer_Addr_Offset_input_write] = list("{0:014b}".format(LOFF_LGD_pool_1))
Instr[Layer_Addr_Offset_weight_read] = list("{0:014b}".format(LOFF_W_conv_4))
Instr = np.flipud(Instr).reshape((64,4))
output = ''
for i in range(64):
    output += "{0:01x}".format(int("".join(Instr[i]),2))
Instruction_list.append(output)

print("Instruction %02d: End of instruction" % len(Instruction_list))
Instr = np.array(list('0' * 256))
Instr[end_of_instructions] = '1'
Instr = np.flipud(Instr).reshape((64,4))
output = ''
for i in range(64):
    output += "{0:01x}".format(int("".join(Instr[i]),2))
Instruction_list.append(output)

print('\n'.join(Instruction_list))
Instr_file_name = "Instr_part_1.txt"
with open(Instr_file_name, 'w') as f:
    f.write('\n'.join(Instruction_list))
    
Instruction_list = []

print("Instruction %02d: Layer: conv_3, upsampling (externally) and mask AD,    bypass MAC operation" % len(Instruction_list))
Instr = np.array(list('0' * 256))
Instr[auto_index] = '1'
Instr[ena_bypass] = '1'
Instr[ena_input_read] = '1'
Instr[ena_transpose] = '1'
Instr[ena_input_saving] = '1'
Instr[ena_xposed_LG_saving] = '1'
Instr[ena_mask_AD] = '1'
Instr[layer_type] = CONV
Instr[Mode] = BACKWARD
Instr[Config_batch_size] = list("{0:09b}".format(BATCH_SIZE))
Instr[Config_in_map_n] = list("{0:012b}".format(MAP_N_conv_3))
Instr[Config_out_map_n] = list("{0:012b}".format(MAP_N_conv_3))
Instr[Config_in_map_x] = list("{0:08b}".format(MAP_X_conv_3))
Instr[Config_in_map_y] = list("{0:08b}".format(MAP_Y_conv_3))
Instr[Config_kernel_size] = list("{0:04b}".format(1))
Instr[Pad_zeros] = list("{0:04b}".format(0))
Instr[LearningRate] = list("{0:016b}".format(int(round(lr / batchsize / scale * (1<<FL_L_BG_conv_3)))))
Instr[Layer_Addr_Offset_input_read] = list("{0:014b}".format(LOFF_LGD_pool_1))
Instr[Layer_Addr_Offset_input_write] = list("{0:014b}".format(LOFF_LGD_conv_3))
Instr[Layer_Addr_Offset_weight_write] = list("{0:014b}".format(XPOSED_LGD))
Instr[Layer_Addr_Offset_bias] = list("{0:07b}".format(LOFF_B_conv_3))
Instr[Layer_Addr_Offset_AD] = list("{0:010b}".format(LOFF_A_conv_3))
Instr[Layer_Addr_Offset_PI] = list("{0:010b}".format(LOFF_P_conv_3))
Instr = np.flipud(Instr).reshape((64,4))
output = ''
for i in range(64):
    output += "{0:01x}".format(int("".join(Instr[i]),2))
Instruction_list.append(output)

print("Instruction %02d: Layer: conv_3, weight update" % len(Instruction_list))
Instr = np.array(list('0' * 256))
Instr[auto_index] = '1'
Instr[ena_input_read] = '1'
Instr[ena_use_LG_xposed] = '1'
Instr[ena_transpose] = '1'
Instr[ena_weight_update] = '1'
Instr[layer_type] = CONV
Instr[Mode] = WEIGHT_UPD
Instr[Bit_Sel] = list("{0:02b}".format(Sel_W_conv_3))
Instr[Config_batch_size] = list("{0:09b}".format(BATCH_SIZE))
Instr[Config_in_map_n] = list("{0:012b}".format(MAP_N_conv_2))
Instr[Config_out_map_n] = list("{0:012b}".format(MAP_N_conv_3))
Instr[Config_in_map_x] = list("{0:08b}".format(MAP_X_conv_2))
Instr[Config_in_map_y] = list("{0:08b}".format(MAP_Y_conv_2))
Instr[Config_kernel_size] = list("{0:04b}".format(KERNEL_SIZE))
Instr[Pad_zeros] = list("{0:04b}".format(PAD_ZEROS))
Instr[LearningRate] = list("{0:016b}".format(int(round(lr / batchsize / scale * (1<<FL_L_WG_conv_3)))))
Instr[Layer_Addr_Offset_input_read] = list("{0:014b}".format(LOFF_ACT_conv_2))
Instr[Layer_Addr_Offset_weight_read] = list("{0:014b}".format(XPOSED_LGD))
Instr[Layer_Addr_Offset_weight_write] = list("{0:014b}".format(LOFF_W_conv_3))
Instr = np.flipud(Instr).reshape((64,4))
output = ''
for i in range(64):
    output += "{0:01x}".format(int("".join(Instr[i]),2))
Instruction_list.append(output)

print("Instruction %02d: Layer: conv_3->conv_2, backward" % len(Instruction_list))
Instr = np.array(list('0' * 256))
Instr[auto_index] = '1'
Instr[ena_input_read] = '1'
Instr[ena_use_weight] = '1'
Instr[ena_transpose] = '1'
Instr[ena_input_saving] = '1'
Instr[ena_xposed_LG_saving] = '1'
Instr[ena_bias_update] = '1'
Instr[ena_mask_AD] = '1'
Instr[layer_type] = CONV
Instr[Mode] = BACKWARD
Instr[Bit_Sel] = list("{0:02b}".format(Sel_D_conv_3))
Instr[Config_batch_size] = list("{0:09b}".format(BATCH_SIZE))
Instr[Config_in_map_n] = list("{0:012b}".format(MAP_N_conv_3))
Instr[Config_out_map_n] = list("{0:012b}".format(MAP_N_conv_2))
Instr[Config_in_map_x] = list("{0:08b}".format(MAP_X_conv_3))
Instr[Config_in_map_y] = list("{0:08b}".format(MAP_Y_conv_3))
Instr[Config_kernel_size] = list("{0:04b}".format(KERNEL_SIZE))
Instr[Pad_zeros] = list("{0:04b}".format(PAD_ZEROS))
Instr[LearningRate] = list("{0:016b}".format(int(round(lr / batchsize / scale * (1<<FL_L_BG_conv_2)))))
Instr[Layer_Addr_Offset_input_read] = list("{0:014b}".format(LOFF_LGD_conv_3))
Instr[Layer_Addr_Offset_input_write] = list("{0:014b}".format(LOFF_LGD_conv_2))
Instr[Layer_Addr_Offset_weight_read] = list("{0:014b}".format(LOFF_W_conv_3))
Instr[Layer_Addr_Offset_weight_write] = list("{0:014b}".format(XPOSED_LGD))
Instr[Layer_Addr_Offset_bias] = list("{0:07b}".format(LOFF_B_conv_2))
Instr[Layer_Addr_Offset_AD] = list("{0:010b}".format(LOFF_A_conv_2))
Instr[Layer_Addr_Offset_PI] = list("{0:010b}".format(LOFF_P_conv_2))
Instr = np.flipud(Instr).reshape((64,4))
output = ''
for i in range(64):
    output += "{0:01x}".format(int("".join(Instr[i]),2))
Instruction_list.append(output)

print("Instruction %02d: Layer: conv_2, weight update" % len(Instruction_list))
Instr = np.array(list('0' * 256))
Instr[auto_index] = '1'
Instr[ena_input_read] = '1'
Instr[ena_use_LG_xposed] = '1'
Instr[ena_transpose] = '1'
Instr[ena_weight_update] = '1'
Instr[layer_type] = CONV
Instr[Mode] = WEIGHT_UPD
Instr[Bit_Sel] = list("{0:02b}".format(Sel_W_conv_2))
Instr[Config_batch_size] = list("{0:09b}".format(BATCH_SIZE))
Instr[Config_in_map_n] = list("{0:012b}".format(MAP_N_pool_0))
Instr[Config_out_map_n] = list("{0:012b}".format(MAP_N_conv_2))
Instr[Config_in_map_x] = list("{0:08b}".format(MAP_X_pool_0))
Instr[Config_in_map_y] = list("{0:08b}".format(MAP_Y_pool_0))
Instr[Config_kernel_size] = list("{0:04b}".format(KERNEL_SIZE))
Instr[Pad_zeros] = list("{0:04b}".format(PAD_ZEROS))
Instr[LearningRate] = list("{0:016b}".format(int(round(lr / batchsize / scale * (1<<FL_L_WG_conv_2)))))
Instr[Layer_Addr_Offset_input_read] = list("{0:014b}".format(LOFF_ACT_pool_0))
Instr[Layer_Addr_Offset_weight_read] = list("{0:014b}".format(XPOSED_LGD))
Instr[Layer_Addr_Offset_weight_write] = list("{0:014b}".format(LOFF_W_conv_2))
Instr = np.flipud(Instr).reshape((64,4))
output = ''
for i in range(64):
    output += "{0:01x}".format(int("".join(Instr[i]),2))
Instruction_list.append(output)

print("Instruction %02d: Layer: conv_2->pool_0, backward" % len(Instruction_list))
Instr = np.array(list('0' * 256))
Instr[auto_index] = '1'
Instr[ena_input_read] = '1'
Instr[ena_use_weight] = '1'
Instr[ena_input_saving] = '1'
Instr[layer_type] = CONV
Instr[Mode] = BACKWARD
Instr[Bit_Sel] = list("{0:02b}".format(Sel_D_conv_2))
Instr[Config_batch_size] = list("{0:09b}".format(BATCH_SIZE))
Instr[Config_in_map_n] = list("{0:012b}".format(MAP_N_conv_2))
Instr[Config_out_map_n] = list("{0:012b}".format(MAP_N_pool_0))
Instr[Config_in_map_x] = list("{0:08b}".format(MAP_X_conv_2))
Instr[Config_in_map_y] = list("{0:08b}".format(MAP_Y_conv_2))
Instr[Config_kernel_size] = list("{0:04b}".format(KERNEL_SIZE))
Instr[Pad_zeros] = list("{0:04b}".format(PAD_ZEROS))
Instr[LearningRate] = list("{0:016b}".format(int(round(lr / batchsize / scale * (1<<FL_L_BG_conv_2)))))
Instr[Layer_Addr_Offset_input_read] = list("{0:014b}".format(LOFF_LGD_conv_2))
Instr[Layer_Addr_Offset_input_write] = list("{0:014b}".format(LOFF_LGD_pool_0))
Instr[Layer_Addr_Offset_weight_read] = list("{0:014b}".format(LOFF_W_conv_2))
Instr = np.flipud(Instr).reshape((64,4))
output = ''
for i in range(64):
    output += "{0:01x}".format(int("".join(Instr[i]),2))
Instruction_list.append(output)

print("Instruction %02d: End of instruction" % len(Instruction_list))
Instr = np.array(list('0' * 256))
Instr[end_of_instructions] = '1'
Instr = np.flipud(Instr).reshape((64,4))
output = ''
for i in range(64):
    output += "{0:01x}".format(int("".join(Instr[i]),2))
Instruction_list.append(output)

print('\n'.join(Instruction_list))
Instr_file_name = "Instr_part_2.txt"
with open(Instr_file_name, 'w') as f:
    f.write('\n'.join(Instruction_list))
    
Instruction_list = []

print("Instruction %02d: Layer: conv_1, upsampling (externally) and mask AD,    bypass MAC operation" % len(Instruction_list))
Instr = np.array(list('0' * 256))
Instr[auto_index] = '1'
Instr[ena_bypass] = '1'
Instr[ena_input_read] = '1'
Instr[ena_transpose] = '1'
Instr[ena_input_saving] = '1'
Instr[ena_xposed_LG_saving] = '1'
Instr[ena_mask_AD] = '1'
Instr[layer_type] = CONV
Instr[Mode] = BACKWARD
Instr[Config_batch_size] = list("{0:09b}".format(BATCH_SIZE))
Instr[Config_in_map_n] = list("{0:012b}".format(MAP_N_conv_1))
Instr[Config_out_map_n] = list("{0:012b}".format(MAP_N_conv_1))
Instr[Config_in_map_x] = list("{0:08b}".format(MAP_X_conv_1))
Instr[Config_in_map_y] = list("{0:08b}".format(MAP_Y_conv_1))
Instr[Config_kernel_size] = list("{0:04b}".format(1))
Instr[Pad_zeros] = list("{0:04b}".format(0))
Instr[LearningRate] = list("{0:016b}".format(int(round(lr / batchsize / scale * (1<<FL_L_BG_conv_1)))))
Instr[Layer_Addr_Offset_input_read] = list("{0:014b}".format(LOFF_LGD_pool_0))
Instr[Layer_Addr_Offset_input_write] = list("{0:014b}".format(LOFF_LGD_conv_1))
Instr[Layer_Addr_Offset_weight_write] = list("{0:014b}".format(XPOSED_LGD))
Instr[Layer_Addr_Offset_bias] = list("{0:07b}".format(LOFF_B_conv_1))
Instr[Layer_Addr_Offset_AD] = list("{0:010b}".format(LOFF_A_conv_1))
Instr[Layer_Addr_Offset_PI] = list("{0:010b}".format(LOFF_P_conv_1))
Instr = np.flipud(Instr).reshape((64,4))
output = ''
for i in range(64):
    output += "{0:01x}".format(int("".join(Instr[i]),2))
Instruction_list.append(output)

print("Instruction %02d: Layer: conv_1, weight update" % len(Instruction_list))
Instr = np.array(list('0' * 256))
Instr[auto_index] = '1'
Instr[ena_input_read] = '1'
Instr[ena_use_LG_xposed] = '1'
Instr[ena_transpose] = '1'
Instr[ena_weight_update] = '1'
Instr[layer_type] = CONV
Instr[Mode] = WEIGHT_UPD
Instr[Bit_Sel] = list("{0:02b}".format(Sel_W_conv_1))
Instr[Config_batch_size] = list("{0:09b}".format(BATCH_SIZE))
Instr[Config_in_map_n] = list("{0:012b}".format(MAP_N_conv_0))
Instr[Config_out_map_n] = list("{0:012b}".format(MAP_N_conv_1))
Instr[Config_in_map_x] = list("{0:08b}".format(MAP_X_conv_0))
Instr[Config_in_map_y] = list("{0:08b}".format(MAP_Y_conv_0))
Instr[Config_kernel_size] = list("{0:04b}".format(KERNEL_SIZE))
Instr[Pad_zeros] = list("{0:04b}".format(PAD_ZEROS))
Instr[LearningRate] = list("{0:016b}".format(int(round(lr / batchsize / scale * (1<<FL_L_WG_conv_1)))))
Instr[Layer_Addr_Offset_input_read] = list("{0:014b}".format(LOFF_ACT_conv_0))
Instr[Layer_Addr_Offset_weight_read] = list("{0:014b}".format(XPOSED_LGD))
Instr[Layer_Addr_Offset_weight_write] = list("{0:014b}".format(LOFF_W_conv_1))
Instr = np.flipud(Instr).reshape((64,4))
output = ''
for i in range(64):
    output += "{0:01x}".format(int("".join(Instr[i]),2))
Instruction_list.append(output)

print("Instruction %02d: Layer: conv_1->conv_0, backward" % len(Instruction_list))
Instr = np.array(list('0' * 256))
Instr[auto_index] = '1'
Instr[ena_input_read] = '1'
Instr[ena_use_weight] = '1'
Instr[ena_transpose] = '1'
Instr[ena_xposed_LG_saving] = '1'
Instr[ena_bias_update] = '1'
Instr[ena_mask_AD] = '1'
Instr[layer_type] = CONV
Instr[Mode] = BACKWARD
Instr[Bit_Sel] = list("{0:02b}".format(Sel_D_conv_1))
Instr[Config_batch_size] = list("{0:09b}".format(BATCH_SIZE))
Instr[Config_in_map_n] = list("{0:012b}".format(MAP_N_conv_1))
Instr[Config_out_map_n] = list("{0:012b}".format(MAP_N_conv_0))
Instr[Config_in_map_x] = list("{0:08b}".format(MAP_X_conv_1))
Instr[Config_in_map_y] = list("{0:08b}".format(MAP_Y_conv_1))
Instr[Config_kernel_size] = list("{0:04b}".format(KERNEL_SIZE))
Instr[Pad_zeros] = list("{0:04b}".format(PAD_ZEROS))
Instr[LearningRate] = list("{0:016b}".format(int(round(lr / batchsize / scale * (1<<FL_L_BG_conv_0)))))
Instr[Layer_Addr_Offset_input_read] = list("{0:014b}".format(LOFF_LGD_conv_1))
Instr[Layer_Addr_Offset_weight_read] = list("{0:014b}".format(LOFF_W_conv_1))
Instr[Layer_Addr_Offset_weight_write] = list("{0:014b}".format(XPOSED_LGD))
Instr[Layer_Addr_Offset_bias] = list("{0:07b}".format(LOFF_B_conv_0))
Instr[Layer_Addr_Offset_AD] = list("{0:010b}".format(LOFF_A_conv_0))
Instr[Layer_Addr_Offset_PI] = list("{0:010b}".format(LOFF_P_conv_0))
Instr = np.flipud(Instr).reshape((64,4))
output = ''
for i in range(64):
    output += "{0:01x}".format(int("".join(Instr[i]),2))
Instruction_list.append(output)

print("Instruction %02d: Layer: conv_0, weight update" % len(Instruction_list))
Instr = np.array(list('0' * 256))
Instr[auto_index] = '1'
Instr[ena_input_read] = '1'
Instr[ena_use_LG_xposed] = '1'
Instr[ena_transpose] = '1'
Instr[ena_weight_update] = '1'
Instr[layer_type] = CONV
Instr[Mode] = WEIGHT_UPD
Instr[Bit_Sel] = list("{0:02b}".format(Sel_W_conv_0))
Instr[Config_batch_size] = list("{0:09b}".format(BATCH_SIZE))
Instr[Config_in_map_n] = list("{0:012b}".format(MAP_N_input))
Instr[Config_out_map_n] = list("{0:012b}".format(MAP_N_conv_0))
Instr[Config_in_map_x] = list("{0:08b}".format(MAP_X_input))
Instr[Config_in_map_y] = list("{0:08b}".format(MAP_Y_input))
Instr[Config_kernel_size] = list("{0:04b}".format(KERNEL_SIZE))
Instr[Pad_zeros] = list("{0:04b}".format(PAD_ZEROS))
Instr[LearningRate] = list("{0:016b}".format(int(round(lr / batchsize / scale * (1<<FL_L_WG_conv_0)))))
Instr[Layer_Addr_Offset_input_read] = list("{0:014b}".format(LOFF_ACT_input))
Instr[Layer_Addr_Offset_weight_read] = list("{0:014b}".format(XPOSED_LGD))
Instr[Layer_Addr_Offset_weight_write] = list("{0:014b}".format(LOFF_W_conv_0))
Instr = np.flipud(Instr).reshape((64,4))
output = ''
for i in range(64):
    output += "{0:01x}".format(int("".join(Instr[i]),2))
Instruction_list.append(output)

print("Instruction %02d: End of instruction" % len(Instruction_list))
Instr = np.array(list('0' * 256))
Instr[end_of_instructions] = '1'
Instr = np.flipud(Instr).reshape((64,4))
output = ''
for i in range(64):
    output += "{0:01x}".format(int("".join(Instr[i]),2))
Instruction_list.append(output)
print('\n'.join(Instruction_list))
Instr_file_name = "Instr_part_3.txt"
with open(Instr_file_name, 'w') as f:
    f.write('\n'.join(Instruction_list))

Instruction_list = Instruction_list[:-1] # remove last end_of_instructions

print("Instruction %02d: Layer: fc, apply weight update" % len(Instruction_list))
Instr = np.array(list('0' * 256))
Instr[auto_index] = '1'
Instr[ena_apply_wu] = '1'
Instr[layer_type] = FC
Instr[Mode] = WEIGHT_UPD
Instr[Config_batch_size] = list("{0:09b}".format(BATCH_SIZE))
Instr[Config_in_map_n] = list("{0:012b}".format(MAP_N_flatten))
Instr[Config_out_map_n] = list("{0:012b}".format(MAP_N_fc))
Instr[Config_in_map_x] = list("{0:08b}".format(MAP_X_flatten))
Instr[Config_in_map_y] = list("{0:08b}".format(MAP_Y_flatten))
Instr[Config_kernel_size] = list("{0:04b}".format(1))
Instr[LearningRate] = list("{0:016b}".format(int(round(1. * scale * (1<<FL_L_WU_fc)))))
Instr[Momentum] = list("{0:016b}".format(int(round(momentum * (1<<FL_M_WU_fc)))))
Instr[Layer_Addr_Offset_weight_write] = list("{0:014b}".format(LOFF_W_fc))
Instr = np.flipud(Instr).reshape((64,4))
output = ''
for i in range(64):
    output += "{0:01x}".format(int("".join(Instr[i]),2))
Instruction_list.append(output)

# print("Instruction %02d: Layer: fc   apply bias update" % len(Instruction_list))
# Instr = np.array(list('0' * 256))
# Instr[auto_index] = '1'
# Instr[ena_apply_bu] = '1'
# Instr[layer_type] = FC
# Instr[Mode] = WEIGHT_UPD
# Instr[Config_batch_size] = list("{0:09b}".format(BATCH_SIZE))
# Instr[Config_in_map_n] = list("{0:012b}".format(MAP_N_flatten))
# Instr[Config_out_map_n] = list("{0:012b}".format(MAP_N_fc))
# Instr[Config_in_map_x] = list("{0:08b}".format(MAP_X_flatten))
# Instr[Config_in_map_y] = list("{0:08b}".format(MAP_Y_flatten))
# Instr[Config_kernel_size] = list("{0:04b}".format(1))
# Instr[LearningRate] = list("{0:016b}".format(int(round(1. * (1<<FL_L_BU_fc)))))
# Instr[Momentum] = list("{0:016b}".format(int(round(momentum * (1<<FL_M_BU_fc)))))
# Instr[Layer_Addr_Offset_bias] = list("{0:07b}".format(LOFF_B_fc))
# Instr = np.flipud(Instr).reshape((64,4))
# output = ''
# for i in range(64):
    # output += "{0:01x}".format(int("".join(Instr[i]),2))
# Instruction_list.append(output)

print("Instruction %02d: Layer: conv_5, apply weight update" % len(Instruction_list))
Instr = np.array(list('0' * 256))
Instr[auto_index] = '1'
Instr[ena_apply_wu] = '1'
Instr[layer_type] = CONV
Instr[Mode] = WEIGHT_UPD
Instr[Config_batch_size] = list("{0:09b}".format(BATCH_SIZE))
Instr[Config_in_map_n] = list("{0:012b}".format(MAP_N_conv_4))
Instr[Config_out_map_n] = list("{0:012b}".format(MAP_N_conv_5))
Instr[Config_in_map_x] = list("{0:08b}".format(MAP_X_conv_4))
Instr[Config_in_map_y] = list("{0:08b}".format(MAP_Y_conv_4))
Instr[Config_kernel_size] = list("{0:04b}".format(KERNEL_SIZE))
Instr[Pad_zeros] = list("{0:04b}".format(PAD_ZEROS))
Instr[LearningRate] = list("{0:016b}".format(int(round(1. * scale * (1<<FL_L_WU_conv_5)))))
Instr[Momentum] = list("{0:016b}".format(int(round(momentum * (1<<FL_M_WU_conv_5)))))
Instr[Layer_Addr_Offset_weight_write] = list("{0:014b}".format(LOFF_W_conv_5))
Instr = np.flipud(Instr).reshape((64,4))
output = ''
for i in range(64):
    output += "{0:01x}".format(int("".join(Instr[i]),2))
Instruction_list.append(output)

# print("Instruction %02d: Layer 6, conv apply bias update" % len(Instruction_list))
# Instr = np.array(list('0' * 256))
# Instr[auto_index] = '1'
# Instr[ena_apply_bu] = '1'
# Instr[layer_type] = CONV
# Instr[Mode] = WEIGHT_UPD
# Instr[Config_batch_size] = list("{0:09b}".format(BATCH_SIZE))
# Instr[Config_in_map_n] = list("{0:012b}".format(MAP_N_conv_4))
# Instr[Config_out_map_n] = list("{0:012b}".format(MAP_N_conv_5))
# Instr[Config_in_map_x] = list("{0:08b}".format(MAP_X_conv_4))
# Instr[Config_in_map_y] = list("{0:08b}".format(MAP_Y_conv_4))
# Instr[Config_kernel_size] = list("{0:04b}".format(KERNEL_SIZE))
# Instr[Pad_zeros] = list("{0:04b}".format(PAD_ZEROS))
# Instr[LearningRate] = list("{0:016b}".format(int(round(1. * (1<<FL_L_BU_conv_5)))))
# Instr[Momentum] = list("{0:016b}".format(int(round(momentum * (1<<FL_M_BU_conv_5)))))
# Instr[Layer_Addr_Offset_bias] = list("{0:07b}".format(LOFF_B_conv_5))
# Instr = np.flipud(Instr).reshape((64,4))
# output = ''
# for i in range(64):
    # output += "{0:01x}".format(int("".join(Instr[i]),2))
# Instruction_list.append(output)


print("Instruction %02d: Layer: conv_4, apply weight update" % len(Instruction_list))
Instr = np.array(list('0' * 256))
Instr[auto_index] = '1'
Instr[ena_apply_wu] = '1'
Instr[layer_type] = CONV
Instr[Mode] = WEIGHT_UPD
Instr[Config_batch_size] = list("{0:09b}".format(BATCH_SIZE))
Instr[Config_in_map_n] = list("{0:012b}".format(MAP_N_pool_1))
Instr[Config_out_map_n] = list("{0:012b}".format(MAP_N_conv_4))
Instr[Config_in_map_x] = list("{0:08b}".format(MAP_X_pool_1))
Instr[Config_in_map_y] = list("{0:08b}".format(MAP_Y_pool_1))
Instr[Config_kernel_size] = list("{0:04b}".format(KERNEL_SIZE))
Instr[Pad_zeros] = list("{0:04b}".format(PAD_ZEROS))
Instr[LearningRate] = list("{0:016b}".format(int(round(1. * scale * (1<<FL_L_WU_conv_4)))))
Instr[Momentum] = list("{0:016b}".format(int(round(momentum * (1<<FL_M_WU_conv_4)))))
Instr[Layer_Addr_Offset_weight_write] = list("{0:014b}".format(LOFF_W_conv_4))
Instr = np.flipud(Instr).reshape((64,4))
output = ''
for i in range(64):
    output += "{0:01x}".format(int("".join(Instr[i]),2))
Instruction_list.append(output)

# print("Instruction %02d: Layer 5, conv apply bias update" % len(Instruction_list))
# Instr = np.array(list('0' * 256))
# Instr[auto_index] = '1'
# Instr[ena_apply_bu] = '1'
# Instr[layer_type] = CONV
# Instr[Mode] = WEIGHT_UPD
# Instr[Config_batch_size] = list("{0:09b}".format(BATCH_SIZE))
# Instr[Config_in_map_n] = list("{0:012b}".format(MAP_N_pool_1))
# Instr[Config_out_map_n] = list("{0:012b}".format(MAP_N_conv_4))
# Instr[Config_in_map_x] = list("{0:08b}".format(MAP_X_pool_1))
# Instr[Config_in_map_y] = list("{0:08b}".format(MAP_Y_pool_1))
# Instr[Config_kernel_size] = list("{0:04b}".format(KERNEL_SIZE))
# Instr[Pad_zeros] = list("{0:04b}".format(PAD_ZEROS))
# Instr[LearningRate] = list("{0:016b}".format(int(round(1. * (1<<FL_L_BU_conv_4)))))
# Instr[Momentum] = list("{0:016b}".format(int(round(momentum * (1<<FL_M_BU_conv_4)))))
# Instr[Layer_Addr_Offset_bias] = list("{0:07b}".format(LOFF_B_conv_4))
# Instr = np.flipud(Instr).reshape((64,4))
# output = ''
# for i in range(64):
    # output += "{0:01x}".format(int("".join(Instr[i]),2))
# Instruction_list.append(output)

print("Instruction %02d: Layer: conv_3, apply weight update" % len(Instruction_list))
Instr = np.array(list('0' * 256))
Instr[auto_index] = '1'
Instr[ena_apply_wu] = '1'
Instr[layer_type] = CONV
Instr[Mode] = WEIGHT_UPD
Instr[Config_batch_size] = list("{0:09b}".format(BATCH_SIZE))
Instr[Config_in_map_n] = list("{0:012b}".format(MAP_N_conv_2))
Instr[Config_out_map_n] = list("{0:012b}".format(MAP_N_conv_3))
Instr[Config_in_map_x] = list("{0:08b}".format(MAP_X_conv_2))
Instr[Config_in_map_y] = list("{0:08b}".format(MAP_Y_conv_2))
Instr[Config_kernel_size] = list("{0:04b}".format(KERNEL_SIZE))
Instr[Pad_zeros] = list("{0:04b}".format(PAD_ZEROS))
Instr[LearningRate] = list("{0:016b}".format(int(round(1. * scale * (1<<FL_L_WU_conv_3)))))
Instr[Momentum] = list("{0:016b}".format(int(round(momentum * (1<<FL_M_WU_conv_3)))))
Instr[Layer_Addr_Offset_weight_write] = list("{0:014b}".format(LOFF_W_conv_3))
Instr = np.flipud(Instr).reshape((64,4))
output = ''
for i in range(64):
    output += "{0:01x}".format(int("".join(Instr[i]),2))
Instruction_list.append(output)

# print("Instruction %02d: Layer 4, conv apply bias update" % len(Instruction_list))
# Instr = np.array(list('0' * 256))
# Instr[auto_index] = '1'
# Instr[ena_apply_bu] = '1'
# Instr[layer_type] = CONV
# Instr[Mode] = WEIGHT_UPD
# Instr[Config_batch_size] = list("{0:09b}".format(BATCH_SIZE))
# Instr[Config_in_map_n] = list("{0:012b}".format(MAP_N_conv_2))
# Instr[Config_out_map_n] = list("{0:012b}".format(MAP_N_conv_3))
# Instr[Config_in_map_x] = list("{0:08b}".format(MAP_X_conv_2))
# Instr[Config_in_map_y] = list("{0:08b}".format(MAP_Y_conv_2))
# Instr[Config_kernel_size] = list("{0:04b}".format(KERNEL_SIZE))
# Instr[Pad_zeros] = list("{0:04b}".format(PAD_ZEROS))
# Instr[LearningRate] = list("{0:016b}".format(int(round(1. * (1<<FL_L_BU_conv_3)))))
# Instr[Momentum] = list("{0:016b}".format(int(round(momentum * (1<<FL_M_BU_conv_3)))))
# Instr[Layer_Addr_Offset_bias] = list("{0:07b}".format(LOFF_B_conv_3))
# Instr = np.flipud(Instr).reshape((64,4))
# output = ''
# for i in range(64):
    # output += "{0:01x}".format(int("".join(Instr[i]),2))
# Instruction_list.append(output)

print("Instruction %02d: Layer: conv_2, apply weight update" % len(Instruction_list))
Instr = np.array(list('0' * 256))
Instr[auto_index] = '1'
Instr[ena_apply_wu] = '1'
Instr[layer_type] = CONV
Instr[Mode] = WEIGHT_UPD
Instr[Config_batch_size] = list("{0:09b}".format(BATCH_SIZE))
Instr[Config_in_map_n] = list("{0:012b}".format(MAP_N_pool_0))
Instr[Config_out_map_n] = list("{0:012b}".format(MAP_N_conv_2))
Instr[Config_in_map_x] = list("{0:08b}".format(MAP_X_pool_0))
Instr[Config_in_map_y] = list("{0:08b}".format(MAP_Y_pool_0))
Instr[Config_kernel_size] = list("{0:04b}".format(KERNEL_SIZE))
Instr[Pad_zeros] = list("{0:04b}".format(PAD_ZEROS))
Instr[LearningRate] = list("{0:016b}".format(int(round(1. * scale * (1<<FL_L_WU_conv_2)))))
Instr[Momentum] = list("{0:016b}".format(int(round(momentum * (1<<FL_M_WU_conv_2)))))
Instr[Layer_Addr_Offset_weight_write] = list("{0:014b}".format(LOFF_W_conv_2))
Instr = np.flipud(Instr).reshape((64,4))
output = ''
for i in range(64):
    output += "{0:01x}".format(int("".join(Instr[i]),2))
Instruction_list.append(output)

# print("Instruction %02d: Layer 3, conv apply bias update" % len(Instruction_list))
# Instr = np.array(list('0' * 256))
# Instr[auto_index] = '1'
# Instr[ena_apply_bu] = '1'
# Instr[layer_type] = CONV
# Instr[Mode] = WEIGHT_UPD
# Instr[Config_batch_size] = list("{0:09b}".format(BATCH_SIZE))
# Instr[Config_in_map_n] = list("{0:012b}".format(MAP_N_pool_0))
# Instr[Config_out_map_n] = list("{0:012b}".format(MAP_N_conv_2))
# Instr[Config_in_map_x] = list("{0:08b}".format(MAP_X_pool_0))
# Instr[Config_in_map_y] = list("{0:08b}".format(MAP_Y_pool_0))
# Instr[Config_kernel_size] = list("{0:04b}".format(KERNEL_SIZE))
# Instr[Pad_zeros] = list("{0:04b}".format(PAD_ZEROS))
# Instr[LearningRate] = list("{0:016b}".format(int(round(1. * (1<<FL_L_BU_conv_2)))))
# Instr[Momentum] = list("{0:016b}".format(int(round(momentum * (1<<FL_M_BU_conv_2)))))
# Instr[Layer_Addr_Offset_bias] = list("{0:07b}".format(LOFF_B_conv_2))
# Instr = np.flipud(Instr).reshape((64,4))
# output = ''
# for i in range(64):
    # output += "{0:01x}".format(int("".join(Instr[i]),2))
# Instruction_list.append(output)

print("Instruction %02d: Layer: conv_1, apply weight update" % len(Instruction_list))
Instr = np.array(list('0' * 256))
Instr[auto_index] = '1'
Instr[ena_apply_wu] = '1'
Instr[layer_type] = CONV
Instr[Mode] = WEIGHT_UPD
Instr[Config_batch_size] = list("{0:09b}".format(BATCH_SIZE))
Instr[Config_in_map_n] = list("{0:012b}".format(MAP_N_conv_0))
Instr[Config_out_map_n] = list("{0:012b}".format(MAP_N_conv_1))
Instr[Config_in_map_x] = list("{0:08b}".format(MAP_X_conv_0))
Instr[Config_in_map_y] = list("{0:08b}".format(MAP_Y_conv_0))
Instr[Config_kernel_size] = list("{0:04b}".format(KERNEL_SIZE))
Instr[Pad_zeros] = list("{0:04b}".format(PAD_ZEROS))
Instr[LearningRate] = list("{0:016b}".format(int(round(1. * scale * (1<<FL_L_WU_conv_1)))))
Instr[Momentum] = list("{0:016b}".format(int(round(momentum * (1<<FL_M_WU_conv_1)))))
Instr[Layer_Addr_Offset_weight_write] = list("{0:014b}".format(LOFF_W_conv_1))
Instr = np.flipud(Instr).reshape((64,4))
output = ''
for i in range(64):
    output += "{0:01x}".format(int("".join(Instr[i]),2))
Instruction_list.append(output)

# print("Instruction %02d: Layer 2, conv apply bias update" % len(Instruction_list))
# Instr = np.array(list('0' * 256))
# Instr[auto_index] = '1'
# Instr[ena_apply_bu] = '1'
# Instr[layer_type] = CONV
# Instr[Mode] = WEIGHT_UPD
# Instr[Config_batch_size] = list("{0:09b}".format(BATCH_SIZE))
# Instr[Config_in_map_n] = list("{0:012b}".format(MAP_N_conv_0))
# Instr[Config_out_map_n] = list("{0:012b}".format(MAP_N_conv_1))
# Instr[Config_in_map_x] = list("{0:08b}".format(MAP_X_conv_0))
# Instr[Config_in_map_y] = list("{0:08b}".format(MAP_Y_conv_0))
# Instr[Config_kernel_size] = list("{0:04b}".format(KERNEL_SIZE))
# Instr[Pad_zeros] = list("{0:04b}".format(PAD_ZEROS))
# Instr[LearningRate] = list("{0:016b}".format(int(round(1. * (1<<FL_L_BU_conv_1)))))
# Instr[Momentum] = list("{0:016b}".format(int(round(momentum * (1<<FL_M_BU_conv_1)))))
# Instr[Layer_Addr_Offset_bias] = list("{0:07b}".format(LOFF_B_conv_1))
# Instr = np.flipud(Instr).reshape((64,4))
# output = ''
# for i in range(64):
    # output += "{0:01x}".format(int("".join(Instr[i]),2))
# Instruction_list.append(output)

print("Instruction %02d: Layer: conv_0, apply weight update" % len(Instruction_list))
Instr = np.array(list('0' * 256))
Instr[auto_index] = '1'
Instr[ena_apply_wu] = '1'
Instr[layer_type] = CONV
Instr[Mode] = WEIGHT_UPD
Instr[Config_batch_size] = list("{0:09b}".format(BATCH_SIZE))
Instr[Config_in_map_n] = list("{0:012b}".format(MAP_N_input))
Instr[Config_out_map_n] = list("{0:012b}".format(MAP_N_conv_0))
Instr[Config_in_map_x] = list("{0:08b}".format(MAP_X_input))
Instr[Config_in_map_y] = list("{0:08b}".format(MAP_Y_input))
Instr[Config_kernel_size] = list("{0:04b}".format(KERNEL_SIZE))
Instr[Pad_zeros] = list("{0:04b}".format(PAD_ZEROS))
Instr[LearningRate] = list("{0:016b}".format(int(round(1. * scale * (1<<FL_L_WU_conv_0)))))
Instr[Momentum] = list("{0:016b}".format(int(round(momentum * (1<<FL_M_WU_conv_0)))))
Instr[Layer_Addr_Offset_weight_write] = list("{0:014b}".format(LOFF_W_conv_0))
Instr = np.flipud(Instr).reshape((64,4))
output = ''
for i in range(64):
    output += "{0:01x}".format(int("".join(Instr[i]),2))
Instruction_list.append(output)

# print("Instruction %02d: Layer 1, conv apply bias update" % len(Instruction_list))
# Instr = np.array(list('0' * 256))
# Instr[auto_index] = '1'
# Instr[ena_apply_bu] = '1'
# Instr[layer_type] = CONV
# Instr[Mode] = WEIGHT_UPD
# Instr[Config_batch_size] = list("{0:09b}".format(BATCH_SIZE))
# Instr[Config_in_map_n] = list("{0:012b}".format(MAP_N_input))
# Instr[Config_out_map_n] = list("{0:012b}".format(MAP_N_conv_0))
# Instr[Config_in_map_x] = list("{0:08b}".format(MAP_X_input))
# Instr[Config_in_map_y] = list("{0:08b}".format(MAP_Y_input))
# Instr[Config_kernel_size] = list("{0:04b}".format(KERNEL_SIZE))
# Instr[Pad_zeros] = list("{0:04b}".format(PAD_ZEROS))
# Instr[LearningRate] = list("{0:016b}".format(int(round(1. * (1<<FL_L_BU_conv_0)))))
# Instr[Momentum] = list("{0:016b}".format(int(round(momentum * (1<<FL_M_BU_conv_0)))))
# Instr[Layer_Addr_Offset_bias] = list("{0:07b}".format(LOFF_B_conv_0))
# Instr = np.flipud(Instr).reshape((64,4))
# output = ''
# for i in range(64):
    # output += "{0:01x}".format(int("".join(Instr[i]),2))
# Instruction_list.append(output)

print("Instruction %02d: End of instruction" % len(Instruction_list))
Instr = np.array(list('0' * 256))
Instr[end_of_instructions] = '1'
Instr = np.flipud(Instr).reshape((64,4))
output = ''
for i in range(64):
    output += "{0:01x}".format(int("".join(Instr[i]),2))
Instruction_list.append(output)

print('\n'.join(Instruction_list))
Instr_file_name = "Instr_part_3a.txt"
with open(Instr_file_name, 'w') as f:
    f.write('\n'.join(Instruction_list))
    
# Generate instructions for inference
Instruction_list = []

print("Instruction %02d: Layer: input->conv_0, forward" % len(Instruction_list))
Instr = np.array(list('0' * 256))
Instr[auto_index] = '1' 
Instr[ena_input_read] = '1' 
Instr[ena_use_weight] = '1'
Instr[ena_input_saving] = '1'
Instr[ena_gen_AD] = '1'
Instr[ena_relu] = '1'
Instr[layer_type] = CONV
Instr[Mode] = FORWARD
Instr[Bit_Sel] = list("{0:02b}".format(Sel_A_conv_0))
Instr[Config_batch_size] = list("{0:09b}".format(BATCH_SIZE))
Instr[Config_in_map_n] = list("{0:012b}".format(MAP_N_input))
Instr[Config_out_map_n] = list("{0:012b}".format(MAP_N_conv_0))
Instr[Config_in_map_x] = list("{0:08b}".format(MAP_X_input))
Instr[Config_in_map_y] = list("{0:08b}".format(MAP_Y_input))
Instr[Config_kernel_size] = list("{0:04b}".format(KERNEL_SIZE))
Instr[Pad_zeros] = list("{0:04b}".format(PAD_ZEROS))
Instr[Layer_Addr_Offset_input_read] = list("{0:014b}".format(LOFF_ACT_input))
Instr[Layer_Addr_Offset_input_write] = list("{0:014b}".format(LOFF_ACT_conv_0))
Instr[Layer_Addr_Offset_weight_read] = list("{0:014b}".format(LOFF_W_conv_0))
Instr[Layer_Addr_Offset_bias] = list("{0:07b}".format(LOFF_B_conv_0))
Instr[Layer_Addr_Offset_AD] = list("{0:010b}".format(LOFF_A_conv_0))
Instr[Layer_Addr_Offset_PI] = list("{0:010b}".format(LOFF_P_conv_0))
Instr = np.flipud(Instr).reshape((64,4))
output = ''
for i in range(64):
    output += "{0:01x}".format(int("".join(Instr[i]),2))
Instruction_list.append(output)

print("Instruction %02d: Layer: conv_0->conv_1->pool_0, forward" % len(Instruction_list))
Instr = np.array(list('0' * 256))
Instr[auto_index] = '1' 
Instr[ena_input_read] = '1' 
Instr[ena_use_weight] = '1'
Instr[ena_input_saving] = '1'
Instr[ena_gen_AD] = '1'
Instr[ena_relu] = '1'
Instr[ena_pooling] = '1'
Instr[layer_type] = CONV
Instr[Mode] = FORWARD
Instr[Bit_Sel] = list("{0:02b}".format(Sel_A_conv_1))
Instr[Config_batch_size] = list("{0:09b}".format(BATCH_SIZE))
Instr[Config_in_map_n] = list("{0:012b}".format(MAP_N_conv_0))
Instr[Config_out_map_n] = list("{0:012b}".format(MAP_N_conv_1))
Instr[Config_in_map_x] = list("{0:08b}".format(MAP_X_conv_0))
Instr[Config_in_map_y] = list("{0:08b}".format(MAP_Y_conv_0))
Instr[Config_kernel_size] = list("{0:04b}".format(KERNEL_SIZE))
Instr[Pad_zeros] = list("{0:04b}".format(PAD_ZEROS))
Instr[Layer_Addr_Offset_input_read] = list("{0:014b}".format(LOFF_ACT_conv_0))
Instr[Layer_Addr_Offset_input_write] = list("{0:014b}".format(LOFF_ACT_pool_0))
Instr[Layer_Addr_Offset_weight_read] = list("{0:014b}".format(LOFF_W_conv_1))
Instr[Layer_Addr_Offset_bias] = list("{0:07b}".format(LOFF_B_conv_1))
Instr[Layer_Addr_Offset_AD] = list("{0:010b}".format(LOFF_A_conv_1))
Instr[Layer_Addr_Offset_PI] = list("{0:010b}".format(LOFF_P_conv_1))
Instr = np.flipud(Instr).reshape((64,4))
output = ''
for i in range(64):
    output += "{0:01x}".format(int("".join(Instr[i]),2))
Instruction_list.append(output)

print("Instruction %02d: Layer: pool_0->conv_2, forward" % len(Instruction_list))
Instr = np.array(list('0' * 256))
Instr[auto_index] = '1' 
Instr[ena_input_read] = '1' 
Instr[ena_use_weight] = '1'
Instr[ena_input_saving] = '1'
Instr[ena_gen_AD] = '1'
Instr[ena_relu] = '1'
Instr[layer_type] = CONV
Instr[Mode] = FORWARD
Instr[Bit_Sel] = list("{0:02b}".format(Sel_A_conv_2))
Instr[Config_batch_size] = list("{0:09b}".format(BATCH_SIZE))
Instr[Config_in_map_n] = list("{0:012b}".format(MAP_N_pool_0))
Instr[Config_out_map_n] = list("{0:012b}".format(MAP_N_conv_2))
Instr[Config_in_map_x] = list("{0:08b}".format(MAP_X_pool_0))
Instr[Config_in_map_y] = list("{0:08b}".format(MAP_Y_pool_0))
Instr[Config_kernel_size] = list("{0:04b}".format(KERNEL_SIZE))
Instr[Pad_zeros] = list("{0:04b}".format(PAD_ZEROS))
Instr[Layer_Addr_Offset_input_read] = list("{0:014b}".format(LOFF_ACT_pool_0))
Instr[Layer_Addr_Offset_input_write] = list("{0:014b}".format(LOFF_ACT_conv_2))
Instr[Layer_Addr_Offset_weight_read] = list("{0:014b}".format(LOFF_W_conv_2))
Instr[Layer_Addr_Offset_bias] = list("{0:07b}".format(LOFF_B_conv_2))
Instr[Layer_Addr_Offset_AD] = list("{0:010b}".format(LOFF_A_conv_2))
Instr[Layer_Addr_Offset_PI] = list("{0:010b}".format(LOFF_P_conv_2))
Instr = np.flipud(Instr).reshape((64,4))
output = ''
for i in range(64):
    output += "{0:01x}".format(int("".join(Instr[i]),2))
Instruction_list.append(output)

print("Instruction %02d: Layer: conv_2->conv_3->pool_1, forward" % len(Instruction_list))
Instr = np.array(list('0' * 256))
Instr[auto_index] = '1' 
Instr[ena_input_read] = '1' 
Instr[ena_use_weight] = '1'
Instr[ena_input_saving] = '1'
Instr[ena_gen_AD] = '1'
Instr[ena_relu] = '1'
Instr[ena_pooling] = '1'
Instr[layer_type] = CONV
Instr[Mode] = FORWARD
Instr[Bit_Sel] = list("{0:02b}".format(Sel_A_conv_3))
Instr[Config_batch_size] = list("{0:09b}".format(BATCH_SIZE))
Instr[Config_in_map_n] = list("{0:012b}".format(MAP_N_conv_2))
Instr[Config_out_map_n] = list("{0:012b}".format(MAP_N_conv_3))
Instr[Config_in_map_x] = list("{0:08b}".format(MAP_X_conv_2))
Instr[Config_in_map_y] = list("{0:08b}".format(MAP_Y_conv_2))
Instr[Config_kernel_size] = list("{0:04b}".format(KERNEL_SIZE))
Instr[Pad_zeros] = list("{0:04b}".format(PAD_ZEROS))
Instr[Layer_Addr_Offset_input_read] = list("{0:014b}".format(LOFF_ACT_conv_2))
Instr[Layer_Addr_Offset_input_write] = list("{0:014b}".format(LOFF_ACT_pool_1))
Instr[Layer_Addr_Offset_weight_read] = list("{0:014b}".format(LOFF_W_conv_3))
Instr[Layer_Addr_Offset_bias] = list("{0:07b}".format(LOFF_B_conv_3))
Instr[Layer_Addr_Offset_AD] = list("{0:010b}".format(LOFF_A_conv_3))
Instr[Layer_Addr_Offset_PI] = list("{0:010b}".format(LOFF_P_conv_3))
Instr = np.flipud(Instr).reshape((64,4))
output = ''
for i in range(64):
    output += "{0:01x}".format(int("".join(Instr[i]),2))
Instruction_list.append(output)

print("Instruction %02d: Layer: pool_1->conv_4, forward" % len(Instruction_list))
Instr = np.array(list('0' * 256))
Instr[auto_index] = '1' 
Instr[ena_input_read] = '1' 
Instr[ena_use_weight] = '1'
Instr[ena_input_saving] = '1'
Instr[ena_gen_AD] = '1'
Instr[ena_relu] = '1'
Instr[layer_type] = CONV
Instr[Mode] = FORWARD
Instr[Bit_Sel] = list("{0:02b}".format(Sel_A_conv_4))
Instr[Config_batch_size] = list("{0:09b}".format(BATCH_SIZE))
Instr[Config_in_map_n] = list("{0:012b}".format(MAP_N_pool_1))
Instr[Config_out_map_n] = list("{0:012b}".format(MAP_N_conv_4))
Instr[Config_in_map_x] = list("{0:08b}".format(MAP_X_pool_1))
Instr[Config_in_map_y] = list("{0:08b}".format(MAP_Y_pool_1))
Instr[Config_kernel_size] = list("{0:04b}".format(KERNEL_SIZE))
Instr[Pad_zeros] = list("{0:04b}".format(PAD_ZEROS))
Instr[Layer_Addr_Offset_input_read] = list("{0:014b}".format(LOFF_ACT_pool_1))
Instr[Layer_Addr_Offset_input_write] = list("{0:014b}".format(LOFF_ACT_conv_4))
Instr[Layer_Addr_Offset_weight_read] = list("{0:014b}".format(LOFF_W_conv_4))
Instr[Layer_Addr_Offset_bias] = list("{0:07b}".format(LOFF_B_conv_4))
Instr[Layer_Addr_Offset_AD] = list("{0:010b}".format(LOFF_A_conv_4))
Instr[Layer_Addr_Offset_PI] = list("{0:010b}".format(LOFF_P_conv_4))
Instr = np.flipud(Instr).reshape((64,4))
output = ''
for i in range(64):
    output += "{0:01x}".format(int("".join(Instr[i]),2))
Instruction_list.append(output)

print("Instruction %02d: Layer: conv_4->conv_5->pool_2->flatten, forward" % len(Instruction_list))
Instr = np.array(list('0' * 256))
Instr[auto_index] = '1' 
Instr[ena_input_read] = '1' 
Instr[ena_use_weight] = '1'
Instr[ena_input_saving] = '1'
Instr[ena_gen_AD] = '1'
Instr[ena_relu] = '1'
Instr[ena_pooling] = '1'
Instr[ena_reshape] = '1'
Instr[layer_type] = CONV
Instr[Mode] = FORWARD
Instr[Bit_Sel] = list("{0:02b}".format(Sel_A_conv_5))
Instr[Config_batch_size] = list("{0:09b}".format(BATCH_SIZE))
Instr[Config_in_map_n] = list("{0:012b}".format(MAP_N_conv_4))
Instr[Config_out_map_n] = list("{0:012b}".format(MAP_N_conv_5))
Instr[Config_in_map_x] = list("{0:08b}".format(MAP_X_conv_4))
Instr[Config_in_map_y] = list("{0:08b}".format(MAP_Y_conv_4))
Instr[Config_kernel_size] = list("{0:04b}".format(KERNEL_SIZE))
Instr[Pad_zeros] = list("{0:04b}".format(PAD_ZEROS))
Instr[Layer_Addr_Offset_input_read] = list("{0:014b}".format(LOFF_ACT_conv_4))
Instr[Layer_Addr_Offset_input_write] = list("{0:014b}".format(LOFF_ACT_flatten))
Instr[Layer_Addr_Offset_weight_read] = list("{0:014b}".format(LOFF_W_conv_5))
Instr[Layer_Addr_Offset_bias] = list("{0:07b}".format(LOFF_B_conv_5))
Instr[Layer_Addr_Offset_AD] = list("{0:010b}".format(LOFF_A_conv_5))
Instr[Layer_Addr_Offset_PI] = list("{0:010b}".format(LOFF_P_conv_5))
Instr = np.flipud(Instr).reshape((64,4))
output = ''
for i in range(64):
    output += "{0:01x}".format(int("".join(Instr[i]),2))
Instruction_list.append(output)

print("Instruction %02d: Layer: flatten->fc,   forward,  evaluate output" % len(Instruction_list))
Instr = np.array(list('0' * 256))
Instr[auto_index] = '1' 
Instr[ena_input_read] = '1' 
Instr[ena_use_weight] = '1'
Instr[ena_transpose] = '1'
Instr[ena_input_saving] = '1'
Instr[layer_type] = FC
Instr[Mode] = FORWARD
Instr[Bit_Sel] = list("{0:02b}".format(Sel_A_fc))
Instr[Config_batch_size] = list("{0:09b}".format(BATCH_SIZE))
Instr[Config_in_map_n] = list("{0:012b}".format(MAP_N_flatten))
Instr[Config_out_map_n] = list("{0:012b}".format(MAP_N_fc))
Instr[Config_in_map_x] = list("{0:08b}".format(MAP_X_flatten))
Instr[Config_in_map_y] = list("{0:08b}".format(MAP_Y_flatten))
Instr[Config_kernel_size] = list("{0:04b}".format(KERNEL_SIZE))
Instr[Pad_zeros] = list("{0:04b}".format(PAD_ZEROS))
Instr[Layer_Addr_Offset_input_read] = list("{0:014b}".format(LOFF_ACT_flatten))
Instr[Layer_Addr_Offset_input_write] = list("{0:014b}".format(LOFF_LGD_fc)) 
Instr[Layer_Addr_Offset_weight_read] = list("{0:014b}".format(LOFF_W_fc))
Instr = np.flipud(Instr).reshape((64,4))
output = ''
for i in range(64):
    output += "{0:01x}".format(int("".join(Instr[i]),2))
Instruction_list.append(output)


print("Instruction %02d: End of instruction" % len(Instruction_list))
Instr = np.array(list('0' * 256))
Instr[end_of_instructions] = '1'
Instr = np.flipud(Instr).reshape((64,4))
output = ''
for i in range(64):
    output += "{0:01x}".format(int("".join(Instr[i]),2))
Instruction_list.append(output)

print('\n'.join(Instruction_list))
Instr_file_name = "Instr_inference_only.txt"
with open(Instr_file_name, 'w') as f:
    f.write('\n'.join(Instruction_list))