# PSTonFPGA_demo
This repo include files used to perform progressive segmented training on CIFAR-10 dataset. Progressive segmented training significantly reduces the computation cost of the trianing and this is demonstrated on a Intel Stratix-10 MX FPGA.

### Enviroments:

Pytorch: 0.4.0 or higher

Python 2.7

CUDA 8.0+

To run CIFAR-10 full dataset fixed point training experiment, run:
	
	python train_CIFAR10_CNN_fixed_point.py -task_division 10,0
	
	
To run CIFAR-10 X+Y (e.g., 9+1) fixed point training,  where X denotes cloud data (the knowledge inheritance), run:
	
	python train_CIFAR10_CNN_fixed_point.py -task_division 9,1 -ne 45

To load a pre-trained model and do importance sampling to generate a mask, which will be used on incremental learning later, run:

	python importance_sampling.py -task_division 9,1
	
	
To perfom the online learning on the rest edge classes with partial weights frozen (for example, 9 classes correspond to 90% weight frozen), run:
	
	python incremental_learning.py 
	
To check whether the mask is correct, run: 

	python check_mask.py

To obtain interdiate results (post-activation value after each layer), run 

	python intermediate_data.py

All the generated files are saved in ./result

You will also need a cifar10 dataset file in .mat. Since the file is too large to upload to Github, please email the authours to request the dataset.
	
	
