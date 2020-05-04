# PSTonFPGA_demo

### Enviroments:

Pytorch: 0.4.0 or higher

Python 2.7

CUDA 8.0+

To run CIFAR-10 full dataset fixed point training experiment, perform:
	
	python train_CIFAR10_CNN_fixed_point.py -task_division 10,0
	
	
To run CIFAR-10 X+Y (e.g., 9+1) fixed point training,  where X denotes cloud data (the knowledge inheritance), perform:
	
	python train_CIFAR10_CNN_fixed_point.py -task_division 9,1 -ne 45

To load a pre-trained model and do importance sampling, perform:

	python importance_sampling.py -task_division 9,1

To load incrementally learn 1 class online on top of a pre-trained model, perform:

	python incremetnal_learning.py -task_division 9,1 -ne 2

To check whether the mask is correct, run:
	python check_mask.py
	
To obtain interdiate results (post-activation value after each layer), run
	python intermediate_date.py
	
All the generated files are saved in ./result

You will also need a cifar10 dataset file in .mat. Since the file is too large to upload to Github, please email the authours to request the dataset.

	
	
