import os

# Training Hyperparameters
NUM_CLASSES         = 200

# 64, 128, 256, 512, 1024, 2048
BATCH_SIZE          = 128 
VAL_EVERY_N_EPOCH   = 1

NUM_EPOCHS          = 40

# OPTIMIZER_PARAMS    
# SGD {'type: 'SGD', 'lr': [1e-3, 5e-3, 1e-2], 'momentum': [0.8, 0.9, 0.95]}
# Adam {'type: 'SGD', 'lr': [1e-3, 5e-3, 1e-2], 'betas': [(0.8, 0.99), (0.85, 0.995), (0.9, 0.999)]}
# RMSProp {'type: 'RMSprop', 'lr': [1e-3, 5e-3, 1e-2], 'alpha': [0.9, 0.95, 0.99]}
# Adagrad {'type': 'Adagrad', 'lr': [1e-3, 5e-3, 1e-2], initial_accumulator_value: [0.01, 0.1, 1.0]}
OPTIMIZER_PARAMS    = {'type': 'SGD', 'lr': 0.01, 'momentum': 0.8}

# SCHEDULER_PARAMS   
# StepLR {'type': 'StepLR', 'step_size': [10, 15], 'gamma': [0.01, 0.1, 0.2, 0.5]} 
# MultiStepLR {'type': 'MultiStepLR', 'milestones': [[10, 20], [15, 30], [20, 30], 'gamma': [0.1, 0.2, 0.5]}
# ExponentialLR {'type': 'ExponentialLR', 'gamma': [0.9, 0.95, 0.97, 0.99, 0.999]}
SCHEDULER_PARAMS    = {'type': 'MultiStepLR', 'milestones': [20, 30], 'gamma': 0.5}

# Dataset
DATASET_ROOT_PATH   = 'datasets/'
NUM_WORKERS         = 8

# Augmentation
IMAGE_ROTATION      = 20
IMAGE_FLIP_PROB     = 0.5
IMAGE_NUM_CROPS     = 64
IMAGE_PAD_CROPS     = 4
IMAGE_MEAN          = [0.4802, 0.4481, 0.3975]
IMAGE_STD           = [0.2302, 0.2265, 0.2262]

# MODEL_NAME
# alexnet, vgg11/13/16/19, vgg11/13/16/19_bn, resnet18/34/50/101/152, densenet121/161/169/201, efficientnet_b0 ~ 7,  efficientnet_v2_s/m/l, swin_t/s/b, swin_v2_t/s
MODEL_NAME          = 'MyNetwork'
# MODEL_NAME          = 'efficientnet_b7'

# Compute related
ACCELERATOR         = 'gpu'
DEVICES             = [7]
PRECISION_STR       = '32-true'

# Logging
WANDB_PROJECT       = 'aue8088-pa1'
WANDB_ENTITY        = os.environ.get('WANDB_ENTITY')
WANDB_SAVE_DIR      = 'wandb/'
WANDB_IMG_LOG_FREQ  = 50

# Basic Setting
# WANDB_NAME          = f'{MODEL_NAME}-B{BATCH_SIZE}-E{NUM_EPOCHS}-{OPTIMIZER_PARAMS["type"]}'
# WANDB_NAME         += f'-{SCHEDULER_PARAMS["type"]}{OPTIMIZER_PARAMS["lr"]:.1E}'

# Optimizer
# WANDB_NAME          = f'{MODEL_NAME}-{OPTIMIZER_PARAMS["type"]}-LR{OPTIMIZER_PARAMS["lr"]:.1E}'
# WANDB_NAME         += f'-M{OPTIMIZER_PARAMS["momentum"]}' # SGD
# WANDB_NAME         += f'-B{OPTIMIZER_PARAMS["betas"]}' # Adam
# WANDB_NAME         += f'-A{OPTIMIZER_PARAMS["alpha"]}' # RMSprop
# WANDB_NAME         += f'-IAV{OPTIMIZER_PARAMS["initial_accumulator_value"]}' # Adagrad

# Scheduler
WANDB_NAME          = f'{MODEL_NAME}-{SCHEDULER_PARAMS["type"]}'
# WANDB_NAME         += f'-S{SCHEDULER_PARAMS["step_size"]:.1E}-G{SCHEDULER_PARAMS["gamma"]}' # StepLR
# WANDB_NAME         += f'-M{SCHEDULER_PARAMS["milestones"]}-G{SCHEDULER_PARAMS["gamma"]}' # MultiStepLR
# WANDB_NAME         += f'-G{SCHEDULER_PARAMS["gamma"]}' # ExponentialLR