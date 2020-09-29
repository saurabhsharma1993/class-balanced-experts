import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import torch
from sklearn.metrics import f1_score
import importlib
import pdb
import os
import random
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

data_root = {'ImageNet': '/BS/max-interpretability/nobackup/data/imagenet',
             'Places': '/BS/databases/places2'}

def source_import(file_path):
    """This function imports python module directly from source code using importlib"""
    spec = importlib.util.spec_from_file_location('', file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def print_write(print_str, log_file):
    print(*print_str)
    with open(log_file, 'a') as f:
        print(*print_str, file=f)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def init_weights(model, weights_path, caffe=False, classifier=False):  
    """Initialize weights"""
    print('Pretrained %s weights path: %s' % ('classifier' if classifier else 'feature model',
                                              weights_path))    
    weights = torch.load(weights_path)   
    if not classifier:
        if caffe:
            weights = {k: weights[k] if k in weights else model.state_dict()[k] 
                       for k in model.state_dict()}
        else:
            weights = weights['state_dict_best']['feat_model']
            weights = {k: weights['module.' + k] if 'module.' + k in weights else model.state_dict()[k] 
                       for k in model.state_dict()}
    else:      
        weights = weights['state_dict_best']['classifier']
        weights = {k: weights['module.fc.' + k] if 'module.fc.' + k in weights else model.state_dict()[k] 
                   for k in model.state_dict()}
    model.load_state_dict(weights)   
    return model

def plot_curves(results, save_dir):

    # plot cross entropy losses curve
    fig = plt.figure()
    ax = fig.add_subplot('111')

    ax.plot(np.arange(1, len(results['train_losses'])+1), results['train_losses'], color='red', linewidth=1, label='Train_loss')
    if('test_losses' in results):
        ax.plot(np.arange(1, len(results['train_losses'])+1), results['test_losses'], color='blue', linewidth=1, label='Val_loss')

    ax.set_ylabel('Cross entropy loss per label')
    ax.set_xlabel('Epochs')
    plt.grid(b=True, which='major')
    plt.legend(loc='upper right')
    fig.set_dpi(500)
    path = os.path.join(save_dir, 'training curve')
    fig.savefig(path)

    # plot accuracies
    fig = plt.figure()
    ax = fig.add_subplot('111')

    ax.plot(np.arange(1, len(results['train_losses'])+1), results['train_accuracies'], color='red', linewidth=1, label='Train_accuracies')
    if('test_accuracies' in results):
        ax.plot(np.arange(1, len(results['train_losses'])+1), results['test_accuracies'], color='blue', linewidth=1, label='Val_accuracies') 
    best_acc, best_epoch = results['best_acc'], results['best_epoch']
    ax.annotate('{:.4f}'.format(best_acc), (best_epoch, best_acc))

    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Epochs')
    plt.grid(b=True, which='major')
    plt.legend(loc='lower right')

    fig.set_dpi(500)
    path = os.path.join(save_dir, 'accuracy curve')
    fig.savefig(path)

    plt.close('all')

def seed_everything(seed=5021):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True

def set_weights(module_name, module, state ):

    tot_params = 0
    
    for param in module.parameters():
        param.requires_grad = state 
        tot_params += param.numel()
    
    if ( state == False ):
        print('Freezing {} weights. Num of parameters  : {} '.format( module_name, tot_params ) )
    else :
        print('Learning {} weights. Num of parameters  : {} '.format( module_name, tot_params ) )
