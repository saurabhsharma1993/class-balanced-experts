from __future__ import print_function
import argparse
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import pickle
import os
from pprint import pprint
from utils import data_root, plot_curves, seed_everything
from itertools import chain
import copy
import matplotlib.pyplot as plt
from inspect import signature
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

from dataloader import Threshold_Dataset, data_transforms
from models import create_model_resnet10, create_model_resnet152, DotProduct_Classifier

def gen_logits( args, feature_extractor, classifier, device, loader ):

    feature_extractor.eval()
    classifier.eval()

    total_features = torch.empty((0, 512)).to(device)
    total_logits = torch.empty((0, classifier.num_classes)).to(device)
    total_labels = torch.empty((0), dtype=torch.long).to(device)

    print('Generating logits : {}'.format(len(loader.dataset)))

    with torch.no_grad():
        for ind, (data, target, _,  train_count ) in enumerate(loader):

            # print('Doing something.')
            sys.stdout.flush()

            data, target = data.to(device), target.to(device)
            features, _ = feature_extractor(data)
            logits = classifier(features)  

            total_logits = torch.cat((total_logits, logits))
            total_labels = torch.cat((total_labels, target))

            if (ind % args.log_interval == 0):
                print('Progress: {:.2f}%'.format( 100 * (ind + 1) * data.shape[0] / len(loader.dataset) ) )

            # break

    return total_logits, total_labels

def main():

	########################################################################
    ######################## training parameters ###########################
    ########################################################################

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ImageNet', metavar='N', help='dataset to run experiments on')
    parser.add_argument('--batch_size', type=int, default=256, metavar='N', help='input batch size for training (default: 256; note that batch_size 64 gives worse performance for imagenet, so don\'t change this. )') 
    parser.add_argument('--exp', type=str, default='default', metavar='N', help='name of experiment')
    parser.add_argument('--epochs', type=int, default=30, metavar='N', help='number of epochs to train (default: 10)')
    parser.add_argument('--load_model', type=str, default=None, help='model to initialise from')
    parser.add_argument('--model_name', type=str, default=None, help='name of model : manyshot | mediumshot | lowshot | general')
    parser.add_argument('--data_split', type=str, default=None, help='train | val | test_aligned')
    parser.add_argument('--caffe', action='store_true', default=False, help='caffe pretrained model')
    parser.add_argument('--test', action='store_true', default=False, help='run in test mode')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=5021, metavar='S', help='random seed (default: 5021)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',help='how many batches to wait before logging training status')
    parser.add_argument('--stopping_criterion', type=int, default=30, metavar='N',)
    parser.add_argument('--low_threshold', type=int, default=0, metavar='N', )
    parser.add_argument('--high_threshold', type=int, default=100000, metavar='N', )
    parser.add_argument('--picker', type=str, default='generalist', help='dataloader or model picker - experts | generalist : experts uses manyshot, medianshot, lowshot partitioning; \
                                                                    generalist uses the generalist model', )

    args = parser.parse_args()

    print("\n==================Options=================")
    pprint(vars(args), indent=4)
    print("==========================================\n")

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    # torch.manual_seed(args.seed)
    # make everything deterministic, reproducible
    if (args.seed is not None):
        print('Seeding everything with seed {}.'.format(args.seed))
        seed_everything(args.seed)
    else:
        print('Note : Seed is random.')

    device = torch.device("cuda" if use_cuda else "cpu")

    exp_dir = os.path.join('checkpoint', args.exp)
    if not os.path.isdir(exp_dir):
        os.makedirs(exp_dir)

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # batch size settings : note that these are important for memory and performance reasons
    if(args.dataset.lower()=='imagenet' ):
        args.batch_size = 256
    elif (args.dataset.lower() == 'places' ):
        args.batch_size = 32

    ########################################################################
    ######################## load data and pre-trained models ##############
    ########################################################################

    # all loaders must have shuffle false so that data is aligned across different models !!!!

    print('Loading train loader.')
    train_loader = torch.utils.data.DataLoader( Threshold_Dataset(root=data_root[args.dataset], orig_txt='./data/{}_LT/{}_LT_train.txt'.format(args.dataset, args.dataset), txt='./data/{}_LT/{}_LT_train.txt'.format(args.dataset, args.dataset ),
                                                             use_open = False, transform=data_transforms['train'], picker='experts', sampling=args.sampling ), batch_size = args.batch_size, shuffle = False, **kwargs )
    
    print('Loading val loader.')
    val_loader = torch.utils.data.DataLoader( Threshold_Dataset(root=data_root[args.dataset], orig_txt='./data/{}_LT/{}_LT_train.txt'.format(args.dataset, args.dataset), txt='./data/{}_LT/{}_LT_val.txt'.format(args.dataset, args.dataset),
                      use_open=False, transform=data_transforms['val'], picker='experts', sampling=args.sampling), batch_size=args.batch_size, shuffle=False, **kwargs )

    print('Loading test loader.')
    test_loader = torch.utils.data.DataLoader( Threshold_Dataset(root=data_root[args.dataset], orig_txt='./data/{}_LT/{}_LT_train.txt'.format(args.dataset, args.dataset), txt='./data/{}_LT/{}_LT_test.txt'.format(args.dataset, args.dataset),
                      use_open=False, transform=data_transforms['test'], picker='experts', sampling=args.sampling), batch_size=args.batch_size, shuffle=False, **kwargs )
  
    # using many/medium/few shot loaders for test sets for reporting metrics

    print('Loading test loader many shot.')
    test_loader_manyshot = torch.utils.data.DataLoader( Threshold_Dataset(root=data_root[args.dataset], orig_txt='./data/{}_LT/{}_LT_train.txt'.format(args.dataset, args.dataset), txt='./data/{}_LT/{}_LT_test.txt'.format(args.dataset, args.dataset),
                                                                       low_threshold = 100, use_open = False, transform=data_transforms['test'], picker='experts' ), batch_size=args.batch_size, shuffle=False, **kwargs )

    print('Loading test loader medium shot.')
    test_loader_mediumshot = torch.utils.data.DataLoader( Threshold_Dataset(root=data_root[args.dataset], orig_txt='./data/{}_LT/{}_LT_train.txt'.format(args.dataset, args.dataset), txt='./data/{}_LT/{}_LT_test.txt'.format(args.dataset, args.dataset),
                                                                       low_threshold = 20, high_threshold = 100, use_open = False, transform=data_transforms['test'], picker='experts' ), batch_size=args.batch_size, shuffle=False, **kwargs )

    print('Loading test loader low shot.')
    test_loader_fewshot = torch.utils.data.DataLoader( Threshold_Dataset(root=data_root[args.dataset], orig_txt='./data/{}_LT/{}_LT_train.txt'.format(args.dataset, args.dataset), txt='./data/{}_LT/{}_LT_test.txt'.format(args.dataset, args.dataset),
                                                                       high_threshold= 20, use_open = False, transform=data_transforms['test'], picker='experts' ), batch_size=args.batch_size, shuffle=False, **kwargs )

    tot_num_classes = 1000 if args.dataset.lower() == 'imagenet' else 365

    if(args.model_name=='manyshot'):
        class_mask = test_loader_manyshot.dataset.class_mask
    elif (args.model_name == 'mediumshot'):
        class_mask = test_loader_mediumshot.dataset.class_mask
    elif (args.model_name == 'lowshot'):
        class_mask = test_loader_fewshot.dataset.class_mask
    elif (args.model_name == 'general'):
        class_mask = torch.BoolTensor( [ True for i in range(tot_num_classes) ] )

    if (args.dataset.lower() == 'imagenet'):
        feature_extractor = create_model_resnet10().to(device)  # use this for imagenet
        if(args.model_name != 'general'):    
            classifier = DotProduct_Classifier(num_classes=int(class_mask.sum() + 1), feat_dim=512, use_logits=True).to(device)            # for experts with oe training
        else:
            classifier = DotProduct_Classifier(num_classes=1000, feat_dim=512, use_logits=True).to(device)
    else:
        feature_extractor = create_model_resnet152(caffe=True).to(device)  # use this for places. pass caffe=true to load pretrained imagenet model
        if (args.model_name != 'general'):
            classifier = DotProduct_Classifier(num_classes=int( class_mask.sum() + 1 ), feat_dim=512, use_logits=True).to(device)           # for experts with oe training
        else:
            classifier = DotProduct_Classifier(num_classes=365, feat_dim=512, use_logits=True).to(device)

    # load pretrained model
    if (args.load_model is not None):

        if(not args.caffe):

            pretrained_model = torch.load(args.load_model)
            
            weights_feat = pretrained_model['state_dict_best']['feat_model']
            weights_feat = {k: weights_feat['module.' + k] if 'module.' + k in weights_feat else weights_feat[k] for k in feature_extractor.state_dict()}
            feature_extractor.load_state_dict(weights_feat)  # loading feature extractor weights

            weights_class = pretrained_model['state_dict_best']['classifier']
            if (args.model_name != 'general'):
                weights_class = {k: weights_class['module.' + k] if 'module.' + k in weights_class else weights_class[k] for k in classifier.state_dict()}
            else:
                weights_class = {k: weights_class['module.' + k][:-1] if 'module.' + k in weights_class else weights_class[k][:-1] for k in classifier.state_dict()}  # due to a bug, there was an extra neuron for the open class even in the generalist, so must slice it away
            classifier.load_state_dict(weights_class)
            
    # #####################################################################################
    # ######################## generate logit scores           ############################
    # #####################################################################################
    
    results = {}
    if( args.data_split=='train' ):
        logits, labels = gen_logits(args, feature_extractor, classifier, device, train_loader))
    if( args.data_split=='val' ):
        logits, labels = gen_logits(args, feature_extractor, classifier, device, val_loader))
    if( args.data_split=='test_aligned' ):
        manyshot_logits, manyshot_labels = gen_logits(args, feature_extractor, classifier, device, test_loader))
        mediumshot_logits, mediumshot_labels = gen_logits(args, feature_extractor, classifier, device, test_loader))
        fewshot_logits, fewshot_labels = gen_logits(args, feature_extractor, classifier, device, test_loader))
        logits = torch.cat((manyshot_logits, mediumshot_logits, fewshot_logits), dim=0)     
        labels = torch.cat((manyshot_labels, mediumshot_labels, fewshot_labels), dim=0)
    else:
        raise Exception('Invalid data split.')

    results['logits'] = logits
    results['labels'] = labels
    results['class_mask'] = class_mask
    torch.save(results, os.path.join(exp_dir, 'results_{}_{}.pickle'.format(args.data_split, args.model_name)))

if __name__ == '__main__':
    main()

