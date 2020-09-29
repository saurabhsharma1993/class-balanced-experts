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
from utils import plot_curves, seed_everything
from itertools import chain
import copy
import matplotlib.pyplot as plt
from inspect import signature
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

from dataloader import Threshold_Dataset, data_transforms
from models import create_model_resnet10
from models import create_model_resnet152
from models import DotProduct_Classifier

data_root = {'ImageNet': '/BS/max-interpretability/nobackup/data/imagenet',
             'Places': '/BS/databases/places2'}

def gen_features_and_probs( args, feature_extractor, classifier, device, loader ):

    feature_extractor.eval()
    classifier.eval()

    total_features = torch.empty((0, 512)).to(device)
    # total_probs = torch.empty((0, classifier.num_classes)).to(device)
    total_logprobs = torch.empty((0, classifier.num_classes)).to(device)
    total_labels = torch.empty((0), dtype=torch.long).to(device)

    print('Generating features : {}'.format(len(loader.dataset)))

    with torch.no_grad():
        for ind, (data, target, _,  train_count ) in enumerate(loader):

            # print('Doing something.')
            sys.stdout.flush()

            data, target = data.to(device), target.to(device)
            features, _ = feature_extractor(data)
            # probs = F.softmax( classifier(features), dim=1 ) # classifier returns logprobs
            logprobs = classifier(features)  # for specialist use logprobs for logit tuning if required

            # for pr curve analysis        
            total_features = torch.cat((total_features, features))
            # total_probs = torch.cat((total_probs, probs))
            total_logprobs = torch.cat((total_logprobs, logprobs))
            total_labels = torch.cat((total_labels, target))

            if (ind % args.log_interval == 0):
                print('Progress: {:.2f}%'.format( 100 * (ind + 1) * data.shape[0] / len(loader.dataset) ) )

            # break

    # return total_features, total_probs, total_labels
    return total_features, total_logprobs, total_labels

def main():

	########################################################################
    ######################## training parameters ###########################
    ########################################################################

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ImageNet', metavar='N', help='dataset to run experiments on')
    parser.add_argument('--batch_size', type=int, default=256, metavar='N', help='input batch size for training (default: 256; note that batch_size 64 gives worse performance for imagenet, so don\'t change this. )') 
    parser.add_argument('--exp', type=str, default='default', metavar='N', help='name of experiment')
    parser.add_argument('--epochs', type=int, default=30, metavar='N', help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR', help='learning rate (default: 0.01)')
    parser.add_argument('--weight_decay', type=float, default=5*1e-4, help='weight_decay (default: 1e-5)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.5)')
    parser.add_argument('--step_size', type=float, default=10, metavar='M', help='SGD momentum (default: 0.5)')
    parser.add_argument('--gamma', type=float, default=0.1, metavar='M', help='SGD momentum (default: 0.5)')
    parser.add_argument('--load_model', type=str, default=None, help='model to initialise from')
    parser.add_argument('--model_name', type=str, default=None, help='name of model : manyshot | mediumshot | lowshot | general')
    parser.add_argument('--caffe', action='store_true', default=False, help='caffe pretrained model')
    parser.add_argument('--test', action='store_true', default=False, help='run in test mode')
    parser.add_argument('--ensemble_inference', action='store_true', default=True, help='run in ensemble inference mode') # testing is always in ensemble inference mode anyways !
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=5021, metavar='S', help='random seed (default: 5021)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',help='how many batches to wait before logging training status')
    parser.add_argument('--stopping_criterion', type=int, default=30, metavar='N',)
    parser.add_argument('--low_threshold', type=int, default=0, metavar='N', )
    parser.add_argument('--high_threshold', type=int, default=100000, metavar='N', )
    parser.add_argument('--open_ratio', type=int, default=1, help='ratio of closed_set to open_set data', )
    parser.add_argument('--picker', type=str, default='simple', help='dataloader or model picker - systematic | simple | generalist : simple uses manyshot, mediumshot, lowshot partitioning; \
                                                                    systematic uses overlapping classwise equally distributed splits; generalist uses the generalist model', )
    parser.add_argument('--index', type=int, default='-1', help='index of specialist, for systematic split : 0:num_block, or for parallel ensemble inference')
    parser.add_argument('--ei_mode', type=str, default='parallel', help='ensemble inference mode : sequential | parallel ')
    parser.add_argument('--num_learnable', type=int, default='-1', help='number of learnable layers : -1 ( all ) | 1 ( only classifier ) | 2 ( classifier and last fc ) | 3 - 6 ( classifier, fc + $ind - 2$ resnet super-blocks ) ')
    parser.add_argument('--sampling', type=str, default=None, help=' sampling : over/under ')
    parser.add_argument('--for_OEmodel', action='store_true', default=False, help='run for experts with oe training' )

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

    # print('Loading train loader.')
    # train_loader = torch.utils.data.DataLoader( Threshold_Dataset(root=data_root[args.dataset], orig_txt='./data/{}_LT/{}_LT_train.txt'.format(args.dataset, args.dataset), txt='./data/{}_LT/{}_LT_train.txt'.format(args.dataset, args.dataset ),
    #                                                          use_open = False, transform=data_transforms['train'], picker='experts', sampling=args.sampling ), batch_size = args.batch_size, shuffle = False, **kwargs )
    #
    # print('Loading val loader.')
    # val_loader = torch.utils.data.DataLoader( Threshold_Dataset(root=data_root[args.dataset], orig_txt='./data/{}_LT/{}_LT_train.txt'.format(args.dataset, args.dataset), txt='./data/{}_LT/{}_LT_val.txt'.format(args.dataset, args.dataset),
    #                   use_open=False, transform=data_transforms['val'], picker='experts', sampling=args.sampling), batch_size=args.batch_size, shuffle=False, **kwargs )

    # print('Loading test loader.')
    # test_loader = torch.utils.data.DataLoader( Threshold_Dataset(root=data_root[args.dataset], orig_txt='./data/{}_LT/{}_LT_train.txt'.format(args.dataset, args.dataset), txt='./data/{}_LT/{}_LT_test.txt'.format(args.dataset, args.dataset),
    #                   use_open=False, transform=data_transforms['test'], picker='experts', sampling=args.sampling), batch_size=args.batch_size, shuffle=False, **kwargs )

    # using many/medium/few shot loaders for val sets for reporting metrics - barely used this ever

    # print('Loading val loader many shot.')
    # val_loader_manyshot = torch.utils.data.DataLoader( Threshold_Dataset(root=data_root[args.dataset], orig_txt='./data/{}_LT/{}_LT_train.txt'.format(args.dataset, args.dataset), txt='./data/{}_LT/{}_LT_val.txt'.format(args.dataset, args.dataset),
    #                                                         low_threshold = 100,
    #                                                        use_open = False, transform=data_transforms['val'], picker='experts' ), batch_size=args.batch_size, shuffle=False, **kwargs )
    
    # print('Loading val loader medium shot.')
    # val_loader_mediumshot = torch.utils.data.DataLoader( Threshold_Dataset(root=data_root[args.dataset], orig_txt='./data/{}_LT/{}_LT_train.txt'.format(args.dataset, args.dataset), txt='./data/{}_LT/{}_LT_val.txt'.format(args.dataset, args.dataset),
    #                                                         low_threshold = 20, high_threshold = 100,
    #                                                        use_open = False, transform=data_transforms['val'], picker='experts' ), batch_size=args.batch_size, shuffle=False, **kwargs )
    
    # print('Loading val loader few shot.')
    # val_loader_fewshot = torch.utils.data.DataLoader( Threshold_Dataset(root=data_root[args.dataset], orig_txt='./data/{}_LT/{}_LT_train.txt'.format(args.dataset, args.dataset), txt='./data/{}_LT/{}_LT_val.txt'.format(args.dataset, args.dataset),
    #                                                         high_threshold = 20,
    #                                                        use_open = False, transform=data_transforms['val'], picker='experts' ), batch_size=args.batch_size, shuffle=False, **kwargs )
    
    
    # using many/medium/few shot loaders for test sets for reporting metrics

    print('Loading test loader many shot.')
    test_loader_manyshot = torch.utils.data.DataLoader( Threshold_Dataset(root=data_root[args.dataset], orig_txt='./data/{}_LT/{}_LT_train.txt'.format(args.dataset, args.dataset), txt='./data/{}_LT/{}_LT_test.txt'.format(args.dataset, args.dataset),
                                                                       low_threshold = 100,
                                                                       use_open = False, transform=data_transforms['test'], picker='experts' ), batch_size=args.batch_size, shuffle=False, **kwargs )

    print('Loading test loader medium shot.')
    test_loader_mediumshot = torch.utils.data.DataLoader( Threshold_Dataset(root=data_root[args.dataset], orig_txt='./data/{}_LT/{}_LT_train.txt'.format(args.dataset, args.dataset), txt='./data/{}_LT/{}_LT_test.txt'.format(args.dataset, args.dataset),
                                                                       low_threshold = 20, high_threshold = 100,
                                                                       use_open = False, transform=data_transforms['test'], picker='experts' ), batch_size=args.batch_size, shuffle=False, **kwargs )

    print('Loading test loader low shot.')
    test_loader_fewshot = torch.utils.data.DataLoader( Threshold_Dataset(root=data_root[args.dataset], orig_txt='./data/{}_LT/{}_LT_train.txt'.format(args.dataset, args.dataset), txt='./data/{}_LT/{}_LT_test.txt'.format(args.dataset, args.dataset),
                                                                       high_threshold= 20,
                                                                       use_open = False, transform=data_transforms['test'], picker='experts' ), batch_size=args.batch_size, shuffle=False, **kwargs )

    tot_num_classes = 1000 if args.dataset.lower() == 'imagenet' else 365

    if(args.picker=='experts' or args.picker=='generalist'):

        if(args.model_name=='manyshot'):
            class_mask = test_loader_manyshot.dataset.class_mask
        elif (args.model_name == 'mediumshot'):
            class_mask = test_loader_mediumshot.dataset.class_mask
        elif (args.model_name == 'lowshot'):
            class_mask = test_loader_fewshot.dataset.class_mask
        elif (args.model_name == 'general'):
            class_mask = torch.BoolTensor( [ True for i in range(tot_num_classes) ] )

    if (args.dataset.lower() == 'imagenet'):
        feature_extractor = create_model(use_selfatt=False, use_fc=True).to(device)  # use this for imagenet
        if(args.model_name != 'general'):    
            classifier = DotProduct_Classifier(num_classes=int(class_mask.sum() ), feat_dim=512).to(device)            # for experts with oe training
        else:
            classifier = DotProduct_Classifier(num_classes=1000, feat_dim=512).to(device)
    else:
        feature_extractor = create_model_resnet152(use_selfatt=False, use_fc=True, caffe=True).to(device)  # use this for places. pass caffe=true to load pretrained imagenet model
        if (args.model_name != 'general'):
            classifier = DotProduct_Classifier(num_classes=int( class_mask.sum() ), feat_dim=512).to(device)           # for experts with oe training
        else:
            classifier = DotProduct_Classifier(num_classes=365, feat_dim=512).to(device)

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
    # ######################## generate features for train set ############################
    # #####################################################################################

    # results = {}
    # train_features, train_probs, train_labels = gen_features_and_probs(args, feature_extractor, classifier, device, train_loader)
    # results['features'] = train_features
    # # results['probs'] = train_probs  # for generalist models
    # results['logits'] = train_probs  # for specialists we use logits, logprobs above is legacy
    # results['labels'] = train_labels
    # results['class_mask'] = class_mask
    # torch.save(results, os.path.join(exp_dir, 'results_train_{}.pickle'.format(args.model_name)))

    # #####################################################################################
    # ######################## generate features for val set ##############################
    # #####################################################################################

    # results = {}
    # val_features, val_probs, val_labels = gen_features_and_probs(args, feature_extractor, classifier, device, val_loader)
    # results['features'] = val_features
    # # results['probs'] = val_probs          # for generalist models
    # results['logits'] = val_probs           # for specialists we use logits, logprobs above is legacy
    # results['labels'] = val_labels
    # results['class_mask'] = class_mask
    # torch.save(results, os.path.join(exp_dir, 'results_val_{}.pickle'.format(args.model_name)))

    # #####################################################################################
    # ######################## generate features for test set #############################
    # #####################################################################################

    # # aligned ( screws up labels for specialists ! ) : fix using class_mask. Or use labels already stored to disk in prior experiments.
    results = {}
    test_features_manyshot, test_probs_manyshot, test_labels_manyshot = gen_features_and_probs(args, feature_extractor, classifier, device, test_loader_manyshot)
    test_features_mediumshot, test_probs_mediumshot, test_labels_mediumshot = gen_features_and_probs(args, feature_extractor, classifier, device, test_loader_mediumshot)
    test_features_fewshot, test_probs_fewshot, test_labels_fewshot = gen_features_and_probs(args, feature_extractor, classifier, device, test_loader_fewshot)

    results['features'] = torch.cat((test_features_manyshot, test_features_mediumshot, test_features_fewshot), dim=0)
    # results['probs'] = torch.cat((test_probs_manyshot, test_probs_mediumshot, test_probs_fewshot), dim=0)   # for generalist models
    results['logits'] = torch.cat((test_probs_manyshot, test_probs_mediumshot, test_probs_fewshot), dim=0)      # for specialists we use logits, logprobs above is legacy
    results['labels'] = torch.cat((test_labels_manyshot, test_labels_mediumshot, test_labels_fewshot), dim=0)
    results['class_mask'] = class_mask
    torch.save(results, os.path.join(exp_dir, 'results_test_aligned_{}.pickle'.format(args.model_name)))

if __name__ == '__main__':
    main()

