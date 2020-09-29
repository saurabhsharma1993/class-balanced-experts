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
from utils import data_root, plot_curves, seed_everything, min_kldiv, plot_confusion_matrix, find_avg_scores, set_weights
from itertools import chain
import copy
import matplotlib.pyplot as plt
from inspect import signature
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

from dataloader import Threshold_Dataset, data_transforms
from models import DotProduct_Classifier, create_model_resnet10, create_model_resnet152

def train( args, feature_extractor, classifier, device, train_loader, optimizer, scheduler, epoch ):

    feature_extractor.train()
    classifier.train()
    train_loss = 0
    correct = 0

    for batch_idx, (data, target, _, train_count ) in enumerate(train_loader):

        sys.stdout.flush()

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        features, _ = feature_extractor(data)
        output = classifier(features)

        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        train_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
        pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))

    scheduler.step()

    train_loss /= len(train_loader.dataset)
    train_err = 100*correct/len(train_loader.dataset)

    print('\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        train_loss, correct, len(train_loader.dataset),
        100. * correct / len(train_loader.dataset)))

    return train_loss, train_err

def test( args, feature_extractor, classifier, device, test_loader ):

    feature_extractor.eval()
    classifier.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for ind, (data, target, _,  train_count ) in enumerate(test_loader):

            sys.stdout.flush()

            data, target = data.to(device), target.to(device)
            features, _ = feature_extractor(data)
            output = classifier(features)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

            if (ind % args.log_interval == 0):
                print('Test set: Accuracy: {}/{} ({:.2f}%)'.format(
                    correct, (ind + 1) * data.shape[0],
                             100. * correct / ((ind + 1) * data.shape[0])))

    test_loss /= len(test_loader.dataset)
    test_err = 100. * correct / len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    return test_loss, test_err

def main():

    ########################################################################
    ######################## training parameters ###########################
    ########################################################################

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ImageNet', metavar='N', help='dataset to run experiments on')
    parser.add_argument('--batch_size', type=int, default=256, metavar='N', help='input batch size for training (default: 256; note that batch_size 64 gives worse performance for imagenet, so don\'t change this. )') 
    parser.add_argument('--exp', type=str, default='default', metavar='N', help='name of experiment')
    parser.add_argument('--epochs', type=int, default=30, metavar='N', help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.2, metavar='LR', help='learning rate (default: 0.01)')
    parser.add_argument('--weight_decay', type=float, default=5*1e-4, help='weight_decay (default: 1e-5)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.5)')
    parser.add_argument('--step_size', type=float, default=10, metavar='M', help='SGD momentum (default: 0.5)')
    parser.add_argument('--gamma', type=float, default=0.1, metavar='M', help='SGD momentum (default: 0.5)')
    parser.add_argument('--load_model', type=str, default=None, help='model to initialise from')
    parser.add_argument('--caffe', action='store_true', default=False, help='caffe pretrained model')
    parser.add_argument('--test', action='store_true', default=False, help='run in test mode')
    parser.add_argument('--ensemble_inference', action='store_true', default=True, help='run in ensemble inference mode') # testing is always in ensemble inference mode anyways !
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=5021, metavar='S', help='random seed (default: 5021)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',help='how many batches to wait before logging training status')
    parser.add_argument('--stopping_criterion', type=int, default=15, metavar='N',)
    parser.add_argument('--low_threshold', type=int, default=0, metavar='N', )
    parser.add_argument('--high_threshold', type=int, default=100000, metavar='N', )
    parser.add_argument('--open_ratio', type=int, default=1, help='ratio of closed_set to open_set data', )
    parser.add_argument('--picker', type=str, default='generalist', help='dataloader or model picker - experts | generalist : experts uses manyshot, medianshot, lowshot partitioning; \
                                                                    generalist uses the generalist model', )
    parser.add_argument('--num_learnable', type=int, default='-1', help='number of learnable layers : -1 ( all ) | 1 ( only classifier ) | 2 ( classifier and last fc ) | 3 - 6 ( classifier, fc + $ind - 2$ resnet super-blocks ) ')
    parser.add_argument('--scheduler', type=str, default='stepLR', help=' stepLR | cosine lr scheduler')
    parser.add_argument('--max_epochs', type=int, default=None, help='max number of epochs, for cosine lr scheduler')

    args = parser.parse_args()

    print("\n==================Options=================")
    pprint(vars(args), indent=4)
    print("==========================================\n")

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    # torch.manual_seed(args.seed)
    # make everything deterministic
    
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
    if(args.dataset.lower()=='imagenet' and args.test):
        args.batch_size = 64
    elif (args.dataset.lower() == 'imagenet' and not(args.test)):
        args.batch_size = 256
    elif (args.dataset.lower() == 'places' and not(args.test)):
        args.batch_size = 32
    elif (args.dataset.lower() == 'places' and args.test):
        args.batch_size = 8

    ########################################################################
    ######################## load data and pre-trained models ##############
    ########################################################################

    print('Loading train loader.')
    train_loader = torch.utils.data.DataLoader( Threshold_Dataset(root=data_root[args.dataset], orig_txt='./data/{}_LT/{}_LT_train.txt'.format(args.dataset, args.dataset), txt='./data/{}_LT/{}_LT_train.txt'.format(args.dataset, args.dataset),
                                                             low_threshold=args.low_threshold, high_threshold=args.high_threshold, open_ratio= args.open_ratio, transform=data_transforms['train'], picker=args.picker), 
                                                             batch_size = args.batch_size, shuffle = True, **kwargs )
    print('Loading val loader.')
    val_loader = torch.utils.data.DataLoader( Threshold_Dataset(root=data_root[args.dataset], orig_txt='./data/{}_LT/{}_LT_train.txt'.format(args.dataset, args.dataset), txt='./data/{}_LT/{}_LT_val.txt'.format(args.dataset, args.dataset),
                                                            low_threshold=args.low_threshold, high_threshold=args.high_threshold, open_ratio= 1, transform=data_transforms['val'], picker=args.picker), 
                                                            batch_size=args.batch_size, shuffle=True, **kwargs )
    
    num_classes = train_loader.dataset.num_classes + 1 - int(args.picker == 'generalist') # add 1 for the open/dustbin class if not generalist model
    if (args.dataset.lower() == 'imagenet'):
        feature_extractor = create_model_resnet10().to(device)  # use this for imagenet
        args.lr = 1e-1
    else:
        feature_extractor = create_model_resnet152(caffe=True).to(device)  # use this for places. pass caffe=true to load pretrained imagenet model
        args.lr = 1e-2

    print('Learning rate : {:.4f}'.format(args.lr))
    classifier = DotProduct_Classifier(num_classes=num_classes, feat_dim=512).to(device)
    optimizer = torch.optim.SGD(chain(feature_extractor.parameters(),classifier.parameters()),lr=args.lr, momentum= args.momentum, weight_decay=args.weight_decay)
    
    if(args.scheduler == 'stepLR'):
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma) # high learning rate decay, useful or not ?
        print( 'Using StepLR scheduler with params, stepsize : {}, gamma : {}'.format( args.step_size, args.gamma ) )
    elif(args.scheduler == 'cosine'):
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = args.max_epochs ) 
        print( 'Using CosineAnnealingLR scheduler with params, T_max : {}'.format( args.max_epochs ) )
    else:
        raise Exception('Invalid scheduler argument.')

    # load pretrained model
    if (args.load_model is not None):

        if(not args.caffe):

            pretrained_model = torch.load(args.load_model)
            
            weights_feat = pretrained_model['state_dict_best']['feat_model']
            weights_feat = {k: weights_feat['module.' + k] if 'module.' + k in weights_feat else weights_feat[k] for k in feature_extractor.state_dict()}
            feature_extractor.load_state_dict(weights_feat)  # loading feature extractor weights
            
            weights_class = pretrained_model['state_dict_best']['classifier']
            weights_class = {k: weights_class['module.' + k] if 'module.' + k in weights_class else weights_class[k] for k in classifier.state_dict()}

            if (classifier.state_dict()['fc.weight'].shape == weights_class['fc.weight'].shape):
                classifier.load_state_dict(weights_class)  # loading classifier weights if classifiers match
            else:
                print('Classifiers of pretrained model and current model are different with dimensions : ',
                      classifier.state_dict()['fc.weight'].shape, weights_class['fc.weight'].shape)

            print('Loaded pretrained model on entire dataset from epoch : {:d} with acc : {:.4f}'
                  .format(pretrained_model['best_epoch'], pretrained_model['best_acc']))
        else:

            weights_feat = torch.load(args.load_model)
            weights_feat = {k: weights_feat[k] if k in weights_feat else feature_extractor.state_dict()[k] for k in feature_extractor.state_dict()}
            feature_extractor.load_state_dict(weights_feat)  # loading feature extractor weights
            print('Loaded imagenet pretrained model from Caffe.')

    ########################################################################
    ######################## set learnable layers ##########################
    ########################################################################

    if ( args.num_learnable==-1 ):
        print('Learning feature extractor and classifier.')

    elif ( args.num_learnable >= 1 and args.num_learnable <= 6 ):

        if ( args.num_learnable == 1 ):
            
            set_weights( 'feature_extractor', feature_extractor, False )
            set_weights( 'classifier', classifier, True )

        elif ( args.num_learnable == 2 ):

            print( 'Setting feature extractor weights.')
            for ind, (name, layer) in enumerate(feature_extractor.named_children() ):
                if( ind == 9 ):
                    set_weights( name, layer, True )
                else:        
                    set_weights( name, layer, False )
            set_weights( 'classifier', classifier, True )

        else :

            print( 'Setting feature extractor weights.')
            for ind, (name, layer) in enumerate(feature_extractor.named_children() ):
                if( ind >= 10 - args.num_learnable ):
                    set_weights( name, layer, True )
                else:        
                    set_weights( name, layer, False )
            set_weights( 'classifier', classifier, True )

    else:
        raise Exception('Invalid num_learnable layers : {}'.format( args.num_learnable ) )

    ########################################################################
    ######################## training with early stopping ##################
    ########################################################################
    if(not args.test):

        results = vars(args)
        results['train_losses'] = []
        results['train_accuracies'] = []
        results['test_losses'] = []
        results['test_accuracies'] = []
        best_acc, best_epoch = 0,0

        epoch = 1
        while(True): # use for early stopping

            sys.stdout.flush()
            train_loss, train_err = train(args, feature_extractor, classifier, device, train_loader, optimizer, scheduler, epoch )
            test_loss, test_err = test(args, feature_extractor, classifier, device, val_loader) # validation must be done on the validation set !!!!!!!

            results['train_losses'].append(train_loss)
            results['test_losses'].append(test_loss)
            results['train_accuracies'].append(train_err)
            results['test_accuracies'].append(test_err)

            if (test_err > best_acc):
                best_acc = test_err
                best_epoch = epoch
                results['best_acc'], results['best_epoch'] = best_acc, best_epoch

                # save best model
                best_model_weights = {}
                best_model_weights['feat_model'] = copy.deepcopy(feature_extractor.state_dict())
                best_model_weights['classifier'] = copy.deepcopy(classifier.state_dict())
                model_states = {'epoch': epoch,
                                'best_epoch': best_epoch,
                                'state_dict_best': best_model_weights,
                                'best_acc': best_acc,}
                torch.save(model_states, os.path.join(exp_dir,"best_model.pt"))

            elif (epoch > best_epoch + args.stopping_criterion):
                print('Best model obtained. Error : ', best_acc)
                plot_curves(results,exp_dir) # plot
                break

            elif ( args.scheduler == 'cosine' and epoch == args.max_epochs):
                print('Best model obtained. Error : ', best_acc)
                plot_curves(results,exp_dir) # plot
                break

            savepath = os.path.join(exp_dir, 'results.pickle')
            with open(savepath, 'wb') as f:
                pickle.dump(results, f)
            plot_curves(results,exp_dir) # plot
            epoch = epoch + 1

if __name__ == '__main__':
    main()
