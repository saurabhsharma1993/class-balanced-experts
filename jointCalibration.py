import numpy as np
import torch
from torch import nn, optim
import torch.nn as nn
from torch.nn import functional as F
import argparse
from dataloader import Calibration_Dataset
from pprint import pprint
from utils import plot_curves, seed_everything, weights_init
import os
import sys
import copy
import pickle
from models import CalibrateExperts
import matplotlib
import matplotlib.pyplot as plt


def train(args, model, device, dataloader, optimizer, scheduler, epoch):

    model.train()
    total_loss = 0
    correct = 0
    criterion = nn.NLLLoss().cuda()

    features = torch.from_numpy(dataloader.dataset.features).float()
    labels = torch.from_numpy(dataloader.dataset.labels).long()
    batch_size = dataloader.batch_size

    for batch_idx, (data, target) in enumerate(dataloader):

        sys.stdout.flush()

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)

        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += F.nll_loss(output, target, reduction='sum').item()       	           # sum up batch loss
        pred = output.max(1, keepdim=True)[1]  									           # get the index of the max logit score
        correct += pred.eq(target.view_as(pred)).sum().item()

        if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(dataloader.dataset),
                           100. * batch_idx / (len(dataloader)), loss.item()))
              
    scheduler.step()

    total_loss /= len(dataloader.dataset)
    acc = 100*correct / len(dataloader.dataset)

    print('\nTrain set: Loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format( total_loss, correct, len(dataloader.dataset), 100. * correct / len(dataloader.dataset)))

    return total_loss, acc

def test(args, model, device, dataloader):

    model.eval()
    total_loss = 0
    correct = 0
    num_classes = (dataloader.dataset.labels).max() + 1
    criterion = nn.NLLLoss().cuda()
    total_preds = torch.empty((0), dtype=torch.long).to(device)

    for batch_idx, (data, target) in enumerate(dataloader):

        sys.stdout.flush()

        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = criterion(output, target)

        total_loss += F.nll_loss(output, target, reduction='sum').item()       	           # sum up batch loss
        pred = output.max(1, keepdim=True)[1]  									           # get the index of the max logit score
        total_preds = torch.cat((total_preds, pred))
        correct += pred.eq(target.view_as(pred)).sum().item()

        if batch_idx % args.log_interval == 0:
            print('Test set: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(batch_idx * len(data), len(dataloader.dataset),100. * batch_idx / len(dataloader), loss.item()))

    total_loss /= len(dataloader.dataset)
    acc = 100*correct / len(dataloader.dataset)

    print('\nTest set: Loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        total_loss, correct, len(dataloader.dataset), 100. * correct / len(dataloader.dataset)))

    return total_loss, acc, total_preds

def main():

    ########################################################################
    ######################## training parameters ###########################
    ########################################################################

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ImageNet', metavar='N', help='dataset to run experiments on')
    parser.add_argument('--batch_size', type=int, default=256, metavar='N', help='input batch size for training (default: 256; note that batch_size 64 gives worse performance for imagenet, so don\'t change this. )')
    parser.add_argument('--exp', type=str, default='default', metavar='N', help='name of experiment')
    parser.add_argument('--logits_exp', type=str, default='default', metavar='N', help='name of experiment containing logits')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=5021, metavar='S', help='random seed (default: 5021)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR', help='learning rate (default: 0.01)')
    parser.add_argument('--weight_decay', type=float, default=5*1e-4, help='weight_decay (default: 1e-5)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.5)')
    parser.add_argument('--step_size', type=float, default=30, metavar='M', help='SGD momentum (default: 0.5)')
    parser.add_argument('--gamma', type=float, default=0.1, metavar='M', help='SGD momentum (default: 0.5)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',help='how many batches to wait before logging training status')
    parser.add_argument('--stopping_criterion', type=int, default=30, metavar='N',)
    parser.add_argument('--test', action='store_true', default=False, help='test mode')
    parser.add_argument('--load_model', type=str, default=None, help='model to initialise from')

    args = parser.parse_args()

    print("\n==================Options=================")
    pprint(vars(args), indent=4)
    print("==========================================\n")

    use_cuda = not args.no_cuda and torch.cuda.is_available()
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
    dataset = args.dataset
    num_classes = 1000 if dataset.lower() == 'imagenet' else 365

    ########################################################################
    ########################         load data  		####################
    ########################################################################
    
    datadir = './checkpoint/{}'.format(args.logits_exp)

    if (not args.test):
        data_manyshot = torch.load('{}/results_val_manyshot.pickle'.format(datadir))        # for experts with reject option
        data_mediumshot = torch.load('{}/results_val_mediumshot.pickle'.format(datadir))    # for experts with reject option
        data_fewshot = torch.load('{}/results_val_fewshot.pickle'.format(datadir))          # for experts with reject option

    else:
        data_manyshot = torch.load('{}/results_test_aligned_manyshot.pickle'.format(datadir))        # for experts with reject option
        data_mediumshot = torch.load('{}/results_test_aligned_mediumshot.pickle'.format(datadir))    # for experts with reject option
        data_fewshot = torch.load('{}/results_test_aligned_fewshot.pickle'.format(datadir))          # for experts with reject option
        data_general = torch.load('{}/results_test_aligned_general.pickle'.format(dataset.lower()))

    manyshot_logits = data_manyshot['logits'].clone().detach()
    mediumshot_logits = data_mediumshot['logits'].clone().detach()
    fewshot_logits = data_fewshot['logits'].clone().detach()
    labels = data_manyshot['labels'] if not args.test else data_general['labels']

    manyshotClassMask, mediumshotClassMask, fewshotClassMask = data_manyshot['class_mask'], data_mediumshot['class_mask'], data_fewshot['class_mask']        
   
    # logit tuning to correct for open set sampling ratio
    if(dataset.lower()=='imagenet'):
        manyshot_logits[:, -1] = manyshot_logits[:, -1]  - np.log(2/ (1+16))
        mediumshot_logits[:, -1] = mediumshot_logits[:, -1]  - np.log(2/ (1+16))
        fewshot_logits[:, -1] = fewshot_logits[:, -1]  - np.log(2/ (1+16))
    
    else:
        manyshot_logits[:, -1] = manyshot_logits[:, -1]  - np.log(2/ (1+16))
        mediumshot_logits[:, -1] = mediumshot_logits[:, -1]  - np.log(2/ (1+8))
        fewshot_logits[:, -1] = fewshot_logits[:, -1]  - np.log(2/ (1+8))
    
    manyshot_features = manyshot_logits.data.cpu().numpy()                                        
    mediumshot_features = mediumshot_logits.data.cpu().numpy()
    fewshot_features = fewshot_logits.data.cpu().numpy()
    labels = labels.data.cpu().numpy()
       
    if(not args.test):
        train_loader = torch.utils.data.DataLoader(Calibration_Dataset(orig_txt='./data/{}_LT/{}_LT_train.txt'.format(args.dataset, args.dataset), manyshot_features=manyshot_features, mediumshot_features=mediumshot_features,
                                       fewshot_features=fewshot_features, labels=labels), batch_size=args.batch_size, shuffle=True, **kwargs)
    else:
        test_loader = torch.utils.data.DataLoader(Calibration_Dataset(orig_txt='./data/{}_LT/{}_LT_train.txt'.format(args.dataset, args.dataset), manyshot_features=manyshot_features, mediumshot_features=mediumshot_features,
                                       fewshot_features=fewshot_features, labels=labels), batch_size=args.batch_size, shuffle=False, **kwargs)         # dont shuffle test set as usual
       
    ########################################################################
    ######################## initialise model and optimizer ################
    ########################################################################

    model = CalibrateExperts(args.dataset.lower(), manyshotClassMask, mediumshotClassMask, fewshotClassMask).cuda()
    optimizer = torch.optim.SGD(model.parameters() ,lr=args.lr, momentum= args.momentum, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    print( 'Using StepLR scheduler with params, stepsize : {}, gamma : {}'.format( args.step_size, args.gamma ) )

    if(args.test):
        pretrained_model = torch.load(args.load_model)
        weights = pretrained_model['state_dict_best']['model']
        weights = {k: weights['module.' + k] if 'module.' + k in weights else weights[k] for k in model.state_dict()}
        model.load_state_dict(weights)                                                                            # loading model weights
        print('Loaded pretrained model.')

    ########################################################################
    ######################## training with early stopping ##################
    ########################################################################

    if(not args.test):

        results = vars(args)
        results['train_losses'], results['train_accuracies']  = [], []
        best_acc, best_epoch = 0, 0

        epoch = 1
        while (True): 

            sys.stdout.flush()

            train_loss, train_acc = train(args, model, device, train_loader, optimizer, scheduler, epoch)

            results['train_losses'].append(train_loss)
            results['train_accuracies'].append(train_acc)

            if (train_acc > best_acc):
                best_acc = train_acc
                best_epoch = epoch
                results['best_acc'], results['best_epoch'] = best_acc, best_epoch

                # save best model
                best_model_weights = {}
                best_model_weights['model'] = copy.deepcopy(model.state_dict())
                model_states = {'epoch': epoch,
                                'best_epoch': best_epoch,
                                'state_dict_best': best_model_weights,
                                'best_acc': best_acc, }
                torch.save(model_states, os.path.join(exp_dir, "best_model.pt"))

            elif (epoch > best_epoch + args.stopping_criterion):
                print('Best model obtained. Error : ', best_acc)
                plot_curves(results, exp_dir)  # plot
                break

            savepath = os.path.join(exp_dir, 'results.pickle')
            with open(savepath, 'wb') as f:
                pickle.dump(results, f)
            plot_curves(results, exp_dir)  # plot
            epoch = epoch + 1

    ########################################################################
    ########################        testing         ########################
    ########################################################################

    else:

        loss, acc, preds = test(args, model, device, test_loader)
        
        if(dataset=='ImageNet'):
            split_ranges = {'manyshot' : [0, 19550], 'medianshot' : [19550, 43200], 'fewshot' : [43200,50000], 'all' : [0, 50000]} # imagenet
        else:
            split_ranges = {'manyshot' : [0, 13200], 'medianshot' : [13200, 29400], 'fewshot' : [29400,36500], 'all' : [0, 36500]} # places
        
        for split_name, split_range in split_ranges.items():
            
            gt_target = torch.from_numpy(labels[int(split_range[0]):int(split_range[1])]).cuda()
            split_preds = preds[int(split_range[0]):int(split_range[1])]

            correct = split_preds.eq(gt_target.view_as(split_preds)).sum().item()
            accuracy = 100*( correct / ( split_range[1]-split_range[0] ) )

            print('{} accuracy : {:.2f}'.format(split_name, accuracy ) )

main()






