from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
import os
from PIL import Image
import numpy as np
import torch
import sys

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

# looks up data count in orig data file and creates the dataset according to the specified thresholds
class Threshold_Dataset(Dataset):

    def __init__(self, root, orig_txt, txt, low_threshold=0, high_threshold=100000, open_ratio=1, use_open=True, transform=None, picker='generalist'):

        # loading train file (from orig_txt)
        self.orig_labels = []
        with open(orig_txt) as f:
            for line in f:
                self.orig_labels.append(int(line.split()[1]))
        self.orig_labels = np.array(self.orig_labels).astype(int)

        # loading class counts
        self.tot_num_classes = self.orig_labels.max() + 1
        self.train_class_count = np.zeros(self.tot_num_classes, dtype=np.int32)
        for l in np.unique(self.orig_labels):
            self.train_class_count[l] = len(self.orig_labels[self.orig_labels == l])

        self.img_path, open_set = [], []
        self.labels = []
        self.transform = transform
        self.use_open = use_open
        self.open_ratio = open_ratio

        # loading data from txt file
        with open(txt) as f:
            
            for line in f:
                img_path, label = os.path.join(root, line.split()[0]), int(line.split()[1])
                
                if(picker=='experts' or picker=='generalist'):

                    if(self.train_class_count[label]>=low_threshold and self.train_class_count[label]<high_threshold):
                        self.img_path.append(img_path)
                        self.labels.append(label)
                    else:
                        open_set.append(img_path)

                else:

                    raise Exception('Invalid picker.')
 
        self.num_classes, self.closed_set_length = np.unique(np.array(self.labels).astype(int)).shape[0], len(self.labels)
 
        # appending openset examples from rest of the classes as open set instances
        if(self.use_open):
            if(self.open_ratio==-1):
                open_samples = np.random.permutation(np.arange(len(open_set)))[:self.closed_set_length//self.num_classes] # 1:n openset sampling
            else:
                open_samples = np.random.permutation(np.arange(len(open_set)))[:self.closed_set_length//self.open_ratio] # 1:open_ratio openset sampling
            for index in open_samples:
                self.img_path.append(open_set[index])
                self.labels.append(1001) # label for open class

        # creating class mask for fusion during ensemble inference
        self.class_mask = torch.BoolTensor([True if label in np.array( self.labels ) else False for label in range(self.tot_num_classes)])

        # creating new labels for classification
        sorted_labels = np.sort(np.array(np.unique(self.labels)))
        new_labels = []
        for ind,label in enumerate(self.labels):
            new_label = np.where(sorted_labels==label)[0][0]
            new_labels.append(new_label)
        self.labels = new_labels

        print('Created dataset split: {:d} closed set samples from {:d}({:d}) classes, {:d} open set samples.'.format(self.closed_set_length, self.num_classes,
                                                                                                            self.tot_num_classes, len(self.labels) - self.closed_set_length))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):

        path = self.img_path[index]
        label = self.labels[index]
        train_count = self.train_class_count[label]

        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, label, path, train_count

class Calibration_Dataset(Dataset):

    def __init__(self, orig_txt, manyshot_features, mediumshot_features, lowshot_features, labels = None):

        # loading train file ( from orig_txt )
        self.orig_labels = []
        with open(orig_txt) as f:
            for line in f:
                self.orig_labels.append(int(line.split()[1]))
        self.orig_labels = np.array(self.orig_labels).astype(int)

        # loading class counts
        self.tot_num_classes = self.orig_labels.max() + 1
        self.train_class_count = np.zeros(self.tot_num_classes, dtype=np.int32)
        for l in np.unique(self.orig_labels):
            self.train_class_count[l] = len(self.orig_labels[self.orig_labels == l])

        self.manyshot_features, self.mediumshot_features, self.lowshot_features = manyshot_features, mediumshot_features, lowshot_features
        self.labels = labels
        self.features = np.concatenate((self.manyshot_features, self.mediumshot_features, self.lowshot_features), axis=1)                                                                                          # concatenating features from the three experts
       
        print('Created dataset: {:d} manyshot samples, {:d} mediumshot samples, {:d} lowshot samples. Size of features: {:d}'.format((self.expertLabels==0).sum(), (self.expertLabels==1).sum(), (self.expertLabels==2).sum(), self.features.shape[1]))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):

        feature = self.features[index]
        label = self.labels[index]
        return feature, label                                              

    
