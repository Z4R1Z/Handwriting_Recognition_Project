import torch
import torchvision
from torchvision import transforms, datasets
from torchvision.transforms import Normalize, ToTensor
import torch.nn as nn  # neural network
import torch.optim as optim  # optimization layer
import torch.nn.functional as F  # activation functions
import argparse
import time
from collections import OrderedDict
from PIL import Image
import cv2
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score
import pandas as pd

class ConvNet(nn.Module):
    '''Define your convolutional neural network'''
    def __init__(self, **kwargs):
        super(ConvNet, self).__init__()
        self.layers = []
        self.layers += [nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1)]
        self.layers += [nn.BatchNorm2d(64)]
        self.layers += [nn.ReLU()]
        self.layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        
        self.layers += [nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)]
        self.layers += [nn.BatchNorm2d(128)]
        self.layers += [nn.ReLU()]
        self.layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        self.layers += [nn.Dropout(0.3)]
        
        self.layers += [nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)]
        self.layers += [nn.BatchNorm2d(256)]
        self.layers += [nn.ReLU()]
        self.layers += [nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)]
        self.layers += [nn.BatchNorm2d(256)]
        self.layers += [nn.ReLU()]
        self.layers += [nn.MaxPool2d(kernel_size=2, stride=2)]

        self.layers += [nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)]
        self.layers += [nn.BatchNorm2d(512)]
        self.layers += [nn.ReLU()]
        self.layers += [nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)]
        self.layers += [nn.BatchNorm2d(512)]
        self.layers += [nn.ReLU()]
        self.layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        
        self.sq = nn.Sequential(*self.layers)
        
        self.layers_l = []
        self.layers_l += [nn.Linear(512, 256)]
        self.layers_l += [nn.ReLU()]
        self.layers_l += [nn.Linear(256, 47)]
        #self.layers_l += nn.LogSoftmax(dim=1)
        self.sq2 = nn.Sequential(*self.layers_l)
     
    def forward(self, x):
        x = self.sq(x)
        x = torch.flatten(x,1)
        x = self.sq2(x)
        #x = self.logSoftmax(x)
        return x

def train(net, optimizer, criterion):
    '''
    Returns validation loss and accuracy
    
        Parameters:
            net (CNN): a convolutional neural network to train
            optimizer: optimizer
            criterion (loss function): a loss function to evaluate the model on
            args (ArgumentParser): hyperparameters
        
        Returns:
            net (CNN): a trained model
            train_loss (float): train loss
            train_acc (float): train accuracy
    '''
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
    
    net.train()
    
    correct = 0
    total = 0
    train_loss = 0
    
    for i, data in enumerate(train_loader):
        inputs, labels = data
        inputs = inputs.cuda()
        labels = labels.cuda()
        outputs = net(inputs)
        
        optimizer.zero_grad()
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        
        # the class with the highest value is the prediction
        _, prediction = torch.max(outputs.data, 1)  # grab prediction as one-dimensional tensor
        total += labels.size(0)
        correct += (prediction == labels).sum().item()

    train_loss = train_loss / len(train_loader)
    train_acc = 100 * correct / total
    
    return net, train_loss, train_acc  # net is returned to be fed to the test function later
def validate(net, criterion):
    '''
    Returns validation loss and accuracy
    
        Parameters:
            net (CNN): a convolutional neural network to validate
            criterion (loss function): a loss function to evaluate the model on
            args (ArgumentParser): hyperparameters
        
        Returns:
            val_loss (float): validation loss
            val_acc (float): validation accuracy
    '''
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=64, shuffle=True)
    
    net.eval()

    correct = 0
    total = 0
    val_loss = 0 
    
    with torch.no_grad():
        for data in val_loader:
            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()
            outputs = net(inputs)

            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, prediction = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (prediction == labels).sum().item()

        val_loss = val_loss / len(val_loader)
        val_acc = 100 * correct / total

    return val_loss, val_acc

def test(net):
    '''
    Returns test accuracy
    
        Parameters:
            net (CNN): a trained model
            args (ArgumentParser): hyperparameters
        
        Returns:
            test_acc (float): test accuracy of a trained model
    '''
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=True)

    net.eval()
    
    y_pred = []
    y_true = []
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()
            outputs = net(inputs)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            output = (torch.max(torch.exp(outputs), 1)[1]).data.cpu().numpy()
            y_pred.extend(output) # Save Prediction
        
            labels = labels.data.cpu().numpy()
            y_true.extend(labels) # Save Truth


        test_acc = 100 * correct / total
        cf_matrix = confusion_matrix(y_true, y_pred)
        classes = ('0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','X','W','Y','Z','a','b','d','e','f', 'g', 'h','n','q','r','t')
        df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix) *10, index = [i for i in classes], columns = [i for i in classes])
        print("F1 : ", f1_score(y_true,y_pred, average='micro'))
        df_cm.to_pickle("conf.pkl")

    return test_acc

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)


# load data in
train_set = datasets.EMNIST(download=True,root="data", split="balanced",
                            train=True, transform=transforms.Compose([ToTensor()])
                           )
test_set = datasets.EMNIST(root="data", split="balanced", 
                           train=False,transform=transforms.Compose([ToTensor()])
                          )
entire_trainset = torch.utils.data.DataLoader(train_set, shuffle=True)

split_train_size = int(0.8*(len(entire_trainset)))  # use 80% as train set
split_valid_size = len(entire_trainset) - split_train_size  # use 20% as validation set

train_set, val_set = torch.utils.data.random_split(train_set, [split_train_size, split_valid_size]) 

print(f'train set size: {split_train_size}, validation set size: {split_valid_size}')

net = ConvNet()
#net = net.cuda()
criterion = nn.CrossEntropyLoss()
    
print(net.summary())
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# containers to keep track of statistics
train_losses = []
val_losses = []
train_accs = []
val_accs = []
time_total = 0
    
for epoch in range(10):  # number of training to be completed
    time_start = time.time()
    net, train_loss, train_acc = train(net, optimizer, criterion)
    val_loss, val_acc = validate(net, criterion)
    time_end = time.time()
    
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)
    
    time_duration = round(time_end - time_start, 2)
    time_total += time_duration
    
    # print results of each iteration
    print(f'Epoch {epoch+1}, Accuracy(train, validation):{round(train_acc, 2), round(val_acc, 2)}, '
            f'Loss(train, validation):{round(train_loss, 4), round(val_loss, 4)}, Time: {time_duration}s')

test_acc = test(net)

results = OrderedDict()
results['train_losses'] = [round(x, 4) for x in train_losses]
results['val_losses'] = [round(x, 4) for x in val_losses]
results['train_accs'] = [round(x, 2) for x in train_accs]
results['val_accs'] = [round(x, 2) for x in val_accs]
results['train_acc'] = round(train_acc, 2)
results['val_acc'] = round(val_acc, 2)
results['test_acc'] = round(test_acc, 2)
results['time_total'] = round(time_total, 2)



print('Test Accuracy: {}'.format(results['test_acc']))
print('Total time duration: {}'.format(results['time_total']))
print()
