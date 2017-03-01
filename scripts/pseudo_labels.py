#from __future__ import print_function
import pickle 
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
import seaborn

#
# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

print('loading data!')
path = '../../data/'
#trainset_labeled = pickle.load(open(path + "train_labeled.p", "rb"))
#validset = pickle.load(open(path + "validation.p", "rb"))
#trainset_unlabeled = pickle.load(open(path + "train_unlabeled.p", "rb"))
#
#train_loader = torch.utils.data.DataLoader(trainset_labeled, batch_size=32, shuffle=True, **kwargs)
#valid_loader = torch.utils.data.DataLoader(validset, batch_size=64, shuffle=True)
#train_unlabeled_loader = torch.utils.data.DataLoader(trainset_unlabeled, batch_size=256, shuffle=False, **kwargs)

def weightingFunction(epoch,T1 = 100.0,T2 = 300.0,alpha = 3./12):
    """
    values from paper: T1 =100, T2 = 300, alpha =3 
    
    """
    if epoch < T1:
        return 0
    elif (epoch >= T1) and (epoch <= T2):
        return alpha * (T1 - epoch) / (T1-T2)
    else:
        return alpha
    
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=2)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=2)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
#        print ('x', x.size())
        x = x.view(-1, 1024)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc3(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc4(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc5(x))
        return F.log_softmax(x)


def train_unlabeled(epoch):
    model.train()
    print("epoch of unlabeled, " , epoch)
    print("weighing function" , weightingFunction(epoch))
    if weightingFunction(epoch) == 0:
        return

    for batch_idx, (data, target) in enumerate(train_unlabeled_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data = Variable(data)

        optimizer.zero_grad()
        output = model(data)
        
        target = output.data.max(1)[1]
#        target = target.view(target.size()[0]) #make 1d array
#        
#        target_np = target.numpy().astype(int)
#        # finds the most likely lable for unlabeled image and its transformations
#        labels = []
#        for i in range(target.size()[0]/num_same_unlabeled):
#            part = target_np[i * num_same_unlabeled : i * num_same_unlabeled + num_same_unlabeled]
#            counts = np.bincount(part)
#            label = np.argmax(counts)
#            labels.extend(np.ones(num_same_unlabeled) * label)
#        
#        target = torch.from_numpy(np.array(labels, dtype = int))
        
#        target = Variable(target)
        loss = weightingFunction(epoch)*F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
            

def train(epoch):
    model.train()
    print("epoch of labeled")

    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)

        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))
        
            
def test(epoch, valid_loader, accuracy_list, name):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in valid_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target).data[0]
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    test_loss /= len(valid_loader) # loss function already averages over batch size
    
    print('\n' + name + ' set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(valid_loader.dataset),
        100. * correct / len(valid_loader.dataset)))
    accuracy_list.append(100. * correct / len(valid_loader.dataset))

model = Net()
if args.cuda:
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
print ('args.lr', args.lr)

train_accuracy = []
valid_accuracy = []

num_same_unlabeled = 4
    
#for testing
args.epochs = 10
args.momentum = 0.95
args.lr = 0.003

for epoch in range(1, args.epochs + 1):
    train(epoch)
    print ('done with labeled')
    train_unlabeled(epoch)
    test(epoch, valid_loader, valid_accuracy, "Validation")
    test(epoch, train_loader, train_accuracy, "Train")

def plot_accuracy(epochs, accuracy_test, accuracy_train, file_name):
    plt.plot(epochs, accuracy_test, label = 'Validation')
    plt.plot(epochs, accuracy_train, label = 'Train')
    plt.title("Validation and Train accuracy")
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12),
               fancybox=True, shadow=True, ncol=5)
    plt.savefig('../plots/' + file_name, bbox_inches='tight')
    plt.show()

def plot_accuracy_zoomin(epochs, accuracy_test, accuracy_train, file_name):
    plt.plot(epochs, accuracy_test, label = 'Validation')
    plt.plot(epochs, accuracy_train, label = 'Train')
    plt.title("Validation and Train accuracy")
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim((90,100))
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12),
               fancybox=True, shadow=True, ncol=5)
    plt.savefig('../plots/' + file_name, bbox_inches='tight')
    plt.show() 
    
plot_accuracy(range(1, args.epochs + 1), valid_accuracy, train_accuracy, 'acc')
plot_accuracy_zoomin(range(1, args.epochs + 1), valid_accuracy, train_accuracy, 'acc_zomm')


# Code to generate a file for submission
testset = pickle.load(open(path + "test.p", "rb"))
test_loader = torch.utils.data.DataLoader(testset,batch_size=64, shuffle=False)

label_predict = np.array([])
model.eval()
for data, target in test_loader:
    data, target = Variable(data, volatile=True), Variable(target)
    output = model(data)
    temp = output.data.max(1)[1].numpy().reshape(-1)
    label_predict = np.concatenate((label_predict, temp))
label_true = test_loader.dataset.test_labels.numpy()

true_label = pd.DataFrame(label_true, columns=['label'])
true_label.reset_index(inplace=True)
true_label.rename(columns={'index': 'ID'}, inplace=True)
predict_label = pd.DataFrame(label_predict, columns=['label'], dtype=int)
predict_label.reset_index(inplace=True)
predict_label.rename(columns={'index': 'ID'}, inplace=True)
predict_label.to_csv('sample_submission.csv', index=False)
true_label.to_csv('true_label.csv', index=False)