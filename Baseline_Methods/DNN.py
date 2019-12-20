
# coding: utf-8

# In[1]:

# Reference : https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
import numpy as np
import os
import pandas as pd
import pickle
import torch.optim as optim
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')


# In[ ]:




# In[2]:

def get_train_data_label(data_set,train,train_data,train_label):
    for i in range(len(train['data'])):
        img = np.reshape(train['data'][i], (3, 32, 32))
        rgb = np.moveaxis(img, 0, 2)
        data_set[train['labels'][i]].append(rgb)
        train_data.append(rgb)
        train_label.append(train['labels'][i])
    return train_data,train_label,data_set


# In[3]:

def get_test_data_label(test_data2):
    test_label=[]
    test_data=[]
    for i in range(len(test_data2['labels'])):
        test_label.append(int(test_data2['labels'][i]))
    
    for i in range(len(test_data2['data'])):
        img = np.reshape(test_data2['data'][i], (3, 32, 32))
        rgb = np.moveaxis(img, 0, 2)
        test_data.append(rgb)
    return test_data,test_label


# In[4]:

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
    return dict


# # Reading the Data for CIFAR

# In[5]:

# length = 10
# train_data = []
# train_label = []
# data_set=[]
# for i in range(length):
#     data_set.append([])
# os.chdir('./SML3/cifar')
# test_data1=unpickle('test_batch')
# train1=unpickle('data_batch_1')
# train2=unpickle('data_batch_2')
# train3=unpickle('data_batch_3')
# train4=unpickle('data_batch_4')
# train5=unpickle('data_batch_5')
# os.chdir('../..')
# train_data,train_label,data_set = get_train_data_label(data_set,train1,train_data,train_label)
# train_data,train_label,data_set = get_train_data_label(data_set,train2,train_data,train_label)
# train_data,train_label,data_set = get_train_data_label(data_set,train3,train_data,train_label)
# train_data,train_label,data_set = get_train_data_label(data_set,train4,train_data,train_label)
# train_data,train_label,data_set = get_train_data_label(data_set,train5,train_data,train_label)
# test_data,test_label = get_test_data_label(test_data1)


# # Visualisation of Dataset

# In[6]:

# def pca_plot(dataset_main,label_main,title,classes):
#     #TSNE Plot for glass dataset
#     tsne = PCA(n_components=2)
#     tsne_results = tsne.fit_transform(dataset_main)

#     df_subset = pd.DataFrame()
#     df_subset['X'] = tsne_results[:,0]
#     df_subset['y']=label_main
#     df_subset['Y'] = tsne_results[:,1]
#     plt.figure(figsize=(6,4))
#     plt.title(title)
#     sns.scatterplot(
#         x="X", y="Y",
#         hue="y",
#         palette=sns.color_palette("hls", classes),
#         data=df_subset,
#         legend="full",
#         alpha=1.0
#     )


# In[7]:

# train_data_main = []
# for i in train_data:
#     train_data_main.append(i.ravel())


# In[8]:

# pca_plot(train_data_main,train_label,"Visualisation for CIFAR Datset",10)


# In[9]:

# #Dataset description for Glass dataset
# dict_label={}
# for i in range(len(train_label)):
#     try:
#         dict_label[train_label[i]] = dict_label[train_label[i]]+1
#     except:
#         dict_label[train_label[i]]=1
# print("Class distribution : ",dict_label)
# labels = list(dict_label.keys())
# index = np.arange(len(list(dict_label.keys())))
# plt.pie(list(dict_label.values()),labels=labels,autopct='%1.0f%%')
# labels = list(dict_label.keys())
# plt.title("Class Distribution in CIFAR dataset")
# plt.show()


# In[10]:

# Initialisation of Parameters
epoch_train = 50
global dropout 
dropout = 0.2
learning_rate = 0.001 
momentum = 0.9
init_weights = "xavier"
global padding 
global activation_function
activation_function = "relu"
padding = 1
batch_size = 64


# # For CIFAR10

# In[11]:

# For CIFAR10
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=10)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=10)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# # For STL10

# In[47]:

# transform = transforms.Compose(
#     [transforms.ToTensor(),
#      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# trainset = torchvision.datasets.STL10(root='./data', split='train',
#                                         download=True, transform=transform)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
#                                           shuffle=True, num_workers=10)

# testset = torchvision.datasets.STL10(root='./data', split='test',
#                                        download=True, transform=transform)
# testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
#                                          shuffle=False, num_workers=10)

# classes = ('airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck')


# In[12]:

# Reference : https://stackoverflow.com/questions/49433936/how-to-initialize-weights-in-pytorch
def init_weights_xavier(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)
        
def init_weights_normal(m):
        classname = m.__class__.__name__
        # for every Linear layer in a model
        if classname.find('Linear') == -1:
            pass
        else:
            y = m.in_features
            m.weight.data.normal_(0.0,np.sqrt(1/y))
            m.bias.data.fill_(0)
        if classname.find('Conv2d') == -1:
            pass
        else:
            y = m.kernel_size[0]*m.kernel_size[1]*m.out_channels
            m.weight.data.normal_(0.0,np.sqrt(2/y))
            m.bias.data.fill_(0)


# In[13]:

class Net_Q1_2(nn.Module):
    def __init__(self):
        global dropout
        global padding
        global activation_function
        super(Net_Q1_2, self).__init__()
        # Encoding layers 
        self.conv1 = nn.Conv2d(3, 32, 3,padding=padding)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32,64, 3,padding=padding)
        self.conv3 = nn.Conv2d(64,128, 3,padding=padding)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(128 * 4 * 4, 90)
        self.fc2 = nn.Linear(90, 10)

    def forward(self, x):
        if activation_function == "relu":
            x = self.dropout(self.pool(F.relu(self.conv1(x))))
            x = self.dropout(self.pool(F.relu(self.conv2(x))))
            x = self.dropout(self.pool(F.relu(self.conv3(x))))
            x = x.view(-1, 128 * 4 * 4)
            x = self.dropout(F.relu(self.fc1(x)))
            x = F.softmax(self.fc2(x))
        elif activation_function == "tanh":
            x = self.dropout(self.pool(F.tanh(self.conv1(x))))
            x = self.dropout(self.pool(F.tanh(self.conv2(x))))
            x = self.dropout(self.pool(F.tanh(self.conv3(x))))
            x = x.view(-1, 128 * 4 * 4)
            x = self.dropout(F.tanh(self.fc1(x)))
            x = F.softmax(self.fc2(x))
        return x


# In[50]:

class Net_Q1_1(nn.Module):
    def __init__(self):
        global dropout
        global padding
        global activation_function
        super(Net_Q1_1, self).__init__()
        # Encoding layers 
        self.conv1 = nn.Conv2d(3, 32, 3,padding=padding)
        self.pool1 = nn.MaxPool2d(2, 2,return_indices=True)
        self.conv2 = nn.Conv2d(32,64, 3,padding=padding)
        self.conv3 = nn.Conv2d(64,128, 3,padding=padding)
        
        # Decoding layers
        self.deconv1 = nn.ConvTranspose2d(128, 64, 3,padding=padding)
        self.pool2 = nn.MaxUnpool2d(2,2)
        self.deconv2 = nn.ConvTranspose2d(64,32, 3,padding=padding)
        self.deconv3 = nn.ConvTranspose2d(32,3,3,padding=padding)

        # FCs
        self.fc1 = nn.Linear(128 * 4 * 4, 100)
        self.fc2 = nn.Linear(100, 10)
        
    def forward(self, x):
        if activation_function == "relu":
             # Encoder
            x = self.conv1(x)
            x,indices1 = self.pool1(F.relu(x))
            x = self.conv2(x)
            x,indices2 = self.pool1(F.relu(x))
            x_encode,indices3 = self.pool1(F.relu(self.conv3(x)))
            
            # Decoder  
            x_decode = self.pool2(x_encode,indices3)
            x_decode = F.relu(self.deconv1(x_decode))
            x_decode = self.pool2(x_decode,indices2)
            x_decode = self.deconv2(x_decode)
            x_decode = self.pool2(F.relu(x_decode),indices1)
            x_decode = self.deconv3(x_decode)
             
            # FC +Softmax
            x_fc = x_encode.view(-1, 128 * 4 * 4)
            x_fc = F.relu(self.fc1(x_fc))
            x_fc = F.softmax(self.fc2(x_fc))
            
        elif activation_function == "tanh":
             # Encoder
            x = self.conv1(x)
            x,indices1 = self.pool1(F.tanh(x))
            x = self.conv2(x)
            x,indices2 = self.pool1(F.tanh(x))
            x_encode,indices3 = self.pool1(F.tanh(self.conv3(x)))
            
            # Decoder  
            x_decode = self.pool2(x_encode,indices3)
            x_decode = F.tanh(self.deconv1(x_decode))
            x_decode = self.pool2(x_decode,indices2)
            x_decode = self.deconv2(x_decode)
            x_decode = self.pool2(F.tanh(x_decode),indices1)
            x_decode = self.deconv3(x_decode)
             
            # FC +Softmax
            x_fc = x_encode.view(-1, 128 * 4 * 4)
            x_fc = F.tanh(self.fc1(x_fc))
            x_fc = F.softmax(self.fc2(x_fc))
        
        
        return x_fc, x_decode




# In[18]:

net = Net_Q1_2()
if init_weights == "xavier":
    net.apply(init_weights_xavier)
if init_weights == "random":
    net.apply(init_weights_normal)
net = net.cuda()


# In[ ]:




# In[19]:

# Define a Loss function and optimizer
criterion_fc=nn.CrossEntropyLoss()
criterion_reconstruction = nn.MSELoss()  #Loss Class
optimizer = optim.Adam(net.parameters(),lr=learning_rate) #optimizer class
size_train = len(trainloader.sampler)
size_test = len(testloader.sampler)


# In[20]:

# Part 2 (Encoder ------> FC + softmax)
train_loss = []
gradients = {}
for epoch in range(epoch_train):  # loop over the dataset multiple times
    
    running_loss = 0.0
    counter  = 0 
    gradients_temp = {}
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = inputs.cuda()
        labels = labels.cuda()
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        output_fc = net(inputs)
        loss = criterion_fc(output_fc, labels)
#         loss_decode = criterion_reconstruction(output_decoder,inputs)
        
        loss.backward()
        
        for n,p in net.named_parameters():
            if n.split('.')[1] =="weight" and n.split('.')[0][:-1] != "batchnorm":
                try:
                    gradients_temp[n.split('.')[0]].append(np.linalg.norm(p.grad))
                except:
                    temp = []
                    temp.append(np.linalg.norm(p.grad))
                    gradients_temp[n.split('.')[0]] = temp 
                    
        if  counter == (size_train/batch_size -1):
        
            for n,p in net.named_parameters():
                if n.split('.')[1] =="weight" and n.split('.')[0][:-1] != "batchnorm":
                    try:
                        gradients[n.split('.')[0]].append(np.sum(gradients_temp[n.split('.')[0]]))
                    except:
                        gradients[n.split('.')[0]] = [np.sum(gradients_temp[n.split('.')[0]])]
            gradients_temp = {}
        counter = counter +1 
        optimizer.step()
        
        # print statistics
        running_loss += loss.item()
    if epoch%10 == 0:
        print("Trained till epoch ",epoch)
        print("Running Loss : ",running_loss)
    train_loss.append(running_loss)

    
print("Finished Training")


# In[ ]:




# In[55]:

# Encoder  ------ > (Decoder & FC + softmax)
train_loss = []
gradients = {}
for epoch in range(epoch_train):  # loop over the dataset multiple times
    
    running_loss = 0.0
    counter  = 0 
    gradients_temp = {}
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = inputs.cuda()
        labels = labels.cuda()
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        output_fc,output_decoder = net(inputs)
        loss_fc = criterion_fc(output_fc, labels)
        loss_decode = criterion_reconstruction(output_decoder,inputs)
        loss = loss_decode + loss_fc
        loss.backward()
        
        for n,p in net.named_parameters():
            if n.split('.')[1] =="weight" and n.split('.')[0][:-1] != "batchnorm":
                try:
                    gradients_temp[n.split('.')[0]].append(np.linalg.norm(p.grad))
                except:
                    temp = []
                    temp.append(np.linalg.norm(p.grad))
                    gradients_temp[n.split('.')[0]] = temp 
                    
        if  counter == (size_train/batch_size -1):
        
            for n,p in net.named_parameters():
                if n.split('.')[1] =="weight" and n.split('.')[0][:-1] != "batchnorm":
                    try:
                        gradients[n.split('.')[0]].append(np.sum(gradients_temp[n.split('.')[0]]))
                    except:
                        gradients[n.split('.')[0]] = [np.sum(gradients_temp[n.split('.')[0]])]
            gradients_temp = {}
                        
        optimizer.step()
        counter = counter +1
        
        # print statistics
        running_loss += loss.item()
    if epoch%10 == 0:
        print("Trained till epoch ",epoch)
        print("Running Loss : ",running_loss)
    train_loss.append(running_loss)
    
print('Finished Training')


# In[21]:

plt.title("Plot forLoss")
plt.plot([j for j in range(len(train_loss))],train_loss)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()


# In[22]:

for i in gradients.keys():
    plt.title("Plot for epoch vs gradients for blocks")
    plt.plot([j for j in range(len(gradients[i]))],gradients[i],label = i)
    plt.xlabel("Epoch")
    plt.ylabel("Gradients")
    plt.legend()
plt.show()


# In[48]:

def accuracy_prediction2(trainloader,net,testloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.cuda()
            labels = labels.cuda()
#             output_fc,output_decoder = net(inputs)
            output_fc = net(images)
            _, predicted = torch.max(output_fc.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    test_accuracy = (correct / float(total)) *100
    correct = 0
    total = 0
    with torch.no_grad():
        for data in trainloader:
            images, labels = data
            images = images.cuda()
            labels = labels.cuda()
#             output_fc,output_decoder = net(inputs)
            output_fc = net(images)
            _, predicted = torch.max(output_fc.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    train_accuracy = (correct / float(total)) *100
    return train_accuracy,test_accuracy


# In[49]:

def accuracy_prediction1(trainloader,net,testloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.cuda()
            labels = labels.cuda()
            output_fc,output_decoder = net(images)
#             output_fc = net(images)
            _, predicted = torch.max(output_fc.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    test_accuracy = (correct / float(total)) *100
    correct = 0
    total = 0
    with torch.no_grad():
        for data in trainloader:
            images, labels = data
            images = images.cuda()
            labels = labels.cuda()
            output_fc,output_decoder = net(images)
#             output_fc = net(images)
            _, predicted = torch.max(output_fc.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    train_accuracy = (correct / float(total)) *100
    return train_accuracy,test_accuracy


# In[50]:

accuracy_prediction1(trainloader,net,testloader)


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:

# transform = transforms.Compose(
#     [transforms.ToTensor(),
#      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# trainset = torchvision.datasets.STL10(root='./data', train=True,
#                                         download=True, transform=None)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
#                                           shuffle=True, num_workers=10)

# testset = torchvision.datasets.STL10(root='./data', train=False,
#                                        download=True, transform=transform)
# testloader = torch.utils.data.DataLoader(testset, batch_size=64,
#                                          shuffle=False, num_workers=10)

# classes = ('plane', 'car', 'bird', 'cat',
#            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# In[ ]:



