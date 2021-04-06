import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import copy
import numpy as np
import random
from model import LeNet, NN

from torch.utils.data import Dataset, DataLoader, sampler
import torchvision.datasets as dsets

client_N = 100
client_select_fraction = 0.1
selected_client_N = (int)(client_N * client_select_fraction)
selected_client = list()

rounds = 100
epochs = 200
FedAvg_iter = 10
learning_rate = 0.001
# LR_scheduler_step_size = 100
optimizer = "SGD"
batch_size = 16
model_used = 'LeNet'

use_CUDA = False
use_CUDA = torch.cuda.is_available()

device = torch.device("cpu")
dtype = torch.FloatTensor

if use_CUDA:
    print("GPU Count : ", torch.cuda.device_count())
    if(torch.cuda.device_count() > 1):
        torch.cuda.set_device(1)
        device = torch.cuda.current_device()
    else:
        torch.cuda.set_device(0)
        device = torch.cuda.current_device()
    print("Selected GPU # :", torch.cuda.current_device())
    print("Selected GPU Name : ", torch.cuda.get_device_name(device=device))
    dtype = torch.cuda.FloatTensor


"""
====================================================================================================
====================================================================================================
Data Loader
====================================================================================================
====================================================================================================
"""


def get_indices(dataset,class_name):
    indices = []
    for i in range(len(dataset.targets)):
        if dataset.targets[i] == class_name:
            indices.append(i)
    return indices


mnist_train = dsets.MNIST(root='data/',
                              train=True,
                              transform=transforms.ToTensor(),
                              download=True)
mnist_test = dsets.MNIST(root='data/',
                              train=False,
                              transform=transforms.ToTensor(),
                              download=True)

print("MNIST Train Data size : ", len(mnist_train))
dataset_size = len(mnist_train)
indices = list(range(dataset_size))

train_datafraction_each = 1/client_N
train_datafraction = [train_datafraction_each for _ in range(client_N)]

random_seed = 42
np.random.seed(random_seed)
np.random.shuffle(indices)
global_fraction = 0

train_datasize = [(int)(len(mnist_train) * x) for x in train_datafraction]
print("Each Train Loader Size : ", train_datasize)


train_dataidx = list()

for i in range(client_N):
    train_dataidx.append(indices[(int)(dataset_size * global_fraction) : (int)(dataset_size*(global_fraction+train_datafraction[i]))])
    global_fraction += train_datafraction[i]

print("train_dataidx : ", len(train_dataidx))

train_loader = list()

for i in range(client_N):
    train_loader.append(DataLoader(
        dataset=mnist_train,\
        batch_size=batch_size,\
        sampler=sampler.SubsetRandomSampler(train_dataidx[i])
    ))

test_loader = DataLoader(
    dataset=mnist_test,\
    batch_size=len(mnist_test), \
    shuffle=True
)
print("Iteration : ", train_loader[0].__len__())


"""
====================================================================================================
====================================================================================================
Model Identify
====================================================================================================
====================================================================================================
"""
def weights_init(m):
    if type(m) == nn.Conv2d:
        torch.nn.init.xavier_uniform(m.weight)

if model_used == 'LeNet':
    model_Server = LeNet()
    model_Server.to(device)
    model_Client = list()
    for idx in range(client_N):
        model_Client.append(LeNet())
        model_Client[idx].to(device)
        with torch.no_grad():
            model_Client[idx] = copy.deepcopy(model_Server)

elif model_used == 'NN':
    model_Server = NN()
    model_Server.to(device)
    model_Client = list()
    for idx in range(client_N):
        model_Client.append(NN())
        model_Client[idx].to(device)
        with torch.no_grad():
            model_Client[idx] = copy.deepcopy(model_Server)


"""
======================
Loss Function, Optimizer
======================
"""
loss_fn = nn.CrossEntropyLoss()

params_list = list()
for idx in range(client_N):
    params_list.append(list(model_Client[idx].parameters()))

opt_list = list()
if(optimizer == "Adam"):
    for idx in range(client_N):
        opt_list.append(optim.Adam(params=params_list[idx], lr=learning_rate))
elif(optimizer == "SGD"):
    for idx in range(client_N):
        opt_list.append(optim.SGD(params=params_list[idx], lr=learning_rate, momentum=0.9))
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer=opt, step_size=LR_scheduler_step_size, gamma=0.1)


"""
======================
[Test Function]
======================
"""
def test_fn():
    with torch.no_grad():
        count_target = [0 for i in range(10)]
        count_pred = [0 for i in range(10)]
        count_correct = [0 for i in range(10)]
        total = 0
        correct = 0

        for iter, data in enumerate(test_loader):
            test_input = data[0].cuda()
            test_target = data[1].cuda()

            if model_used == 'LeNet':
                output = model_Server(test_input)
            elif model_used == 'NN':
                output = model_Server(test_input.view(-1, 28 * 28))

            _, prediction = torch.max(output.data, 1)

            for idx in range(test_target.shape[0]):
                count_target[test_target[idx].data] += 1
                count_pred[prediction[idx].data] += 1
                if(prediction[idx].data == test_target[idx].data):
                    count_correct[prediction[idx].data] += 1

            total += test_target.size(0)
            correct += (prediction == test_target).sum().item()

        print("count_target : ", count_target)
        print("count_pred : ", count_pred)
        print("count_correct : ", count_correct)
        print("total : ", total)
        print("correct : ", correct)
        print("Accuracy : ", correct / total * 100)
        print("\n")


""""
=================================
CLient Selection By Lyapunov
=================================
def client_selector_lyapunov():
    gamma = 20
    val  = -9999
    opt_case = None
"""



"""
====================================================================================================
====================================================================================================
[Training]
====================================================================================================
====================================================================================================
"""
if model_used == 'LeNet':
    conv1_weight_list = list()
    conv2_weight_list = list()
    fc1_weight_list = list()
    fc2_weight_list = list()
    fc3_weight_list = list()
elif model_used == 'NN':
    fc1_weight_list = list()
    fc2_weight_list = list()
    fc3_weight_list = list()


client_list = [i for i in range(client_N)]


for round in range(rounds):
    # Client Selection
    selected_client.clear()
    selected_client = random.sample(client_list, selected_client_N)
    selected_client.sort()

    # Broadcast Model
    with torch.no_grad():
        for idx in range(client_N):
            model_Client[idx] = copy.deepcopy(model_Server)


    # print("Round : ", round, "Selected Clients : ", selected_client)

    for idx in selected_client:
        # iterate with local data of selected Clients
        print("Round : ", round, ", Client : ", idx)

        if (optimizer == "SGD"):
            opt = optim.SGD(params=model_Client[idx].parameters(), lr=learning_rate)
        elif (optimizer == "Adam"):
            opt = optim.Adam(params=model_Client[idx].parameters(), lr=learning_rate)

        for iter1 in range(FedAvg_iter):
            for iter2, data in enumerate(train_loader[idx]):
                train_input = data[0].cuda()
                train_target = data[1].cuda()
                if model_used == 'LeNet':
                    pred = model_Client[idx](train_input)
                elif model_used == 'NN':
                    pred = model_Client[idx](train_input.view(-1, 28*28))
                loss = loss_fn(pred, train_target)
                opt.zero_grad()
                loss.backward()
                opt.step()


    # Aggregate models of Clients
    with torch.no_grad():
        if model_used == 'LeNet':
            for idx in range(selected_client_N):
                conv1_weight_list.append(model_Client[idx].conv1.weight)
                conv2_weight_list.append(model_Client[idx].conv2.weight)
                fc1_weight_list.append(model_Client[idx].fc1.weight)
                fc2_weight_list.append(model_Client[idx].fc2.weight)
                fc3_weight_list.append(model_Client[idx].fc3.weight)
            conv1_weight_mean = torch.mean(torch.stack(conv1_weight_list), dim=0)
            conv2_weight_mean = torch.mean(torch.stack(conv2_weight_list), dim=0)
            fc1_weight_mean = torch.mean(torch.stack(fc1_weight_list), dim=0)
            fc2_weight_mean = torch.mean(torch.stack(fc2_weight_list), dim=0)
            fc3_weight_mean = torch.mean(torch.stack(fc3_weight_list), dim=0)

            # Change Global Model (Server Model)
            model_Server.conv1.weight = torch.nn.Parameter(conv1_weight_mean)
            model_Server.conv2.weight = torch.nn.Parameter(conv2_weight_mean)
            model_Server.fc1.weight = torch.nn.Parameter(fc1_weight_mean)
            model_Server.fc2.weight = torch.nn.Parameter(fc2_weight_mean)
            model_Server.fc3.weight = torch.nn.Parameter(fc3_weight_mean)

            conv1_weight_list.clear()
            conv2_weight_list.clear()
            fc1_weight_list.clear()
            fc2_weight_list.clear()
            fc3_weight_list.clear()

        elif model_used == 'NN':
            for idx in range(selected_client_N):
                fc1_weight_list.append(model_Client[idx].fc1.weight)
                fc2_weight_list.append(model_Client[idx].fc1.weight)
                fc3_weight_list.append(model_Client[idx].fc1.weight)
            fc1_weight_mean = torch.mean(torch.stack(fc1_weight_list), dim=0)
            fc2_weight_mean = torch.mean(torch.stack(fc2_weight_list), dim=0)
            fc3_weight_mean = torch.mean(torch.stack(fc3_weight_list), dim=0)

            # Change Global Model (Server Model)
            model_Server.fc1.weight = torch.nn.Parameter(fc1_weight_mean)
            model_Server.fc2.weight = torch.nn.Parameter(fc2_weight_mean)
            model_Server.fc3.weight = torch.nn.Parameter(fc3_weight_mean)

            fc1_weight_list.clear()
            fc2_weight_list.clear()
            fc3_weight_list.clear()


        # print("Round : ", round)
        test_fn()



print("\n\nStart Evaluation")
test_fn()