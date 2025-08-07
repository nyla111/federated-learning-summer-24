import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.models import resnet18
import copy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class FLServer:
  def __init__(self) -> None:
    self.clients = []

  def add_client(self, fl_client):
    self.clients.append(fl_client)


  def test(self, model):
    # test on each client and then print accuracy
    test_metrics = []
    for flclient in self.clients:
      test_metric = flclient.test(model)
      test_metrics.append(test_metric)
    print(test_metrics)


class FLClient:
  def __init__(self, id, train_data, test_data):
    self.id = id
    self.train_data = train_data
    self.test_data = test_data
    self.train_loader = torch.utils.data.DataLoader(self.train_data, batch_size=256, shuffle=True)
    self.test_loader = torch.utils.data.DataLoader(self.test_data, batch_size=256, shuffle=True)
    # torch.utils.data.DataLoader is used to load and iterate data
    # batch-size is the number of sample. in this case 256 samples is iterated over at once
    # shuffle=True: the data is shuffled before forming batches -> reduce bias

  def train(self, model):
    # optimizer include some algorithms used to reduce loss
    optimizer = optim.SGD(model.parameters(), lr = 0.01) # set an opitmizer using SGD (Stochastic gradient descent) method with learning rate of 0.01
    model.train() # set the model to training mode - should always include in the training phase

    total_loss, correct, num_sample = 0, 0, 0
    for idb, (data, target) in enumerate(self.train_loader):
      data, target = data.to(device), target.to(device) # send the input to device - typically is cuda (gpu) or cpu
      optimizer.zero_grad() # set all the parameters of optimizer to 0
      output = model(data) # give the output after the FORWARD flow of input 'data'

      loss = F.cross_entropy(output, target) # give the loss function using cross entropy method
      total_loss += loss.item() * len(data)
      num_sample += len(data)

      # find the maximum value along the second dimension (1) of 'output':
      pred = output.argmax(1, keepdim = True) # keepdim=True: cause 'pred' the same shape with 'output' (2 dimensional). otherwise may 1 dimension only.

      # count the correct prediction (if it is the same as target value or not)
      correct += pred.eq(target.view_as(pred)).sum().item() # sum() adds all the True value and item() returns a scalar number of it.

      loss.backward() # compute gradient using backpropagation (after using needs to clear all with zeros_grad())
      optimizer.step() # update the model param based on the computed gradient

    total_loss /= num_sample
    accuracy = 100.* correct / num_sample

    return model.state_dict(), {'loss': total_loss, 'accuracy': accuracy, '#data': num_sample, 'correct': correct}

  def test(self, model):
    model.eval() # set the model to testing mode

    total_loss, correct, num_sample = 0, 0, 0
    for idb, (data, target) in enumerate(self.test_loader):
      data, target = data.to(device), target.to(device)
      output = model(data)

      loss = F.cross_entropy(output, target)
      total_loss += loss.item() * len(data)
      num_sample += len(data)

      pred = output.argmax(1, keepdim = True)
      correct += pred.eq(target.view_as(pred)).sum().item()

      loss.backward()

    total_loss /= num_sample
    accuracy = 100.* correct / num_sample

    return {'id': self.id, 'loss': total_loss, 'accuracy': accuracy, '#data': num_sample, 'correct': correct}

class Helper:
  # List of dataset: MNIST, CIFAR-10, Fashion-MNIST
  # CIFAR-10 dataset
  transform = transforms.Compose([ transforms.ToTensor(),
                                   transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

  train_data = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
  test_data = datasets.CIFAR10('./data', train=False, download=True, transform=transform)

  print(len(train_data), len(test_data))

  # statistic about number of samples in each class
  train_stat = {}
  for data, label in train_data:
    if label not in train_stat:
      train_stat[label] = 0
    train_stat[label] += 1
  print(f"Train stat: {train_stat}")

  test_stat = {}
  for data, label in test_data:
    if label not in test_stat:
      test_stat[label] = 0
    test_stat[label] += 1
  print(f"Test stat: {test_stat}")

  # divide the dataset into classes, each with a unique 'label'
  ds_classes = {}
  for i, (_, label) in enumerate(test_data):
    ds_classes.setdefault(label, []).append(i)
    # for each 'label', check if exist (if not then add new KEY of that label to the dictionary with VALUE is empty list [])
    # then add the indice 'i' to the list of according label

  # create non-overlapping subsets of data for each client
  num_clients = 10
  client_train_data = torch.utils.data.random_split(train_data, [len(train_data) // num_clients] * num_clients)
  client_test_data = torch.utils.data.random_split(test_data, [len(test_data) // num_clients] * num_clients)

  # stat data of each client
  for idc in range(num_clients):
    print(f"Client number: {idc}")
    data_client = client_train_data[idc]
    stat_client = {}
    # count the total number of each 'data' (seperated by 'label')
    for data, label in data_client: #list: [[data, label], ...]
      if label not in stat_client:
        stat_client[label] = 0
      stat_client[label] += 1

    # initialize a server
    flserver = FLServer()

    # add clients to server
    for idc in range(num_clients):
      client = FLClient(idc, client_train_data[idc], client_test_data[idc])
      flserver.add_client(client)

    # create a ResNet-18 model (for classification) with 10 classes and send to device (clients)
    global_model = resnet18(weights=None, num_classes = 10).to(device) # weights=None: use random weights rather than pre-trained (fine-tuning)

    num_com_round = 10
    for id_com_round in range(num_com_round):
      print(f"Training at communication round id: {id_com_round}")
      list_local_models = []
      # create a copy of dictionary of current 'state' (value) of parameters (weight and bias) of 'global model'
      global_weights = copy.deepcopy(global_model.state_dict()) # KEYS are param names and VALUES are tensors of param values.

      # for each key, assign new tensor of zeros with the same shape and datatype of float32
      for key in global_weights.keys():
        global_weights[key] = torch.zeros_like(global_weights[key].to(torch.float32))

      for flclient in flserver.clients:
        client_weights, train_metrics = flclient.train(copy.deepcopy(global_model))
        print(f"Finish training client number {flclient.id} with metrics: {train_metrics}")

      # Aggregation step here by FedAvg -> update the global weights by client weights' average update
        for _, key in enumerate(global_weights.keys()):
          update_key = client_weights[key] / num_clients
          global_weights[key].add_(update_key) # not 'add' :))))))) it was tough...

      global_model.load_state_dict(global_weights)
      flserver.test(global_model)

      print('*'*100)

if _name_ == '_main_':
  # set fixed random seed
  random_seed = 42
  torch.manual_speed(random_seed)
  torch.cuda.manual_seed_all(random_seed)

  helper = Helper()
