import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.models import resnet18
import copy
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class FLServer:
  def __init__(self) -> None:
    self.clients = []

  def add_client(self, fl_client):
    self.clients.append(fl_client)

  def test(self, model):
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

  def train(self, model):
    optimizer = optim.SGD(model.parameters(), lr = 0.01)
    model.train()

    total_loss, correct, num_sample = 0, 0, 0
    for idb, (data, target) in enumerate(self.train_loader):
      data, target = data.to(device), target.to(device)
      optimizer.zero_grad()
      output = model(data)

      loss = F.cross_entropy(output, target)
      total_loss += loss.item() * len(data)
      num_sample += len(data)

      pred = output.argmax(1, keepdim = True)

      correct += pred.eq(target.view_as(pred)).sum().item()

      loss.backward()
      optimizer.step()

    total_loss /= num_sample
    accuracy = 100.* correct / num_sample

    return model.state_dict(), {'loss': total_loss, 'accuracy': accuracy, '#data': num_sample, 'correct': correct}

  def test(self, model):
    model.eval()

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
  # CIFAR-10 dataset
  transform = transforms.Compose([ transforms.ToTensor(),
                                   transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

  train_data = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
  test_data = datasets.CIFAR10('./data', train=False, download=True, transform=transform)

  print(len(train_data), len(test_data))


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

  ds_classes = {}
  for i, (_, label) in enumerate(test_data):
    ds_classes.setdefault(label, []).append(i)

  num_clients = 10
  client_train_data = torch.utils.data.random_split(train_data, [len(train_data) // num_clients] * num_clients)
  client_test_data = torch.utils.data.random_split(test_data, [len(test_data) // num_clients] * num_clients)


  for idc in range(num_clients):
    print(f"Client number: {idc}")
    data_client = client_train_data[idc]
    stat_client = {}
    # count the total number of each 'data' (seperated by 'label')
    for data, label in data_client:
      if label not in stat_client:
        stat_client[label] = 0
      stat_client[label] += 1


    flserver = FLServer()
    for idc in range(num_clients):
      client = FLClient(idc, client_train_data[idc], client_test_data[idc])
      flserver.add_client(client)

    global_model = resnet18(weights=None, num_classes = 10).to(device)

    num_com_round = 10
    for id_com_round in range(num_com_round):
      print(f"Training at communication round id: {id_com_round}")
      list_local_models = []
      # create a copy of dictionary of current 'state' (value) of parameters (weight and bias) of 'global model'
      global_weights = copy.deepcopy(global_model.state_dict()) # KEYS are param names and VALUES are tensors of param values.
      copy_gw = copy.deepcopy(global_weights)

      for key in global_weights.keys():
        global_weights[key] = global_weights[key].to(torch.float32)

      for flclient in flserver.clients:
        client_weights, train_metrics = flclient.train(copy.deepcopy(global_model))
        print(f"Finish training client number {flclient.id + 1} with metrics: {train_metrics}")

      # Aggregation step here by FedAvg
        for _, key in enumerate(global_weights.keys()):
          update_key = (client_weights[key].sub_(copy_gw[key])) / num_clients
          global_weights[key].add_(update_key)

      global_model.load_state_dict(global_weights)
      flserver.test(global_model)

      print('*'*100)

if __name__ == '__main__':
  random_seed = 42
  torch.manual_speed(random_seed)
  torch.cuda.manual_seed_all(random_seed)

  helper = Helper()
