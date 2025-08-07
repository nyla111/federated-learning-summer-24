import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.models import resnet18
from collections import defaultdict
import copy
import numpy as np
import random

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
  def __init__(self, id, train_data, test_data, num_clients):
    self.id = id
    self.train_data = train_data
    self.test_data = test_data
    self.num_clients = num_clients

    indices_per_participant = self.sample_dirichlet_train_data(train_data, num_clients, alpha=0.5)
    self.train_loader = [self.get_train(indices) for pos, indices in indices_per_participant.items()]

    self.test_loader = torch.utils.data.DataLoader(self.test_data, batch_size=256, shuffle=True)

  def train(self, model):
    optimizer = optim.SGD(model.parameters(), lr = 0.01)
    model.train()

    total_loss, correct, num_sample = 0, 0, 0
    for idb, (data, target) in enumerate(self.train_loader[self.id]):
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

  def sample_dirichlet_train_data(self, train_dataset, no_participants, alpha=0.5):
    print(f"SAMPLE DIRICHLET TRAIN DATA: {no_participants} with alpha = {alpha}")

    # divide dataset into classes of unique labels
    ds_classes = {}
    for i, (_, label) in enumerate(train_dataset):
      ds_classes.setdefault(label, []).append(i)

    class_size = len(ds_classes[0])  # for cifar: 5000
    per_participant_list = defaultdict(list)
    no_classes = len(ds_classes.keys())  # for cifar: 10

    print(f"Class size: {class_size} - No. classes: {no_classes} - No. participants: {no_participants}")
    image_nums = []  # the number of images per class for each participant

    # loop through the classes:
    for n in range(no_classes):
      image_num = []  # the number of images for current class per participant
      random.shuffle(ds_classes[n])  # shuffle the list of images in class n

      # generate probabilities for each participant using Dirichlet distribution
      sampled_probabilities = class_size * np.random.dirichlet(np.array(no_participants * [alpha]))

      # move images from dataset to each clients
      for user in range(no_participants):
        no_imgs = int(round(sampled_probabilities[user]))  # calculate the number of images for this participant based on its 'sampled_prob'
        sampled_list = ds_classes[n][:min(len(ds_classes[n]), no_imgs)]  # get the subset of images based on the number calculated
        image_num.append(len(sampled_list))  # append the number of images actually assigned to the list
        per_participant_list[user].extend(sampled_list)  # add the selected images to the participant's own list
        ds_classes[n] = ds_classes[n][min(len(ds_classes[n]), no_imgs):]  # remove the assigned images from the class list

      # store the list of images per participant for this class
      image_nums.append(image_num)

    return per_participant_list

  def get_train(self, indices):
    # given indices, return a dataloader
    train_loader = torch.utils.data.DataLoader(self.train_data, batch_size=256, shuffle = False,
                                                sampler=torch.utils.data.sampler.SubsetRandomSampler(indices), pin_memory=True)
    return train_loader

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

  num_clients = 10

  flserver = FLServer()

  #split data
  client_train_data = torch.utils.data.random_split(train_data, [len(train_data) // num_clients] * num_clients)
  client_test_data = torch.utils.data.random_split(test_data, [len(test_data) // num_clients] * num_clients)

  for idc in range(num_clients):
    client = FLClient(idc, client_train_data[idc], client_test_data[idc], num_clients)
    flserver.add_client(client)

  # stat data of each client
  for idc in range(num_clients):
    print(f"Client number: {idc}")
    data_client = client_train_data[idc]
    stat_client = {}

    for data, label in data_client:
      if label not in stat_client:
        stat_client[label] = 0
      stat_client[label] += 1

    global_model = resnet18(weights=None, num_classes = 10).to(device)

    num_com_round = 10
    for id_com_round in range(num_com_round):
      print(f"Training at communication round id: {id_com_round}")
      list_local_models = []
      global_weights = copy.deepcopy(global_model.state_dict())

      for key in global_weights.keys():
        global_weights[key] = torch.zeros_like(global_weights[key].to(torch.float32))

      for flclient in flserver.clients:
        client_weights, train_metrics = flclient.train(copy.deepcopy(global_model))
        print(f"Finish training client number {flclient.id} with metrics: {train_metrics}")
        for _, key in enumerate(global_weights.keys()):
          update_key = client_weights[key] / num_clients
          global_weights[key].add_(update_key)

      global_model.load_state_dict(global_weights)
      flserver.test(global_model)

      print('*'*100)

if __name__ == '__main__':
  # set fixed random seed
  random_seed = 42
  torch.manual_speed(random_seed)
  torch.cuda.manual_seed_all(random_seed)

  helper = Helper()
