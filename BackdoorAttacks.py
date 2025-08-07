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
  def __init__(self, id, train_data, test_data, poison_pattern, target_label):
    self.id = id
    self.train_data = train_data
    self.test_data = test_data
    self.poison_pattern = poison_pattern
    self.target_label = target_label
    self.train_loader = torch.utils.data.DataLoader(self.train_data, batch_size=256, shuffle=True)
    self.test_loader = torch.utils.data.DataLoader(self.test_data, batch_size=256, shuffle=True)

  # Add distributed triggers
  def add_pixel_pattern(self, image):
    for pos in self.poison_patterns:
      image[:, pos[0], pos[1]] = 1
    return image

  def get_poison_batch(self, bptt, fn_dump="./tmp/batch_images.png", evaluation=False):
    # print(f"Starting get poison data by batch for client id: {self.id}")
    images, targets = bptt

    poison_count = 0
    new_images = copy.deepcopy(images)
    new_targets = copy.deepcopy(targets)

    for index in range(0, len(images)):
      if evaluation or index < self.params['poisoning_per_batch']:
        # poison all data when testing;  # poison part of data when training
        new_targets[index] = self.target_label
        new_images[index] = self.add_pixel_pattern(images[index])
        poison_count += 1

    # imshow(make_grid(new_images), save_path=fn_dump)

    new_images = new_images.to(device)
    new_targets = new_targets.to(device).long()

    if evaluation:
      new_images.requires_grad_(False)
      new_targets.requires_grad_(False)

    return new_images, new_targets, poison_count

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

  def train_federated(self, num_clients, local_epochs, lr=0.01):

    server = FlServer(device, self.target_model, self.local_model, num_clients, self.test_loader, self.params)
    # import IPython; IPython.embed(); exit(0)

    for id_client in range(num_clients):
      poison_pattern, target_label = [], 2
      if id_client in self.advasarial_namelist:
        # find id of the attacker, then we have a posion pattern:
        id_attacker = self.advasarial_namelist.index(id_client)
        # poison_pattern = self.params[f'{id_attacker % 4}_poison_pattern']
        poison_pattern = self.params[f'{id_attacker}_poison_pattern']
        # target_label = self.params['poison_label_swap']
        target_label = self.params['poison_labels'][id_attacker]
        print(f"Init attacker with real id: {id_client} and attacker id: {id_attacker} and pattern: {poison_pattern}")

      client = FlClient(id_client, self.train_loaders[id_client], self.test_loader, self.params, poison_pattern, target_label)
      server.add_client(client)
      # time.sleep(0.2)
      # print(f"Done data id: {id_client}")
    # import IPython; IPython.embed(); exit(0)

    print(f"Done loaded data of {len(server.clients)} clients")
    # import IPython; IPython.embed(); exit(0)
    num_rounds = self.params['epochs']
    # self.start_epoch: 201
    save_wandb = self.params.get('save_wandb', False)

    wandb_dict = server.test_all()

    if save_wandb:
      wandb.init(project="neurips24-nba",
        entity="mtuann",
        group=f"{self.params['group_exp_name']}",
        name=f"{self.params['wandb_exp_name']}")
      wandb.log(wandb_dict, step=self.start_epoch - 1)

    for epoch in range(self.start_epoch, num_rounds + 1, 1):
    # for epoch in range(314, 341, 1):
      print(f"Communication round: {epoch}/ {num_rounds}")
      clients_round = self.get_client_by_round(epoch)
      print(f"At communication round: {epoch} participants: {clients_round}")
      server.train(clients_round)
      wandb_dict = server.test_all()
      if save_wandb:
        wandb.log(wandb_dict, step=epoch)

    return server.get_model()

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

  num_clients = 10
  client_train_data = torch.utils.data.random_split(train_data, [len(train_data) // num_clients] * num_clients)
  client_test_data = torch.utils.data.random_split(test_data, [len(test_data) // num_clients] * num_clients)

  # stat data of each client
  for idc in range(num_clients):
    print(f"Client number: {idc}")
    data_client = client_train_data[idc]
    stat_client = {}
    for data, label in data_client:
      if label not in stat_client:
        stat_client[label] = 0
      stat_client[label] += 1

    # initialize a server
    flserver = FLServer()
    for idc in range(num_clients):
      client = FLClient(idc, client_train_data[idc], client_test_data[idc])
      flserver.add_client(client)

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

      # Aggregation step here by FedAvg -> update the global weights by client weights' average update
        for _, key in enumerate(global_weights.keys()):
          update_key = client_weights[key] / num_clients
          global_weights[key].add_(update_key)
      global_model.load_state_dict(global_weights)
      flserver.test(global_model)

      print('*'*100)

if _name_ == '_main_':
  # set fixed random seed
  random_seed = 42
  torch.manual_speed(random_seed)
  torch.cuda.manual_seed_all(random_seed)

  helper = Helper()
