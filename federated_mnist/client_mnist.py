from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets
import argparse
from argparse import Namespace

import flwr as fl
import time

from utils import get_kl_loss, get_net
# from signSGD import signSGD


# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument(
    '--model-type',
    default='full',
    choices=['CP', 'TensorTrain', 'TensorTrainMatrix','Tucker','full'],
    type=str)
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--rank-loss', type=bool, default=False)
parser.add_argument('--kl-multiplier', type=float, default=1.0) #account for the batch size,dataset size, and renormalize
parser.add_argument('--em-stepsize', type=float, default=1.0) #account for the batch size,dataset size, and renormalize
parser.add_argument('--no-kl-epochs', type=int, default=5)
parser.add_argument('--warmup-epochs', type=int, default=50)
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--rank', type=int, default=20)
parser.add_argument('--prior-type', type=str, default='log_uniform')
parser.add_argument('--eta', type=float, default=1.0)
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 1.0)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--dry-run', action='store_true', default=False,
                    help='quickly check a single pass')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=1000, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save-model', action='store_true', default=False,
                    help='For Saving the current Model')
parser.add_argument('--tensorized', type=bool, default=False,
                    help='Run the tensorized model')
parser.add_argument('--optimizer', type=str, default='SGD',
                    help='Optimizer (default: SGD)')
args = parser.parse_args()
use_cuda = not args.no_cuda and torch.cuda.is_available()


# args = Namespace(batch_size=128, dry_run=False, em_stepsize=1.0, epochs=100, eta=1.0, kl_multiplier=5e-05, log_interval=1000, lr=0.001, model_type='TensorTrainMatrix', no_cuda=False, no_kl_epochs=5, prior_type='log_uniform', rank=20, rank_loss=True, save_model=False, seed=1, tensorized=False, test_batch_size=1000, warmup_epochs=50)

# Standard
args = Namespace(batch_size=128, 
                    dry_run=False, 
                    em_stepsize=1.0, 
                    epochs=10, eta=1.0, 
                    kl_multiplier=0.0,
                    log_interval=1000, 
                    # lr=0.001, 
                    lr=0.0001, 
                    model_type='full',
                    no_cuda=False, 
                    no_kl_epochs=5,
                    # optimizer='SGD',
                    optimizer='Adam',
                    prior_type='log_uniform', 
                    rank=20, 
                    rank_loss=False, 
                    save_model=False, 
                    seed=1, 
                    tensorized=False, 
                    test_batch_size=1000, 
                    warmup_epochs=50)

# # TTM
# args = Namespace(batch_size=128, 
#                     dry_run=False, 
#                     em_stepsize=1.0, 
#                     epochs=10, eta=1.0, 
#                     kl_multiplier=5e-05, 
#                     log_interval=1000, 
#                     lr=0.0001, 
#                     # lr=0.001,
#                     model_type='TensorTrainMatrix', 
#                     no_cuda=False, 
#                     no_kl_epochs=5, 
#                     # optimizer='SGD',
#                     optimizer='Adam',
#                     prior_type='log_uniform', 
#                     rank=20, 
#                     rank_loss=True, 
#                     save_model=False, 
#                     seed=1, 
#                     tensorized=True, 
#                     test_batch_size=1000, 
#                     warmup_epochs=50)


print(args)

# torch.manual_seed(args.seed)

device = torch.device("cuda:0" if use_cuda else "cpu")

def load_data():
    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    trainset = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
    testset = datasets.MNIST('../data', train=False,
                       transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset,**train_kwargs)
    testloader = torch.utils.data.DataLoader(testset, **test_kwargs)
    num_examples = {"trainset" : len(trainset), "testset" : len(testset)}
    return trainloader, testloader, num_examples

def train(model, trainloader, epochs, local_epoch, optimizer="signSGD"):
    learning_rate = args.lr

    if optimizer == "signSGD":
        optimizer = signSGD(model.parameters(), lr=args.lr)
    elif optimizer == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr)

    
    for epoch in range(local_epoch, local_epoch+epochs):
        ## Timing
        t = time.time()

        model.train()

        if optimizer == "signSGD":
            if epoch <= args.warmup_epochs:
                if epoch % 10 == 0:
                    learning_rate /= 10.0
                optimizer = signSGD(model.parameters(), lr=learning_rate)
            else:
                optimizer = optim.SGD(model.parameters(), lr=args.lr)

        for batch_idx, (data, target) in enumerate(trainloader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            loss = F.nll_loss(model(data), target)

            if args.rank_loss:
                ard_loss = get_kl_loss(model,args,epoch)
                loss += ard_loss

            loss.backward()
            optimizer.step()
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(trainloader.dataset),
                    100. * batch_idx / len(trainloader), loss.item()))
                if args.dry_run:
                    break
        
        # Timing
        epoch_train_time = time.time()-t
        print("Epoch train time {:.4f} seconds".format(epoch_train_time))
        

        # print("******Tensor Ranks*******")
        # print(model.fc1.tensor.estimate_rank())
        # print(model.fc2.tensor.estimate_rank())
        # print("******Param Savings*******")
        # param_savings_1 = model.fc1.tensor.get_parameter_savings(threshold=1e-4)
        # param_savings_2 = model.fc2.tensor.get_parameter_savings(threshold=1e-4)
        # full_params = 784*512+512+512*10+10
        # print(param_savings_1,param_savings_2)
        # total_savings = sum(param_savings_1)+sum(param_savings_2)
        # print("Savings {} ratio {}".format(total_savings,full_params/(full_params-total_savings)))

        # print("******End rounds stats*******\n")
    # return loss.item()
    return local_epoch

def test(model, testloader):

    ## Timing
    t = time.time()

    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in testloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(testloader.dataset)
    accuracy = 100. * correct / len(testloader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(testloader.dataset),
        accuracy))
    
    # Timing
    epoch_inference_time = time.time()-t
    print("Epoch inference time {:.4f} seconds".format(epoch_inference_time))
    
    return test_loss, accuracy


model = get_net(args).to(device)
print(model)
trainloader, testloader, num_examples = load_data()

class MNISTClient(fl.client.NumPyClient):
    def __init__(self):
        self.cur_timer = time.time()
        self.local_epoch = 0
        self.train_time = 0.0
        self.inference_time = 0.0
        self.communication_time = 0.0

    def get_parameters(self):        
        # print("*******************print state dict************************")
        # for val_name, val in model.state_dict().items():
        #     print(val_name)
        #     print(val.shape)
        # print("*****************end print state dict**********************")
        return [val.cpu().numpy() for _, val in model.state_dict().items()]

    # Update the local model weights with the parameters received from the server
    def set_parameters(self, parameters):
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=False)
    
    def fit(self, parameters, config):
        t1 = time.time() - self.cur_timer
        self.communication_time += t1
        self.set_parameters(parameters)
        print("[Training] Receive parameters after {:.4f} seconds, total communication time: {:.4f} seconds".format(t1, self.communication_time))
        self.cur_timer = time.time()

        self.local_epoch = train(model, trainloader, epochs=args.epochs, local_epoch=self.local_epoch, optimizer=args.optimizer)
        t2 = time.time() - self.cur_timer
        self.train_time += t2
        print("Finish local training after {:.4f} seconds, total train time: {:.4f} seconds".format(t2, self.train_time))

        self.local_epoch += args.epochs
        self.cur_timer = time.time()
        return self.get_parameters(), num_examples["trainset"], {}
    
    def evaluate(self, parameters, config):
        t1 = time.time() - self.cur_timer
        self.communication_time += t1
        self.set_parameters(parameters)
        print("[Inference] Receive parameters after {:.4f} seconds, total communication time: {:.4f} seconds".format(t1, self.communication_time))
        self.cur_timer = time.time()

        loss, accuracy = test(model, testloader)
        t2 = time.time() - self.cur_timer
        self.inference_time += t2
        print("Finish inference after {:.4f} seconds, total train time: {:.4f} seconds".format(t2, self.inference_time))

        self.cur_timer = time.time()
        return float(loss), num_examples["testset"], {"accuracy": float(accuracy)}

fl.client.start_numpy_client("[::]:8080", client=MNISTClient())