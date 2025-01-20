#import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torch.optim import Adam, RAdam, SGD
from sklearn.metrics import accuracy_score, fbeta_score
import matplotlib.pyplot as plt

import data_loader, models

device="cuda"
#data_dir = '../yuetal-arraydata-tnf-pr/Data'
data_dir = '../celeba'

parser = argparse.ArgumentParser(description='Training arguments')
parser.add_argument('--method', default='yu', type=str, choices=['guo', 'sajid', 'yu'], help='method choice')
parser.add_argument('--epoch', default=10, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--opt', '--optimizer', default='adam', type=str, choices=['adam', 'sgd'], required=False, help='optimizer choice')
parser.add_argument('--lr', '--learning-rate', default=8e-5, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.8, type=float, metavar='M')
parser.add_argument('--weight-decay', '--wd', default=1e-5, type=float, metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--batch_size', default=32, type=int, help='Training batch size')
parser.add_argument('--schedule', default=5, type=int, help='number of epochs to reduce LR')
parser.add_argument('--save_model', default='checkpoint', type=str, help='filename for saved model')
parser.add_argument('--tolerance', default = 6, type=int, help='Number of epochs before early stoppage')
parser.add_argument('--seed', default = 3546, type=int, help='Seed for RNG')

args = parser.parse_args()


def reset_weights(m):
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            print(f'Reset trainable parameters of layer = {layer}')
            layer.reset_parameters()

def reset_weights(m):
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            print(f'Reset trainable parameters of layer = {layer}')
            layer.reset_parameters()

def main():
    for i in range(5):
        with LoggingPrinter(f'{args.save_model}_split_{i}.txt'):
            #model = None
            train_loader, val_loader, test_loader, _, _, _ = data_loader.LoadData(data_dir, args.batch_size, args.batch_size, i, args.method)
            torch.manual_seed(args.seed)            
            print(f'Seed: {args.seed}, model: {args.method} et al.')
            if args.method == 'guo':
                model = models.ModelGuo().to(device)
            elif args.method == 'sajid':
                model = models.ModelSajid().to(device)
            else:
                model = models.ModelYu().to(device)
            #model.apply(reset_weights)
            #if args.method == 'sajid':
            #    model.load_state_dict(torch.load('sajid_init.pt'))
            if i == 0:
                if args.method == 'sajid':
                    torch.save(model.state_dict(), f'sajid_init.pt')  
                elif args.method == 'guo':
                    torch.save(model.state_dict(), f'guo_init.pt')  
                else:
                    torch.save(model.state_dict(), f'yu_init.pt') 
            if args.method == 'sajid':
                model.load_state_dict(torch.load('sajid_init.pt'))
            elif args.method == 'guo':
                model.load_state_dict(torch.load('guo_init.pt'))
            else:
                model.load_state_dict(torch.load('yu_init.pt'))
                
            model.cuda()
            if args.method == 'yu':
                if args.opt == 'sgd':    
                    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), args.lr,
                                            momentum=args.momentum,
                                            weight_decay=args.weight_decay)
                elif args.opt == 'adam':
                    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
                                            weight_decay=args.weight_decay)
            else:
                optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            criterion = nn.CrossEntropyLoss()
            #scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.7, threshold=1e-2, patience=5, cooldown=3)
            scheduler = StepLR(optimizer, step_size=args.schedule, gamma=0.7)
            check = 0
            track = 0
            old_lr = 0
            losses = []
            print("SPLIT:", i)
            for epoch in range(args.epoch):
                print("EPOCH:", epoch)
                if track < args.tolerance:
                    loss = train(train_loader, model, criterion, optimizer)
                    #train(train_loader, model, criterion, optimizer)
                    losses.append(loss.cpu().detach().numpy())
                    train_pred, train_target = validate(train_loader, model)
                    if epoch % 10000 == 0:
                        print(f"Train total: {train_target.cpu().shape}, Train positive: {train_target.cpu().sum()}")

                    _, train_pred = torch.max(train_pred.data, 1)
                    train_acc = accuracy_score(train_pred.cpu(), train_target.cpu())
                    print(f"Training acc: {train_acc}")

                    pred_val, target_val = validate(val_loader, model)
                    _, pred_val = torch.max(pred_val.data, 1)

                    #print(f'Loss:{loss}')
                    print(f"Val total: {target_val.cpu().shape}, val positive: {target_val.cpu().sum()}")
                    print(f"Val acc: {accuracy_score(target_val.cpu(), pred_val.cpu())}")
                    print(f"Val F2: {fbeta_score(target_val.cpu(), pred_val.cpu(), beta=2.0)}")

                    #scheduler.step(train_acc)
                    scheduler.step()
                    lr_now = optimizer.param_groups[0]['lr']
                    if lr_now != old_lr:
                        print(f'New LR: {lr_now}')
                        old_lr = lr_now

                    check = accuracy_score(train_target.cpu(), train_pred.cpu())
                    if check < 0.95:
                        track = 0
                    else:
                        track += 1
                elif accuracy_score(target_val.cpu(), pred_val.cpu()) < 0.95:
                    track = 0
                else:
                    print("Stop condition reached!")
                    break

            #plt.plot(losses)
            #plt.savefig(f'{args.save_model}_split_{args.split_id}.png')

            torch.save(model.state_dict(), f'{args.save_model}_split_{i}.pt')    

            pred_test, target_test = test(test_loader, model)
            _, pred_test = torch.max(pred_test.data, 1)
            print(f"Test total: {target_test.cpu().shape}, test positive: {target_test.cpu().sum()}")
            print(f"Test acc: {accuracy_score(target_test.cpu(), pred_test.cpu())}")
            print(f"Test F2: {fbeta_score(target_test.cpu(), pred_test.cpu(), beta=2.0)}")
      
def train(train_loader, model, criterion, optimizer):
    model.train()
    
    if args.method == 'yu':
        for (image_set, target) in train_loader:
            target = target.cuda()
            image_set = image_set.cuda()

            p1, p2, p3, pred = model(image_set)
            loss_temporal = (p2 - p1).abs().sum() + (p3 - p2).abs().sum()
            loss = criterion(pred, target)
            loss = (loss + 0.5 * loss_temporal).abs()

            optimizer.zero_grad()
            #loss.retain_grad()
            loss.backward()
            optimizer.step()
        return loss
    else:
        for (image_set, target) in train_loader:
            target = target.cuda()
            image_set = image_set.cuda()
            
            pred = model(image_set)
            loss = criterion(pred, target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return loss
        
def validate(val_loader, model):
    
    pred_all_frames = []
    target_all_frames = []
    
    model.eval()
    
    with torch.no_grad():
        for (image_set, target) in val_loader:
            target = target.cuda()
            image_set = image_set.cuda()
            if args.method == 'yu':
                _, _, _, pred = model(image_set)
            else:
                pred = model(image_set)
            pred_all_frames.append(pred)
            target_all_frames.append(target)
        pred_all = torch.cat(pred_all_frames, dim=0)
        target_all = torch.cat(target_all_frames, dim=0)
        return pred_all, target_all
    
def test(test_loader, model):
    
    pred_all_frames = []
    target_all_frames = []
    
    model.eval()
    
    with torch.no_grad():
        for (image_set, target) in test_loader:
            target = target.cuda()
            image_set = image_set.cuda()
            if args.method == 'yu':
                _, _, _, pred = model(image_set)
            else:
                pred = model(image_set)
            pred_all_frames.append(pred)
            target_all_frames.append(target)
        pred_all = torch.cat(pred_all_frames, dim=0)
        target_all = torch.cat(target_all_frames, dim=0)
        return pred_all, target_all
    
class LoggingPrinter:
    def __init__(self, filename):
        self.out_file = open(filename, "w")
        self.old_stdout = sys.stdout
        sys.stdout = self
    def write(self, text): 
        self.old_stdout.write(text)
        self.out_file.write(text)
    def __enter__(self): 
        return self
    def __exit__(self, type, value, traceback): 
        sys.stdout = self.old_stdout

if __name__ == '__main__':
    main()
            
            
            
        
        

        