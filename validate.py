import sys
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
#import torch.backends.cudnn as cudnn
from sklearn.metrics import precision_score, f1_score, accuracy_score, recall_score, fbeta_score
from torch.nn.functional import softmax
import argparse

import data_loader, models

device = 'cuda'
data_dir = './Data'
data_dir = '../celeba'

parser = argparse.ArgumentParser(description='Testing arguments')
parser.add_argument('--method', default='guo', type=str, choices=['guo', 'sajid', 'yu'], help='method choice')
parser.add_argument('--load_model', default='checkpoint', type=str, help='Load saved model')

args=parser.parse_args()

def main():
    for i in range(5):
        with LoggingPrinter(f'Results_{args.load_model}_split_{i}.txt'):
            if args.method == 'guo':
                model = models.ModelGuo()
            elif args.method == 'sajid':
                model = models.ModelSajid()
            else:
                model = models.ModelYu()
            print(f"Method: {args.method}, Loaded model: {args.load_model}_split_{i}.pt")
            model.cuda()
            test_loader, _ = data_loader.TestData(data_dir, 64, args.method)
            model.load_state_dict(torch.load(f'./{args.load_model}_split_{i}.pt'))
            pred_test, target_test = test(test_loader, model)

            _, pred_test = torch.max(pred_test.data, 1)
            wrong_pos = (target_test * (target_test != pred_test)).sum()
            wrong_pred = ((target_test != pred_test)).sum()
            
            print(f"Test acc.: {accuracy_score(target_test.cpu(), pred_test.cpu())}")
            print(f"Test prec.: {precision_score(target_test.cpu(), pred_test.cpu())}")
            print(f"Test rec.: {recall_score(target_test.cpu(), pred_test.cpu())}")
            print(f"Test F1: {f1_score(target_test.cpu(), pred_test.cpu())}")  
            print(f"Test F2: {fbeta_score(target_test.cpu(), pred_test.cpu(), beta=2.0)}")
            print(f"Total: {target_test.cpu().shape}, total positive: {target_test.cpu().sum()}")
            print(f"Wrongly predicted: {wrong_pred}, wrongly predicted positive: {wrong_pos}") 
        
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
