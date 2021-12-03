import argparse
import os
from datetime import datetime
from time import time, sleep
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import idr_torch
import pickle
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import pandas as pd
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-05)
    
    args = parser.parse_args()
    
    return args


def main(args):
    
    if idr_torch.rank == 0: start = datetime.now() 
    train(args)
    if idr_torch.rank == 0: print(">>> Training complete in: " + str(datetime.now() - start))      
    
    
class ModelPerso(nn.Module):
    def __init__(self, my_pretrained_model):
        super(ModelPerso, self).__init__()
        self.pretrained = my_pretrained_model
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(in_features=768, out_features=1, bias=True)
        self.act = torch.nn.Sigmoid()
    
    def forward(self, input_ids=None, attention_mask=None,  token_type_ids=None, labels=None):
        x = self.pretrained(input_ids, attention_mask=attention_mask)
        x = self.dropout(x[1])
        x = self.classifier(x)
        x = self.act(x)
        return x
  

class Dataset(Dataset):
    
    def __init__(self, df): 
        self.df = df
            
    def __len__(self):
        return self.df.shape[0]
        
    def __getitem__(self, idx):
        return self.df.iloc[idx, 2], self.df.iloc[idx, 1]
    
    
def get_dataset(source='/gpfswork/idris/sos/ssos022/datasets/imdb/dataset_train.csv', batch_size=32):
      
    df = pd.read_csv(source)  
    dataset = Dataset(df = df)  
    
    
    data_sampler = DistributedSampler(dataset, shuffle=True, num_replicas=idr_torch.size, rank=idr_torch.rank)
    
    
    
    # define DataLoader - optimized parameters
    drop_last = True                       # set to False if it represents important information loss
    num_workers = idr_torch.cpus_per_task  # define number of CPU workers per process
    persistent_workers = True              # set to False if CPU RAM must be released
    pin_memory = True                      # optimize CPU to GPU transfers
    non_blocking = True                    # activate asynchronism to speed up CPU/GPU transfers
    prefetch_factor = 2                    # adjust number of batches to preload
    
    
    dataloader = DataLoader(dataset,
                            sampler=data_sampler,
                            batch_size=batch_size,
                            drop_last=drop_last,
                            num_workers=num_workers,
                            persistent_workers=persistent_workers,
                            pin_memory=pin_memory,
                            prefetch_factor=prefetch_factor)
    
    return dataloader


def get_evaluation(y_true, y_prob):
    # accuracy = accuracy_score(y_true, y_prob)
    y_true = y_true.cpu().detach().numpy()
    y_prob = y_prob.cpu().detach().numpy()
    y_prob = np.where(y_prob <= 0.5, 0, y_prob)
    y_prob = np.where(y_prob > 0.5, 1, y_prob)

    accuracy = 1 - np.sum(np.absolute(y_true - y_prob))/len(y_true)
    return accuracy


def train(args):
    
    dist.init_process_group(backend='nccl', init_method='env://', world_size=idr_torch.size, rank=idr_torch.rank)
    # distribute model
    NTASKS_PER_NODE = int(os.environ['SLURM_NTASKS_PER_NODE'])

    # distribute model
    torch.cuda.set_device(idr_torch.local_rank)
    gpu = torch.device("cuda")
    base_model = torch.load("/gpfswork/idris/sos/ssos022/Models/bert_base_HF/bert-base-uncased.pt")
    model = ModelPerso(my_pretrained_model=base_model).to(gpu)
    ddp_mp_model = DistributedDataParallel(model)
    
    # distribute batch size (mini-batch)
    batch_size_per_proc = args.batch_size
    
    # define loss function (criterion) and optimizer
    criterion = nn.BCELoss()  
    optimizer = torch.optim.AdamW(ddp_mp_model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    
    train_loader = get_dataset(source='/gpfswork/idris/sos/ssos022/datasets/imdb/dataset_train.csv', batch_size=batch_size_per_proc)
    valid_loader = get_dataset(source='/gpfswork/idris/sos/ssos022/datasets/imdb/dataset_val.csv', batch_size=batch_size_per_proc)
    tokenizer = pickle.load( open( "/gpfswork/idris/sos/ssos022/Models/bert_base_HF/tokenizer_bert_unc", "rb" ) )

    # training (timers and display handled by process 0)
    total_step = len(train_loader)
    total_step_valid = len(valid_loader)
    
    first = True
    list_epoch = []
    for epoch in range(args.epochs):
        if idr_torch.rank == 0: start_epoch = time()
        ddp_mp_model.train()
        for i, (texts, labels) in enumerate(train_loader):
            
            if idr_torch.rank == 0: 
                start_dataload = time()

            # distribution of images and labels to all GPUs
            encoding = tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
            input_ids = encoding['input_ids'].to(gpu, non_blocking=True) 
            attention_mask = encoding['attention_mask'].to(gpu, non_blocking=True) 
            labels = labels.to(gpu, non_blocking=True) 
            if idr_torch.rank == 0: 
                stop_dataload = time()
                start_training = time()

            
            # forward pass
            outputs = ddp_mp_model(input_ids=input_ids, attention_mask=attention_mask)
            outputs = outputs.view(1,batch_size_per_proc)[0]
            loss = criterion(outputs, labels.float())
            
                
            # backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if idr_torch.rank == 0: 
                stop_training = time()
                
            if (i + 1) % 1 == 0 and idr_torch.rank == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Time dataload: {:.3f}, Time training step {:.3f}'
                      .format(epoch + 1, args.epochs, i + 1, total_step, loss.item(), stop_dataload-start_dataload, stop_training-start_training))
                      
            if idr_torch.rank == 0: start_dataload = time()
            
        if idr_torch.rank == 0: 
            stop_epoch = time()
            time_epoch = stop_epoch-start_epoch
            list_epoch.append(time_epoch)
            print('\nEpoch [{}/{}], Time epoch: {:.3f}\n'.format(epoch + 1, args.epochs, time_epoch))
              
    #Evaluate model
    ddp_mp_model.eval() 
    with torch.no_grad():
        accuracy = 0
        n = 0
        for i, (texts, labels) in enumerate(valid_loader):
            encoding = tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
            input_ids = encoding['input_ids'].to(gpu, non_blocking=True)
            attention_mask = encoding['attention_mask'].to(gpu, non_blocking=True) 
            labels = labels.to(gpu, non_blocking=True)

            outputs = ddp_mp_model(input_ids=input_ids, attention_mask=attention_mask)
            outputs = outputs.view(1,batch_size_per_proc)[0]

            list_acc = [0 for _ in range(idr_torch.size)]
            acc = get_evaluation(labels, outputs)
            torch.distributed.all_gather_object(list_acc, acc)

            accuracy += np.array(list_acc).sum()
            n += idr_torch.size

        accuracy = accuracy/n
        
        
    if idr_torch.rank == 0:
        print('\nMean time per epoch: ',np.array(list_epoch).mean() ,' Accuracy: ', accuracy)
    
                              
            
if __name__ == '__main__':
    
    args = parse_args()
    # get distributed configuration from Slurm environment
    NODE_ID = os.environ['SLURM_NODEID']
    MASTER_ADDR = os.environ['MASTER_ADDR']
    
    # display info
    if idr_torch.rank == 0:
        print(">>> Training on ", len(idr_torch.hostnames), " nodes and ", idr_torch.size, " processes, master node is ", MASTER_ADDR,'\n')
    print("- Process {} corresponds to GPU {} of node {}\n".format(idr_torch.rank, idr_torch.local_rank, NODE_ID))

    main(args)
