import argparse
import os
import numpy as np
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer
import pickle
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments
from transformers import Trainer
import idr_torch
from datetime import datetime
from time import time, sleep


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-05)
    
    args = parser.parse_args()
    
    return args


def main(args):
    os.environ["WANDB_DISABLED"] = "true"
    
    if idr_torch.rank == 0: start = datetime.now() 
    train(args)
    if idr_torch.rank == 0: print(">>> Training complete in: " + str(datetime.now() - start))

    
def train(args):
    dataset = load_dataset('csv', data_files={'train': '/gpfswork/idris/sos/ssos022/datasets/imdb/dataset_train.csv',
                                             'valid': '/gpfswork/idris/sos/ssos022/datasets/imdb/dataset_val.csv'},
                          cache_dir='/gpfswork/idris/sos/ssos022/.cache')

    tokenizer = pickle.load(open('/gpfswork/idris/sos/ssos022/Models/Auto_model_HF/auto_tokenizer_bert_base_class.hf', 'rb'))

    def tokenize_function(examples):
        return tokenizer(examples["review"], padding="max_length", truncation=True)

    tokenized_train = dataset['train'].map(tokenize_function, batched=True)
    tokenized_valid = dataset['valid'].map(tokenize_function, batched=True)

    metric = load_metric("accuracy", cache_dir='/gpfswork/idris/sos/ssos022/.cache')

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)


    model = pickle.load(open('/gpfswork/idris/sos/ssos022/Models/Auto_model_HF/auto_model_bert_base_class.hf', 'rb'))
    training_args = TrainingArguments("checkpoints", 
                                      evaluation_strategy="epoch", 
                                      per_device_train_batch_size=args.batch_size,
                                      per_device_eval_batch_size=args.batch_size, 
                                      logging_strategy='no', 
                                      save_strategy='no',
                                      num_train_epochs=args.epochs)
    trainer = Trainer(model=model, 
                      args=training_args, 
                      train_dataset=tokenized_train, 
                      eval_dataset=tokenized_valid, 
                      compute_metrics=compute_metrics)

    trainer.train()
    
    
if __name__ == '__main__':
    
    args = parse_args()
    main(args)