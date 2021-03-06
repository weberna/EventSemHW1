import torch 
import logging
from sklearn.metrics import f1_score
import time
import argparse
import random
import torch.nn as nn
import numpy as np
import csv
from torch.utils.data import Dataset, DataLoader
from torchtext.vocab import Vocab
from torchtext.data import Field
from torchtext.data import TabularDataset
from torchtext.data import Iterator, BucketIterator
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from models import PredictionModel


def read_data(path, train, test, dev):

    sentence = Field(sequential=True, tokenize=lambda x: x.lower().split(), include_lengths=True)
    pred_arg = Field(sequential=False, lower=True)
    dependency = Field(sequential=False, lower=True)
    label = Field(sequential=False, use_vocab=False)

    fields = [('sentence', sentence), ('pred_head', pred_arg), ('arg_head', pred_arg), ('dependency', dependency), ('label', label)]
    train_data, dev_data, test_data = TabularDataset.splits(path=path, train=train, validation=dev, test=test, format='csv', skip_header=True, fields=fields)
    sentence.build_vocab(train_data, max_size=10000)
    pred_arg.build_vocab(train_data, max_size=2500)
    dependency.build_vocab(train_data)
    return train_data, dev_data, test_data, sentence.vocab, pred_arg.vocab, dependency.vocab



def validation(args, val_batches, model, loss_func):
    model.eval()
    labels = []
    outputs = []

    with torch.no_grad():
        for v_iteration, instance in enumerate(val_batches):
            model_outputs = model(instance) 
            output = torch.sigmoid(model_outputs)
            outputs.append(output.item())
            labels.append(instance.label[0].item())

            
    pred = lambda x: 1 if x >= 0.15 else 0
    predicted = [pred(x) for x in outputs]
    return f1_score(labels, predicted)


def train(args):
    train_data, dev_data, test_data, sentence_vocab, pred_arg_vocab, _ = read_data(args.path, args.train, args.test, args.dev)

    train_iter = BucketIterator(train_data, 64, sort_key = lambda x: len(x.sentence), train=True, shuffle=True, repeat=False, sort_within_batch=True)
    valid_iter = BucketIterator(dev_data, 1, sort_key = lambda x: len(x.sentence), train=False, repeat=False, sort_within_batch=True)
    test_iter = BucketIterator(test_data, 1, train=False, repeat=False)

    model = PredictionModel(128, len(sentence_vocab.itos), len(pred_arg_vocab.itos))


    optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad, model.parameters()), lr=args.lr)

    loss_func = nn.BCEWithLogitsLoss()

    start_time = time.time() #start of epoch 1
    best_valid_loss= -1 
    best_epoch = args.epochs 

    #MAIN TRAINING LOOP
    for curr_epoch in range(args.epochs):
        prev_losses = []
        for iteration, instance in enumerate(train_iter): 

            model.train()
            model.zero_grad()
            model_outputs = model(instance) 
            loss = loss_func(model_outputs, instance.label*1.0)

            loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
            optimizer.step() 

            prev_losses.append(loss.cpu().data)
            prev_losses = prev_losses[-50:]

            if (iteration % args.log_every == 0) and iteration != 0:
                past_50_avg = sum(prev_losses) / len(prev_losses)
                logging.info("Epoch/iteration {}/{}, Past 50 Average Loss {}, Best Val {} at Epoch {}".format(curr_epoch, iteration, past_50_avg, 'NA' if best_valid_loss == float('inf') else best_valid_loss, 'NA' if best_epoch == args.epochs else best_epoch))

            if (iteration % args.validate_after == 0) and iteration != 0:
                logging.info("Running Validation at Epoch/iteration {}/{}".format(curr_epoch, iteration))
                new_valid_loss = validation(args, valid_iter, model, loss_func)
                logging.info("Validation F1 at Epoch/iteration {}/{}: {:.3f} - Best Validation F1: {:.3f}".format(curr_epoch, iteration, new_valid_loss, best_valid_loss))
                if new_valid_loss > best_valid_loss:
                    logging.info("New Validation Best...Saving Model Checkpoint")  
                    best_valid_loss = new_valid_loss
                    best_epoch = curr_epoch
                    #torch.save(model, "{}.epoch_{}.loss_{:.2f}.pt".format(args.save_model, curr_epoch, best_valid_loss))
                    #torch.save(optimizer, "{}.{}.epoch_{}.loss_{:.2f}.pt".format(args.save_model, "optimizer", curr_epoch, best_valid_loss))
                    torch.save(model, "{}".format(args.save_model))
                    torch.save(optimizer, "{}_optimizer".format(args.save_model))

        #END OF EPOCH
        logging.info("End of Epoch {}, Running Validation".format(curr_epoch))
        new_valid_loss = validation(args, valid_iter, model, loss_func)
        logging.info("Validation f1 at end of Epoch {}: {:.3f} - Best Validation f1 : {:.3f}".format(curr_epoch, new_valid_loss, best_valid_loss))
        if new_valid_loss > best_valid_loss:
            logging.info("New Validation Best...Saving Model Checkpoint")  
            best_valid_loss = new_valid_loss
            best_epoch = curr_epoch
            torch.save(model, "{}".format(args.save_model))
            torch.save(optimizer, "{}_optimizer".format(args.save_model))

        if curr_epoch - best_epoch >= args.stop_after:
            logging.info("No improvement in {} epochs, terminating at epoch {}...".format(args.stop_after, curr_epoch))
            logging.info("Best Validation Loss: {:.2f} at Epoch {}".format(best_valid_loss, best_epoch))
            break

             
def test(args):
    train_data, dev_data, test_data, sentence_vocab, pred_arg_vocab, _ = read_data(args.path, args.train, args.test, args.dev)

    test_iter = Iterator(test_data, 1,  sort_key = lambda x: len(x.sentence), train=False, repeat=False)
    model = torch.load(args.save_model)

    model.eval()

    instances_seen = 0
    labels = []
    outputs = []
    with torch.no_grad():
        for v_iteration, instance in enumerate(test_iter):
            model_outputs = model(instance) 
            output = torch.sigmoid(model_outputs)
            outputs.append(output.item())
            labels.append(instance.label[0].item())

    pred = lambda x: 1 if x >= 0.15 else 0
    predicted = [pred(x) for x in outputs]
    print(f1_score(labels, predicted))


            
  


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str)
    parser.add_argument('--train', type=str)
    parser.add_argument('--test', type=str)
    parser.add_argument('--dev', type=str)
    parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate')
    parser.add_argument('--log_every', type=int, default=500)
    parser.add_argument('--validate_after', type=int, default=5000)
    parser.add_argument('--clip', type=float, default=10.0, help='gradient clipping')
    parser.add_argument('--epochs', type=int, default=40, help='upper epoch limit')
    parser.add_argument('--stop_after', type=int, default=3, help='Stop after this many epochs have passed without decrease in validation loss')
    parser.add_argument('--batch_size', type=int, default=32, metavar='N', help='batch size')
    parser.add_argument('--seed', type=int, default=11, help='random seed') 
    parser.add_argument('-save_model', default='model_checkpoint.pt', help="""Model filename""")


    logging.basicConfig(level=logging.INFO)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    train(args)
    test(args)

