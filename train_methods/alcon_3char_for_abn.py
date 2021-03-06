import sys
import os

import torch
from fastprogress import progress_bar
import pandas as pd

from src.metrics import *
from src.utils import *


def train_alcon_abn(model, optimizer, dataloader, device, loss_fn, eval_fn, epoch, scheduler=None, writer=None, parent=None):
    model.train()
    avg_loss = 0
    avg_accuracy = 0
    three_char_accuracy = 0
    for step, (inputs, targets, indices) in enumerate(progress_bar(dataloader, parent=parent)):
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        logits, ax, attn = model(inputs) # logits.size() = (batch*3, 48)
        preds = logits.view(targets.size(0), 3, -1).softmax(dim=2)
        r_loss = loss_fn(logits, targets.view(-1, targets.size(2)).argmax(dim=1))
        a_loss = loss_fn(ax, targets.view(-1, targets.size(2)).argmax(dim=1))
        loss = r_loss + a_loss
        loss.backward()
        optimizer.step()
        avg_loss += loss.item()
        _avg_accuracy = eval_fn(preds, targets.argmax(dim=2)).item()
        avg_accuracy += _avg_accuracy
        _three_char_accuracy = accuracy_three_character(preds, targets.argmax(dim=2), mean=True).item()
        three_char_accuracy += _three_char_accuracy

        if scheduler is not None:
            if writer is not None:
                writer.add_scalar("data/learning rate", scheduler.get_lr()[0], step + epoch*len(dataloader))
            scheduler.step()

        writer.add_scalars("data/metric/train", {
            'loss': loss.item(),
            'accuracy': _avg_accuracy,
            '3accuracy': _three_char_accuracy
        }, step + epoch*len(dataloader))

    avg_loss /= len(dataloader)
    avg_accuracy /= len(dataloader)
    three_char_accuracy /= len(dataloader)
    print()
    return avg_loss, avg_accuracy, three_char_accuracy


def valid_alcon_abn(model, dataloader, device, loss_fn, eval_fn):
    model.eval()
    with torch.no_grad():
        avg_loss = 0
        avg_accuracy = 0
        three_char_accuracy = 0
        for step, (inputs, targets, indices) in enumerate(dataloader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            logits, ax, attn = model(inputs)  # logits.size() = (batch*3, 48)
            preds = logits.view(targets.size(0), 3, -1).softmax(dim=2)
            r_loss = loss_fn(logits, targets.view(-1, targets.size(2)).argmax(dim=1))
            a_loss = loss_fn(ax, targets.view(-1, targets.size(2)).argmax(dim=1))
            loss = r_loss + a_loss
            avg_loss += loss.item()
            _avg_accuracy = eval_fn(preds, targets.argmax(dim=2)).item()
            avg_accuracy += _avg_accuracy
            _three_char_accuracy = accuracy_three_character(preds, targets.argmax(dim=2), mean=True).item()
            three_char_accuracy += _three_char_accuracy

        avg_loss /= len(dataloader)
        avg_accuracy /= len(dataloader)
        three_char_accuracy /= len(dataloader)
    return avg_loss, avg_accuracy, three_char_accuracy


def pred_alcon_abn(model, dataloader, device):
    output_list = list()
    vocab = get_vocab()

    model.eval()
    with torch.no_grad():
        for step, (inputs, indices) in enumerate(dataloader):
            inputs = inputs.to(device)
            logits,_,_ = model(inputs) # logits.size() = (batch*3, 48)
            logits = logits.view(inputs.size(0), 3, -1) # logits.size() = (batch, 3, 48)
            preds = logits.softmax(dim=2).to('cpu')

            for i in range(inputs.size(0)):
                index = indices[i]
                prediction = dict()
                prediction['ID'] = int(index.item())
                prediction['Unicode1'] = vocab['index2uni'][int(preds[i, 0].argmax(dim=0).item())]
                prediction['Unicode2'] = vocab['index2uni'][int(preds[i, 1].argmax(dim=0).item())]
                prediction['Unicode3'] = vocab['index2uni'][int(preds[i, 2].argmax(dim=0).item())]
                prediction['logit'] = logits[i].detach().to('cpu')
                output_list.append(prediction)
        print()


    return output_list