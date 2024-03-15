import torch
import numpy as np


def train_epoch(model, optimizer, train_data, criterion, device, log_interval: int=100, metrics=[]):
    model.train()
    train_loss = 0.0
    losses = []
    for n_batch, (img, lab) in enumerate(train_data):

        images = img
        if not type(img) in (tuple, list):
            images = (img,)
            
        inputs = tuple(i.cuda() for i in images)

        labels = None
        if len(lab) > 0:
            labels = lab.to(device)

        optimizer.zero_grad()
        outputs = model(*inputs)

        if type(outputs) not in (tuple, list):
            outputs = (outputs,)

        loss_inputs = outputs
        if labels is not None:
            labels = (labels,)
            loss_inputs += labels

        loss_o = criterion(*loss_inputs)

        loss = loss_o
        if type(loss_o) in (tuple, list): loss = loss_o[0]

        losses.append(loss.item())
        train_loss += loss.item()
        loss.backward()
        optimizer.step()

        for metric in metrics:
            metric(outputs, labels, loss_o)
        
        #Aqui en el codigo original, calculan otras metrics
        if n_batch % log_interval == 0:
            log = 'Train: [{}/{} ({:.0f}%)]\tMean Loss: {:.6f}'.format(
                n_batch * len(images[0]), len(train_data.dataset),
                100. * n_batch / len(train_data), np.mean(losses))
            for metric in metrics:
                log += '\t{}: {}'.format(metric.name(), metric.value())

            print(log)
            losses = []

    train_loss /= (n_batch + 1)
    return train_loss


def val_epoch(model,optimizer,val_data,criterion,device,log_interval:int=100):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    for n_batch, (img, lab) in enumerate(val_data):
        
        images = img
        if not type(img) in (tuple, list):
            images = (img,)
            
        inputs = tuple(i.cuda() for i in images)

        labels = None
        if len(lab) > 0:
            labels = lab.to(device)

        outputs = model(*inputs)

        if type(outputs) not in (tuple, list):
            outputs = (outputs,)
            
        loss_inputs = outputs
        if labels is not None:
            target = (labels,)
            loss_inputs += target

        loss_o = criterion(*loss_inputs)

        loss = loss_o
        if type(loss) in (tuple, list): loss = loss_o[0]

        val_loss += loss.item()

        # En el codigo original aqui calculan mas metrics

    return val_loss