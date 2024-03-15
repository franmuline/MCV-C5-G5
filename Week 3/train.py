import torch
import numpy as np


def train_epoch(model,optimizer,train_data,criterion,device,log_interval:int=100):
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0
    losses = []
    for n_batch, (img, lab) in enumerate(train_data):
        if not type(img) in (tuple, list):
            images = (img,)
            
        inputs= images.to(device)

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
        
        #Aqui en el codigo original, calculan otras metrics

        if n_batch % log_interval == 0:
            log = 'Train: [{}/{} ({:.0f}%)]\tMean Loss: {:.6f}'.format(
                n_batch * len(images[0]), len(train_data.dataset),
                100. * n_batch / len(train_data), np.mean(losses))
            print(log)
            losses = []
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    train_accuracy = 100. * correct / total
    train_loss /= (n_batch + 1)
    return train_loss, train_accuracy


def val_epoch(model,optimizer,val_data,criterion,device,log_interval:int=100):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    for n_batch, (img, lab) in enumerate(val_data):
        if not type(img) in (tuple, list):
            images = (img,)
            
        inputs= images.to(device)

        labels = None
        if len(lab) > 0:
            labels = lab.to(device)

        outputs = model(*inputs)

        if type(outputs) not in (tuple, list):
            outputs = (outputs,)
            
        loss_inputs = outputs
        if target is not None:
            target = (target,)
            loss_inputs += target

        loss_o = criterion(*loss_inputs)

        loss = loss_o
        if type(loss) in (tuple, list): loss = loss_o[0]

        val_loss += loss.item()

        # En el codigo original aqui calculan mas metrics
        
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    val_accuracy = 100. * correct / total

    return val_loss, val_accuracy