import torch
from model import DNN
from dataset import COVID19Dataset
from torch.utils.data import dataloader

def train(tr_set, dv_set, model, config, device):
    n_epochs = config['n_epochs']

    # Set up optimizer
    optimizer = getattr(torch.optim, config['optimizer'])(model.parameters(), **config['optim_hparas'])
    min_mse = 1000.
    loss_record = {'train': [], 'dev': []}
    early_stop_cnt = 0
    epoch = 0
    while epoch < n_epochs:
        model.train()
        for x, y in tr_set:
            optimizer.zero_grad()
            x, y = x.to(device), y.to(device)
            pred = model(x)
            mse_loss = model.cal_loss(pred, y)
            mse_loss.backward()
            optimizer.step()
            loss_record['train'].append(mse_loss.detach().cpu().item())

            dev_mse = dev(dv_set, model, device)
            if dev_mse < min_mse:
                min_mse = dev_mse
                print('Saving model (epoch = {:4d}, loss = {:.4f})'.format(epoch + 1, min_mse))
                torch.save(model.state_dict(), config['save_path'])
                early_stop_cnt = 0
            else:
                early_stop_cnt += 1
            epoch += 1
            loss_record['dev'].append(dev_mse)
            if early_stop_cnt > config['early_stop']:
                break
        print('Finished training after {} epochs'.format(epoch))
        return min_mse, loss_record

def dev(dv_set, model, device):
    model.eval()
    total_loss = 0
    for x, y in dv_set:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            pred = model(x)
            mse_loss = model.cal_loss(pred, y)
        total_loss += mse_loss.detach().cpu().item() * len(x)
    total_loss = total_loss / len(dv_set.dataset)

    return total_loss

def testing(tt_set, model, device):
    model.eval()
    preds = []
    for x in tt_set:
        x = x.to(device)
        with torch.no_grad():
            pred = model(x)
            preds.append(pred.detach().cpu())
    preds = torch.cat(preds, dim=0).numpy
    return preds