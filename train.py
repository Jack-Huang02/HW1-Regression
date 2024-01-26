import torch
from model import DNN
from dataset import COVID19Dataset
from torch.utils.data import dataloader
from train_dev_test import train, dev, testing
from torch.utils.data import DataLoader
from plot import plot_pred, plot_learning_curve
import os
import numpy as np

myseed = 42069  # set a random seed for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)

def prep_dataloader(path, mode, batch_size, n_jobs=0, target_only=False):
    dataset = COVID19Dataset(path, mode=mode, target_only=target_only)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=(mode == 'train'),
                            drop_last=False,
                            num_workers=n_jobs,
                            pin_memory=True)
    return dataloader

device = 'cuda' if torch.cuda.is_available() else 'cpu'
os.makedirs('models', exist_ok=True)
target_only = False

config = {
    'n_epochs': 3000,
    'batch_size': 270,
    'optimizer': 'SGD',
    'optim_hparas': {
        'lr': 0.001,
        'momentum': 0.9
    },
    'early_stop': 200,
    'save_path': 'models/model.pth'
}

tr_path = 'data/covid.train.csv'
tt_path = 'data/covid.test.csv'
tr_set = prep_dataloader(tr_path,
                         mode='train',
                         batch_size=config['batch_size'],
                         target_only=target_only
                         )
dv_set = prep_dataloader(tr_path,
                         mode='dev',
                         batch_size=config['batch_size'],
                         target_only=target_only
                         )
tt_set = prep_dataloader(tt_path,
                         mode='test',
                         batch_size=config['batch_size'],
                         target_only=target_only
                         )
model = DNN(tr_set.dataset.dim).to(device)

model_loss, model_loss_record = train(tr_set, dv_set, model, config, device)

plot_learning_curve(model_loss_record, 'loss')

del model

model = DNN(tr_set.dataset.dim).to(device)
ckpt = torch.load(config['save_path'], map_location='cpu')
model.load_state_dict(ckpt)
plot_pred(dv_set, model, device)
