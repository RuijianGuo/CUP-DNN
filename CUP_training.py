import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
from torch.optim import Adam
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import os
from CUP_utils import EarlyStopping, LRScheduler, save_plots
from tqdm import tqdm
def data_preprocessing_train(Xtr, ytr):
    '''
    convert training data into desired format
    '''
    std = StandardScaler()
    data_std = std.fit_transform(Xtr)
    smote = SMOTE(random_state = 1)
    X_smote, y_smote = smote.fit_resample(data_std, ytr)
    X_tensor = torch.from_numpy(X_smote)
    y_tensor = torch.from_numpy(y_smote)    
    return std, X_tensor, y_tensor

def data_preprocessing_test(std, Xt, yt):
    '''
    convert validation data into desired format 
    '''
    X_std = std.transform(Xt)
    X_tensor = torch.from_numpy(X_std)
    y_tensor = torch.from_numpy(yt)
    return X_tensor, y_tensor

def data_loader(X, y, batch_size):
    '''
    function to load training and validation data
    '''
    data_set = TensorDataset(X, y)
    data_loader = DataLoader(data_set, batch_size = batch_size)
    return data_loader

class CUP(nn.Module):
    def __init__(self):
        super(CUP, self).__init__()
        self.linear_relu_stack = nn.Sequential(
        nn.Linear(2387, 1024),
        nn.ReLU(),
        nn.Linear(1024, 512),
        nn.Dropout(0.2),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 32)
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits

def fit(data_loader, model, loss_func, optimizer):
    print('traning')
    model.train()
    size = len(data_loader.dataset)
    num_batches = len(data_loader)
    train_running_loss = 0
    train_running_accuracy = 0
  
    for batch, (batch_x, batch_y) in tqdm(enumerate(data_loader), total = num_batches):
        optimizer.zero_grad()
        y_pred = model(batch_x)
        loss = loss_func(y_pred, batch_y)
        train_running_loss += loss.item()
        _, preds = torch.max(y_pred.data,1)
        train_running_accuracy += (preds == batch_y).sum().item()
        loss.backward()
        optimizer.step() 
    train_loss = train_running_loss / num_batches
    train_accuracy = 100*train_running_accuracy/size
    print(f'Training loss: {train_loss}, Training accuracy:{train_accuracy}')
    return train_loss, train_accuracy
    
def validate(data_loader, model, loss_func):
    print('validating')
    model.eval()
    size = len(data_loader.dataset)
    num_batches = len(data_loader)
    val_running_loss = 0
    val_running_accuracy = 0
    
    with torch.no_grad():
        for batch, (batch_x, batch_y) in tqdm(enumerate(data_loader), total = num_batches):
            y_pred = model(batch_x)
            loss = loss_func(y_pred, batch_y)
            val_running_loss += loss.item()
            _, preds = torch.max(y_pred.data,1)
            val_running_accuracy += (preds==batch_y).sum().item()
        
    val_loss = val_running_loss / num_batches
    val_accuracy = 100*val_running_accuracy/size
    print(f'Validation loss:{val_loss}, Validation accuracy:{val_accuracy}')
    return val_loss, val_accuracy

        
if  __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='CUP model training')
    parser.add_argument('--dataset','-i',dest = 'input', help ='mRNA dataset ')
    parser.add_argument('--learning-rate', '-lr', dest = 'learn_rate', action = 'store_true', help ='training cup model with learnig rate scheduler')
    parser.add_argument('--early-stopping', '-es', dest = 'early_stop', action = 'store_true', help = 'training cup model with early stopping')
    args = parser.parse_args()
    
#   data loading
    rna_data = pd.read_csv(args.input, index_col = 0)
    print(f'Number of columns is:{rna_data.shape[1]}')
    X = rna_data.drop('TCGA_codes', axis = 1)
    y = rna_data['TCGA_codes']
    label = y.unique()
    label.sort()
    label = pd.Series(label).to_dict()
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y)
    std_scaler, Xtr_tensor, ytr_tensor = data_preprocessing_train(X_train, y_train)
    Xt_tensor, yt_tensor = data_preprocessing_test(std_scaler,X_test, y_test)

#   initialize hyperparameters,model,loss function,optimizer
    epochs = 150
    batch_size = 64
    lr = 1e-4

    train_data_loader = data_loader(Xtr_tensor, ytr_tensor, batch_size)
    test_data_loader = data_loader(Xt_tensor, yt_tensor, batch_size)    

    model = CUP().double()
    loss_func = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr)

#    initialize learning rate scheduler and early stopping
    loss_plot_name = 'loss_plot'
    accuracy_plot_name = 'accuracy_plot'
    model_name = 'cup_dnn_model'

    if args.learn_rate:
        lr_scheduler = LRScheduler(optimizer)
        loss_plot_name = 'LR_loss_plot'
        accuracy_plot_name = 'LR_accuracy_plot'
        model_name = 'LR_cup_dnn_model'
    if args.early_stop:
        early_stopping = EarlyStopping()
        loss_plot_name = 'ES_loss_plot'
        accuracy_plot_name = 'ES_accuracy_plot'
        model_name = 'ES_cup_dnn_model'

#   optimization loop
    print('Model training start!')
    train_loss_list = []
    val_loss_list = []
    train_accuracy_list = []
    val_accuracy_list = []
    for epoch in range(epochs):
        train_loss, train_accuracy = fit(data_loader = train_data_loader, model = model, loss_func = loss_func, optimizer = optimizer)
        val_loss, val_accuracy = validate(data_loader = test_data_loader, model = model, loss_func = loss_func)
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        train_accuracy_list.append(train_accuracy)
        val_accuracy_list.append(val_accuracy)
        # learning rate decay
        if args.learn_rate:
            lr_scheduler(val_loss)
        if args.early_stop:
            early_stopping(val_loss, label, std_scaler, epoch, model, optimizer, loss_func)
            if early_stopping.early_stop:
                break    
    print('Model training is finished!')   
    

    print('Saving loss and accuracy plots...')
    save_plots('accuracy', accuracy_plot_name, train_accuracy_list, val_accuracy_list)
    save_plots('loss', loss_plot_name, train_loss_list, val_loss_list)
    
    
