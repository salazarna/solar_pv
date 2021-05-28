#Built-in Python Modules
import datetime
import inspect
import os
import gc
import csv
import math
import random
import warnings
from calendar import monthrange
warnings.filterwarnings(action='ignore')

#Python add-ons
import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib import dates as mpl_dates
from tqdm import tqdm
from scipy import stats
from sklearn.linear_model import LinearRegression, TheilSenRegressor
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score

#PyTorch built-ins
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision.transforms import ToTensor

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

########################################
# 1. Datasets Preparation
########################################

#SysA
file_name = './data_SysA.csv'
data = pd.read_csv(filepath_or_buffer=file_name, 
                   sep=',',
                   decimal='.',
                   header='infer')
data_SysA = pd.DataFrame(data)
data_SysA.set_index(keys='Unnamed: 0', drop=True, inplace=True)
data_SysA.index.name = None

#SysB
'''file_name = './data_SysB.csv'
data = pd.read_csv(filepath_or_buffer=file_name, 
                   sep=',',
                   decimal='.',
                   header='infer')
data_SysB = pd.DataFrame(data)
data_SysB.set_index(keys='Unnamed: 0', drop=True, inplace=True)
data_SysB.index.name = None'''

#Remove zero-irradiance rows
'''
index_names = data_SysA[(data_SysA['Hour'] < 6) | (data_SysA['Hour'] >= 18)].index
data_SysA.drop(index_names, inplace=True)
data_SysA.reset_index(inplace=True)
'''

### 1.1. Train dataset
#### 1.1.1. K-fold splits (10 K-fold)
'''
Randomly split the dataframe into (train+val)-test
    - df: Dataframe to be split
    - kfolds: Scikit-learn KFold list (indexes to be split in dataframes)
    - split: Number of K-fold split to generate dataframes, i.e. [0 - n_splits) from KFold function.
Note: Seed of 'random_state' varies from [0 - len(kfolds)) and makes the train+val split stochastic.
'''
#Function definition
def kfold_dataframes(df, kfolds, split, verbose):
    #Seed
    seed = np.random.randint(low=0, high=len(kfolds), size=None)

    #Datasets: Train - Test from K-fold
    train_val = df.loc[kfolds[0]]
    test = df.loc[kfolds[1]]

    #Datasets: Train 85 - Val 15
    train, val = train_test_split(train_val, test_size=0.15, random_state=seed)

    #Size checking
    if verbose:
        print(f'Train shape: {np.shape(train)}')
        print(f'Val shape: {np.shape(val)}')
        print(f'Test shape: {np.shape(test)}')

    return train, val, test

#### 1.1.2. Set inputs (X) and targets (Y_hat) variables
#Inputs (X) dataset
#X_cols = ['Irrad_poa', 'Tamb', 'Tmod', 'Hour', 'Month', 'Year', 'k', 'i_mp', 'v_mp', 'p_mp', 'ac_power']
X_cols = ['Irrad_poa', 'Tamb', 'Tmod', 'Hour', 'Month', 'k', 'i_mp', 'v_mp', 'p_mp', 'ac_power']

#Targets (Yhat) dataset
Yhat_cols = ['i_sc'] #['IL', 'i_sc', 'v_oc', 'mod_eff_exp', 'inv_eff_exp']

#### 1.1.3. Train inputs (X) and targets (Y_hat) normalization (0 to 1 scaling)
#Scaler
sc = MinMaxScaler(feature_range=(0,1), copy=True)

def normalization(train, val, test, X_cols, Yhat_cols, scaler, scale_targets):
    #Numpy inputs (X) datasets
    X_train = pd.DataFrame(sc.fit_transform(train[X_cols]), columns=X_cols, index=train.index)
    X_val = pd.DataFrame(sc.fit_transform(val[X_cols]), columns=X_cols, index=val.index)
    X_test = pd.DataFrame(sc.fit_transform(test[X_cols]), columns=X_cols, index=test.index)

    #Numpy targets (Yhat) dataset
    if scale_targets:
        Yhat_train = pd.DataFrame(sc.fit_transform(train[Yhat_cols]), columns=Yhat_cols, index=train.index)
        Yhat_val = pd.DataFrame(sc.fit_transform(val[Yhat_cols]), columns=Yhat_cols, index=val.index)
        Yhat_test = pd.DataFrame(sc.fit_transform(test[Yhat_cols]), columns=Yhat_cols, index=test.index)
    else:
        Yhat_train = pd.DataFrame(train[Yhat_cols], columns=Yhat_cols, index=train.index)
        Yhat_val = pd.DataFrame(val[Yhat_cols], columns=Yhat_cols, index=val.index)
        Yhat_test = pd.DataFrame(test[Yhat_cols], columns=Yhat_cols, index=test.index)
    
    return X_train, X_val, X_test, Yhat_train, Yhat_val, Yhat_test

#### 1.1.4. Numpy arrays to PyTorch Tensors
#PyTorch tensors
def df_to_tensor(X_train, X_val, X_test, Yhat_train, Yhat_val, Yhat_test, verbose):
    #Train
    inputs_train = torch.tensor(X_train.values, dtype=torch.float32)
    targets_train = torch.tensor(Yhat_train.values, dtype=torch.float32)

    if verbose:
        print('TRAIN')
        print(f'Inputs Size: {inputs_train.size()} | dtype: {inputs_train.dtype}')
        print(f'Targets Size: {targets_train.size()} | dtype: {targets_train.dtype} \n')

    #Val
    inputs_val = torch.tensor(X_val.values, dtype=torch.float32)
    targets_val = torch.tensor(Yhat_val.values, dtype=torch.float32)
    
    if verbose:
        print('VAL')
        print(f'Inputs Size: {inputs_val.size()} | dtype: {inputs_val.dtype}')
        print(f'Targets Size: {targets_val.size()} | dtype: {targets_val.dtype} \n')

    #Test
    inputs_test = torch.tensor(X_test.values, dtype=torch.float32)
    targets_test = torch.tensor(Yhat_test.values, dtype=torch.float32)

    if verbose:
        print('TEST')
        print(f'Inputs Size: {inputs_test.size()} | dtype: {inputs_test.dtype}')
        print(f'Targets Size: {targets_test.size()} | dtype: {targets_test.dtype} \n')

    return inputs_train, targets_train, inputs_val, targets_val, inputs_test, targets_test

### 1.2. Dataset and DataLoader
'''The `TensorDataset` allows us to access a small section of the training data using the array indexing notation (`[0:3]` in the above code). It returns a tuple with two elements. The first element contains the input variables for the selected rows, and the second contains the targets.

We'll also create a `DataLoader`, which can split the data into batches of a predefined size while training. It also provides other utilities like shuffling and random sampling of the data.'''

#Define datasets
def datasets(inputs_train, targets_train, inputs_val, targets_val, inputs_test, targets_test):
    train = TensorDataset(inputs_train, targets_train)
    val = TensorDataset(inputs_val, targets_val)
    test = TensorDataset(inputs_test, targets_test)

    return train, val, test

#Define train DataLoader
def dataloader(train, val, test, batch_size, num_workers, pin_memory):
    train_loader = DataLoader(dataset=train, 
                              batch_size=batch_size, 
                              shuffle=True,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    val_loader = DataLoader(dataset=val, 
                            batch_size=batch_size*2, 
                            shuffle=False,
                            num_workers=num_workers,
                            pin_memory=pin_memory)

    test_loader = DataLoader(dataset=test, 
                             batch_size=1, #batch_size=1
                             shuffle=False,
                             num_workers=num_workers,
                             pin_memory=pin_memory)
    
    return train_loader, val_loader, test_loader
'''
#To see the first inputs-targets from DataLoader
for xb, yb in train_loader:
    print('xb.device:', xb.device)
    print('xb:', xb)
    print('yb:', yb)
    break
'''

########################################
# 2. Neural Network
########################################

### 2.1. Get device (GPU or CPU)
'''Helper function to ensure that our code uses the GPU if available and defaults to using the CPU if it isn't.'''

def get_default_device():
    '''Pick GPU if available, else CPU'''
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

device = get_default_device()
print(device)

### 2.2. Hyper-parameters
#ANN parameters
input_size = len(X_cols)
hidden1_size = 15 #[6, 9, 12, 15]
hidden2_size = 15 #[6, 9, 12, 15]
output_size = len(Yhat_cols)
dropout = 0.2

#Train parameters
num_epochs = 120
learning_rate = 0.1 #Initial LR
batch_size = 5000 #Datapoints that will go through the ANN

### 2.3. ANN Design
#Defining the model
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden1_size, hidden2_size, output_size):
        super(NeuralNet, self).__init__()
        
        #First layer: Linear -> ReLU -> Dropout
        self.l1 = nn.Linear(in_features=input_size, out_features=hidden1_size, bias=True)
        self.activ1 = nn.LeakyReLU()
        self.dropout1 = nn.Dropout(p=dropout, inplace=False)

        #Second layer: Linear -> ReLU -> Dropout
        self.l2 = nn.Linear(in_features=hidden1_size, out_features=hidden2_size, bias=True)
        self.activ2 = nn.Sigmoid()
        self.dropout2 = nn.Dropout(p=dropout, inplace=False)

        #Third layer: Linear --> Prediction (output)
        self.l3 = nn.Linear(in_features=hidden2_size, out_features=output_size, bias=True)
    
    def forward(self, xb):
        out = self.l1(xb)
        out = self.activ1(out)
        out = self.dropout1(out)

        out = self.l2(out)
        out = self.activ2(out)
        out = self.dropout2(out)

        out = self.l3(out)
        return out
    
    def predict_step(self, batch):
        #Forward pass
        inputs, targets = batch
        inputs, targets = inputs.to(device), targets.to(device)
        predict = self(inputs) #Generate predictions
        loss = criterion(predict, targets) #Calculate loss
        return loss

### 2.4. Model, Loss and Optimizer
#Model
model_ANN = NeuralNet(input_size=input_size,
                      hidden1_size=hidden1_size,
                      hidden2_size=hidden2_size,
                      output_size=output_size).to(device)
print(model_ANN)

#Loss function (criterion)
criterion = torch.nn.MSELoss()

#Optimizer
opt_Adam = torch.optim.Adam(model_ANN.parameters(), 
                            lr=learning_rate,
                            betas=(0.9, 0.999),
                            eps=1e-08,
                            weight_decay=0, 
                            amsgrad=False)

#Learning rate scheduler
sch = lr_scheduler.ReduceLROnPlateau(optimizer=opt_Adam,
                                     mode='min',
                                     factor=0.1,
                                     patience=(num_epochs/100)+2,
                                     threshold=0.0001,
                                     threshold_mode='rel',
                                     cooldown=0,
                                     min_lr=0,
                                     eps=1e-08, 
                                     verbose=True)

### 2.5. Reset weights
'''Try resetting model weights to avoid weight leakage.'''
def reset_weights(model):
    for layers in model.children():
        for layer in layers:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

def weight_reset(model):
    if isinstance(model, nn.Conv2d) or isinstance(model, nn.Linear):
        model.reset_parameters()

### 2.6. Train the model
#Dict to store losses
models = {}

def fit_model(df, kfolds, model, criterion, optimizer, epochs, lr, batch_size, scheduler, X_cols, Yhat_cols, sc, scale_targets, verb_step, verbose):
    #Model train and validation  
    print('Start model training.')
    print('#'*25)

    m_name = f'{Yhat_cols[0]}_{hidden1_size}_{hidden2_size}'
    models[m_name] = {}

    for split, fold in enumerate(kfolds):
        f_name = f'Fold_{split+1}'
        print('-'*25)
        print(f_name)
        print('-'*25)

        #Restart model parameters
        model = model_ANN
        #model.apply(reset_weights)
        model.apply(weight_reset)

        #Lists data
        models[m_name][f_name] = {'train_loss': [], 'val_loss': [], 'predictions': [], 'Yhat_test': [], 'mse': [], 'rmse': [], 'r_square': []}
        train_loss = []
        val_loss = []

        #Step 1: Train, val and test dataframes
        train, val, test = kfold_dataframes(df=df, kfolds=fold, split=split, verbose=verb_step)

        #Step 2: Normalization
        X_train, X_val, X_test, Yhat_train, Yhat_val, Yhat_test = normalization(train=train, 
                                                                                val=val, 
                                                                                test=test, 
                                                                                X_cols=X_cols, 
                                                                                Yhat_cols=Yhat_cols, 
                                                                                scaler=sc, 
                                                                                scale_targets=scale_targets)
        
        #Step 3: Inputs and targets                                                                           
        inputs_train, targets_train, inputs_val, targets_val, inputs_test, targets_test = df_to_tensor(X_train=X_train, 
                                                                                                       X_val=X_val, 
                                                                                                       X_test=X_test, 
                                                                                                       Yhat_train=Yhat_train, 
                                                                                                       Yhat_val=Yhat_val, 
                                                                                                       Yhat_test=Yhat_test,
                                                                                                       verbose=verb_step)
        
        #Step 4: Dataset
        train, val, test = datasets(inputs_train=inputs_train, 
                                    targets_train=targets_train, 
                                    inputs_val=inputs_val, 
                                    targets_val=targets_val, 
                                    inputs_test=inputs_test, 
                                    targets_test=targets_test)

        #Step 5: DataLoader
        train_loader, val_loader, test_loader = dataloader(train=train, 
                                                           val=val,
                                                           test=test,
                                                           batch_size=batch_size,
                                                           num_workers=4,
                                                           pin_memory=True)

        #Step 6: Epoch training run
        optimizer = opt_Adam
        scheduler = sch

        for epoch in tqdm(range(epochs), desc='Epochs', leave=False):
            #Training phase
            train_epoch_loss = 0

            model.train() #Set model to training mode
            for i, (batch) in enumerate(train_loader):
                optimizer.zero_grad() #Clear accumulated gradients

                #Forward pass
                loss = model.predict_step(batch)

                #Backward pass and optimize
                loss.backward() #Back propagation
                optimizer.step() #Updating parameters

                #Save loss progress
                train_epoch_loss += loss.item()
                
            #Validation phase
            with torch.no_grad():
                val_epoch_loss = 0

                model.eval() #Set model to evaluate mode
                for batch in val_loader:
                    #Forward pass
                    loss = model.predict_step(batch)
                    #Save loss progress
                    val_epoch_loss += loss.item()

            models[m_name][f_name]['train_loss'].append(train_epoch_loss/len(train_loader)) #Average loss per epoch
            models[m_name][f_name]['val_loss'].append(val_epoch_loss/len(val_loader)) #Average loss per epoch

            #Print progress if verbose
            total_steps = 10, #total_steps = len(train_loader)
            if verbose:
                if (epoch+1) % total_steps == 0:
                    lr_sch = optimizer.param_groups[0]['lr']
                    print(f'Epoch [{epoch+1+0:03}/{num_epochs}] | Train Loss: {train_epoch_loss/len(train_loader):.4f} | Val Loss: {val_epoch_loss/len(val_loader):.4f} | LR: {lr_sch}')
                
                if epoch == num_epochs:
                    print('-'*25)
                    print('End fold training.')
            
            #Scheduler step
            scheduler.step(val_epoch_loss)
            if optimizer.param_groups[0]['lr'] <= 1e-07:
                optimizer.param_groups[0]['lr'] = learning_rate
        
        #Step 7: Fold test
        print('-'*25)
        print('Start fold testing.')
        print('-'*25)    
        predict_list = []
        Yhat_list = []

        with torch.no_grad():
            model.eval()
            for batch in tqdm(test_loader, desc='Batch', leave=False):
                inputs, _ = batch
                inputs = inputs.to(device)
                predict = model(inputs)
                predict_list.append(predict.cpu().numpy())
                Yhat_list.append(_.cpu().numpy())

        predict_list = [x.squeeze().tolist() for x in predict_list]
    
        #Metrics
        mse = mean_squared_error(y_true=Yhat_test.values, y_pred=predict_list, squared=True)
        rmse = np.sqrt(mse)
        r_square = r2_score(y_true=Yhat_test.values, y_pred=predict_list)

        models[m_name][f_name]['predictions'].append(predict_list)
        models[m_name][f_name]['Yhat_test'].append(Yhat_test.values)
        models[m_name][f_name]['mse'].append(mse)
        models[m_name][f_name]['rmse'].append(rmse)
        models[m_name][f_name]['r_square'].append(r_square)

        #Print progress if verbose
        if verbose:
            print(f'MSE: {mse:.4f} | RMSE: {rmse:.4f} | R2: {r_square:.4f}')

        print('-'*25)
        print('End fold testing.')

        #Reset learning rate
        optimizer.param_groups[0]['lr'] = learning_rate
        scheduler = scheduler

        #Step 8: Data download
        #Predictions and Yhat
        dataf = pd.DataFrame({'predictions': models[m_name][f_name]['predictions'][0]})
        dataf['Yhat_test'] = models[m_name][f_name]['Yhat_test'][0]
        file_name = f'./Fold/{f_name}_predictions.csv'
        dataf.to_csv(file_name, header=True, index=False, decimal='.')

        #Losses
        dataf = pd.DataFrame({'train_loss': models[m_name][f_name]['train_loss'], 
                              'val_loss': models[m_name][f_name]['val_loss']})
        file_name = f'./Fold/{f_name}_losses.csv'
        dataf.to_csv(file_name, header=True, index=False, decimal='.')

        #Metrics
        dataf = pd.DataFrame({'mse': models[m_name][f_name]['mse'], 
                            'rmse': models[m_name][f_name]['rmse'], 
                            'r_square': models[m_name][f_name]['r_square']})
        file_name = f'./Fold/{f_name}_metrics.csv'
        dataf.to_csv(file_name, header=True, index=False, decimal='.')

        #Step 9: Clean cuda cache
        del optimizer #Adam adapta los momentos estadísticos del error
        del scheduler 
        del model
        gc.collect()
        torch.cuda.empty_cache()
    
        #Step 10: Save model and parameters
        save_path = f'./models/{m_name}_{f_name}.pth'
        torch.save(model.state_dict(), save_path)

    if split+1 == len(kfolds):
        print('#'*25)
        print('End model training.')    
    
    return models

#### 3.5.1. Model training in action
#Train
models = {}
kfolds = list(KFold(n_splits=5, shuffle=True, random_state=None).split(data_SysA))
model_trained = fit_model(df=data_SysA, 
                          kfolds=kfolds, 
                          model=model_ANN, 
                          criterion=criterion, 
                          optimizer=opt_Adam, 
                          epochs=num_epochs, 
                          lr=learning_rate, 
                          batch_size=batch_size, 
                          scheduler=sch,
                          X_cols=X_cols,
                          Yhat_cols=['i_sc'], #Yhat_cols = ['IL', 'i_sc', 'v_oc', 'mod_eff_exp', 'inv_eff_exp']
                          sc=sc,
                          scale_targets=False, 
                          verb_step=False,
                          verbose=True)

#Training plot
hor = 8
ver = 5
plt.figure(figsize=(hor,ver))

plt.plot(loss_stats['train_loss'], marker='o', ls='-', markersize=6, linewidth=1, alpha=0.5, label='Train Loss', color='#1580E4')
plt.plot(loss_stats['val_loss'], marker='o', ls='-', markersize=6, linewidth=1, alpha=0.5, label='Validation Loss', color='orangered')

plt.title('Losses Behaviour', fontsize=15)
plt.ylabel('MSE Loss', fontsize=13)
plt.xlabel('Epochs', fontsize=13)

plt.tick_params(direction='in', length=6, width=1, grid_alpha=0.5)
plt.xlim(0, num_epochs)
plt.ylim(0, None)
plt.xticks(rotation=0)
plt.grid(True)
plt.legend(loc='upper right', fontsize=10)
plt.tight_layout;
#plt.savefig('Protocolo_SD.eps', bbox_inches='tight')

#Target-prediction check
with torch.no_grad():
    for i in range(0, 10):
        inputs, targets = val[i]
        inputs = inputs.to(device)
        predict = model(inputs)
        
        print(f'Test {i} | Target: {targets.cpu().numpy()[0]:.4f} | Prediction: {predict.cpu().numpy()[0]:.4f}')
        #print("Input: ", x)
        #print("Target: ", sc.inverse_transform(targets))
        #print("Prediction:", sc.inverse_transform(predict))