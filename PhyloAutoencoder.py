#!/usr/bin/env python3

import matplotlib as plt
import numpy as np
# import pandas as pd
import torch
from torch import optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression


class PhyloAutoencoder(object):
    def __init__(self, model, optimizer, loss_func):
        '''
            model is an object from torch.model
            optimizer is an object from torch.optim
            loss_func is ???
        '''

        self.device    = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model     = model
        self.model.to(self.device)
        self.optimizer = optimizer
        self.loss_func = loss_func

        self.train_loader = None
        self.val_loader   = None

        self.losses = []
        self.val_losses = []
        self.total_epochs = 0

        # step functions. These perform a single gradient descent step:
        # (model(x), then backward) given a batch
        # updates internal state. returns the loss for the batch
        self.train_step = self._make_train_step_func()
        self.val_step   = self._make_val_func()

        self.writer = None

    def train(self, num_epochs, seed = 1):
        self.set_seed(seed)
        for epoch in range(num_epochs):
            self.total_epochs += 1
            epoch_train_loss = self._mini_batch()

            with torch.no_grad():
                epoch_validation_loss = self._mini_batch(validation=True)

            self.losses.append(epoch_train_loss)
            self.val_losses.append(epoch_validation_loss)

            if self.writer:
                scalars = {'training':epoch_train_loss}
                if epoch_validation_loss is not None:
                    scalars.update({'validation':epoch_validation_loss})

                self.writer.add_scalars(main_tag='loss', tag_scaler_dict = scalars, 
                                        global_step = epoch)
        
        if self.writer:
            self.writer.flush()

    def _mini_batch(self, validation = False):

        if validation:
            data_loader = self.val_loader
            step_fn = self.val_step
        else:
            data_loader = self.train_loader
            step_fn = self.train_step

        if data_loader == None:
            return None

        mini_batch_losses = []
        for x_batch, y_batch in data_loader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            loss = step_fn(x_batch, y_batch)
            mini_batch_losses.append(loss)

        return np.mean(mini_batch_losses)

    def predict(self, x):
        self.model.eval() 
        x_tensor = torch.as_tensor(x).float().to(self.device)
        y_hat_tensor = self.model(x_tensor)
        self.model.train()
        return y_hat_tensor.detach().cpu().numpy()

    def _make_train_step_func(self):

        def train_step(x, y):
            self.model.train()
            yhat = self.model(x)
            loss = self.loss_func(yhat, y)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            return loss.item()
        
        return train_step

    def _make_val_func(self):

        def evaluate(x, y):
            self.model.eval()
            yhat = self.model(x)
            loss = self.loss_func(yhat, y)
            return loss.item()

        return evaluate

    def to_device(self, device):
        try:
            self.device = device
            self.model.to(self.device)
        except RuntimeError:
            print(f"Didn't work, sending to {self.device} instead.")

    def set_data_loaders(self, train_loader, val_loader = None):
        self.train_loader = train_loader
        self.val_loader   = val_loader

    def plot_losses(self):
        fig = plt.figure(figsize=(11,8))
        plt.plot(self.losses, label = 'Training Loss', c="b")
        if self.val_loader:
            plt.plot(self.val_losses, label = 'Validation Loss', c='r')
        plt.yscale('log')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout()
        return fig

    def make_graph(self):
        if self.train_loader and self.writer:
            x_sample, y_sample = next(iter(self.train_loader))
            self.writer.add_graph(self.model, x_sample.to(self.device))

    def set_seed(self, seed = 1):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(seed)
        np.random.seed(seed)

    def save_checkpoint(self, filename):
        checkpoint = {'epoch':self.total_epochs,
                      'model_state_dict':self.model.state_dict(),
                      'optimizer_state_dict':self.optimizer.state_dict(),
                      'loss':self.losses,
                      'val_loss':self.val_losses}
        torch.save(checkpoint, filename)

    def load_checkpoint(self, filename):
        checkpoint = torch.load(filename)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.total_epochs = checkpoint['epoch']
        self.losses = checkpoint['loss']
        self.val_losses = checkpoint['val_loss']
        self.model.train()



'''
    TESTING
'''

if __name__ == "__main__":

    print("Testing...")
    torch.manual_seed(1)

    # simulated data
    data_shape = (1000, 1)
    label_shape = (1000, 1)
    x = np.random.random(data_shape)
    y = -2 + 7 * x + np.random.normal(loc = 0, 
                                      scale = 0.1, 
                                      size = label_shape)
    test_x = np.random.random((10,1))
    test_y = -2 + 7 * test_x + np.random.normal(loc=0,scale=0.1,size=(10,1))


    # split and standardize data by train data moments
    prop_train = 0.8
    prop_test = 1-prop_train
    x_train, x_val, y_train, y_val = train_test_split(x, y, 
                                                      test_size = prop_test, 
                                                      random_state = 1)
    scx = StandardScaler()
    scx.fit(x_train)
    scy = StandardScaler()
    scy.fit(y_train)
    x_train = scx.transform(x_train)
    x_val   = scx.transform(x_val)
    y_train = scy.transform(y_train)
    y_val   = scy.transform(y_val)


    # data processing
    x_train_tensor = torch.as_tensor(x_train).float()
    y_train_tensor = torch.as_tensor(y_train).float()
    x_val_tensor   = torch.as_tensor(x_train).float()
    y_val_tensor   = torch.as_tensor(y_train).float()

     
    
    # data set objects
    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    val_dataset   = TensorDataset(x_val_tensor, y_val_tensor)
    # prop_train = 0.8
    # n_total = data_shape[0]
    # n_train = int(prop_train * n_total)
    # n_val = n_total - n_train
    # train_data, val_data = torch.utils.data.random_split(dataset, [n_train, n_val])

    

    # data loaders
    train_data_loader = DataLoader(train_dataset, 
                                   batch_size=16, 
                                   shuffle=True)
    val_data_loader = DataLoader(val_dataset, 
                                 batch_size = 16, 
                                 shuffle=False)

    # model
    lr =  0.1
    model = torch.nn.Sequential(torch.nn.Linear(1,1))
    optimizer = optim.SGD(model.parameters(), lr=lr)
    loss_func = torch.nn.MSELoss(reduction="mean")
    print("initial state")
    print(model.state_dict())

    tree_nn = PhyloAutoencoder(model=model, 
                               optimizer=optimizer, 
                               loss_func=loss_func)
    tree_nn.set_data_loaders(train_loader=train_data_loader, 
                             val_loader=val_data_loader)
    tree_nn.train(num_epochs=10)
    print("final state")
    print(model.state_dict())
    # tree_nn.plot_losses()

    # prediction test
    pred = tree_nn.predict(scx.transform(test_x))
    pred = scy.inverse_transform(pred)
    print(np.hstack((pred, test_y)))
    
