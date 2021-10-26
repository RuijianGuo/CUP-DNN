import torch
import matplotlib
import matplotlib.pyplot as plt
class LRScheduler():
    """
    Learning rate scheduler. If the validation loss does not decrease for the 
    given number of `patience` epochs, then the learning rate will decrease by
    by given `factor`.
    """
    def __init__(
        self, optimizer, patience=2, min_lr=1e-5, factor=0.9
    ):
        """
        new_lr = old_lr * factor
        :param optimizer: the optimizer we are using
        :param patience: how many epochs to wait before updating the lr
        :param min_lr: least lr value to reduce to while updating
        :param factor: factor by which the lr should be updated
        """
        self.optimizer = optimizer
        self.patience = patience
        self.min_lr = min_lr
        self.factor = factor
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau( 
                self.optimizer,
                mode='min',
                patience=self.patience,
                factor=self.factor,
                min_lr=self.min_lr,
                verbose=True
            )
    def __call__(self, val_loss):
        self.lr_scheduler.step(val_loss)

class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    def __init__(self, min_epoch = 30, patience=10, min_delta=0):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.min_epoch = min_epoch
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False


    def __call__(self, val_loss,label, scaler, epoch, model, optimizer, loss_func):
        if self.best_loss == None:
            self.best_loss = val_loss
        if epoch >= self.min_epoch:
            if self.best_loss - val_loss > self.min_delta:
                self.best_loss = val_loss
            # reset counter if validation loss improves
                self.save_model(label, scaler, epoch, model, optimizer, loss_func)
                self.counter = 0
            else:
                self.counter += 1
                print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
                if self.counter >= self.patience:
                    print('INFO: Early stopping')
                    self.early_stop = True
    @staticmethod
    def save_model(label,scaler, epoch, model, optimizer, loss_func):
        '''
        function to save the trained model to the disk
        '''
        torch.save({
                    'label' : label,
                    'scaler': scaler,
                    'epochs': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'loss_func': loss_func},
                    'outputs/cup_dnn_model.pkl')

def save_plots(plot_type,plot_name,train,validation):
    '''
    function to save accuracy and loss plots
    '''
    plt.figure(figsize=(10,7))
    plt.plot(train, color ='#d62728', linestyle = '-', label = f'training {plot_type}')
    plt.plot(validation, color = '#2ca02c', linestyle = '-', label = f'validation {plot_type}')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel(f'{plot_type}')
    plt.savefig(f'outputs/{plot_name}_{plot_type}_plot.png')

