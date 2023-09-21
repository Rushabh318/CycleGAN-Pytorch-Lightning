import numpy as np
import pathlib
import os
import torch
from tqdm import tqdm, trange

class Trainer:
    def __init__(self,
                 model: torch.nn.Module,
                 device: torch.device,
                 loss_function: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 training_DataLoader: torch.utils.data.Dataset,
                 validation_DataLoader: torch.utils.data.Dataset = None,
                 lr_scheduler: torch.optim.lr_scheduler = None,
                 num_epochs: int = 100,
                 logger=None,
                 **kwargs                
                 ):

        self.model = model        
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.training_DataLoader = training_DataLoader
        self.validation_DataLoader = validation_DataLoader
        self.device = device
        self.num_epochs = num_epochs
        self.logger = logger
        
        if "start_epoch" in kwargs.keys():
            self.start_epoch = kwargs["start_epoch"]
        else:
            self.start_epoch = 0
        
        if "path" in kwargs["checkpoints"].keys() and "freq" in kwargs["checkpoints"].keys():
            self.checkpoint_params = {"path": pathlib.Path(kwargs["checkpoints"]["path"]),
                                      "freq": kwargs["checkpoints"]["freq"]}
        elif "path" in kwargs["checkpoints"].keys():
            self.checkpoint_params = {"path": pathlib.Path(kwargs["checkpoints"]["path"]),
                                      "freq": 100}
        elif "freq" in kwargs["checkpoints"].keys():
            self.checkpoint_params = {"path": pathlib.Path("out/tmp_checkpoints"),
                                      "freq": kwargs["checkpoints"]["freq"]}
        else:
            self.checkpoint_params = {"path": pathlib.Path("out/tmp_checkpoints"), "freq": 100}   

        if not pathlib.Path.is_dir(self.checkpoint_params["path"]):
            pathlib.Path.mkdir(self.checkpoint_params["path"], parents=True)

        # initiate some structures
        self.model.to(self.device)
        self.training_loss = []
        self.validation_loss = []
        self.learning_rate = []

    def run_trainer(self):
        progressbar = trange(self.num_epochs, desc='Progress')
        for i in progressbar:
            """Epoch counter"""
            self.start_epoch += 1

            """Training block"""
            self._train()
            loss = {"train_loss": self.training_loss[i], }

            """Validation block"""
            if self.validation_DataLoader is not None:
                self._validate()
                loss["val_loss"] = self.validation_loss[i]

            if np.remainder(i, self.checkpoint_params["freq"]) == 0:
                checkpoint = {"epoch": self.start_epoch, 
                              "model_state_dict": self.model.state_dict(),
                              "optimizer_state_dict": self.optimizer.state_dict(),
                               **loss,
                                }
                
                torch.save(checkpoint, 
                            self.checkpoint_params["path"] / "checkpoint_epoch_{}.pt".format(i))
                
            if self.logger is not None:
                self.logger(epoch=self.start_epoch, loss=loss)

            """Learning rate scheduler block"""
            if self.lr_scheduler is not None:
                if self.validation_DataLoader is not None and self.lr_scheduler.__class__.__name__ == 'ReduceLROnPlateau':
                    self.lr_scheduler.batch(self.validation_loss[i])  # learning rate scheduler step with validation loss
                else:
                    self.lr_scheduler.batch()  # learning rate scheduler step
                    
        torch.save({"epoch": self.start_epoch, 
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    **loss}, 
                   self.checkpoint_params["path"] / "checkpoint_last.pt")
               
        if self.logger is not None:
            self.logger.release()
                    
        return self.training_loss, self.validation_loss, self.learning_rate

    def _train(self):
        # train mode
        self.model.train() 
        
        train_losses = []  
        batch_iter = tqdm(enumerate(self.training_DataLoader), 
                          'Training', 
                          total=len(self.training_DataLoader),
                          leave=False)

        for i, (x, y) in batch_iter:
            # send to device (GPU or CPU)
            data, target = x.to(self.device), y.to(self.device)  
         
             # zerograd the parameters
            self.optimizer.zero_grad() 
            
            # foward pass
            out = self.model(data) 
            
            loss = self.loss_function(out, target)  
            loss_value = loss.item()
            train_losses.append(loss_value)
            
            # backward pass
            loss.backward()  
            self.optimizer.step()

             # update progressbar
            batch_iter.set_description(f'Training: (loss {loss_value:.4f})') 

        self.training_loss.append(np.mean(train_losses))
        self.learning_rate.append(self.optimizer.param_groups[0]['lr'])

        batch_iter.close()

    def _validate(self):
        # evaluation mode
        self.model.eval() 
         
        valid_losses = []  
        batch_iter = tqdm(enumerate(self.validation_DataLoader), 
                          'Validation', 
                          total=len(self.validation_DataLoader),
                          leave=False)

        for i, (x, y) in batch_iter:
            # send to device (GPU or CPU)
            data, target = x.to(self.device), y.to(self.device)  

            with torch.no_grad():
                out = self.model(data)
                loss = self.loss_function(out, target)
                loss_value = loss.item()
                valid_losses.append(loss_value)

                batch_iter.set_description(f'Validation: (loss {loss_value:.4f})')

        self.validation_loss.append(np.mean(valid_losses))

        batch_iter.close()