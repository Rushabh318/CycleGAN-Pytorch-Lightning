import numpy as np
import torch
import csv
import os
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from copy import deepcopy

def write_metrics_csv(metrics: dict, basepath: str, filename: str = "metrics.csv"):
    os.makedirs(basepath, exist_ok=True)
    path = os.path.join(basepath, filename)
    with open(path, "w+") as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(metrics.keys())
        writer.writerow(metrics.values())

class CrossValTrainer(Trainer):

    def __init__(self, num_folds: int = None, *args, **kwargs):
        super(CrossValTrainer, self).__init__(*args, **kwargs)
        self.num_folds = num_folds

    def fit(self, model, datamodule=None, **kwargs):
        #all_results = []
        metrics = {}
        for k in range(self.num_folds):

            print("--------------------------------------------------------------------------------")
            print('Fold ' + str(k+1) + ' of ' + str(self.num_folds))
            print("--------------------------------------------------------------------------------")

            train_dataloader = datamodule.train_dataloader(k)
            val_dataloader = datamodule.val_dataloader(k)

            #self.fit_loop.global_step = 0
            #self.logger_connector.logged_metrics = {}
            #self.fit_loop.iteration_count = 0
            

            #local_model = deepcopy(model)
            local_model = deepcopy(model)

            for logger in self.logger:
                
                print(logger)

                metrics_path_base, metrics_file_name = os.path.split(logger.experiment.metrics_file_path)

                if k > 0:
                    logger.experiment.log_dir, _ = os.path.split(logger.experiment.log_dir)
                    metrics_path_base, _ = os.path.split(metrics_path_base)

                logger.experiment.log_dir = os.path.join(logger.experiment.log_dir, "fold_" + str(k))
                logger.experiment.metrics_file_path = os.path.join(metrics_path_base, "fold_" + str(k), metrics_file_name)
                logger.experiment.metrics = []

                os.makedirs(logger.experiment.log_dir, exist_ok=True)

            for callback in self.callbacks:
                if isinstance(callback, ModelCheckpoint):
                    if k > 0:
                        callback.dirpath ,_ = os.path.split(callback.dirpath)
                    
                    callback.monitor = None
                    callback.current_score = None
                    callback.best_k_models = {}
                    callback.kth_best_model_path = ""
                    callback.best_model_score = None
                    callback.best_model_path = ""
                    callback.last_model_path = ""
                    callback.dirpath = os.path.join(callback.dirpath, "fold_" + str(k))

            
            super().fit_loop.epoch_loop.reset()
            super().fit_loop.reset()
            super().fit_loop.epoch_loop.global_step = 0
            super().fit_loop.global_step = 0
            super().fit_loop.current_epoch = 0
            #self.trainer.logger.
            #self.logger_connector._logged_metrics = {}
            #self.checkpoint_callback.
            #super().checkpoint_callback.on_init_start(self)
            #super().checkpoint_callback.setup(self, local_model)

            super().fit(local_model,
                     train_dataloaders=train_dataloader,
                     val_dataloaders=val_dataloader,
                     **kwargs
                     )

            results = super().validate(local_model, 
                                    dataloaders=val_dataloader, 
                                    **kwargs
                                    )
            
            metrics["fold_" + str(k)] = results

        mean_dir, _ = os.path.split(self.logger[0].experiment.log_dir)
        
        mean_metrics = {}
        
        for i in range(len(metrics["fold_0"])):
            print(i)
            for key in metrics["fold_0"][i].keys():
                print(key)
                metric_sum = 0
                for k in range(self.num_folds):
                    metric_sum += metrics["fold_" + str(k)][i][key]
                metric_mean = metric_sum / self.num_folds
                mean_metrics[key] = metric_mean

        write_metrics_csv(mean_metrics, mean_dir)
