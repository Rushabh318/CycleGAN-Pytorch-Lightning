import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_gan_details(value_dict, value_range=None, block=False):
    
    fig = plt.figure(figsize=(12, 6))   
    for i, key in enumerate(value_dict):
        name = key.split("_loss")[0]
        
        plt.subplot(2, 3, i + 1)
        plt.title(name)
        values = value_dict[key]
        
        if value_range is not None:
            values = values[value_range[0]:value_range[1]]
        
        plt.plot(np.arange(len(values)), values, "bx-")
     
    plt.show(block=block)

def plot_metrics(csv_file):
    data = pd.read_csv(csv_file, index_col="epoch")
    #print(data.columns)
    
    train = data[[col for col in data.columns if "train_loss_epoch" in col]]
    tmp = {}
    num_epochs = np.max(np.unique(train["g_tot_train_loss_epoch"].index)) + 1
    
    for col in train.columns:
        tmp[col] = [train[col][i].iloc[-1] for i in range(num_epochs)]
    train = tmp

    tmp = {}
    val = data[[col for col in data.columns if "val" in col or "epoch" == col]]
    for col in val.columns:
        tmp[col] = [val[col][i].iloc[-2] for i in range(num_epochs)]
    val = tmp
   
    plot_gan_details(train, value_range=(0, 100), block=False)
    plot_gan_details(val, value_range=(0, 100), block=True)
    

if __name__ == "__main__":
    # old
    #path = pathlib.Path("logs/cyclegan_test_ct/version_0/metrics.csv")
    
    #path = pathlib.Path("logs/cyclegan_cluster_run/version_1/metrics.csv")
    #path = pathlib.Path("logs/cyclegan_cluster_run/version_2/metrics.csv")
    path = pathlib.Path("logs/cyclegan_cluster_run/version_3/metrics.csv")
    #path = pathlib.Path("logs/cyclegan_cluster_run/version_4/metrics.csv")
        
    plot_metrics(path)
