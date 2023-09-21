import numpy as np
import pathlib
import os
import pandas as pd

from utils.files import parse_loss_csv

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

RWTH_colors = ["#00549f", "#f6a800", "#57ab27"]

def plot_histogram(data, compute=False, ylim=None, num_bin=None, block=True, **kwargs):
    if "img_figsize" in kwargs.keys():
        img_figsize = kwargs["img_figsize"]
    else:
        img_figsize = (10,10)
    
    fig = plt.figure(figsize=img_figsize)
  
    # if compute == False:
    #     plt.plot(np.arange(len(data)), data, "r-")
    #     plt.ylim(0, 2000000)
    # else:
    #     plt.hist(data, 1000)
    
    if num_bin is None:
        num_bin = len(data)
        
    plt.plot(np.arange(num_bin), data, "b-")
    
    if ylim is not None:
        plt.ylim(ylim)
    
    if "iso50" in kwargs.keys():
        if ylim is not None:
            ymax = ylim[1]
        else:
            ymax = np.max(data)
        
        plt.vlines(x=kwargs["iso50"], ymin=0.05, ymax=ymax, color="r")
        
    plt.show(block=block)

def plot_img(img, img_cmap=None, mask=None, block=True, cbar=False, **kwargs):
    # create img
    if "img_figsize" in kwargs.keys():
        img_figsize = kwargs["img_figsize"]
    else:
        img_figsize = (10,10)
    
    fig = plt.figure(figsize=img_figsize)
    
    # show image
    if img_cmap is not None:
        img_plt = plt.imshow(img, cmap=img_cmap)
    else:
        img_plt = plt.imshow(img, cmap="gray")
        
    # mask options
    if mask is not None:
        if mask.shape != img.shape:
            raise ValueError("Mask has not the same shape as the image!")
        
        mask_kwargs = {}
        for key in kwargs.keys():
            if key.startswith("mask"):
                mask_kwargs[key.split("mask_")[1]] = kwargs[key]
            
        mask_plt = plt.imshow(mask, **mask_kwargs)
    
    if "title" in kwargs.keys():
        plt.title(kwargs["title"])
    
    if cbar:
        plot_cbar(img_plt)
    
    if "save_path" in kwargs:
        path = pathlib.Path(kwargs["save_path"])
        plt.savefig(path)
    else:
        plt.show(block=block)    
    
def plot_cbar(img):
    ax = plt.gca()
    cax = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.05)
    
    plt.colorbar(img, cax=cax)
    
def plot_img_comp(img, comp, block=True, **kwargs):    
    if type(comp) != list:
        raise ValueError()
    
    #num_plots = len(comp) + 1
    num_rows = 1 
    num_col = len(comp) + 1
    if num_col == 4:
        num_col = 2
        num_rows = 2
    
    if "title" in kwargs.keys():
        title = kwargs["title"]
    else:
        title = ["Image", ]
    
    if "img_cmap" in kwargs:
        img_cmap = kwargs["img_cmap"]
    else:
        img_cmap = ["gray",]
    
    if "value_range" in kwargs:
        vmin = kwargs["value_range"][0]
        vmax = kwargs["value_range"][1]
    else:
        vmin = np.min(img)
        vmax = np.max(img)
    
    # plot image
    fig = plt.figure(figsize=(5 * num_col, 5 * num_rows))
    plt.subplot(num_rows, num_col, 1)
    plt.title(title[0])
    
    img = plt.imshow(img, cmap=img_cmap[0], vmin=vmin, vmax=vmax)
    plot_cbar(img)
    
    # iterate over the given comparisons
    for i, data in enumerate(comp):
        plt.subplot(num_rows, num_col, i+2)
        
        if len(title) == 1 or len(title) < i + 1:
            tmp_title = "Comparison data {}".format(i+1)
        else:
            tmp_title = title[i+1]
        plt.title(tmp_title)
        
        if len(img_cmap) == 1 or len(img_cmap) < i + 1 :
            tmp_cmap = "gray"
        else:
            tmp_cmap = img_cmap[i+1]
             
        if "value_range" in kwargs:
            vmin = kwargs["value_range"][0]
            vmax = kwargs["value_range"][1]
        else:
            vmin = np.min(data)
            vmax = np.max(data)
                        
        img = plt.imshow(data, cmap=tmp_cmap, vmin=vmin, vmax=vmax)
        
        plt.subplots_adjust(wspace=0.25)
        plot_cbar(img)
        
    if "save_path" in kwargs:
        path = pathlib.Path(kwargs["save_path"])
        if not path.parent.is_dir():
            os.makedirs(path.parent)
        
        plt.savefig(path)
    else:
        plt.show(block=block)    
    plt.close('all')

      
def plot_losses(csv_file_list, legend=None):
    if type(csv_file_list) is not list:
        raise TypeError()
     
    train = {}
    val = {} 
    num_epochs = np.inf    
    for csv_file in csv_file_list:
    
        data = parse_loss_csv(csv_file)
        num_epochs = np.minimum(num_epochs, data["num_epochs"])
        
        for key in data["train"].keys():
            if key not in train.keys():
                train[key] = [data["train"][key]]
            else:
                train[key].append(data["train"][key])
                
        for key in data["val"].keys():
            if key not in val.keys():
                val[key] = [data["val"][key]]
            else:
                val[key].append(data["val"][key])
                
    fig = plt.figure(figsize=(18, 9))
    num_plots = len(train.keys())
    for i, key in enumerate(train.keys()):
        plt.subplot(2, num_plots, i + 1)
        plt.title(key)
        values = np.asarray(train[key])
        
        plt.plot(np.arange(num_epochs), values.T, "x-")
        if legend is not None:
            plt.legend(legend)
    
    for i, key in enumerate(val.keys()):
        plt.subplot(2, num_plots, num_plots + i + 1)
        plt.title(key)
        values = np.asarray(val[key])
        
        plt.plot(np.arange(num_epochs), values.T, "x-")
        if legend is not None:
            plt.legend(legend)
    plt.show()
        
def plot_metrics_boxplots(csv_file_list, labels=None, fig_title=None, 
                          metrics=None,
                          save_path=None, 
                          block=True, 
                          **kwargs
                          ):
    if type(csv_file_list) is not list:
        raise TypeError()
     
    # read out data
    results = {} 
    text = ""
    for csv_file in csv_file_list:
    
        data = pd.read_csv(csv_file)
        
        for column in data.columns:
            if column.startswith("Unnamed"):
               continue 
            
            if column not in results.keys():
                results[column] = [data[column]]
            else:
                results[column].append(data[column])
             
            text += "{} \t Mean {}: {:,.4f} \t Std: {:,.4f}:  \n".format(csv_file.name, 
                                                                         column, 
                                                                         np.mean(data[column]),
                                                                         np.std(data[column]))
    print(text)
      
    # get the metrics to plot; if given, plot only desired ones       
    if metrics is not None:
        keys = metrics
    else:
        keys = results.keys()
    num_keys = len(keys)
    
    # set-up plot
    figsize = (9 * num_keys, 9)
        
    fig = plt.figure(figsize=figsize)
    if fig_title is not None:
        fig.suptitle(fig_title)
    
    if labels is None:
        labels = [file.name for file in csv_file_list]
    
    for i, key in enumerate(keys):
        plt.subplot(1, num_keys, i + 1)
        
        if "plot_title" in kwargs.keys():
            plt.title(kwargs["plot_title"])       
        
        #plt.boxplot(results[key], labels=labels)
        plt.boxplot(results[key])
        plt.ylim(0, 1)
        
        if "ylabel" in kwargs.keys():
            plt.ylabel(kwargs["ylabel"])
        else:
            plt.ylabel(key)
        
        ax = plt.gca()
        ax.set_xticklabels(labels)
    
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show(block=block)
        
def gen_bar_plot(data, block=True, **kwargs):
    if type(data) is not list:
        raise ValueError
    
    num_plots = len(data)
       
    fig = plt.figure(figsize=(6, 6))
    if "ylim" in kwargs.keys():
        plt.ylim(0, kwargs["ylim"])
    
    if "ylabel" in kwargs.keys():
        plt.ylabel(kwargs["ylabel"])
    
    if "xlabel" in kwargs.keys():
        plt.xlabel(kwargs["xlabel"])
        
    if "plot_title" in kwargs.keys():
        plt.title(kwargs["plot_title"]) 
   
    for i in range(num_plots):        
        plt.bar(np.arange(len(data[i])), data[i], color=RWTH_colors[i], alpha=0.5) 

    if "legend" in kwargs.keys():
        plt.legend(kwargs["legend"])    
        
    if "save_path" in kwargs.keys() and kwargs["save_path"] is not None:
        plt.savefig(kwargs["save_path"])
    else:
        plt.show(block=block)
