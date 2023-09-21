import numpy as np
import pathlib
import argparse
import yaml
from datetime import datetime
import os

def param_parser():
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument("--config_file", "-f", type=str)
    
    parser.add_argument("--data_path", "-d", type=str)
    parser.add_argument("--num_epochs", "-n", type=int)
    parser.add_argument("--learning_rate", "-lr", type=np.float32)
    parser.add_argument("--batch_size", "-bs", type=int)
        
    # parse given arguments and transform them into dict
    args = parser.parse_args()
    args = vars(args)
    
    # create an empty params dict
    params = {"data": {},
              "experiment": {},
              "logging": {},
              "checkpoints": {}}
        
    # first, check for a yaml-configruation file
    if "config_file" in args.keys() and args["config_file"] is not None:
        config_file = pathlib.Path(args["config_file"])
      
        if not config_file.is_file():
            raise FileNotFoundError
        
        with open(config_file, "r") as f:
            configs = yaml.safe_load(f)
        
        for key in configs.keys():
            params[key] = configs[key]     

    else:
        # check whether all neccessary arguments have been set; if not, use
        # default values
        if "path" not in params["data"].keys():
            raise KeyError("No explicit data path found!") 
        
        default_exp = {"learning_rate": 0.01, 
                       "batch_size": 1,
                       "num_epochs": 10}
        
        for key in default_exp:
            if key not in params["experiment"]:
                if key in args.keys():
                    params["experiment"][key] = args[key]
                else:
                    params["experiment"][key] = default_exp[key]
        
        # setup directories for checkpoints and logging
        if "name" not in params["experiment"].keys():
            params["experiment"]["name"] = datetime.now().strftime("%y-%m-%d_%H-%M-%S")
            
        if "path" in params["checkpoints"].keys():
            check_path = pathlib.Path(params["checkpoints"]["path"]) /  params["experiment"]["name"] / "checkpoints"
            params["checkpoints"]["path"] = check_path.as_posix() 
                
        if "path" in params["logging"].keys():
            logs_path = pathlib.Path(params["logging"]["path"]) /  params["experiment"]["name"] / "logs"
            params["logging"]["path"] = logs_path.as_posix()
         
    return params

def dump_params(path, params):
    if not os.path.isdir(path.parent):
        os.makedirs(path.parent)
    
    with open(path, "w") as f: 
        yaml.dump(params, f)