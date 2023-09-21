from tests.test_unet import test_unet
from utils.params import param_parser

if __name__ == "__main__":
    params = param_parser() 
    test_unet(params)