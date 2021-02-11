import splitfolders
from info_logging import log
import os
def train_test_split():
    #Creating Train / Test folders (One time use)
    log("########### Train Test Val Script started ###########")
    root_dir = 'C:\\Final Project\\processed_dataset'
    processed_dir = 'C:\\Final Project\\Images Sharpen'
    #Define test ratio
    test_ratio = 0.10
    splitfolders.ratio(processed_dir, output=root_dir, seed=1337, ratio=(1-test_ratio, test_ratio,), group_prefix=None)
    #rename val folder to test 
    os.rename(root_dir+"\\val",root_dir+"\\test")
    log("########### Train Test Val Script Ended ###########")

if __name__ == '__main__':
    train_test_split()
