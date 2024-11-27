import argparse
import pandas as pd
from ECM_func import *


def main():
    '''This is the main function, which reads the arguments from the command line, obtain the data files and load the data files to the file path'''
    parser = argparse.ArgumentParser(description="obtain the data files and load the data files to the given path") #read the arguments from the command line
    parser.add_argument('folder_path', type=str, help="The folderpath") #load the feature file to the path

    args = parser.parse_args() #deliver the arguments to args

    EIS_curve_fit(args.folder_path)



