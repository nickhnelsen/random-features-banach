# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 11:58:13 2020

@author: nickh

Written by:
Nicholas H. Nelsen
California Institute of Technology
Email: nnelsen@caltech.edu

Utility functions: file input-output
    
Last updated: Apr. 09, 2020 
"""

import os
import sys

class Logger(object):
    '''
    Reference: https://stackoverflow.com/questions/11325019/how-to-output-to-the-console-and-file
    '''
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush() # If you want the output to be visible immediately
    def flush(self) :
        for f in self.files:
            f.flush()

def fileio_init(dir_name, USER_PARAMS_DICT):

    # Make directory to store plots
    os.makedirs(dir_name, exist_ok=True)
   
    # Write hyperparameters to text file
    filename_vars = 'hyperparameters.txt'
    with open(dir_name + filename_vars, 'w') as f:
            f.write(repr(USER_PARAMS_DICT)+'\n')
    
    # Write console output to log file in test directory
    stdoutOrigin = sys.stdout 
    log_file = open(dir_name + 'log.txt', 'w')
    sys.stdout = Logger(sys.stdout, log_file)
    
    return log_file, stdoutOrigin

def fileio_end(log_file, stdoutOrigin):
    log_file.close()
    sys.stdout = stdoutOrigin