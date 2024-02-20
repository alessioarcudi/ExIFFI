

import time
from typing import Type
import pickle
import numpy as np
import pandas as pd
import os
from collections import namedtuple
from model_reboot.EIF_reboot import ExtendedIsolationForest

Precisions = namedtuple("Precisions",["direct","inverse","dataset","model"])


def save_element(element, directory_path, filename="", filetype="pickle"):
    assert filetype in ["pickle", "npz"], "filetype must be either 'pickle' or 'npz'"
    t = time.localtime()
    current_time = time.strftime("%d-%m-%Y_%H-%M-%S", t)
    filename = current_time + '_' + filename
    path = directory_path + '/' + filename
    if filetype == "pickle":
        with open(path, 'wb') as fl:
            pickle.dump(element, fl)
    elif filetype == "npz":
        np.savez(path, element=element)
        
def get_most_recent_file(directory_path):
    files = sorted(os.listdir(directory_path))
    return directory_path+"/"+files[-1]

def open_element(file_path, filetype="pickle"):
    assert filetype in ["pickle", "npz"], "filetype must be either 'pickle' or 'npz'"
    if filetype == "pickle":
        with open(file_path, 'rb') as fl:
            element = pickle.load(fl)
    elif filetype == "npz":
        element = np.load(file_path)['element']
    return element