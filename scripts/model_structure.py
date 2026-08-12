"""
Script setting model structure

Copyright (C) 2026 PyPSA-Nepal contributors, licensed under a MIT license
"""

import os
import shutil
import logging


import pandas as pd


pd.set_option('future.no_silent_downcasting', True)

logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)



if __name__ == "__main__":

#%% Levers and macros
    
    path_data = "../data/network/"
    path_structure = "../data/raw_data/others/model_structure/"
    
    
#%% Copy files
    # FIXME: this currently just copies files, this should be based on
    # temporal resolution based on config file, etc.

    files = os.listdir(path_structure)
    
    for f in files:
        path = os.path.join(path_structure, f)
        if os.path.isfile(path):
            shutil.copy(path, path_data)