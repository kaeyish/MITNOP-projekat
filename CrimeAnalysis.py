# -*- coding: utf-8 -*-
"""
Created on Mon May 26 22:25:08 2024

@author: Dušanka
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path

path = Path(__file__).parent / "Crime.csv"

dataframe = pd.read_csv(path)

print(dataframe)