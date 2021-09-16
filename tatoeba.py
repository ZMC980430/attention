#%%
import requests
from zipfile import ZipFile
import os
import torch
from torch import nn


url=r'http://d2l-data.s3-accelerate.amazonaws.com/fra-eng.zip'

myzipfile = requests.get(url)
file_path = r'.\data\fra-eng.zip'

os.makedirs(r'.\data', exist_ok=True)
 
with open(file_path, 'wb') as f:
    f.write(myzipfile.content)
# %%
