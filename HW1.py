import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


plt.rcParams['figure.figsize'] = (14.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# Read comma separated data
df = pd.read_csv('C:\\Users\\goldi\\OneDrive\\Weizmann\\ML\\hw1\\kc_house_data.csv') # Relative paths are sometimes better than absolute paths.
# df stands for dataframe, which is the default format for datasets in pandas

df.head(n=10)
df.describe

























