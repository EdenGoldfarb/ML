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
df.describe()

columns = df.columns


price = df['price']

type(price)
df = df.loc[:,'price':]
X = np.array(df.loc[:,'bedrooms':])
y = np.array(df['price'])
type(df)


bla = (x[875,0] - x[:,0].min()) / (x[:,0].max()-x[:,0].min())
bla = x / x.max(axis = 0)

np.where(x[:,0]==x[:,0].min())

def preprocess(X,y):
    X = (X-X.min(axis = 0)) / (X.max(axis = 0) -X.min(axis = 0))
    y = (y-y.min())/ (y.max() - y.min())
    return X,y

test = preprocess(X,y)


fig, ax = plt.subplots()
ax.scatter(X[:,2], y)
ax.set_xlabel('Square Feet')
ax.set_ylabel("Price")
ax.set_title('Square feet VS Price')












