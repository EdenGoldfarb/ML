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


X = np.c_[np.ones(len(X)),X ]
type(X)
X.shape

 def compute_cost(X, y, theta):
    """
    Computes the average squared difference between an obserbation's actual and
    predicted values for linear regression.  

    Input:
    - X: Inputs  (n features over m instances).
    - y: True labels (1 value over m instances).
    - theta: The parameters (weights) of the model being learned.

    Output:
    - J: the cost associated with the current set of parameters (single number).
    """
    
    J = 0  # Use J for the cost.
    ###########################################################################
    # TODO: Implement the MSE cost function.                                  #
    ###########################################################################
    m = X.shape[0]
    n = X.shape[1]
    J = (((theta.dot(X.transpose()) - y)**2).sum())/(2*m)
        
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return J

theta = np.array([2,1,1])
z = np.arange(1,7,1).reshape(2,3).transpose()
theta.dot(z)

theta = np.array(np.random.random(size=X.shape[1]))
import time

def gradient_descent(X, y, theta, alpha, num_iters):
    """
    Learn the parameters of the model using gradient descent. Gradient descent
    is an optimization algorithm used to minimize a (loss) function by 
    iteratively moving in the direction of steepest descent as defined by the
    opposite direction of the gradient. Instead of performing a constant number
    of iterations, stop the training process once the loss improvement from
    one iteration to the next is smaller than `1e-8`.
    
    Input:
    - X: Inputs  (n features over m instances).
    - y: True labels (1 value over m instances).
    - theta: The parameters (weights) of the model being learned.
    - alpha: The learning rate of the model.
    - num_iters: The number of iterations performed.

    Output:
    - theta: The learned parameters of the model.
    - J_history: the loss value in each iteration.
    """
    
    J_history = [] # Use a python list to save cost in every iteration
    ###########################################################################
    # TODO: Implement the gradient descent optimization algorithm.            #
    ###########################################################################
    J_history =  [compute_cost(X, y, theta)]
    m = X.shape[0]
    n = X.shape[1]
    j=0
    for f in np.arange(num_iters):
        if j % 18 == 0:
            j=0
        theta[j] = theta[j]-(alpha/m)*((theta.dot(X.transpose()) - y)*X[:,j]).sum()
        MSE =  compute_cost(X, y, theta) 
        J_history.append(MSE)
        J_delta = J_history[f] - J_history[f+1] 
        print(J_delta)
        if J_delta < 1e-8:
            break 
        j=+1
        
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return theta, J_history


num_iters = 40000
alpha = 0.00000001
a = np.arange(1,num_iters+1)

np.random.seed(42)
theta = np.random.random(size=X.shape[1])
iterations = 40000
alpha = 0.001

start_time = time.time()
theta, J_history = gradient_descent(X ,y, theta, alpha, iterations)
print("--- %s seconds ---" % (time.time() - start_time))


fig, ax = plt.subplots()
ax.scatter(np.arange(len(J_history)), np.array(J_history))
ax.set_xlabel('Iteration')
ax.set_ylabel("MSE")
ax.set_title('MSE convergence')











