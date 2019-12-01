import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



plt.rcParams['figure.figsize'] = (14.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# Part 1: Data Preprocessing (5 Points)

# Read comma separated data
df = pd.read_csv('C:\\Users\\goldi\\OneDrive\\Weizmann\\ML\\hw1\\kc_house_data.csv') # Relative paths are sometimes better than absolute paths.
# df stands for dataframe, which is the default format for datasets in pandas

### Data Exploration

# Print the first 10 entries of the dataframe. 

# Your code starts here
df.head(n=10)
# Your code ends here

# Show the statistics of the dataset. 

# Your code starts here
df.describe()
# Your code ends here



X = None # placeholder for the variables
y = None # placeholder for the target values

# Your code starts here
df = df.loc[:,'price':]
X = np.array(df.loc[:,'bedrooms':])
y = np.array(df['price'])

# Your code ends here

def preprocess(X, y):
    """
    Perform min-max scaling for both the data and the targets.
    Input:
    - X: Inputs (n features, m instances).
    - y: True labels (1 target, m instances).

    Output:
    - X: The scaled inputs.
    - y: The scaled labels.
    """
    ###########################################################################
    # TODO: Implement Min-Max Scaling.                                        #
    ###########################################################################
    X = (X-X.min(axis = 0)) / (X.max(axis = 0) -X.min(axis = 0))
    y = (y-y.min())/ (y.max() - y.min())
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return X, y

X, y = preprocess(x, y)


## Data Visualization

# Choose one fearture an plot the target price as a function of that feature
# Your code starts here
fig, ax = plt.subplots()
ax.scatter(X[:,2], y)
ax.set_xlabel('Square Feet')
ax.set_ylabel("Price")
ax.set_title('Square feet VS Price')

# Your code ends here

## Bias Trick

# Your code starts here
X = np.c_[np.ones(len(X)),X ]
# Your code ends here


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
    m = X.shape[0]; n = X.shape[1]
    J = (((theta.dot(X.transpose()) - y)**2).sum())/(2*m)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return J
"""
def gradient_descent(X, y, theta, alpha, num_iters):
    
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
    
    
    J_history = [] # Use a python list to save cost in every iteration
    ###########################################################################
    # TODO: Implement the gradient descent optimization algorithm.            #
    ###########################################################################
    J_history =  [compute_cost(X, y, theta)]
    m = X.shape[0]
    j=0
    for f in np.arange(num_iters):
        if j % 18 == 0:
            j=0
        theta[j] = theta[j]-(alpha/m)*((theta.dot(X.transpose()) - y)*X[:,j]).sum()
        MSE =  compute_cost(X, y, theta) 
        J_history.append(MSE)
        J_delta = J_history[f] - J_history[f+1] 
        if J_delta < 1e-8:
            #print("Bingo" + "- num of iterations",f)
            break 
        j=+1
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    
    return theta, J_history
"""


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
    for f in np.arange(num_iters):
        derivative = np.dot(X.transpose(),(np.dot(X,theta) - y))
        theta = theta - alpha*(1/m)*(derivative)
        MSE =  compute_cost(X, y, theta) 
        J_history.append(MSE)
        J_delta = J_history[f] - J_history[f+1] 
        if J_delta < 1e-8:
            #print("Bingo" + "- num of iterations",f)
            break 
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    
    return theta, J_history



np.random.seed(42)
theta = np.random.random(size=X.shape[1])
iterations = 40000
alpha = 0.1
theta, J_history = gradient_descent(X ,y, theta, alpha, iterations)


# Your code starts here
fig, ax = plt.subplots()
ax.scatter(np.arange(len(J_history)), np.array(J_history))
ax.set_xlabel('Iteration')
ax.set_ylabel("MSE")
ax.set_title('MSE convergence')


# Your code ends here


def pinv(X, y):
    """
    Calculate the optimal values of the parameters using the pseudoinverse
    approach as you saw in class.

    Input:
    - X: Inputs  (n features over m instances).
    - y: True labels (1 value over m instances).

    Outpu:
    - theta: The optimal parameters of your model.

    ########## DO NOT USE numpy.pinv ##############
    """
    pinv_theta = [] # Use a python list to save cost in every iteration
    ###########################################################################
    # TODO: Implement the pseudoinverse algorithm.                            #
    ###########################################################################
    h=np.matrix(X.T.dot (X))
    pi=(h.getI().dot(X.T))
    pinv_theta = np.array(pi.dot(y))
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return pinv_theta

theta_pinv = pinv(X,y)
J_pinv = compute_cost(X, y, theta_pinv)
theta_pinv

fig, ax = plt.subplots()
ax.scatter(np.arange(len(J_history)), np.array(J_history))
ax.set_xlabel('Iteration')
ax.set_ylabel("MSE")
ax.set_title('MSE convergence')

ax.axhline(J_pinv, c='r');
ax.annotate('pseudo inverse', xy=(0,0), xytext=(2000,J_pinv+0.05), size=14)
ax.annotate('gradient descent', xy=(0,0), xytext=(700,1), size=14)


def find_best_alpha(X, y, iterations):
    """
    Iterate over the provided values of alpha and maintain a python 
    dictionary with alpha as the key and the final loss as the value.
    For consistent results, use the same theta value for all runs.

    Input:
    - X: Inputs (n features over m instances).
    - y: True labels (1 value over m instances).
    - num_iters: The number of iterations performed.

    Output:
    - alpha_dict: A python dictionary containing alpha as the 
                  key and the final loss as the value
    """
    
    alphas = [0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 2, 3]
    alpha_dict = {}
    alpha_dict.fromkeys(alphas) 
    np.random.seed(42)
    theta = np.random.random(size=X.shape[1])
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    for alpha in alphas:
        np.random.seed(42) # seeding the random number generator allows us to obtain reproducible results
        theta = np.array(np.random.random(size=X.shape[1]))
        alpha_dict[alpha]=gradient_descent(X, y, theta, alpha, iterations)[1][-1]
        
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return alpha_dict

alpha_dict = find_best_alpha(X, y, 40000)

best_alpha = None
# Your code starts here
best_alpha = min(alpha_dict,key = alpha_dict.get)
# Your code ends here

# Your code starts here
fig, ax = plt.subplots()
ax.set_xlabel('Iteration')
ax.set_ylabel("MSE")
ax.set_title('MSE convergence')
alpha_list = []
for i in range(2,-1,-1):
    alpha= sorted(alpha_dict.items(), key=lambda x: x[1]).pop(i)[0]
    alpha_list.append('α = '+ str(alpha))
    np.random.seed(42) 
    theta = np.array(np.random.random(size=X.shape[1]))
    best_theta, J_history = gradient_descent(X ,y, theta, alpha, 40000)
    ax.scatter(np.arange(len(J_history)), np.array(J_history))
ax.legend(alpha_list);

# Your code ends here
    


pseudo_pred = theta_pinv.dot(X.transpose())
descent_pred = best_theta.dot(X.transpose())
fig, ax = plt.subplots(1, 2, figsize=(14,5))
ax[0].scatter(y, descent_pred)
ax[1].scatter(pseudo_pred, descent_pred)
ax[0].set_ylabel('Gradient Descent Prediction')
ax[0].set_xlabel("Targets")
ax[1].set_ylabel('Gradient Descent Prediction')
ax[1].set_xlabel("Pseudo-Inverse Prediction")
ax[0].set_title('Targets VS GD')
ax[1].set_title('PINV VS GD')


prices_list = [y,descent_pred,pseudo_pred]
legends = ["Target","Prediction","Pinv"]
fig, axes = plt.subplots(2,2, figsize=(16,8))
axes[0,0].hist(prices_list, bins = 3);
axes[0,1].hist(prices_list, bins = 9);
axes[1,0].hist(prices_list, bins = 20);
axes[1,1].hist(prices_list, bins = 40);
axes[0,0].set_xlabel("Scaled Price")
axes[0,1].set_xlabel("Scaled Price")
axes[1,0].set_xlabel("Scaled Price")
axes[1,1].set_xlabel("Scaled Price")
axes[0,0].legend(legends)
axes[0,1].legend(legends)
axes[1,0].legend(legends)
axes[1,1].legend(legends)



import itertools as it


triplets = list(it.combinations(range(1,X.shape[1]), 3))



df_2 = df.loc[:,'bedrooms':]

train=df_2.sample(frac=0.8,random_state=200) 
test=df_2.drop(train.index)

train_idx = train.index
test_idx  = test.index


X_train = X[train_idx]
X_test  = X[test_idx]
y_train = y[train_idx]
y_test  = y[test_idx]


def find_best_triplet(X_train, y_train, X_test, y_test, triplets, alpha, num_iter):
    """
    Iterate over all possible triplets and find the triplet that best 
    minimizes the cost function. You should first preprocess the data 
    and obtain a array containing the columns corresponding to the
    triplet. Don't forget the bias trick.

    Input:
    - X_train: training dataset.
    - y_train: training labels.
    - X_test: testinging dataset.
    - y_test: testing labels.
    - triplets: a list of three features in X.
    - alpha: The value of the best alpha previously found.
    - num_iters: The number of updates performed.

    Output:
    - The best triplet.
    """
    best_triplet = None
    jz = []
    np.random.seed(42) # seeding the random number generator allows us to obtain reproducible results
    theta = np.array(np.random.random(size=4))
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    for triple in triplets:
        triple = (0,)+triple
        thetaco = gradient_descent(X_train[:,triple],y_train,theta,best_alpha,num_iter)[0]
        jtriple = compute_cost(X_test[:,triple],y_test,thetaco)
        jz.append(jtriple)
    
    best_triplet = triplets[jz.index(max(jz))]
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return best_triplet


find_best_triplet(X_train, y_train, X_test, y_test, triplets, alpha=best_alpha, num_iter=20000)

colnames = df.columns
num_iter = 20000
np.random.seed(42) # seeding the random number generator allows us to obtain reproducible results
theta = np.array(np.random.random(size=X.shape[1]))
best_feature_index = [0]
jz = []

features_ind = np.arange(1,X.shape[1])
for i in range(3):
    for j in features_ind:
       best_feature_index.append(j)
       thetaco = gradient_descent(X_train[:,best_feature_index],y_train,theta[best_feature_index],best_alpha,num_iter)[0]
       jz.append(compute_cost(X_test[:,best_feature_index],y_test,thetaco))
       best_feature_index.pop()
       
    best_feature_index.append(features_ind[jz.index(min(jz))])
    features_ind = np.delete(features_ind,jz.index(min(jz)))
    jz=[]
colnames[best_feature_index[1:]]





jz = []
num_iter = 20000
features_ind = np.arange(1,X.shape[1])
for i in range(X.shape[1]-4):
        for j in features_ind:
            thetaco = gradient_descent(X_train[:,np.append(0,np.delete(features_ind,np.where(features_ind==j)))], y_train,theta[np.append(0,np.delete(features_ind,np.where(features_ind==j)))],best_alpha,num_iter)[0]
            jz.append(compute_cost(X_test[:,np.append(0,np.delete(features_ind,np.where(features_ind==j)))],y_test,thetaco))
            
        features_ind = np.delete(features_ind,jz.index(max(jz)))
        jz=[]
colnames[features_ind]
        
    
    



np.random.seed(42) # seeding the random number generator allows us to obtain reproducible results
theta = np.array(np.random.random(size=X.shape[1]))

j=1
theta[j] = theta[j]-(alpha/X.shape[0])*((theta.dot(X.transpose()) - y)*X[:,j]).sum()

derivative = np.dot(X[:,j].transpose(),(np.dot(X,theta) - y))
theta[j] = theta[j] - alpha*(1/X.shape[j])*(derivative)






















