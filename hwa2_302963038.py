# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 10:55:32 2019

@author: goldi
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# make matplotlib figures appear inline in the notebook
#%matplotlib inline
#plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
#plt.rcParams['image.interpolation'] = 'nearest'
#plt.rcParams['image.cmap'] = 'gray'

# Make the notebook automatically reload external python modules
#%load_ext autoreload
#%autoreload 2



class Node(object):
    def __init__(self, data):
        self.data = data
        self.children = []

    def add_child(self, node):
        self.children.append(node)


n = Node(5)
n.data
p = Node(6)
q = Node(7)
p.add_child(q)
n.add_child(p)
n.add_child(q)
n.children[0].children
p.children


from sklearn import datasets
from sklearn.model_selection import train_test_split

# load dataset
X, y = datasets.load_breast_cancer(return_X_y = True)
X = np.column_stack([X,y]) # the last column holds the labels

# split dataset
X_train, X_test = train_test_split(X, random_state=99)

print("Training dataset shape: ", X_train.shape)
print("Testing dataset shape: ", X_test.shape)


def calc_gini(data):
    """
    Calculate gini impurity measure of a dataset.
 
    Input:
    - data: any dataset where the last column holds the labels.
 
    Returns the gini impurity.    
    """
    gini = 0
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    
    labels = data[:,-1]
    ones = (labels==1).sum()/len(labels)
    zeroes = (labels==0).sum()/len(labels)
    gini = 1-(((ones)**2)+((zeroes)**2))
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return gini



data=subset_below
def calc_entropy(data):
    """
    Calculate the entropy of a dataset.

    Input:
    - data: any dataset where the last column holds the labels.

    Returns the entropy of the dataset.    
    """
    entropy = 0.0
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    labels = data[:,-1]
    ones = (labels==1).sum()/len(labels)
    zeroes = (labels==0).sum()/len(labels)
    if ones==0 or zeroes==0:
        return 0
    else:
        entropy = -ones*np.log2(ones)-zeroes*np.log2(zeroes)
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return entropy

test = np.zeros((10,3))
test[9,-1] = 1       
test[5:10,-1] = [1,1,1,1,1]        
calc_entropy(test)    

def gain_info(above, below, node_uncertainty):
    gini_above = calc_gini(above)
    gini_below = calc_gini(below)

    return node_uncertainty - ((len(above[:,-1])/len(X[:,-1]))*gini_above)+ ((len(below[:,-1])/len(X[:,-1]))*gini_below)

### test
self = DecisionNode(X_train)
#self=node
impurity_measure = calc_gini
impurity_measure = calc_entropy
###
    
class DecisionNode:
    '''
    This class will hold everyhing you need to construct a node in a DT. You are required to 
    support basic functionality as previously described. It is highly recommended that you  
    first read and understand the entire exercises before diving into this class.
    You are allowed to change the structure of this class as you see fit.
    '''
 
    def __init__(self, data,feature=None,count=None,value=None,childrenData=None,height = None,leaf = "A node" ,left_child = None,right_child = None, prediction = None,left_branch = None,right_branch = None,left_leaf=None,right_leaf = None):
        # you should take more arguments as inputs when initiating a new node
        self.data = data
        self.children = []
        self.left_branch = left_branch
        self.right_branch = right_branch
        self.left_leaf = left_leaf
        self.right_leaf = right_leaf
        self.childrenData = childrenData
        self.feature = feature
        self.threshold = value
        self.count = count
        self.height = height 
        self.leaf = leaf
        self.left_child = left_child
        self.right_child = right_child
        self.prediction = prediction
        
    def add_child(self, node):
        self.children.append(node)
        
    def counter(self):
        outcomes = self.data[:,-1]
        zeroes = (outcomes==0).sum()
        ones = (outcomes==1).sum()
        self.count = [ones,zeroes]
        
    def leafer(self):
    	self.leaf = "A leaf!" 
            
    def predictioner(self):
        outcomes = self.data[:,-1]
        zeroes = (outcomes==0).sum()
        ones = (outcomes==1).sum()
        self.prediction = {'0':(zeroes/len(outcomes))*100,'1':(ones/len(outcomes))*100}
     
    def check_split(self, feature, value, returnChild):
        # this function divides the data according to a specific feature and value
        # you should use this function while testing for the optimal split
            subset_above = self.data[self.data[:,feature]>=value]           
            subset_below = self.data[self.data[:,feature]<=value]
            if returnChild == True:
                return [[(subset_above[:,-1]==1).sum(),(subset_above[:,-1]==0).sum()],[(subset_below[:,-1]==1).sum(),(subset_below[:,-1]==0).sum()]]
            else:    
                return (subset_above,subset_below)

    
    def split(self, impurity_measure):
        # this function goves over all possible features and values and finds
        # the optimal split according to the impurity measure. Note: you can
        # send a function as an argument
        
        b_col, b_val, b_score = None, None, 1
        nfeatures = self.data.shape[1]-1
        for i in range(nfeatures):
            feature = np.sort(self.data[:,i])
            avg = [sum(feature[i:i + 2])/2 for i in range(len(feature) - 2 + 1)] 
            for j in range(len(avg)):
                subset_above, subset_below = self.check_split(i,avg[j],False)
                score_above = impurity_measure(subset_above)
                score_below = impurity_measure(subset_below)
                score = ((len(subset_above[:,-1])/len(self.data[:,-1]))*score_above)+ ((len(subset_below[:,-1])/len(self.data[:,-1]))*score_below)
                if score < b_score:
                    b_col, b_val, b_score = i, avg[j], score
                    
        self.children = self.check_split(b_col,b_val,True)
        self.childrenData = self.check_split(b_col,b_val,False)
        self.feature = b_col
        self.threshold = b_val
        self.counter()
        
    def split_child(self,impurity_measure,height = 0):
        global hight
        left, right = self.childrenData
        left_count, right_count = self.children
        self.childrenData = None
    	# check for a no split
        #if not left.any() and not right.any():
         #   self.leafer()
          #  self.predictioner()
        #if not left.any() or not right.any():
            # self.left_child = self.right_child = leaf(left + right)
         #   if left.any():
          #      self.left_branch = DecisionNode(left)
           # else:
                #self.add_child(DecisionNode(right))
             #   self.right_branch = DecisionNode(right)                
            #return
       # if left.any() and right.any():
            #self.add_child(DecisionNode(left))
            #self.add_child(DecisionNode(right))
        #    self.left_branch = DecisionNode(left)
        #   self.right_branch = DecisionNode(right)  
    	# left branch      
        if 0 in left_count:
            node = DecisionNode(left)
            node.leafer()
            node.predictioner()
            node.count = self.children[0]
            self.left_leaf = node
            self.height = height
            node.height = height
        else:
            node = DecisionNode(left)
            self.left_branch = node
            self.height = height
            node.split(impurity_measure)
            node.split_child(impurity_measure,height+1)
    	# right branch
        if 0 in right_count:
            node = DecisionNode(right)
            node.leafer()
            node.predictioner()
            node.count = self.children[1]
            self.right_leaf = node
            self.height = height
            node.height = height
        else:
            node = DecisionNode(right)
            self.right_branch = node
            self.height = height
            node.split(impurity_measure)
            node.split_child(impurity_measure,height+1)

        
        
def build_tree(data,impurity_measure):
    """
    Build a tree using the given impurity measure and training dataset. 
    You are required to fully grow the tree until all leaves are pure. 
 
    Input:
    - data: the training dataset.
    - impurity: the chosen impurity measure. Notice that you can send a function
                as an argument in python.
 
    Output: the root node of the tree.
    """
    root = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    
    root = DecisionNode(data)
    root.split(impurity_measure)
    root.split_child(impurity_measure,0)
    return root

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return root        
        


tree = build_tree(X_train,calc_gini)



tree_gini = build_tree(data=X_train, impurity_measure=calc_gini) 
tree_entropy = build_tree(data=X_train, impurity_measure=calc_entropy)
X_test[0,:][0]


tree.children
tree.height
tree.right_branch
tree.right_branch.feature
tree.right_branch.threshold
tree.right_branch.height
tree.right_branch.children
tree.right_branch.left_branch.height
tree.right_branch.left_branch.children
tree.right_branch.left_branch.left_branch.children
tree.right_branch.right_branch.height
tree.right_branch.right_branch.children
tree.right_branch.right_branch.right_branch.height
tree.right_branch.right_branch.right_branch.children
tree.right_branch.right_branch.right_branch.right_branch.height
tree.right_branch.right_branch.right_branch.right_branch.children
tree.right_branch.right_branch.right_branch.right_branch.right_leaf.leaf == 'A leaf!'
tree.right_branch.right_branch.right_branch.right_branch.right_branch.children
tree.right_branch.right_branch.right_branch.right_branch.right_branch.right_branch.height



tree.left_branch.height
tree.left_branch.children
tree.left_branch.left_branch.height
tree.left_branch.left_branch.children
tree.left_branch.left_branch.left_branch.height
tree.left_branch.left_branch.left_branch.children
tree.left_branch.left_branch.left_branch.left_leaf.leaf
tree.left_branch.left_branch.left_branch.right_branch.children




node = tree
node = node.right_branch
node = node.left_branch
instance = X_test[0,:]
def predict(node, instance):
    """
    Predict a given instance using the decision tree
 
    Input:
    - root: the root of the decision tree.
    - instance: an row vector from the dataset. 
 
    Output: the prediction of the instance.
    """
    pred = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    if node.left_leaf is not None or node.right_leaf is not None:
        if node.left_leaf is not None:
            pred = node.left_leaf.prediction
            return pred
        else:
            pred = node.right_leaf.prediction
            return pred
            

    if instance[node.feature]>=node.threshold:
        return predict(node.left_branch,instance)
    else:
        return predict(node.right_branch,instance)
    
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return pred



b = X_test[0,:]
b[-1]=0
predict(tree,X_test[2,:])
max(a)
len(X_test[0:3,:])


row = X_test[0,:]
def calc_accuracy(node, dataset):
    """
    Predict a given dataset using the decision tree
 
    Input:
    - node: a node in the decision tree.
    - dataset: the dataset on which the accuracy is evaluated
 
    Output: the accuracy of the decision tree on the given dataset (%).
    """
    accuracy = 0
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    for row in dataset:
        pred = max(predict(node,row))
        if pred == str(int(row[-1])):
            accuracy+=1
            
    accuracy = (accuracy/len(dataset))*100
        
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return accuracy

tree_gini = build_tree(X_train, calc_gini) 
tree_entropy = build_tree(X_train, calc_entropy)

tree_gini.children
str(tree_gini.right_branch.right_branch.right_branch.right_branch.count)
tree_entropy.right_branch.right_branch.right_branch.right_branch.children


predict(tree,X_test[20,:])
calc_accuracy(tree_gini,X_test)
calc_accuracy(tree_entropy,X_test)

tree_gini.children
tree_entropy.children

node.children
node.threshold
node = tree_gini
node = node.left_branch
node = node.right_branch

flip=False
flip=True
anchor = True
anchor = False

def print_tree(node,flip,anchor, spacing=""):
    """
    Prints the tree similar to the example above.
    As long as the print is clear, any printing scheme will be fine
    
    Input:
    - node: a node in the decision tree.
 
    Output: This function has no return value.
    """
    
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    if anchor:
        global root
        root = node
        anchor = False 
        
    if not flip:
        if node.left_leaf is not None and node.right_leaf is not None:
            print (spacing ,node.left_leaf.leaf + ' left', node.left_leaf.count)
            print (spacing ,node.right_leaf.leaf + ' right', node.right_leaf.count)
            return
        else:
            if node.left_leaf is not None or node.right_leaf is not None:
                if node.left_leaf is not None:
                    print (spacing ,node.left_leaf.leaf + ' left', node.left_leaf.count)
                    print_tree(node.right_branch,False,False, spacing + "  ")
                if node.right_leaf is not None:
                    print (spacing ,node.right_leaf.leaf + ' right', node.right_leaf.count)
                    print_tree(node.left_branch,False,False, spacing + "  ")
                return    
        print (spacing + 'Node ' + str(node.count))     
        if node.left_branch is not None:  
            print (spacing + 'Feature ' + str(node.feature))
            print (spacing + 'Cutoff ' + str(node.threshold))
            print (spacing + '--> Left:')
            print_tree(node.left_branch,False,False, spacing + "  ")
            
        
        if node.right_branch is not None:  
            print (spacing + 'Feature ' + str(node.feature))
            print (spacing + 'Cutoff ' + str(node.threshold))
            print (spacing + '--> Right:')
            print_tree(node.right_branch,False,False, spacing + "  ")
            
        return print_tree(root, True,False, spacing + "  ")
    else:
        if node.left_leaf is not None and node.right_leaf is not None:
            print (spacing ,node.left_leaf.leaf + ' left', node.left_leaf.count)
            print (spacing ,node.right_leaf.leaf + ' right', node.right_leaf.count)
            return
        else:
            if node.left_leaf is not None or node.right_leaf is not None:
                if node.left_leaf is not None:
                    print (spacing ,node.left_leaf.leaf + ' left', node.left_leaf.count)
                    print_tree(node.right_branch,True,False, spacing + "  ")
                if node.right_leaf is not None:
                    print (spacing ,node.right_leaf.leaf + ' right', node.right_leaf.count)
                    print_tree(node.left_branch,True,False, spacing + "  ")
                return    
        print (spacing + 'Node ' + str(node.count))          
        if node.right_branch is not None:
            print (spacing + 'Feature ' + str(node.feature))
            print (spacing + 'Cutoff ' + str(node.threshold))
            print (spacing + '--> Right:')
            print_tree(node.right_branch,True,False, spacing + "  ")
        
        if node.left_branch is not None: 
            print (spacing + 'Feature ' + str(node.feature))
            print (spacing + 'Cutoff ' + str(node.threshold))
            print (spacing + '--> Left:')
            print_tree(node.left_branch,True,False, spacing + "  ")
            
        return
    
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return


print_tree(tree_gini,flip=False,anchor = True)

tree.leaf












from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO 
from IPython.display import Image 
from pydot import graph_from_dot_data
import pandas as pd
import numpy as np

dt = DecisionTreeClassifier()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
dt.fit(X_train, y_train)

dot_data = StringIO()
export_graphviz(dt, out_file=dot_data)
(graph, ) = graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())



my_data=[['slashdot','USA','yes',18,'None'],
        ['google','France','yes',23,'Premium'],
        ['reddit','USA','yes',24,'Basic'],
        ['kiwitobes','France','yes',23,'Basic'],
        ['google','UK','no',21,'Premium'],
        ['(direct)','New Zealand','no',12,'None'],
        ['(direct)','UK','no',21,'Basic'],
        ['google','USA','no',24,'Premium'],
        ['slashdot','France','yes',19,'None'],
        ['reddit','USA','no',18,'None'],
        ['google','UK','no',18,'None'],
        ['kiwitobes','UK','no',19,'None'],
        ['reddit','New Zealand','yes',12,'Basic'],
        ['slashdot','UK','no',21,'None'],
        ['google','UK','yes',18,'Basic'],
        ['kiwitobes','France','yes',19,'Basic']]




class Person:
  def __init__(mysillyobject, name, age):
    mysillyobject.name = name
    mysillyobject.age = age

  def myfunc(abc,bla):
    print("Hello my name is " + abc.name + bla)

p1 = Person("John", 36)

