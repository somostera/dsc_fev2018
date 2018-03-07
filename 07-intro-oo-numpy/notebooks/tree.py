## 
## Description: 
##    Script designed to train a given DecisionTreeClassifier and extract it's parameters 
##    to a specific format.
## 
## Author: Allan Dieguez (allandieguez@gmail.com)
## 
## Script based on code provided at Scikit-Learn's website on March 3rd, 2018:
## 
##    'http://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html';
## 
import numpy as np


class BinaryTree:
    
    def __init__(self, id, value=None, parent=None):
        self.id = id
        self.value = value
        self.parent = parent
        self.left_child = None
        self.right_child = None
    
    def append_left_child(self, tree):
        self.left_child = tree
        tree.parent = self
        return self
    
    def append_right_child(self, tree):
        self.right_child = tree
        tree.parent = self
        return self
            
    def is_leaf(self):
        return self.left_child is None and self.right_child is None
    
    def is_root(self):
        return self.parent is None
    
    def depth(self):
        if self.is_root():
            return 0
        return self.parent.depth() + 1
    
    def height(self):
        h = 0
        if self.left_child:
            h = max(h, self.left_child.height())
        if self.right_child:
            h = max(h, self.right_child.height())
        return h
            
    def __str__(self):
        """ string representation of the object
        """ 
        resp = "ID: {} - Value: {}".format(self.id, self.value)
        
        if self.is_root():
            resp = "{} - Root Node".format(resp)
        else:
            resp = "{} - Parent: {}".format(resp, self.parent.id)
            
        if self.is_leaf():
            resp = "{} - Leaf Node".format(resp)
        else:
            children = {}
            if self.left_child:
                children["L"] = self.left_child.id
            if self.right_child:
                children["R"] = self.right_child.id
            resp = "{} - Children ({}): {}".format(resp, len(children), children)
            
        return resp

    
def extract_tree_from_model(model, tree_type=BinaryTree):
    """ Extracts the parameters from a scikit-learn's trained model into 
        a BinaryTree inherited object.        
    """
    n_nodes = model.tree_.node_count
    children_left = model.tree_.children_left
    children_right = model.tree_.children_right
    feature = model.tree_.feature
    threshold = model.tree_.threshold    
    decision = model.tree_.value
    root = tree_type(
        id=0, 
        value={"feature": feature[0], "threshold": threshold[0]}
    )
    stack = [root]    
    while len(stack) > 0:
        node = stack.pop()
        n_id = node.id
        if children_left[n_id] > -1:
            c_id = children_left[n_id]
            child =  tree_type(
                id=c_id, 
                value={"feature": feature[c_id], "threshold": threshold[c_id]}
            )
            node.append_left_child(child)
            stack.append(child)
        if children_right[n_id] > -1:
            c_id = children_right[n_id]
            child = tree_type(
                id=c_id, 
                value={"feature": feature[c_id], "threshold": threshold[c_id]}
            )
            node.append_right_child(child)
            stack.append(child)
        if node.is_leaf():
            output = decision[n_id][0]
            node.value = {
                "probability": max(output / output.sum()),
                "decision": np.argmax(output)
            }
    return root
    
    
def print_tree(tree):
    """ Recursively prints a BinaryTree.
    """
    if tree:
        print(tree.depth() * "\t" + str(tree))
        print_tree(tree.left_child)
        print_tree(tree.right_child)
