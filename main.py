from scipy.io import loadmat
import numpy as np
from scipy.stats import norm
from bayes_classifier import *
from fda_classifier import Fisher
from fda_classifier_2nd import *
from decisionTree import *

data_train = loadmat("data/Data_Train.mat")['Data_Train']
label_train = loadmat("data/Label_Train.mat")['Label_Train']
data_test = loadmat("data/Data_test.mat")['Data_test']

# bayes classifier
bayes = Bayes(data_train, label_train, num_classes=3)
bayes.train()

bayes_accuracy = bayes.cal_acc()
bayes_res = bayes.predict(data_test)
print("--------------------------------Bayes-----------------------------------")
print("Accuracy of Bayes on training set: ", bayes_accuracy)
print("Prediction of Bayes on test set: \n", bayes_res)

# fisher classifier
fisher = Fisher(data_train, label_train, num_classes=3)
fisher_accuracy = fisher.cal_acc()
fisher_res = fisher.classify(data_test)
print("--------------------------------Fisher-----------------------------------")
print("Accuracy of Bayes on training set: ", fisher_accuracy)
print("Prediction of Fisher on test set: \n", fisher_res)

# decision tree
print("-----------------------------Decision Tree--------------------------------")
tree = DecisionTreeClassifier()
tree.fit(data_train, label_train)
tree_accuracy = tree.score(data_train, label_train)
tree_res = list(tree.predict(data_test))
print("Accuracy of Decision Tree on training set: ", tree_accuracy)
print("Prediction of Decision Tree on test set: \n", tree_res)
