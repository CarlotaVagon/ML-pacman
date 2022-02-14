# classifierAgents.py
# parsons/07-oct-2017
#
# Version 1.0
#
# Some simple agents to work with the PacMan AI projects from:
#
# http://ai.berkeley.edu/
#
# These use a simple API that allow us to control Pacman's interaction with
# the environment adding a layer on top of the AI Berkeley code.
#
# As required by the licensing agreement for the PacMan AI we have:
#
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

# The agents here are extensions written by Simon Parsons, based on the code in
# pacmanAgents.py

# some of the resources I used:
# https://machinelearningmastery.com/implement-random-forest-scratch-python/
# https://github.com/olivierroncalez/pacman
# Note to self my mental health was not healthy during this project, due to stress and family health issues
# This most likely lead me to procrastinate, under perform and have a few breakdowns
# In the future I will try my best to time myself well, and bring the motivation from within to decrease my stress
# and improve my work.
#

# own marking check list:
# runs? YES
# classifier? YES :random forest , train_test_split
# classifier without a library random_forest function
# sophisticated? unsure
# runs on other data? unsure
# commented: YES
# Good programming style? unsure


from pacman import Directions
from game import Agent
import api
import random
import game
import util
import sys
import os
import csv
import numpy as np
from random import seed
from random import randrange
from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


# ClassifierAgent
#
# An agent that runs a classifier to decide what to do.
class ClassifierAgent(Agent):

    # Constructor. This gets run when the agent starts up.
    def __init__(self):
        print "Initialising"

    # Take a string of digits and convert to an array of
    # numbers. Exploits the fact that we know the digits are in the
    # range 0-4.
    #
    # There are undoubtedly more elegant and general ways to do this,
    # exploiting ASCII codes.

    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # my attempt to a classifier

    # Cross validation data split
    # this splits the data into n folds
    # Split a dataset into k folds
    def cross_validation_split(self, data, n_folds):
        data_split = list()
        data_copy = list(data)
        fold_size = int(len(data) / n_folds)
        for i in range(n_folds):
            fold = list()
            while len(fold) < fold_size:
                index = randrange(len(data_copy))
                fold.append(data_copy.pop(index))
            data_split.append(fold)
        return data_split

    # Calculate accuracy percentage
    def accuracy_metric(self, actual, predicted):
        correct = 0
        for i in range(len(actual)):
            if actual[i] == predicted[i]:
                correct += 1
        return correct / float(len(actual)) * 100.0

    # Evaluate an algorithm using a cross validation split
    def evaluate_algorithm(self, data, algorithm, n_folds, *args):
        folds = self.cross_validation_split(data, n_folds)
        scores = list()
        for fold in folds:
            train_set = list(folds)
            train_set.remove(fold)
            train_set = sum(train_set, [])
            test_set = list()
            for row in fold:
                row_copy = list(row)
                test_set.append(row_copy)
                row_copy[-1] = None
            predicted = algorithm(train_set, test_set, *args)
            actual = [row[-1] for row in fold]
            accuracy = self.accuracy_metric(actual, predicted)
            scores.append(accuracy)
        return scores

    # Split a dataset based on an attribute and an attribute value
    def test_split(index, value, data):
        left, right = list(), list()
        for row in data:
            if row[index] < value:
                left.append(row)
            else:
                right.append(row)
        return left, right

    # Calculate the Gini index for a split dataset
    def gini_index(groups, classes):
        # count all samples at split point
        n_instances = float(sum([len(group) for group in groups]))
        # sum weighted Gini index for each group
        gini = 0.0
        for group in groups:
            size = float(len(group))
            # avoid divide by zero
            if size == 0:
                continue
            score = 0.0
            # score the group based on the score for each class
            for class_val in classes:
                p = [row[-1] for row in group].count(class_val) / size
                score += p * p
            # weight the group score by its relative size
            gini += (1.0 - score) * (size / n_instances)
        return gini

    # Select the best split point for a dataset
    def get_split(self, data, n_features):
        class_values = list(set(row[-1] for row in data))
        b_index, b_value, b_score, b_groups = 999, 999, 999, None
        features = list()
        while len(features) < n_features:
            index = randrange(len(data[0]) - 1)
            if index not in features:
                features.append(index)
        for index in features:
            for row in data:
                groups = self.test_split(index, row[index], data)
                gini = self.gini_index(groups, class_values)
                if gini < b_score:
                    b_index, b_value, b_score, b_groups = index, row[index], gini, groups
        return {'index': b_index, 'value': b_value, 'groups': b_groups}

    # Create a terminal node value
    def to_terminal(group):
        outcomes = [row[-1] for row in group]
        return max(set(outcomes), key=outcomes.count)

    # Create child splits for a node or make terminal
    def split(self, node, max_depth, min_size, n_features, depth):
        left, right = node['groups']
        del (node['groups'])
        # check for a no split
        if not left or not right:
            node['left'] = node['right'] = self.to_terminal(left + right)
            return
        # check for max depth
        if depth >= max_depth:
            node['left'], node['right'] = self.to_terminal(left), self.to_terminal(right)
            return
        # process left child
        if len(left) <= min_size:
            node['left'] = self.to_terminal(left)
        else:
            node['left'] = self.get_split(left, n_features)
            self.split(node['left'], max_depth, min_size, n_features, depth + 1)
        # process right child
        if len(right) <= min_size:
            node['right'] = self.to_terminal(right)
        else:
            node['right'] = self.get_split(right, n_features)
            self.split(node['right'], max_depth, min_size, n_features, depth + 1)

    # Build a decision tree
    def build_tree(self, train, max_depth, min_size, n_features):
        root = self.get_split(train, n_features)
        self.split(root, max_depth, min_size, n_features, 1)
        return root

    # Make a prediction with a decision tree
    def predict(self, node, row):
        if row[node['index']] < node['value']:
            if isinstance(node['left'], dict):
                return self.predict(node['left'], row)
            else:
                return node['left']
        else:
            if isinstance(node['right'], dict):
                return self.predict(node['right'], row)
            else:
                return node['right']

    # Create a random subsample from the dataset with replacement
    def subsample(self, data, ratio):
        sample = list()
        n_sample = round(len(data) * ratio)
        while len(sample) < n_sample:
            index = randrange(len(data))
            sample.append(data[index])
        return sample

    # Make a prediction with a list of bagged trees
    def bagging_predict(self, trees, row):
        predictions = [self.predict(tree, row) for tree in trees]
        return max(set(predictions), key=predictions.count)

    # Random Forest Algorithm
    def random_forest(self, train, test, max_depth, min_size, sample_size, n_trees, n_features):
        trees = list()
        for i in range(n_trees):
            sample = self.subsample(train, sample_size)
            tree = self.build_tree(sample, max_depth, min_size, n_features)
            trees.append(tree)
        predictions = [self.bagging_predict(trees, row) for row in test]
        return predictions

    def convertToArray(self, numberString):
        numberArray = []
        for i in range(len(numberString) - 1):
            if numberString[i] == '0':
                numberArray.append(0)
            elif numberString[i] == '1':
                numberArray.append(1)
            elif numberString[i] == '2':
                numberArray.append(2)
            elif numberString[i] == '3':
                numberArray.append(3)
            # elif numberString[i] == '4':
            # numberArray.append(4)

        return numberArray

    # This gets run on startup. Has access to state information.
    #
    # Here we use it to load the training data.
    def registerInitialState(self, state):

        # open datafile, extract content into an array, and close.
        self.datafile = open('good-moves.txt', 'r')
        content = self.datafile.readlines()
        self.datafile.close()

        # Now extract data, which is in the form of strings, into an
        # array of numbers, and separate into matched data and target
        # variables.
        self.data = []
        self.target = []

        #X_train, X_test, y_train, y_test = train_test_split(self.data, self.target, test_size=0.2, random_state=0)

       # regresor = RandomForestRegressor(n_estimators=20, random_state=0)
       # regresor.fit(X_train, y_train)
        #y_pred = regresor.predict(X_test)

        # Turn content into nested lists
        for i in range(len(content)):
            lineAsArray = self.convertToArray(content[i])
            dataline = []
            for j in range(len(lineAsArray) - 1):
                dataline.append(lineAsArray[j])

            self.data.append(dataline)
            targetIndex = len(lineAsArray) - 1
            self.target.append(lineAsArray[targetIndex])

        # data and target are both arrays of arbitrary length.
        #
        # data is an array of arrays of integers (0 or 1) indicating state.
        #
        # target is an array of imtegers 0-3 indicating the action
        # taken in that state.

    # Tidy up when Pacman dies
    def final(self, state):

        print "I'm done!"

        # *********************************************
        #
        # Any code you want to run at the end goes here.
        #
        # *********************************************

    # Turn the numbers from the feature set into actions:
    def convertNumberToMove(self, number):
        if number == 0:
            return Directions.NORTH
        elif number == 1:
            return Directions.EAST
        elif number == 2:
            return Directions.SOUTH
        elif number == 3:
            return Directions.WEST

    # Here we just run the classifier to decide what to do
    def getAction(self, state):

        # How we access the features.
        features = api.getFeatureVector(state)

        # *****************************************************

        seed(2)
        n_folds = 5
        max_depth = 10
        min_size = 1
        sample_size = 1.0
        n_features = int(sqrt(len(self.data[0]) - 1))
        for n_trees in [1, 5, 10]:
            scores = self.evaluation_algorithm(self.data, self.random_forest, n_folds, max_depth, min_size, sample_size,
                                             n_trees, n_features)
            print('Trees: %d' % n_trees)
            print('Scores: %s' % scores)
            print('Mean Accuracy: %.3f%%' % (sum(scores) / float(len(scores))))

        # *******************************************************

        classification = self.random_forest(self.data, self.test, max_depth, min_size, sample_size, n_trees, n_features)
        # classification into a move
        best_move = self.convertNumberToMove(classification)

        # Get the actions we can try.
        legal = api.legalActions(state)

        # is the move possible?
        # yes - pacman moves
        # no - it will try the next best move all based on the predictions
        # if no moves are possible then a random move is returned
        if best_move in legal:
            return api.makeMove(best_move, legal)
        else:
            return api.makeMove(random.choice(legal), legal)

        # getAction has to return a move. Here we pass "STOP" to the
        # API to ask Pacman to stay where they are. We need to pass
        # the set of legal moves to teh API so it can do some safety
        # checking.
        #return api.makeMove(Directions.STOP, legal)

#python pacman.py --pacman ClassifierAgent