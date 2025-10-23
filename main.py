import pandas as pd
import numpy as np
import argparse
import sys

#NAIVE BAYES CLASSIFIER IMPLEMENTATION
class NaiveBayesClassifier:
    def __init__(self):
        self.priors = {} #class->prior probability
        self.likelihoods = {} #feature_name, feature_value, class -> likelihood
        self.feature_values = {} #feature_name -> set of possible values
        self.classes = [1, 2, 3, 4, 5] #ratings

        self.train_data = None


    def load_data(self, filepath):
        self.train_data = pd.read_json(filepath)

        #hardcode the small features values for now
        self.feature_values["user_gender"] = {"M", "F"}
        self.feature_values["user_occupation"] = set(self.train_data["user_occupation"].unique())

        print(self.feature_values)


        for feature in self.train_data.columns:
            print(f"Feature: {feature}")

    def train(self):
        #Calculate priors
        total_count = len(self.train_data)
        for c in self.classes:
            class_count = len(self.train_data[self.train_data["rating"] == c])
            self.priors[c] = class_count / total_count
            print(f"Prior for class {c}: {self.priors[c]}")

        #Calculate likelihoods
        for feature in self.feature_values.keys():
            if feature not in self.likelihoods:
                self.likelihoods[feature] = {} #define dictionary within likelihoods
            for value in self.feature_values[feature]:
                if value not in self.likelihoods[feature]:
                    self.likelihoods[feature][value] = {}
                for c in self.classes:
                    count_feature_class = len(self.train_data[(self.train_data[feature] == value) & (self.train_data["rating"] == c)])
                    count_class = len(self.train_data[self.train_data["rating"] == c])
                    #Using Laplace smoothing
                    likelihood = (count_feature_class + 1) / (count_class + len(self.feature_values[feature]))
                    self.likelihoods[feature][value][c] = likelihood
                    print(f"P({feature}={value}|class={c}) = {likelihood}")

    #given filepath to test set, predict ratings
    def predict(self, filepath, print_results=False):
        test_data = pd.read_json(filepath)
        predictions = []
        
        # Iterate over each row in the test data
        for index, row in test_data.iterrows():
            scores = [] # best score predicts class
            for c in self.classes:
                #find class with maximum score
                score = np.log(self.priors[c]) #P(c)
                for feature in self.feature_values.keys():
                    feature_value = row[feature]
                    if feature_value in self.likelihoods[feature]:
                        likelihood = self.likelihoods[feature][feature_value][c]
                    else: #Not sure about this...
                        # Handle unseen feature values with Laplace smoothing
                        likelihood = 1 / (len(self.train_data[self.train_data["rating"] == c]) + len(self.feature_values[feature]))
                    score += np.log(likelihood) #P(f|c)

                scores.append(score)
            
            predicted_class = self.classes[np.argmax(scores)]
            if print_results:
                print(f"Predicted class for test instance {index}: {predicted_class}")
            predictions.append(predicted_class)


        return predictions
    
    def evaluate(self, true_labels, predicted_labels):
        correct = sum(t == p for t, p in zip(true_labels, predicted_labels))
        total = len(true_labels)
        accuracy = correct / total
        print(f"Accuracy: {accuracy * 100:.2f}%")
        return accuracy


if __name__ == "__main__":
    train_set = sys.argv[1]
    test_set = sys.argv[2]
    

    #load into valid pandas dataframes
    

    print(f"datasets are {train_set} and {test_set}")
    classifier = NaiveBayesClassifier()
    classifier.load_data(train_set)
    classifier.train()
    predictions = classifier.predict(test_set, print_results=False)
    classifier.evaluate(pd.read_json(test_set)["rating"].tolist(), predictions)
