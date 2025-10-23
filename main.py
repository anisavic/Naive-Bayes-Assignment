import pandas as pd
import numpy as np
import argparse
import sys

#NAIVE BAYES CLASSIFIER IMPLEMENTATION
class NaiveBayesClassifier:
    def __init__(self):
        self.priors = {} #class->prior probability
        self.likelihoods = {} #feature_name, feature_value, class -> likelihood
        self.features = {} #feature_name -> set of possible values
        self.classes = [1, 2, 3, 4, 5] #ratings

        self.train_data = None


    def load_data(self, filepath):
        self.train_data = pd.read_json(filepath)

        # #hardcode the small features values for now, change this
        # self.features["user_gender"] = {"M", "F"}
        # self.features["user_occupation"] = set(self.train_data["user_occupation"].unique())

        print(self.features)


    def train(self, features):
        #Extract features and their values
        for feature in features:
            self.features[feature] = set(self.train_data[feature].unique())

        #Calculate priors (new)
        total_count = len(self.train_data)
        class_counts = self.train_data["rating"].value_counts()

        for c in self.classes:
            self.priors[c] = class_counts[c] / total_count
            print(f"Prior for class {c}: {self.priors[c]}")

        #Calculate likelihoods
        for feature in self.features.keys():
            num_unique_values = len(self.features[feature])
            if feature not in self.likelihoods:
                self.likelihoods[feature] = {} #initialize the new feature dict
            for value in self.features[feature]:
                if value not in self.likelihoods[feature]:
                    self.likelihoods[feature][value] = {}
                for c in self.classes:
                    count_feature_class = len(self.train_data[(self.train_data[feature] == value) & (self.train_data["rating"] == c)])
                    count_class = class_counts[c]
                    #Using Laplace smoothing
                    likelihood = (count_feature_class + 1) / (count_class + num_unique_values)
                    self.likelihoods[feature][value][c] = likelihood
                    print(f"P({feature}={value}|class={c}) = {likelihood}")

    #given filepath to test set, predict ratings
    def predict(self, filepath, print_results=False):
        test_data = pd.read_json(filepath)
        predictions = []
        
        # Iterate over each row in the test data
        for index, instance in test_data.iterrows():
            log_scores = {c: 0.0 for c in self.classes} # best score predicts class
            for c in self.classes:
                #find class with maximum score
                log_scores[c] = np.log(self.priors[c]) #P(c)
                for feature in self.features.keys(): #for each feature Fi
                    value = instance[feature] #get the person's feature value f (occupation = "writer")
                    if value in self.likelihoods[feature]: #get likelihood P(f|c)
                        log_scores[c] += np.log(self.likelihoods[feature][value][c])
                    else: #If we have never seen this value before MAYBE DELETE THIS< UNNESESARY?
                        log_scores[c] += np.log(1 / (len(self.train_data[self.train_data["rating"] == c]) + len(self.features[feature])))


            predicted_class = max(log_scores, key=log_scores.get)
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
    features = ["user_gender", "user_occupation", "user_id", "item_id"]
    classifier.load_data(train_set)
    classifier.train(features)
    predictions = classifier.predict(test_set, print_results=False)
    classifier.evaluate(pd.read_json(test_set)["rating"].tolist(), predictions)

    # #Pandas prep
    # td = pd.read_json(train_set)

    # #print first 5
    # print(td.head())

    # #get total number of instances
    # total_instances = len(td)
    # print(f"Total instances in training set: {total_instances}")

    # #get all features
    # feature_names = td.columns.tolist()
    # print(f"Feature names: {feature_names}")

    # #get a specific value from specific feature/class
    # sample_occupation = td["user_occupation"].iloc[0]
    # print(f"Sample occupation from first instance: {sample_occupation}")

    # #get unique values for specific feature
    # #NOT WORKING?
    # unique_occupations = td["user_occupation"].unique()
    # print(f"Unique occupations in training set: {unique_occupations}")

    # # #get count of unique vlaues for specific feature
    # # num_unique_occupations = td["user_occupation"].nunique()
    # # print(f"Number of unique occupations: {num_unique_occupations} or also just use {len(unique_occupations)}")

    # #get number of occurences of each feature value (useful for priors)
    # ratings = td["rating"].value_counts()
    # print(f"Rating counts:\n{ratings}")
    # ratings[5] #get like a dict value

    # #get subset of data where class is a specific value
    # td_rating_5 = td[td['rating'] == 5]
    # list_occupations_rating_5 = td_rating_5["user_occupation"].value_counts()
    # print(f"Occupations for rating 5:\n{list_occupations_rating_5}")
    # #
    # numwriters_rating_5 = list_occupations_rating_5["writer"]
    # print(f"Number of writers with rating 5: {numwriters_rating_5}")

