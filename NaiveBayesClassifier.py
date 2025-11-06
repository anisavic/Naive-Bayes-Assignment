import pandas as pd
import numpy as np
import string
import sys
import re


stop_words = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
    'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
    'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
    'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
    'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
    'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
    'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
    'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
    'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here',
    'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few',
    'most', 'other', 'some', 'such', 'no', 'nor', 'only', 'own', 'same', 'so',
    'than', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'm', 'll', 're', 've', 'y'
}

#define punctuation translator
translator = str.maketrans('','', string.punctuation)
#given text, return unique set of words
def preprocess_text(text):
    
    #lowercase
    text = text.lower()

    #punctuation removal
    text = text.translate(translator)

    text = re.sub(r'\d+', '', text)

    tokens = re.findall(r'\b\w+\b', text)

    tokens = [word for word in tokens if word not in stop_words]


    return tokens

#finds optimal categorical features, from feature list plus userreview
def find_optimal_features(features, train, test):
    best_feature_list = []
    features_list = []
    max_accuracy = 0.0

    for train in ["training1.json", "training2.json"]:
        for i in range(1, 2**(len(features))):
            current_features = []
            feature_i = 0
            format_size = "0" + str(len(features)) + "b"
            for c in format(i, format_size):
                if c == "1":
                    current_features.append(features[feature_i]) #change
                feature_i+=1
            current_model = NaiveBayesClassifier()
            current_model.load_data(train, current_features, True)
            current_model.train()
            predictions = current_model.predict(test)
            current_accuracy = current_model.evaluate(pd.read_json(test)["rating"].tolist(), predictions)
            if current_accuracy > max_accuracy:
                best_feature_list = current_features
                max_accuracy = current_accuracy
            features_list.append([current_accuracy, current_features])

            
        
        print(f"optimal feature list based on highest accuracy ({max_accuracy}) : {best_feature_list}")
        features_list.sort()
        print(features_list[-5:])

def preprocess_zipcode(zipcode, digits=5):
    return str(zipcode)[:digits]


#NAIVE BAYES CLASSIFIER IMPLEMENTATION
class NaiveBayesClassifier:
    def __init__(self):
        self.priors = {} #class->prior probability
        self.likelihoods = {} #feature_name, feature_value, class -> likelihood
        self.categorical_features = {}
        self.user_review = False
        self.classes = [1, 2, 3, 4, 5] #ratings

        self.vocab = set()
        self.word_count_class = {}
        self.total_words_class = {c: 0 for c in self.classes} # Initialize counts for each class

        self.train_data = None


    def load_data(self, filepath, features, user_review = False):
        self.train_data = pd.read_json(filepath)


        word_frequencies = {}

        #Extract features and their values
        for feature in features:
                if feature == "user_zip_code":
                    self.categorical_features[feature] = set(
                        preprocess_zipcode(zip_code) 
                        for zip_code in self.train_data[feature].unique()
                    )
                else:
                    self.categorical_features[feature] = set(self.train_data[feature].unique())

        #add all words to vocab
        if user_review:
            self.user_review = True
            # First pass: count frequencies
            for index, instance in self.train_data.iterrows():
                review_words = preprocess_text(instance["user_review"])
                for word in review_words:
                    word_frequencies[word] = word_frequencies.get(word, 0) + 1
            
            # Second pass: only add words that appear at least 5 times
            MIN_FREQUENCY = 20
            for index, instance in self.train_data.iterrows():
                review_words = preprocess_text(instance["user_review"])
                rating_class = instance["rating"]
                for word in review_words:
                    if word_frequencies[word] >= MIN_FREQUENCY:  # Only add frequent words
                        if word not in self.vocab:
                            self.vocab.add(word)
                            self.word_count_class[word] = {c: 0 for c in self.classes}
                        
                        if rating_class not in self.word_count_class[word]:
                            self.word_count_class[word][rating_class] = 0
                        self.word_count_class[word][rating_class] += 1
                        self.total_words_class[rating_class] += 1         


    def train(self):

        #Calculate priors (new)
        total_count = len(self.train_data)
        class_counts = self.train_data["rating"].value_counts()

        for c in self.classes:
            self.priors[c] = class_counts[c] / total_count

        #Calculate likelihoods for non-word features
        for feature in self.categorical_features.keys():
            num_unique_values = len(self.categorical_features[feature])
            if feature not in self.likelihoods:
                self.likelihoods[feature] = {} #initialize the new feature dict
            for value in self.categorical_features[feature]:
                if value not in self.likelihoods[feature]:
                    self.likelihoods[feature][value] = {}
                for c in self.classes:
                    count_feature_class = len(self.train_data[(self.train_data[feature] == value) & (self.train_data["rating"] == c)])
                    count_class = class_counts[c]
                    #Using Laplace smoothing
                    likelihood = (count_feature_class + 1) / (count_class + num_unique_values)
                    self.likelihoods[feature][value][c] = likelihood
                    # print(f"P({feature}={value}|class={c}) = {likelihood}")

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
                for feature in self.categorical_features.keys(): #for each feature Fi
                    value = instance[feature] #get the person's feature value f (occupation = "writer")
                    if value in self.likelihoods[feature]: #get likelihood P(f|c)
                        log_scores[c] += np.log(self.likelihoods[feature][value][c])
                
                #Now, check whether user_review feature is on, if so, take the review,preprocess, and add those probabilities to it using wcc and twc
                if self.user_review:
                    u_review = instance["user_review"]
                    words = preprocess_text(u_review)
                    #for each word, if in vocab, add that log likelihood
                    for word in words:
                        if word in self.vocab: #only add if we have encountered word
                            m = 0
                            #find m for laplacian smoothing (m is mumber of instances of that word amongst all classes)
                            for class_i in self.classes:
                                m += self.word_count_class[word][class_i]

                            likelihood = (self.word_count_class[word][c] + 1)/ (self.total_words_class[c] + m)
                            log_scores[c] += np.log(likelihood)

            predicted_class = max(log_scores, key=log_scores.get)
            if print_results:
                print(predicted_class)
            predictions.append(predicted_class)


        return predictions
    
    def evaluate(self, true_labels, predicted_labels):
        correct = sum(t == p for t, p in zip(true_labels, predicted_labels))
        total = len(true_labels)
        accuracy = correct / total
        print(f"Accuracy: {accuracy * 100:.2f}%")
        return accuracy
    


                    
            # print(features_to_add)



if __name__ == "__main__":
    train_set = sys.argv[1]
    test_set = sys.argv[2]
    
    # print(preprocess_text("Hi... I am sasha, how-are you? I feel pretty good about myself today."))

    classifier = NaiveBayesClassifier()
    # features = ["user_gender", "user_occupation", "user_id", "item_id", "user_zip_code"]
    #For submission _UNCOMMENT_
    features = ["user_zip_code", "user_id", "item_id"] #OPETIMAL FEATURE SET
    classifier.load_data(train_set, features, True)
    classifier.train()
    predictions = classifier.predict(test_set, print_results=True)


    # #Optimization, comment when submission
    # features = ["user_gender", "user_occupation", "user_id", "item_id", "user_zip_code"]
    # feature_opt = [""]
    # classifier.load_data(train_set, features, True)
    # classifier.train()
    # print(classifier.categorical_features)
    # predictions = classifier.predict(test_set, print_results=False)
    # classifier.evaluate(pd.read_json(test_set)["rating"].tolist(), predictions)
    # find_optimal_features(features, train_set, test_set)

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

