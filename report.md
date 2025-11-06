# CS165A - Programming Assignment 1: Naive Bayes Classification
**Name:** Sasha Nisavic
**Email:** anisavic@ucsb.edu
**Perm Number:** 8855504

---
## 1. Feature Design and Pre-processing (Maximum 1 Page)

### Selected Features

Initial development focused on basic categorical features provided in the dataset, such as `user_gender` and `user_occupation`. To systematically determine the most impactful features, an automated selection process was implemented. This process evaluated various combinations of the standard categorical features.

The combination yielding the highest observed accuracy on the validation set included:
* `item_id`: Unique identifier for the movie.
* `user_zip_code`: Categorical representation of user location.
* `user_review`: Raw text review provided by the user.

While features like `user_gender` and `user_occupation` were tested, they did not contribute as significantly to the final accuracy as the selected combination. Features like seemed redundant given `item_id`. `user_age` was not used due to the need for binning, and text features like `movie_description` were omitted in favor of `user_review`.

### Rationale & Pre-processing

* **`item_id` & `user_zip_code`:** These categorical features were handled directly. Unique values were identified, and likelihoods were calculated using counts from the training data with Laplace smoothing. The rationale is that individual movie preferences (`item_id`) and potentially geographic factors (`user_zip_code`) correlate with rating behavior.
* **`user_review`:** This text feature was processed using a Multinomial Naive Bayes approach. The rationale is that specific words within a review strongly indicate positive or negative sentiment, directly influencing the rating. Pre-processing involved:
    1.  Converting text to lowercase.
    2.  Removing punctuation.
    3.  Tokenizing text into words.
    4.  Removed stop words to build the vocabulary.
    
  Each unique word in the resulting vocabulary was treated as a feature, with likelihoods calculated based on word counts within each rating class, using appropriate smoothing for text.

---
## 2. Naive Bayes Implementation and Results

### Implementation Overview

A Python class `NaiveBayesClassifier` was implemented. The core logic involves separate handling for standard categorical features and the text-based `user_review` feature (using a Multinomial Naive Bayes approach). The implementation calculates log-probabilities to prevent underflow.

### Pseudocode

```plaintext
Algorithm: Naive Bayes Movie Rating Prediction

// --- Initialization ---
Initialize structures for priors, likelihoods (categorical & word), vocabulary, counts.

// --- Training ---
Function Train(training_data, features_to_use):
  Calculate Log Priors: Log(P(Rating=C)) for C in {1..5} using class counts.

  Calculate Log Likelihoods (Categorical):
    FOR F in categorical_features:
      FOR V in unique_values(F):
        FOR C in {1..5}:
          Calculate smoothed P(F=V|C) using counts + Laplace smoothing.
          Store Log(P(F=V|C)).

  Calculate Log Likelihoods (Words - Multinomial):
    Build vocabulary `vocab` from preprocessed training reviews.
    Count word occurrences per class -> word_counts[W][C].
    Count total words per class -> total_words[C].
    FOR W in `vocab`:
      FOR C in {1..5}:
        find m by finding mumber of times word appears
        Calculate smoothed P(W|C) = (count(W,C)+1) / (total_words[C] + m).
        Store Log(P(W|C)).

// --- Prediction ---
Procedure Predict(test_data):
  FOR instance in test_data:
    Initialize log_scores[C] = priors

    // Add categorical likelihoods
    FOR F in categorical_features:
      V = instance[F]
      IF V is known THEN log_scores[C] += Log_Likelihood(F, V, C).

    // Add word likelihoods (Multinomial)
    Words = Preprocess(instance["user_review"])
    FOR W in Words:
      IF W in `vocab` THEN
          FOR C in {1..5}:
              log_scores[C] += Log_Likelihood(W, C). // Use pre-calculated word likelihood

    Predict class C with Max(log_scores).
    Store prediction.
  RETURN predictions.

// --- Helper ---
Function Preprocess(text): // Lowercase, remove punct, tokenize.

```
### Probability Calculations & Classification

* **Priors P(C):** Calculated as `Count(Class C) / Total Instances`.
* **Categorical Likelihoods P(F=V|C):** Estimated using counts with Laplace (Add-1) smoothing: `(Count(F=V, C) + 1) / (Count(C) + |Unique Values for F|)`.
* **Word Likelihoods P(W|C)$ (Multinomial):** Estimated using word counts with Laplace smoothing suitable for text: `(Count(W, C) + 1) / (Total Words in C + |How many times word appears|)`.
* **Classification:** The final rating `c` is predicted by finding the class maximizing the sum of the log prior and the log likelihoods for observed features (categorical and words present in the review), following the Naive Bayes formula:
    `argmax_c [ Log(P(C)) + Σ Log(P(Feature_i | C)) + Σ Log(P(Word_j | C)) ]`

### Results

Using the features `item_id`, `user_zip_code`, and `user_review` (processed via Multinomial Naive Bayes), the classifier achieved **96.33%** accuracy on the `validation.json` dataset. The `user_review` feature significantly boosted performance, confirming the predictive power of text sentiment for this task.