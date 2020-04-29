# Perform-Sentiment-Analysis-with-scikit-learn
Build a logistic regression model to classify movie reviews as either positive or negative. We will use the popular IMDB data set. Our goal is to use a simple logistic regression estimator from scikit-learn for document classification.

## Task 1: Introduction and Importing the Data
1. Introduction to the data set and the problem overview.
2. Import essential modules and helper functions from NumPy, Matplotlib, and scikit-learn.

## Task 2: Transforming Documents into Feature Vectors
1. Represent text data using the bag-of-words model from natural language processing and information retrieval.
2. Construct the vocabulary of the bag-of-words model and transform the provided sample sentences into sparse feature vectors.

## Task 3: Term Frequency-Inverse Document Frequency
1. In information retrieval and text mining, we often observe words that crop up across our corpus of documents. These words can lead to bad performance during training and test time because they usually don’t contain useful information.
2. Understand and implement a useful statistical technique, Term frequency-inverse document frequency (tf-idf), to downweight these class of words in the feature vector representation. The tf-idf is the product of the term frequency and the inverse document frequency.

## Task 4: Calculate TF-IDF of the Term 'Is'
1. Manually calculate the tf-idf of an example.
2. Apply scikit-learn’s TfidfTransformer to convert sample text into a vector of tf-idf values and apply the L2-normalization to it.

## Task 5: Data Preparation
1. Cleaning and pre-processing text data is a vital process in data analysis and especially in natural language processing tasks.
2. Strip the data set of reviews of irrelevant characters including HTML tags, punctuation, and emojis using regular expressions.

## Task 6: Tokenization of Documents
1. Ensures that k-means image compression is performed only on the slider widget's mouse release events.
2. Repurpose the data preprocessing and k-means clustering logic from previous tasks to operate on images of your choice.
3. Visualize how the image changes as the number of clusters fed to the k-means algorithm is varied.

## Task 7: Document Classification Using Logistic Regression
1. First, split the data into training and test sets of equal size.
2. Then create a pipeline to build a logistic regression model.
3. To estimate the best parameters and model, we employ cross-validated grid-search over a parameter grid.

## Task 8: Load Saved Model from Disk
1. Although the time it takes to train logistic regression models is very little, estimating the best parameters for our model using GridSearchCV can take hours given the size of our training set.
2. In this task, load a pre-trained model that will later be used to find the best parameter settings, cross validation score, and the test accuracy.

## Task 9: Model Accuracy
1. In this final task, we take a look at the best parameter settings, cross-validation score, and how well our model classifies the sentiments of reviews it has never seen before from the test set
