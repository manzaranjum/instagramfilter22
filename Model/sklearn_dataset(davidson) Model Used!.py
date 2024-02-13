import pandas as pd
import re
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

# Load the dataset
train_data = pd.read_csv("dataset (davidson).csv")

# Rename columns for clarity
train_data.rename(columns={'tweet': 'text', 'class': 'category'}, inplace=True)

# Create a mapping for category labels
category_mapping = {0: 'hate_speech', 1: 'offensive_language', 2: 'neither'}
train_data['category'] = train_data['category'].map(category_mapping)

# Perform lemmatization on text data
train_data['text_lem'] = [''.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]', ' ', text)) for text in lis]) for lis in train_data['text']]

# Create category_id for encoding categories
train_data['category_id'] = train_data['category'].factorize()[0]

# Define the TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(sublinear_tf=True, min_df=50, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')

# Vectorize text data and obtain labels
features = tfidf_vectorizer.fit_transform(train_data['text_lem']).toarray()
labels = train_data['category_id']

# Split data into training and testing sets
X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, train_data.index, test_size=0.33, random_state=0)

# Define machine learning models
models = [
    RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0),
    LinearSVC(),
    MultinomialNB(),
    LogisticRegression(random_state=0),
]

# Perform cross-validation for each model
CV = 5
cv_df = pd.DataFrame(index=range(CV * len(models)))
entries = []

for model in models:
    model_name = model.__class__.__name__
    accuracies = cross_val_score(model, X_train, y_train, scoring='accuracy', cv=CV)
    for fold_idx, accuracy in enumerate(accuracies):
        entries.append((model_name, fold_idx, accuracy))

cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])

# Plot accuracy of models
plt.figure(figsize=(10, 6))
sns.boxplot(x='model_name', y='accuracy', data=cv_df)
sns.stripplot(x='model_name', y='accuracy', data=cv_df, size=8, jitter=True, edgecolor="gray", linewidth=2)
plt.title('Model Accuracy Comparison')
plt.xlabel('Model Name')
plt.ylabel('Accuracy')
plt.show()

# Evaluate the best model on the test set
best_model = LogisticRegression(random_state=0)
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

# Perform predictions on sample sentences
sample_sentences = ["looking beautiful"]
text_features = tfidf_vectorizer.transform(sample_sentences)

predictions = best_model.predict(text_features)
for text, predicted in zip(sample_sentences, predictions):
    print(f'Input: "{text}"')
    print(f'Predicted Category: {id_to_category[predicted]}')
    print()
