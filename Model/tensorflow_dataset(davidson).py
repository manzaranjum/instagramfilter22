# Import necessary libraries
from keras.preprocessing.text import text_to_word_sequence
import pandas as pd
from keras.preprocessing.text import Tokenizer
import numpy as np
import re
import nltk
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from keras.preprocessing import sequence
from keras.models import Sequential
from nltk.stem import WordNetLemmatizer
from keras.layers import LSTM, SpatialDropout1D, Dense, Dropout, Activation, Embedding
from keras.utils.np_utils import to_categorical

# Read the input dataset
train_data = pd.read_csv("dataset (davidson).csv")

# Rename columns for clarity
train_data.rename(columns={'tweet': 'text', 'class': 'category'}, inplace=True)

# Create a mapping for category labels
category_mapping = {0: 'hate_speech', 1: 'offensive_language', 2: 'neither'}
train_data['category'] = train_data['category'].map(category_mapping)

# Perform lemmatization on text data
train_data['text_lem'] = [''.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]',' ',text)) for text in lis]) for lis in train_data['text']]

# Create category_id for encoding categories
train_data['category_id'] = train_data['category'].factorize()[0]

# Label Encoding categorical data for the classification category
label_encoder = LabelEncoder()
train_data['label'] = label_encoder.fit_transform(train_data['label'])

# Encode the text with word sequences - Preprocessing step 1
tk = Tokenizer(num_words=200, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=" ")
tk.fit_on_texts(train_data['text_lem'])
x = tk.texts_to_sequences(train_data['text_lem'])
x = sequence.pad_sequences(x, maxlen=200)

# Perform one hot encoding
y = to_categorical(train_data['label'], num_classes=3)

# Split data into training and testing sets
x_train, x_test, y_train, y_test, indices_train, indices_test = train_test_split(x, y, train_data.index, test_size=0.33, random_state=42)
x_train = sequence.pad_sequences(x_train, maxlen=200)
x_test = sequence.pad_sequences(x_test, maxlen=200)

# Define model parameters
max_features = 24783
maxlen = 200
embedding_dims = 50

# Build the model
model = Sequential()
model.add(Embedding(max_features, embedding_dims, input_length=200))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(200, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
epochs = 10
batch_size = 64
history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test))

# Perform predictions on a sample sentence
new_complaint = ['you are not beautiful. you are a black woman trying to be white']
seq = tk.texts_to_sequences(new_complaint)
padded = sequence.pad_sequences(seq, maxlen=200)
pred = np.argmax(model.predict(padded), axis=-1)

print(f'Predicted Label: {pred}')

# Save the model
model.save('my_model.h5')
