import pandas as pd 
import numpy as np
import string
import re
from collections import Counter
from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras 
from keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Conv1D,Embedding,Dropout,GlobalMaxPooling1D, concatenate,Dense,TextVectorization

df = pd.read_csv("news_classification.csv")




def custom_standardization(input_data):
	lowercase = tf.strings.lower(input_data)
	stripped_html = tf.strings.regex_replace(lowercase, "<br />","")
	return tf.strings.regex_replace(
			stripped_html, f"[{re.escape(string.punctuation)}]",""
		)

#model constants
max_features = 20000
embedding_dim = 128
sequence_length = 35


vectorization_layer = TextVectorization(
	standardize = custom_standardization,
	max_tokens = max_features,
	output_mode = "int",
	output_sequence_length = sequence_length
	)



def remove_punc(text):

	text = str(text)
	translation = str.maketrans("","",string.punctuation)
	return text.translate(translation)

features = df['category'].unique()

print(len(features))
i = 0
nums = []
for cats in features:
	nums.append(i)
	i += 1

df['category'].replace(features,nums , inplace=True)


df['headline'] = df.headline.map(remove_punc)
#df['short_description'] = df.short_description.map(remove_punc)



def counter_words(text_col):

	count = Counter()
	for text in text_col.values:
		for word in text.split():
			count[word] += 1

	return count

counter_1 = counter_words(df.headline)


num_unique_words_1 = len(counter_1)




#train_size = int(df.shape[0] * 0.8)

x_ = df.drop('category',1)
y_ = df['category']


y_train_int = []


x_train, x_test, y_train,y_test = train_test_split(x_,y_, test_size = 0.2)


y = y_train.to_numpy()

for i in y:
	y_train_int.append([int(i)])

x_train_head = np.array(x_train['headline'])


y_train_ = np.array(y_train_int, dtype='int64')

#print(y_train_)


def tokenize_words(text, max_len, state):

	tokenizer = Tokenizer(char_level=False, filters='')
	tokenizer  = Tokenizer(num_words=max_len)
	tokenizer.fit_on_texts(text)

	word_index = tokenizer.word_index

	train_seq = tokenizer.texts_to_sequences(text)

	if state == 1:
		return train_seq, word_index

	else:
		return train_seq


def vectorize_text(text):
	text = tf.expand_dims(text, -1)
	return vectorization_layer(text)



def reverse_words(word_index, sequences):

	#reversing the indices

	reverse_word_index = dict([(idx,word) for (word,idx) in word_index.items()])

	return " ".join([reverse_word_index.get(idx,"?") for idx in sequences])



train_head_seq = tokenize_words(x_train_head,num_unique_words_1, 0)

max_len = 35

train_head_padded = pad_sequences(train_head_seq, max_len,padding='post',truncating='post')


model = Sequential()
model.add(keras.Input(shape=[None], dtype='int64'))
model.add(Embedding(num_unique_words_1, 32, input_length=max_len))
model.add(Dropout(0.5))
model.add(Conv1D(128, 3, padding="valid", activation='relu', strides=3))
model.add(Conv1D(128, 3, padding="valid", activation='relu', strides=3))
model.add(Conv1D(128, 3, padding="valid", activation='relu', strides=3))
model.add(Dense(100, activation='relu'))
model.add(Dense(41, activation='softmax'))
model.add(GlobalMaxPooling1D())



model.compile(loss="sparse_categorical_crossentropy", optimizer='adam', metrics=['accuracy'])

model.fit(x=train_head_padded, y = y_train_, epochs=50, validation_split=0.2 , batch_size=32)

model.save("new_classification_1.h5")



saved_model = keras.models.load_model('new_classification_1.h5')

def predict_class(text, model):

	tokenizer = Tokenizer(char_level=False, filters='')
	token_list = tokenizer.texts_to_sequences([text])[0]
	tokenizer  = Tokenizer(num_words=max_len)
	tokenizer.fit_on_texts(text)

	#word_index = tokenizer.word_index

	train_seq = tokenizer.texts_to_sequences(text)
	print(train_seq)
	train_seq = pad_sequences(train_seq, max_len,padding='post',truncating='post')

	#print(train_seq)
	probs = model.predict(train_seq, verbose=0)[0]

	#print(probs)

predict_class("the war was almost over", saved_model)
