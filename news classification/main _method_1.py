import pandas as pd 
import numpy as np
import string
from collections import Counter
from sklearn.model_selection import train_test_split
import keras 
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Conv1D,Embedding,Dropout,GlobalMaxPooling1D, concatenate,Dense


df = pd.read_csv("news.csv")

def remove_punc(text):

	text = str(text)
	translation = str.maketrans("","",string.punctuation)
	return text.translate(translation)


#print(df.columns)
cols = ['category', 'headline', 'short_description']

features = list(set(df['category']))
i = 0
for cats in features:
	df.loc[df['category'] == cats, 'category'] = i

	i += 1

df['headline'] = df.headline.map(remove_punc)
df['short_description'] = df.short_description.map(remove_punc)
df['category'] = df.category.map(remove_punc)



def counter_words(text_col):

	count = Counter()
	for text in text_col.values:
		for word in text.split():
			count[word] += 1

	return count

counter_1 = counter_words(df.headline)
counter_2 = counter_words(df.short_description)
counter_3 = counter_words(df.category)

num_unique_words_1 = len(counter_1)
num_unique_words_2 = len(counter_2)
num_unique_words_3 = len(counter_3)



#train_size = int(df.shape[0] * 0.8)

x_ = df.drop('category',1)
y_ = df['category']


y_train_int = []




x_train, x_test, y_train,y_test = train_test_split(x_,y_, test_size = 0.2)

y = y_train.to_numpy()
for i in y:
	y_train_int.append([int(i)])

x_train_head = np.array(x_train['headline'])
x_train_short = np.array(x_train['short_description'])

y_train_ = np.array(y_train_int, dtype='int64')


print(y_train_)

def tokenize_words(text, max_len, state):

	tokenizer  = Tokenizer(num_words=max_len)
	tokenizer.fit_on_texts(text)

	word_index = tokenizer.word_index

	train_seq = tokenizer.texts_to_sequences(text)

	if state == 1:
		return train_seq, word_index

	else:
		return train_seq



def reverse_words(word_index, sequences):

	#reversing the indices

	reverse_word_index = dict([(idx,word) for (word,idx) in word_index.items()])

	return " ".join([reverse_word_index.get(idx,"?") for idx in sequences])








train_head_seq = tokenize_words(x_train_head,num_unique_words_1, 0)
train_short_seq = tokenize_words(x_train_head,num_unique_words_2, 0)
#train_label_seq, word_index = tokenize_words(y_train,num_unique_words_3, 1)

max_len = 35

train_short_padded = pad_sequences(train_short_seq, max_len,padding='post',truncating='post')
train_head_padded = pad_sequences(train_head_seq, max_len,padding='post',truncating='post')
#train_label_padded = pad_sequences(train_label_seq, 3,padding='post',truncating='post')

print(y_train[10:25])

input_1 = keras.Input(shape=[None],dtype='int64')
x = Embedding(num_unique_words_1, 32, input_length=max_len)(input_1)
x = Dropout(0.5)(x)
x = Conv1D(64, 3, padding="valid", activation='relu', strides=3)(x)
x = Conv1D(64, 3, padding="valid", activation='relu', strides=3)(x)
x = Conv1D(64, 3, padding="valid", activation='relu', strides=3)(x)
x = GlobalMaxPooling1D()(x)
x = keras.Model(inputs=input_1, outputs=x)


input_2 = keras.Input(shape=[None], dtype='int64')
y = Embedding(num_unique_words_2, 32, input_length=max_len)(input_2)
y = Dropout(0.5)(y)
y = Conv1D(64, 3, padding="valid", activation='relu', strides=3)(y)
y = Conv1D(64, 3, padding="valid", activation='relu', strides=3)(y)
y = Conv1D(64, 3, padding="valid", activation='relu', strides=3)(y)
y = GlobalMaxPooling1D()(y)
y = keras.Model(inputs=input_2, outputs=y)


combined = concatenate([x.output, y.output])

z = Dense(128, activation='relu')(combined)
z = Dense(64, activation='relu')(z)
z = Dense(41, activation='relu')(z)

model = keras.Model(inputs=[x.input, y.input], outputs=z)

model.compile(loss="sparse_categorical_crossentropy", optimizer='adam', metrics=['accuracy'])

model.fit(x=[train_head_padded, train_short_padded], y = y_train_, epochs=50, validation_split=0.2 , batch_size=32)

model.save("news_classification_2.h5")





#decode_text = reverse_words(word_index,train_label_seq[10])
#print(train_label_seq[17])
#print(decode_text)