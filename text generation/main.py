import re
import numpy as np
from keras.utils import np_utils
from keras.preprocessing.text import Tokenizer
from keras.layers import Dense, LSTM,Input,Embedding,Dropout
from keras.models import Model, load_model


file_name = "./dataset/book_2.txt"

with open(file_name, encoding="utf-8-sig") as f:
	text = f.read()


seq_length = 20
start_story = '|'*seq_length



#clean up

text = text.lower()
text = start_story + text
text = text.replace("\n\n\n\n\n",start_story)
text = text.replace("\n"," ")
text = re.sub(" +",". ",text).strip()
text = text.replace("..",".")
text = re.sub('([!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~])', r' \1 ', text)
text = re.sub('\\s{2}',' ',text)


#tokenization

tokenizer = Tokenizer(char_level=False, filters='')
tokenizer.fit_on_texts([text])
total_words = len(tokenizer.word_index) + 1
token_list = tokenizer.texts_to_sequences([text])[0]


def generate_sequences(token_list, step):
	X = []
	y = []


	for i in range(0, len(token_list) - seq_length, step):

		X.append(token_list[i:i + seq_length])
		y.append(token_list[i + seq_length])


	y = np_utils.to_categorical(y, num_classes = total_words)

	num_seq = len(X)
	print(f"Number of sequences : {num_seq}\n")


	return X,y,num_seq


step = 1
seq_length = 20

X, y, num_seq = generate_sequences(token_list, step)

X = np.array(X)
y = np.array(y)



n_units = 128
embedding_size = 100


#text_in = Input(shape=(None,))

#x = Embedding(total_words, embedding_size)(text_in)
#x = LSTM(n_units, return_sequences=True)(x)
#x = LSTM(n_units)(x)
#x = Dropout(0.2)(x)

#text_out = Dense(total_words, activation = "softmax")(x)

#model = Model(text_in, text_out)

#model.compile(loss="categorical_crossentropy",optimizer="adam", metrics=['accuracy'])
#model.summary()

#epochs = 100

#batch_size = 32



#model.fit(X,y, epochs=epochs, shuffle=True)

#model.save("create_text.model")

model = load_model("create_text.model")



#model usage

def sample_with_temp(preds, temperature=1.0):

	#helper function to sample a index from probability array

	preds = np.asarray(preds).astype('float64')
	pred = np.log(preds) / temperature
	exp_preds = np.exp(preds)

	preds = exp_preds / np.sum(exp_preds)
	probs = np.random.multinomial(1,preds,1)

	return np.argmax(probs)


def generate_text(seed_text, next_words, model,max_sequence_len,temp):

	output_text = seed_text
	seed_text = start_story + seed_text
#token_list
	for _ in range(next_words):

		token_list = tokenizer.texts_to_sequences([seed_text])[0]
		token_list = token_list[-max_sequence_len:]
		token_list = np.reshape(token_list, (1, max_sequence_len))


		probs = model.predict(token_list, verbose=0)[0]
		y_class = sample_with_temp(probs, temperature=temp)

		output_word = tokenizer.index_word[y_class] if y_class > 0 else ' '

		if output_word == "|":
			break

		seed_text += output_word + ' '
		output_text += output_word + ' '

	return output_text


seed = "baby i need to know  "

created_text = generate_text(seed,10,model, 3,1)

print(created_text)









