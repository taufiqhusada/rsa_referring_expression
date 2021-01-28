import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import numpy as np
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
import re
from keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Embedding

train_dat = np.load('train_label_tokenized.npy', allow_pickle=True)
test_dat = np.load('test_label_tokenized.npy', allow_pickle=True)

combined_data = np.load('combined_label_tokenized.npy', allow_pickle=True)
combined_data = list(combined_data)

vocab = set()
for sent in combined_data:
    for w in sent:
        vocab.add(w.lower())
vocab.add('')

vocab_size = len(vocab)

train_dat = list(train_dat)
train_data = []
for data_point in train_dat:
    d_point = [w.lower() for w in data_point[::-1]]
    train_data.append(d_point)
test_dat = list(test_dat)
test_data = []
for data_point in test_dat:
    d_point = [w.lower() for w in data_point[::-1]]
    test_data.append(d_point)

processed_data = []
for data_point in combined_data:
    processed_data_point = [w.lower() for w in data_point[::-1]]
    processed_data.append(processed_data_point)

three_gram_train = []
for sentence in train_data:
    sentence.insert(0, '')
    sentence.insert(0, '')
    sentence.append('')
    for i in range(len(sentence)-3):
        three_gram_train.append([sentence[i:i+3], sentence[i+3]])
three_gram_test = []
for sentence in test_data:
    sentence.insert(0, '')
    sentence.insert(0, '')
    sentence.append('')
    for i in range(len(sentence)-3):
        three_gram_test.append([sentence[i:i+3], sentence[i+3]])

word_to_idx = {}
for sentence in processed_data:
    for word in sentence:
        if word.lower() not in word_to_idx:
            word_to_idx[word] = len(word_to_idx)
word_to_idx[''] = len(word_to_idx)

three_gram_train_inputs = []
three_gram_train_outputs = []
for sent, next_word in three_gram_train:
    sentence_in = np.array([word_to_idx[w] for w in sent])
    three_gram_train_inputs.append(sentence_in)
    next_word_out = np.array([word_to_idx[next_word]])
    three_gram_train_outputs.append(next_word_out)
three_gram_train_inputs = np.array(three_gram_train_inputs)
three_gram_train_outputs = to_categorical(three_gram_train_outputs, num_classes=vocab_size)


sequence_length = 3
model = Sequential()
model.add(Embedding(vocab_size, sequence_length, input_length=sequence_length))
model.add(LSTM(50,return_sequences=True))
model.add(LSTM(50))
model.add(Dense(50,activation='relu'))
model.add(Dense(vocab_size, activation='softmax'))
print(model.summary())
# compile network
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(three_gram_train_inputs,three_gram_train_outputs,epochs=500,verbose=1)
model.save("mymodel.h5")