#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import pandas as pd
import gc
from nltk import FreqDist
import time
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, load_model
from keras.layers import Activation, TimeDistributed, Dense, RepeatVector, Embedding, Bidirectional
from keras.optimizers import RMSprop
from keras.layers.recurrent import LSTM
from utils import *
from matplotlib import pyplot as plt
from attenton_decoder import *
# In[6]:


input_vocab_size = 250
target_vocab_size = 346    # 1000 in full dataset
# target_vocab_size = 1000
num_samples = 500000
context_size = 3
padding_entity = [0]
self_sil_retention_percent = 0.5
X_seq_len = 60
y_seq_len = 20
hidden = 256
layers = 2
epochs = 5    # used just 1 for full dataset
batch_size = 128
val_split = 0.1
learning_rate = 0.1


# ## LSTM encoder/decoder for text normalization
# 
# Given a large corpus of written text aligned to its normalized spoken form, I will train an RNN to learn the correct normalization function. My model is a encoder-decoder with layers of bidirectional LSTMs.
# 
# Here is a quick summary of my model approach and architecture:
# 
# + Character level input sequence, and word level output sequence
# + Input character vocabulary of 250 distinct characters. Output word vocabulary of 346 (the number of unique) distinct words.
# + Add a context window of 3 words to the left and right with a distinctive tag to separately identity the key token. I do this to manage the input sequence length reasonably.
# + Input sequence zero padding to a maximum of length 60. Output sequence padding to a maximum of length 20. I do this to create fixed-length sequences.
# + Model architecture with two components: an encoder and a decoder.
#   - 256 hidden units in each layer of the encoder and decoder
#   - Three bidirectional LSTM layers in the encoder
#   - Two LSTM layers in the decoder

# ### 1. Create the encoder/decoder model

# In[3]:


# let's make the model
model = Sequential()

# creating encoder network
model.add(Embedding(input_vocab_size+2, hidden, input_length=X_seq_len, mask_zero=True))
print('Embedding layer created')
model.add(Bidirectional(LSTM(hidden, return_sequences = True), merge_mode = 'concat'))
model.add(Bidirectional(LSTM(hidden, return_sequences = True), merge_mode = 'concat'))
model.add(AttentionDecoder(hidden, return_sequences=True, output_dim=512))
model.add(Bidirectional(LSTM(hidden), merge_mode = 'concat'))

model.add(RepeatVector(y_seq_len))
print('Encoder layer created')

# creating decoder network
for _ in range(layers):
    model.add(LSTM(hidden, return_sequences=True))
model.add(TimeDistributed(Dense(target_vocab_size+1)))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
print('Decoder layer created')

# checking the model summary
model.summary()


# ### 2. Load the training data and do a quick exploratory analysis

# In[13]:


start = time.time()

# load training data
X_train_data = pd.read_csv("en_train.csv")
X_train_data['before'] = X_train_data['before'].apply(str)
X_train_data['after'] = X_train_data['after'].apply(str)

print('Training data loaded in {0} s.'.format(time.time()-start))

print(X_train_data.shape)
X_train_data = X_train_data.iloc[:num_samples]
print(X_train_data.shape)
X_train_data.head()


# In[15]:


import seaborn as sns

# let's see how many training tokens fall into each class by
# plotting a count graph for the "Class" column of the data
fig = plt.figure(figsize=(16,6))
sns.countplot(x='class',data = X_train_data)


# In[16]:


# the counts for number of sample tokens in each class group
X_train_data['class'].value_counts().sort_values(ascending = False)


# ### 3. Create the input and target vocabularies
# 
# These are the vocabularies that the model can draw input from and decode output to.

# In[5]:


start = time.time()

# create vocabularies
# target vocab
y = list(np.where(X_train_data['class'] == "PUNCT", "sil.",
      np.where(X_train_data['before'] == X_train_data['after'], "<self>",
               X_train_data['after'])))

y = [token.split() for token in y]
dist = FreqDist(np.hstack(y))
temp = dist.most_common(target_vocab_size-1)
temp = [word[0] for word in temp]
temp.insert(0, 'ZERO')
temp.append('UNK')

target_vocab = {word:ix for ix, word in enumerate(temp)}
print(len(target_vocab))
target_vocab_reversed = {ix:word for word,ix in target_vocab.items()}

# input vocab
X = list(X_train_data['before'])
X = [list(token) for token in X]

dist = FreqDist(np.hstack(X))
temp = dist.most_common(input_vocab_size-1)
temp = [char[0] for char in temp]
temp.insert(0, 'ZERO')
temp.append('<norm>')
temp.append('UNK')

input_vocab = {char:ix for ix, char in enumerate(temp)}

gc.collect()

print('Vocab created in {0} s.'.format(time.time()-start))


# In[6]:


start = time.time()

# Converting input and target tokens to index values
X = index(X, input_vocab)
y = index(y, target_vocab)

gc.collect()

print('Replaced tokens with integers in {0} s.'.format(time.time()-start))


# ### 4. Data preprocessing
# 
# Add the context window (3 tokens previous and 3 tokens in the future) for each token of the training data. Pad these sequences to make each input sequence have a fixed length, and convert to an

# In[7]:


start = time.time()

# adding a context window of 3 words in input, with token separated by <norm>
X = add_context_window(X, context_size, padding_entity, input_vocab)

print('Added context window to X in {0} s.'.format(time.time()-start))


# In[8]:


start = time.time()

# padding
X = padding_batchwise(X, X_seq_len)
y = padding_batchwise(y, y_seq_len)

# convert to integer array, batch-wise (converting full data to array at once takes a lot of time)
X = np.array(X)
y = np.array(y)
y_sequences = np.asarray(sequences(y, y_seq_len, target_vocab))

print('Added padding and converted to np array in {0} s.'.format(time.time()-start))


# ### 5. Training the LSTM encoder/decoder
# 
# I'll train 500,000 sample tokens for 5 epochs, with both model checkpointing (saving the model) after each epoch and early stop checking (stop training early if the validation accuracy gets worse- this means the model overfit!).

# In[12]:


start = time.time()
from keras.callbacks import ModelCheckpoint, EarlyStopping

# fitting the model on the validation data with batch size set to 128 for a total of 5 epochs:
print('Fitting model...')
checkpointer = ModelCheckpoint(filepath='saved_model.hdf5', verbose=1, save_best_only=True)
earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=1, mode='auto')
callbacks_list = [checkpointer, earlystop]

history = model.fit(X, y_sequences, batch_size=batch_size, epochs=epochs, validation_split=val_split, callbacks=callbacks_list, verbose=1)


# In[14]:


from matplotlib import pyplot as plt 

# let's double check that the model didn't overfit by comparing the training and validation loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model train vs validation loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()


# ### 6a. Predicting text normalization for test data
# 
# I'll use my trained model to predict normalization for the full (10,000 samples) test set.

# In[3]:


# load weights from training
model = load_model('saved_model.hdf5')


# In[9]:


# prepare test data in the right format
X_test_data = pd.read_csv("en_test.csv")
X_test_data['before'] = X_test_data['before'].apply(str)
print(X_test_data.shape)
X_test_data = X_test_data[:100000]
print(X_test_data.shape)
X_test_data.head(10)


# In[55]:


X_test = list(X_test_data['before'])
X_test = [list(token) for token in X_test]

X_test = index(X_test, input_vocab) # Convert to integer index
X_test = add_context_window(X_test, context_size, padding_entity, input_vocab) # Add context window
X_test = batch_wise_padding(X_test, X_seq_len) # Padding

# convert X_test to integer array, batch-wise (converting full data to array at once takes a lot of time)
# X_test = array_batchwise(X_test, X_seq_len)


# In[56]:


# make predictions
test_predictions = np.argmax(model.predict(np.asarray(X_test), batch_size = 64, verbose=1), axis=2)

predicted_test_sequences = []
for prediction in test_predictions:
    sequence = ' '.join([target_vocab_reversed[index] for index in prediction if index > 0])
    predicted_test_sequences.append(sequence)
np.savetxt('test_predictions', predicted_test_sequences, fmt='%s')


# ### 6b. Investigating the model's predictions: what went wrong?
# 
# I'll focus on the "interesting" cases - where the model didn't predict <self\> or sil. These cass include dates, measures, money, and other cases that are not already trivially normalized. Per the concern I bring up in my write-up analysis that my model is only memorizing normalizations, I will also look at novel test cases that don't appear in my training data.

# In[17]:


with open('test_predictions', 'r') as f:
    predicted_test_sequences = f.readlines()
    # need to get rid of "\n" from loading
    predicted_test_sequences = [pred.rstrip('\n') for pred in predicted_test_sequences]
    
pred = pd.Series(predicted_test_sequences)

X_test_data['after'] = pred.values

X_test_data.head(10)


# In[18]:


# boring_cases = ['<self>', 'sil.']
# let's look at the interesting cases in the predicted for test data
interesting_cases = X_test_data[X_test_data['after'] != '<self>']
interesting_cases = interesting_cases[interesting_cases['after'] != 'sil.']
interesting_cases.head(30)


# In[22]:


# now let's look at the novel test cases- those that didn't appear in the training data
common = X_train_data.merge(X_test_data,on=['before'])
novel_cases = X_test_data[(~X_test_data.before.isin(common.before))]
print('There are {0} novel cases in the test set.'.format(novel_cases.shape[0]))
novel_cases.head(30)


# In[25]:


# and now novel important cases
# I want to see if my model generalizes or is just memorizing the training data
novel_important_cases = novel_cases[novel_cases['after'] != '<self>']
novel_important_cases = novel_important_cases[novel_important_cases['after'] != 'sil.']
print('There are {0} novel important cases in the test set.'.format(novel_important_cases.shape[0]))
novel_important_cases.head(30)


# In[ ]:




