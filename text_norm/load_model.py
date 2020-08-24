from numpy import loadtxt
from keras.models import load_model

#load saved model
model = load_model('saved_model.hdf5')

#summarize model
model.summary()



