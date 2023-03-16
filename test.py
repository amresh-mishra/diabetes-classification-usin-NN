import numpy as np
from numpy import loadtxt
from keras.models import model_from_json

dataset= loadtxt('pima-indians-diabetes.csv', delimiter=',')
x=dataset[ :,0:8]
y=dataset[:,8]
json_file= open('model.json', 'r')
loaded_model_json= json_file.read()

model =model_from_json(loaded_model_json)
model.load_weights('model.h5')
print('loaded model from disk ')

pridictions =np.argmax(model.predict(x), axis=-1)


for i in range (5, 10):
    print('%s => %d (Original Class: %d)' % (x[i].tolist(), pridictions[i], y[i]))
