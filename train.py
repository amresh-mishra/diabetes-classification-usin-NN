from numpy import loadtxt 
from keras.models import Sequential 
from keras.layers import Dense 

dataset = loadtxt ('pima-indians-diabetes.csv', delimiter=',')
x=dataset[:,0:8]
y=dataset[:,8]
print (x)

model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu')) # 12 neuron on first and 8 on second layer
model.add(Dense(8, activation='relu')) 
model.add(Dense(1, activation='sigmoid'))  # last layer showing person is diabitic or not 

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x,y, epochs=50, batch_size= 20)

_, accuracy = model .evaluate(x, y) 
print ("Accuracy : % .2f" % (accuracy * 100))
 
model_json = model.to_json()
with open ("model.json", 'w') as json_file :
    json_file.write(model_json)
model.save_weights ('model.h5')
print("saved model to disk")
