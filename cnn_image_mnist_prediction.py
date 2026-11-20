# import lib 
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D,Input
(x_train,y_train),(x_test,y_test) = mnist.load_data()

x_train = x_train/255.0 
x_test = x_test/255.0
model = Sequential() 
model.add(Input(shape=(28,28,1)))
model.add(Conv2D(1,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(32,activation='relu'))
model.add(Dense(10,activation='softmax'))

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

model.fit(x_train,y_train,epochs=5)
loss,accuracy = model.evaluate(x_test,y_test)
print(loss,accuracy)

pred = model.predict(x_test[:3])

proba = list(pred[0])
print(max(proba))

print(proba.index(max(proba)))