import keras
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()

n_cols = 0  # change

target = []  # change
predictors = []  # change
data = []  # change

model.add(Dense(5, activation='relu', input_shape=(n_cols,)))
model.add(Dense(5, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(predictors, target)

prediction = model.predict(data)
