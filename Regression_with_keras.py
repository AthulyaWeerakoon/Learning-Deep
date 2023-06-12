import keras as ker
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()

n_cols = 1
predictors = [0, 10, 20, 30, 40, 50, 60, 70, 80]
target = [0, 0.14, 0.18, 0.26, 0.3, 0.5, 0.56, 0.64, 0.78]

model.add(Dense(4, activation='relu', input_shape=(1,)))
model.add(Dense(4, activation='relu'))
model.add(Dense(1))

model.compile(optimizer='adan', loss='mean_squared_error')
model.fit(predictors, target)

predictions = model.predict(90)

print(predictions)
