import pandas as pd
from keras.models import Sequential
from keras.layers import Dense

concrete_data = pd.read_csv('https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass'
                            '/DL0101EN/labs/data/concrete_data.csv')
concrete_data.head()

model = Sequential()

predictors = concrete_data[concrete_data.columns[concrete_data.columns != 'Strength']]
target = concrete_data['Strength']
predictors_z = (predictors - predictors.mean())/predictors.std()
n_cols = predictors_z.shape[1]

model.add(Dense(50, activation='relu', input_shape=(n_cols,)))
model.add(Dense(50, activation='relu'))
model.add(Dense(1, input_shape=(1,)))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(predictors_z, target, validation_split=0.3, epochs=100, verbose=1)

extracted_row = concrete_data.iloc[:2]
predict_for = extracted_row[extracted_row.columns[extracted_row.columns != 'Strength']]
predict_for_z = (predict_for - predictors.mean())/predictors.std()
predictions = model.predict(predict_for_z)

print(predictions)
