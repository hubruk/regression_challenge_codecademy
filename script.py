from xml.sax.xmlreader import InputSource
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow	import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

import uuid

df = pd.read_csv('D:/GitHub/regression-challenge/regression-challenge/regression-challenge/regression-challenge-starter/admissions_data.csv')
#print(df.head(25))
#print(df.describe())
#print(df.dtypes)
labels = df.iloc[:,-1]
features = df.iloc[:, 1:-1]
#print(features.head())

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.33, random_state=42)

scale = StandardScaler()
X_train_scale = scale.fit_transform(X_train)
X_test_scale = scale.fit_transform(X_test)

_, m = X_train.shape
###
school_model = Sequential()
school_model.add(keras.Input(shape=(m,)))
school_model.add(layers.Dense(32, activation='relu'))
school_model.add(layers.Dropout(0.1))
school_model.add(layers.Dense(32, activation='relu'))
school_model.add(layers.Dropout(0.2))
school_model.add(layers.Dense(1))
#school_model.summary()

opt = keras.optimizers.Adam(learning_rate=0.01)
school_model.compile(optimizer=opt, loss='mse', metrics=['mae'])

stop = EarlyStopping(monitor='loss', mode='min', patience=10)
history = school_model.fit(x=X_train_scale, y=y_train, batch_size=5, epochs=100, validation_split=0.25, verbose=1, callbacks=[stop])

val_mse, val_mae = school_model.evaluate(x=X_test_scale, y=y_test, verbose = 0)
print('MSE val:',val_mse,"MAE val:", val_mae)

# How well the features in regression model make predictions. 
# An R-squared value near close to 1 suggests a well-fit regression model, while a value closer to 0 suggests that the regression model does not fit the data well.
y_pred = school_model.predict(X_test_scale)
print(r2_score(y_test,y_pred))

print(history.history.keys())
fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1) #2 wiersze, 1 kolumna, wykres 1
ax1.plot(history.history['mae'])
ax1.plot(history.history['val_mae'])
ax1.set_title('model mae')
ax1.set_ylabel('MAE')
ax1.set_xlabel('epoch')
ax1.legend(['train', 'validation'], loc='upper left')

ax2 = fig.add_subplot(2, 1, 2) #2 wiersze, 1 kolumna, wykres 2
ax2.plot(history.history['loss'])
ax2.plot(history.history['val_loss'])
ax2.set_title('model loss')
ax2.set_ylabel('loss')
ax2.set_xlabel('epoch')
ax2.legend(['train', 'validation'], loc='upper left')
fig.savefig('images\plot_{}.png'.format(str(uuid.uuid4())))
plt.show()

