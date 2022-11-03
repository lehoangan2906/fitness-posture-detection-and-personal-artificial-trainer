import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt

# Function
def load_data(pushup,label,X,y):
    data = pushup.iloc[:,1:].values
    n_sample = len(data)
    for i in range(no_of_timesteps,n_sample):
        X.append(data[i-no_of_timesteps:i,:])
        y.append(label)

# Initialization
no_of_timesteps = 70
X_train,y_train = list(),list()
X_test,y_test = list(),list()

# Load dataset

# Squat
squat_chum_chan_1 = pd.read_csv('Data TXT/Squat/squat_chum_chan_1.txt')
squat_chum_chan_2 = pd.read_csv('Data TXT/Squat/squat_chum_chan_2.txt')
squat_chum_chan_3 = pd.read_csv('Data TXT/Squat/squat_chum_chan_3.txt')
squat_cong_lung_1 = pd.read_csv('Data TXT/Squat/squat_cong_lung_1.txt')
squat_cong_lung_2 = pd.read_csv('Data TXT/Squat/squat_cong_lung_2.txt')
squat_cong_lung_3 = pd.read_csv('Data TXT/Squat/squat_cong_lung_3.txt')
squat_dung_1 = pd.read_csv('Data TXT/Squat/squat_dung_1.txt')
squat_dung_2 = pd.read_csv('Data TXT/Squat/squat_dung_2.txt')
squat_dung_3 = pd.read_csv('Data TXT/Squat/squat_dung_3.txt')

# Push Up
pushup_dung_left_1 = pd.read_csv('Data TXT/Push Up/pushup_dung_left_1.txt')
pushup_dung_left_2 = pd.read_csv('Data TXT/Push Up/pushup_dung_left_2.txt')
pushup_dung_left_3 = pd.read_csv('Data TXT/Push Up/pushup_dung_left_3.txt')
pushup_cong_lung_left_1 = pd.read_csv('Data TXT/Push Up/pushup_cong_lung_left_1.txt')
pushup_cong_lung_left_2 = pd.read_csv('Data TXT/Push Up/pushup_cong_lung_left_2.txt')
pushup_cong_lung_left_3 = pd.read_csv('Data TXT/Push Up/pushup_cong_lung_left_3.txt')
pushup_vong_lung_left_1 = pd.read_csv('Data TXT/Push Up/pushup_vong_lung_left_1.txt')
pushup_vong_lung_left_2 = pd.read_csv('Data TXT/Push Up/pushup_vong_lung_left_2.txt')
pushup_vong_lung_left_3 = pd.read_csv('Data TXT/Push Up/pushup_vong_lung_left_3.txt')

# Sit Up
pushup_dung_left_1 = pd.read_csv('Data TXT/Sit Up/gap_bung_dung_left_1.txt')
pushup_dung_left_2 = pd.read_csv('Data TXT/Sit Up/gap_bung_dung_left_2.txt')
pushup_dung_left_3 = pd.read_csv('Data TXT/Sit Up/gap_bung_dung_left_3.txt')
pushup_cong_lung_left_1 = pd.read_csv('Data TXT/Sit Up/gap_bung_thap_left_1.txt')
pushup_cong_lung_left_2 = pd.read_csv('Data TXT/Sit Up/gap_bung_thap_left_2.txt')
pushup_cong_lung_left_3 = pd.read_csv('Data TXT/Sit Up/gap_bung_thap_left_3.txt')
pushup_vong_lung_left_1 = pd.read_csv('Data TXT/Sit Up/gap_bung_vung_tay_left_1.txt')
pushup_vong_lung_left_2 = pd.read_csv('Data TXT/Sit Up/gap_bung_vung_tay_left_2.txt')
pushup_vong_lung_left_3 = pd.read_csv('Data TXT/Sit Up/gap_bung_vung_tay_left_3.txt')

load_data(squat_chum_chan_1,0,X_train,y_train)
load_data(squat_chum_chan_2,0,X_train,y_train)
load_data(squat_chum_chan_3,0,X_test,y_test)
load_data(squat_cong_lung_1,1,X_train,y_train)
load_data(squat_cong_lung_2,1,X_train,y_train)
load_data(squat_cong_lung_3,1,X_test,y_test)
load_data(squat_dung_1,2,X_train,y_train)
load_data(squat_dung_2,2,X_train,y_train)
load_data(squat_dung_3,2,X_test,y_test)

X_train,y_train = np.array(X_train),np.array(y_train)
X_test,y_test = np.array(X_test),np.array(y_test)
y_train = to_categorical(y_train, num_classes=3)
y_test = to_categorical(y_test, num_classes=3)

# X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

# Build model
model = Sequential()

model.add(LSTM(units=50,return_sequences=True,input_shape = (X_train.shape[1],X_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units=50,return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50,return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=3,activation="softmax"))
model.compile(optimizer="rmsprop",metrics=["accuracy"],loss="categorical_crossentropy")

# Train model
history = model.fit(X_train,y_train,epochs = 50,batch_size = 16,validation_data=(X_test,y_test))
history_frame = pd.DataFrame(history.history)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Squat Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Squat Loss')
plt.ylabel('Model loss')
plt.xlabel('Epochs')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

# Save model
model.save("model_squat.h5")

