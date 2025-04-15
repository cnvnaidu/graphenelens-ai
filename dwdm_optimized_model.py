import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

X_train = np.load("X_train_dwdm.npy")
y_train = np.load("y_train_dwdm.npy")

# Use MLP (because input is 1D now)
model = Sequential([
    Dense(512, activation='relu', input_shape=(500,)),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(4, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, validation_split=0.2, batch_size=32)
