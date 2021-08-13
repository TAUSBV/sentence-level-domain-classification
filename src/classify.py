import numpy as np
import tensorflow_hub as hub
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from utils import create_dataset


def main():

    # create a labeled data set from sentence-level data
    X, y = create_dataset()

    # convert string labels to a categorical representation
    le = LabelEncoder()
    y = to_categorical(le.fit_transform(y))

    # split data set into TRAIN, DEV, and TEST sets
    X_train, X_dev, y_train, y_dev = train_test_split(X, y, test_size=0.2, random_state=42)  # 20% for DEV and TEST
    # split DEV further into DEV and TEST
    X_dev, X_test, y_dev, y_test = train_test_split(X_dev, y_dev, test_size=0.5, random_state=42)

    # load embeddings model from Tensorflow Hub
    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

    # convert sentences to embeddings
    X_train = embed(X_train)
    X_dev = embed(X_dev)
    X_test = embed(X_test)

    # build Sequential model with 3 layers
    model = Sequential()
    model.add(Dense(units=32, activation="relu"))  # input layer
    model.add(Dense(units=64, activation="relu"))  # hidden layer
    model.add(Dense(units=5, activation="softmax"))  # output layer, no. of units equals no. of classes
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # fit (train) the model
    model.fit(x=X_train, y=y_train,
              epochs=10,
              validation_data=(X_dev, y_dev),
              batch_size=32,
              verbose=1)

    # evaluate the model
    predictions = np.argmax(model.predict(X_test), axis=-1)
    y_test = le.inverse_transform([np.argmax(y) for y in y_test])  # reconstruct original string labels
    predictions = le.inverse_transform(predictions)
    report = classification_report(y_test, predictions)
    print(report)

    pass


if __name__ == "__main__":

    main()
