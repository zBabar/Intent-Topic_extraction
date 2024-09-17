from sklearn.naive_bayes import MultinomialNB
import tensorflow as tf
import numpy as np
class Models:

    def get_simple_Classifier(self):
        return MultinomialNB()
    
    def get_simple_NN(self, output_size, vocab=5000):

        # Build the neural network model
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Embedding(input_dim=vocab, output_dim=128, input_length=20))
        model.add(tf.keras.layers.SpatialDropout1D(0.2))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(128, activation='relu'))
        model.add(tf.keras.layers.Dense(output_size, activation='softmax'))

        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        return model
    
    
