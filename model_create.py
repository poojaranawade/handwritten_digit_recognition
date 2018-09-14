from keras.utils import np_utils
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import BatchNormalization
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import Model
from keras.layers.merge import concatenate
from keras.utils import plot_model
from keras.datasets import mnist
import calendar, time

# class to create CNN model with given architecture
class model_create:
    # add generic convolution layer with given parameters 
    # along with 'ReLu' activation and Batch normalization
    def _conv_layer(self, prev_layer, nb_filter, num_row, num_col,
                    padding='valid', strides=(1, 1)):
        x = Conv2D(nb_filter, (num_row, num_col), strides=strides,
                   padding=padding, activation='relu')(prev_layer)
        x = BatchNormalization(axis=-1)(x)
        return x

    # save model arciture in a local jpg file
    def _give_model_chart(self, model):
        filename = 'model_current' + str(calendar.timegm(time.gmtime())) + '.jpg'
        print('\nModel image saved in',filename)
        plot_model(model, to_file=filename, show_shapes=True,
                   show_layer_names=True)

    # create and compile the CNN model
    def _get_model(self):
        inputs = Input(shape=(28, 28, 1))

        conv1 = self._conv_layer(inputs, 32, 3, 3)

        conv1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2_1 = self._conv_layer(conv1, 64, 3, 3)
        conv2_1 = MaxPooling2D(pool_size=(2, 2))(conv2_1)

        conv2_2 = self._conv_layer(conv1, 64, 3, 3)
        conv2_2 = MaxPooling2D(pool_size=(2, 2))(conv2_2)

        conv3_1 = self._conv_layer(conv2_1, 256, 3, 3)
        conv3_2 = self._conv_layer(conv2_2, 256, 3, 3)

        conv_out = concatenate([conv3_1, conv3_2], axis=-1)

        conv_out = Flatten()(conv_out)

        full_1 = Dense(units=1000, activation='relu')(conv_out)
        full_2 = Dense(units=500, activation='relu')(full_1)

        outputs = Dense(units=10, activation='softmax')(full_2)

        model = Model(inputs=[inputs], outputs=[outputs])
        
        model.compile(optimizer='adam', loss='categorical_crossentropy',
                      metrics=['accuracy'])

        self._give_model_chart(model)
        return model

    def train_model(self):
        # reading data and labels from MNIST dataset
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        
        # reshape to be [num_of_samples][num_pixels][width][height]
        X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
        X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')
        
        # normalize inputs from 0-255 to 0-1
        X_train = X_train / 255
        X_test = X_test / 255
        
        # one hot encode outputs
        y_train = np_utils.to_categorical(y_train)
        y_test = np_utils.to_categorical(y_test)
        
        # build model
        model = self._get_model()
        
        # training the model for 5 epochs and with batch size 512
        model.fit(epochs=5, x=X_train, y=y_train, batch_size=512,
                  validation_data=(X_test, y_test))
        
        # evaluate model performance
        scores = model.evaluate(X_test, y_test, verbose=0)
        print('\nModel Acuuracy: ', (scores[1] * 100))

        model.save('my_model.h5')
        print('\nTrained model saved to file name my_model.h5')
        print('To load use model = load_model(\'my_model.h5\')')
