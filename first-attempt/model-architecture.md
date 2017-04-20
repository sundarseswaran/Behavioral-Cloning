ubuntu@ip-172-31-46-176:~/code$ python three.py
Using TensorFlow backend.

_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
cropping2d_1 (Cropping2D)    (None, 100, 220, 3)       0
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 50, 110, 24)       1824
_________________________________________________________________
activation_1 (Activation)    (None, 50, 110, 24)       0
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 49, 109, 24)       0
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 25, 55, 36)        21636
_________________________________________________________________
activation_2 (Activation)    (None, 25, 55, 36)        0
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 24, 54, 36)        0
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 12, 27, 48)        43248
_________________________________________________________________
activation_3 (Activation)    (None, 12, 27, 48)        0
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 11, 26, 48)        0
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 11, 26, 64)        27712
_________________________________________________________________
activation_4 (Activation)    (None, 11, 26, 64)        0
_________________________________________________________________
max_pooling2d_4 (MaxPooling2 (None, 10, 25, 64)        0
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 10, 25, 64)        36928
_________________________________________________________________
activation_5 (Activation)    (None, 10, 25, 64)        0
_________________________________________________________________
max_pooling2d_5 (MaxPooling2 (None, 9, 24, 64)         0
_________________________________________________________________
flatten_1 (Flatten)          (None, 13824)             0
_________________________________________________________________
dense_1 (Dense)              (None, 1164)              16092300
_________________________________________________________________
activation_6 (Activation)    (None, 1164)              0
_________________________________________________________________
dense_2 (Dense)              (None, 100)               116500
_________________________________________________________________
activation_7 (Activation)    (None, 100)               0
_________________________________________________________________
dense_3 (Dense)              (None, 50)                5050
_________________________________________________________________
activation_8 (Activation)    (None, 50)                0
_________________________________________________________________
dense_4 (Dense)              (None, 10)                510
_________________________________________________________________
activation_9 (Activation)    (None, 10)                0
_________________________________________________________________
dense_5 (Dense)              (None, 1)                 11
=================================================================
Total params: 16,345,719
Trainable params: 16,345,719
Non-trainable params: 0
_________________________________________________________________
Train on 4204 samples, validate on 1051 samples
Epoch 1/10
W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE3 instructions, but these are available on your machine and could speed up CPU computations.
W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.1 instructions, but these are available on your machine and could speed up CPU computations.
W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.2 instructions, but these are available on your machine and could speed up CPU computations.
W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX instructions, but these are available on your machine and could speed up CPU computations.
4204/4204 [==============================] - 74s - loss: 14.9931 - val_loss: 0.0283
Epoch 2/10
4204/4204 [==============================] - 74s - loss: 0.0180 - val_loss: 0.0256
Epoch 3/10
4204/4204 [==============================] - 74s - loss: 0.0181 - val_loss: 0.0306
Epoch 4/10
4204/4204 [==============================] - 74s - loss: 0.0169 - val_loss: 0.0268
Epoch 5/10
4204/4204 [==============================] - 74s - loss: 0.0162 - val_loss: 0.0269
Epoch 6/10
4204/4204 [==============================] - 77s - loss: 0.0156 - val_loss: 0.0259
Epoch 7/10
4204/4204 [==============================] - 80s - loss: 0.0152 - val_loss: 0.0281
Epoch 8/10
4204/4204 [==============================] - 77s - loss: 0.0152 - val_loss: 0.0268
Epoch 9/10
4204/4204 [==============================] - 78s - loss: 0.0140 - val_loss: 0.0273
Epoch 10/10
4204/4204 [==============================] - 77s - loss: 0.0144 - val_loss: 0.0267
