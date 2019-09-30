import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from skimage import io
import matplotlib.pyplot as plt

#Las imagenes de entrenamiento de nminst viven en un espacio vectorial de dimension 784
#El dataset se puede pensar 55000 filas  y 784 columnas 
#Cada dato del dataset es un numero real entre 0 y 1 
mnist = input_data.read_data_sets("MNIST data", one_hot=True)
imagen = mnist.train.images[0] 

print(len(mnist.train.images))
print(len(mnist.test.images))
print(mnist.train.labels[0]) #El uno en la posicion 8 quiere decir que es un 7
plt.subplot(1,1, 1)
plt.imshow(np.reshape(imagen, (28,28)))
plt.show()



