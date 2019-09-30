# import tensorflow as tf
import os
import skimage.data as imd
from skimage import transform
from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
import random
from functools import reduce
import tensorflow as tf

def cargarData(ruta):
    archivos = np.array(os.listdir(ruta))
    labels = []
    data = []
    
    for archivo in archivos:
        indices = [pos for pos, char in enumerate(archivo) if char == '_']
        if (len(indices)>2):
            label = int(archivo[0: indices[0]] + archivo[indices[0]+1:indices[1]] + archivo[indices[1]+1:indices[2]])
        else:
            label = int(archivo[0: indices[0]] + archivo[indices[0]+1:indices[1]])
        labels.append(label)
        data.append(imd.imread(os.path.join(ruta, archivo)))
        # for directorio in directorios:
    #     labelDir = os.path.join(ruta, directorio)
    #     nombreDeArchivos = [os.path.join(labelDir, archivo)
    #                         for archivo in os.listdir(labelDir)]
    #     for archivo in nombreDeArchivos:
    #         images.append(imd.imread(archivo))
    #         labels.append(int(directorio))
    return np.array(data), np.array(labels)


def getImagenesRandom(array, numeroDeElementos):
    imagenesRandom = random.sample(range(0, len(array)), numeroDeElementos)

    for i in range(len(imagenesRandom)):
        imagen = array[imagenesRandom[i]]
        #Agregar numeroDeElementos subplots para las numeroDeElementos imagenes (filas, columnas, index) el index es i+1 porque index debe iniciar en 1 no en 0
        plt.subplot(1, len(imagenesRandom), i+1)
        plt.axis('off')
        plt.imshow(imagen, cmap="gray")
        #Dar espacio entre imagenes
        plt.subplots_adjust(wspace= .5)
        #Esto imprime la forma: (height, width, los canales de color por ejemplo: 3= rgb), min: pixel con menor cantidad de colores (negro), max: pixel con mayor cantidad de colores (blanco)  
        print('forma:{0}, min:{1}, max:{2}'.format(imagen.shape, imagen.min(), imagen.max()))
    plt.show()
    
    

def preProcesado(imagenesArr):
    imagenesProc = [transform.resize(imagen, (150, 150)) for imagen in imagenesArr]
    imagenesProc = np.array(imagenesProc)
    # imagenesProc = rgb2gray(imagenesProc)
    return imagenesProc






data, labels = cargarData('./assets/faces/')

labels.sort()
x = -1
last = 0
for i in range(len(labels)):
    if(labels[i] != last):
        x += 1
    last = labels[i]
    labels[i] = x
    
imagenes30 = preProcesado(data)


#Modelo de red neuronal con TF

x = tf.placeholder(dtype= tf.float32, shape = [None, 30, 30])
y = tf.placeholder(dtype = tf.int32, shape=[None])

imagenesFlat = tf.contrib.layers.flatten(x)
logits = tf.contrib.layers.fully_connected(imagenesFlat, len(set(labels)), tf.nn.relu)

#Mide probabilidad de error en una funcion discreta y la funcion reduce_mean calcula el promedio de todos los elementos a travez de todas las dimensiones del tensor
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits))

#Funcion de optimizador, trata de minimizar la funcion de perdidas (loss)
trainOpt = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)

#Convertir los logits a etiquetas con un solo elemento
prediccionFinal = tf.arg_max(logits, 1)

accuracy = tf.reduce_mean(tf.cast(prediccionFinal, tf.float32))


#Entrenamiento del modelo
tf.set_random_seed(1234)

#Definicion de bucles de entrenamiento
session = tf.Session()
session.run(tf.global_variables_initializer())
cantidadEpochs = 300

for i in range(cantidadEpochs):
    _, accuracy_val = session.run([trainOpt, accuracy], feed_dict={
        x: imagenes30,
        y: list(labels)
    })
#     _, loss_val = session.run([trainOpt, loss], feed_dict={
#         x: imagenes30,
#         y: list(labels)
#     })
    if(i%50==0): 
        print("EPOCH", i)
        print("Eficacia: ", accuracy_val)
#         print("Loss: ", loss_val)
#     print("Fin del EPOCH ", i)

#Evaluacion de la red neuronal
#Extraemos 16 elementos
indicesMuestra = random.sample(range(len(imagenes30)), 40)
imagenesMuestra = [imagenes30[i] for i in indicesMuestra]
labelsMuestra = [labels[i] for i in indicesMuestra]

prediccion = session.run([prediccionFinal], feed_dict={
    x: imagenesMuestra
})[0] 


plt.figure(figsize=(16,20))
for i in range(len(imagenesMuestra)):
    plt.subplot(10, 4, i+1)
    plt.axis("off")
    color = "green" if labelsMuestra[i] == prediccion[i] else "red"
    plt.text(32, 15, "Real:        {0}\nPrediccion:{1}".format(labelsMuestra[i], prediccion[i]), fontsize=14, color=color)
    plt.imshow(imagenesMuestra[i], cmap="gray")
plt.show()



#Termina red neuronal



