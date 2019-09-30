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
    directorios = [directorio for directorio in os.listdir(ruta)
            if(os.path.isdir(os.path.join(ruta, directorio)))]
    # print(dirs)
    labels = []
    images = []
    for directorio in directorios:
        labelDir = os.path.join(ruta, directorio)
        nombreDeArchivos = [os.path.join(labelDir, archivo)
                            for archivo in os.listdir(labelDir)
                            if archivo.endswith('.ppm')]
        for archivo in nombreDeArchivos:
            images.append(imd.imread(archivo))
            labels.append(int(directorio))
    return images, labels



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
            
    
directorioRaiz = './assets/belgian/'
trainDataDir = os.path.join(directorioRaiz, 'Training')
testDataDir = os.path.join(directorioRaiz, 'Testing')

#images y labels son de tipo lista
imagenes, labels = cargarData(trainDataDir)

#Asi que hay que convertir las listas a arrays
imagenes = np.array(imagenes)
labels = np.array(labels)

print('----------------------Comienzan las impresiones-----------------------')

#Saber cuantos bits se han usado en la carga a la memoria
# print(imagenes.nbytes/imagenes.itemsize)

#Hacer grafico de frecuencias por categoria (labels hace referencia a los datos y len(set(labels)) hace referencia a la cantidad de categorias ya que la funcion set devuelve objetos unicos en un array y len la longitud de el array de objetos unicos) 
#Importante recordar que las etiquetas estan en base a las carpetas (cada etiqueta o carpeta son una categoria de senales de trafico)
# plt.hist(labels, len(set(labels)))
# plt.show()

#Comienza la investigacion de los datos

# labelsUnicas = set(labels)
# plt.figure(figsize=(16, 16))
# i = 1
# for label in labelsUnicas:
#     imagen = imagenes[list(labels).index(label)]
#     plt.subplot(8, 8, i)
#     plt.axis("off")
#     plt.title('clase:{0} ({1})'.format(label, list(labels).count(label)))
#     plt.subplots_adjust(wspace= 2)
#     i+=1
    # plt.imshow(imagen)

# plt.show()

#Resumen del analisis exploratorio
# -No todas las imagenes son de diferente tamaño
# -Hay 62 clases de señales de trafico. Del 0 al 61
# -La distribucion de señales no es uniforme, es decir hay muchas mas de unas que de otros tipos

#Termina la investigacion de datos

#Inicia el preprosesado del dataset

# def getWidthImagen(imagen):
#     return imagen.shape[1] 

# def getHeightImagen(imagen):
#     return imagen.shape[0] 

# minWidth = min(list(map(getWidthImagen, imagenes)))
# minHeight = min(list(map(getHeightImagen, imagenes)))

# print("Medidas minimas de imagenes en dataset: {0}x{1}".format(minWidth, minHeight))

#Reescalado de imagenes, se uso un formato 30x30 arbitrariamente pero basado en minWidth = 20, minHeight = 22
def preProcesado(imagenesArr):
    imagenesProc = [transform.resize(imagen, (30, 30)) for imagen in imagenesArr]
    imagenesProc = np.array(imagenesProc)
    imagenesProc = rgb2gray(imagenesProc)
    return imagenesProc

imagenes30 = preProcesado(imagenes)
# getImagenesRandom(imagenes30, 6)

#Finaliza el preprosesado del dataset







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
cantidadEpochs = 1400

for i in range(cantidadEpochs):
    _, accuracy_val = session.run([trainOpt, accuracy], feed_dict={
        x: imagenes30,
        y: list(labels)
    })
#     _, loss_val = session.run([trainOpt, loss], feed_dict={
#         x: imagenes30,
#         y: list(labels)
#     })
#     if(i%10==0): 
#         print("EPOCH", i)
#         print("Eficacia: ", accuracy_val)
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

print('Prediccion: ', prediccion)
print('Reales: ', labelsMuestra)

plt.figure(figsize=(16,20))
for i in range(len(imagenesMuestra)):
    plt.subplot(10, 4, i+1)
    plt.axis("off")
    color = "green" if labelsMuestra[i] == prediccion[i] else "red"
    plt.text(32, 15, "Real:        {0}\nPrediccion:{1}".format(labelsMuestra[i], prediccion[i]), fontsize=14, color=color)
    plt.imshow(imagenesMuestra[i], cmap="gray")
plt.show()



#Validacion de modelo
imagenesMuestra, labelsMuestra = cargarData(testDataDir)
imagenesMuestra = np.array(imagenesMuestra)
labelsMuestra = np.array(labelsMuestra)
testImagenes30 = preProcesado(imagenesMuestra)
prediccion = session.run([prediccionFinal], feed_dict={
    x: testImagenes30    
})[0]

coincidencias = sum([int(l0 == lp) for l0, lp in zip(labelsMuestra, prediccion)])
print("Coincidencias {0} de {1} con una eficacia de {2}".format(coincidencias, len(labelsMuestra), ((coincidencias/len(labelsMuestra))*100)))
#Termina red neuronal