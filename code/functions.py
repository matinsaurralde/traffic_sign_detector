import numpy as np
import matplotlib.pyplot as plt
import random


def show_one_image(images,index):
    """muestra una imagen del array que uno le pasa con el correspondiente index

    Arguments:
        images {[np.arrays]} -- [contiene todas las imagenes posibles a mostrar]
        index {[int]} -- [indice que elije cual imagen mostrar del array]
    """
    if index > len(images):
        index = len(images)
    fig = plt.figure(figsize=(3, 3))
    plt.imshow(images[index], cmap='gray')
    plt.show()



def show_images(images, cols = 1, titles = None):
    """muestra imagenes
    Buscar cols¿

    Arguments:
        images {[np.arrays]} -- [contiene las imagenes a mostrar]

    Keyword Arguments:
        cols {int} -- [NO LA ENTIENDO (SOLO SÉ QUE ES "Number of columns in figure" PERO NO QUEDA CLARO QUE ES ESO)] (default: {1})
        titles {[list]} -- [contiene los titulos de las imagenes] (default: {None})
    """

   
    assert((titles is None)or (len(images) == len(titles)))
    
    n_images = len(images)
    
    if titles is None: titles = ['Image (%d)' % i for i in range(1,n_images + 1)]
    
    fig = plt.figure(figsize=(2, 2))
    
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images/float(cols)), n + 1)
        a.grid(False)
        a.axis('off')
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image, cmap='gray')
        a.set_title(title)
    
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.show()


def select_random_images_by_classes(features, labels, n_features):
    """selecciona imagenes random por cada una de las clases
    

    Arguments:
        features {[numpy.ndarray]} -- [sería el X_train]
        labels {[numpy.ndarray]} -- [sería el y_train]
        n_features {[int]} -- [numero de features]
    """
    indexes = []
    _classes = np.unique(labels)
  
    while len(indexes) < len(_classes):
        
        index = random.randint(0, n_features-1)
        _class = labels[index]

        for i in range(0, len(_classes)):

            if _class == _classes[i]:
                _classes[i] = -1
                indexes.append(index)
                break

    images = []
    titles = []

    for i in range(0, len(indexes)):
        images.append(features[indexes[i]])
        titles.append(str(labels[indexes[i]]))

    show_images(images, titles = titles)

def distribution_chart(x, y, xlabel, ylabel):
    plt.figure(figsize=(15,7))
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel(ylabel, fontsize=18)
    plt.bar(x, y, 0.7, color='red')
    plt.show()