import numpy as np
import pickle
import torch
from torchvision import transforms
import scipy
from scipy import ndimage
from matplotlib import pyplot as plt
from sub import subMNIST 
import random
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter



def randomRotation(image, biggestAngle=30):
    degree = random.uniform(-biggestAngle, biggestAngle)
    return ndimage.rotate(image, degree, reshape=False)


def randomCrop(image):
    if np.random.random()<.5:
        part = image[4: 28, 4: 28]
        return scipy.misc.imresize(part, 1.2)
    else:
        part = image[0: 24, 0: 24]
        return scipy.misc.imresize(part, 1.2)


import numpy as np
from scipy.ndimage import zoom


def clipped_zoom(img, zoom_factor, **kwargs):
    """
    
    function comes from http://stackoverflow.com/questions/37119071/scipy-rotate-and-zoom-an-image-without-changing-its-dimensions
    """

    h, w = img.shape[:2]

    # width and height of the zoomed image
    zh = int(np.round(zoom_factor * h))
    zw = int(np.round(zoom_factor * w))

    # for multichannel images we don't want to apply the zoom factor to the RGB
    # dimension, so instead we create a tuple of zoom factors, one per array
    # dimension, with 1's for any trailing dimensions after the width and height.
    zoom_tuple = (zoom_factor,) * 2 + (1,) * (img.ndim - 2)

    # zooming out
    if zoom_factor < 1:
        # bounding box of the clip region within the output array
        top = (h - zh) // 2
        left = (w - zw) // 2
        # zero-padding
        out = np.zeros_like(img)
        out[top:top+zh, left:left+zw] = zoom(img, zoom_tuple, **kwargs)

    # zooming in
    elif zoom_factor > 1:
        # bounding box of the clip region within the input array
        top = (zh - h) // 2
        left = (zw - w) // 2
        out = zoom(img[top:top+zh, left:left+zw], zoom_tuple, **kwargs)
        # `out` might still be slightly larger than `img` due to rounding, so
        # trim off any extra pixels at the edges
        trim_top = ((out.shape[0] - h) // 2)
        trim_left = ((out.shape[1] - w) // 2)
        out = out[trim_top:trim_top+h, trim_left:trim_left+w]

        # out = out.flatten().reshape(28,28)

    # if zoom_factor == 1, just return the input array
    else:
        out = img
    print out.shape
    return out

def randomZoom(image, minZoom=.7,maxZoom=1.2):
    zoom_factor = np.random.uniform(minZoom,maxZoom)
    clipped_zoom(image, zoom_factor)


def rotate(image, degree):
    return ndimage.rotate(image, degree, reshape=False)

def rotate_pos(image):
    degree = random.uniform(10, 30)
    return ndimage.rotate(image, degree, reshape=False)
    
def rotate_neg(image):
    degree = random.uniform(-30, -10)
    return ndimage.rotate(image, degree, reshape=False)

def gaus_filter(image, sigma=1.5):
    return ndimage.gaussian_filter(image, sigma)
    
def crop_1(image):
    part = image[4: 28, 4: 28]
    return scipy.misc.imresize(part, 1.2)
    
def crop_2(image):
    part = image[0: 24, 0: 24]
    return scipy.misc.imresize(part, 1.2)


def elastic_transform(image, alpha, sigma, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
       source: https://gist.github.com/chsasank/4d8f68caf01f041a6453e67fb30f8f5a
    """
    assert len(image.shape)==2

    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    indices = np.reshape(x+dx, (-1, 1)), np.reshape(y+dy, (-1, 1))
    
    return map_coordinates(image, indices, order=1).reshape(shape)
    
if __name__ == "__main__":   
    print('loading data!')
    path = '../data/'
    trainset_labeled = pickle.load(open(path + "train_labeled.p", "rb"))
    print('done')
    train_loader = torch.utils.data.DataLoader(trainset_labeled, batch_size=3000, shuffle=True)
    data = train_loader.dataset.train_data
    data_images = data.numpy()
    target = train_loader.dataset.train_labels
    target = target.numpy()
    
    image = data_images[0]
    new_images = []
    new_labels = []
    
    numRotations = 0
    numElastic = 0
    numZoom = 10
    
    #perform all transformations on every image
    for i in xrange(data_images.shape[0]):
        created_images = []
        image = data_images[i]
        created_images.append(rotate(image, 0))
        # for k in range(numRotations): # numRotations
        #     created_images.append(randomRotation(image, biggestAngle=30))
            # created_images.append(rotate_pos(image))

            # created_images.append(rotate_neg(image))
        # created_images.append(gaus_filter(image))
        # created_images.append(crop_1(image))
        # created_images.append(crop_2(image))
        # for j in range(numElastic): # numElastic
        #     created_images.append(elastic_transform(image, 34, 4))
        for l in range(numZoom): # numElastic
            created_images.append(randomZoom(image))

        new_images.extend(np.array(created_images))
        new_labels.extend([target[i]] * len(created_images))
    

    size = len(new_images)
    print 'new size:', size
      
    new_train_data = np.array(new_images)
    new_train_labels = np.array(new_labels)
    print new_train_data.shape
    print 'images added:', size - 3000
    print 'labels:', new_train_labels.shape
    
    new_train_data = torch.from_numpy(new_train_data)
    new_train_labels = torch.from_numpy(new_train_labels)
    
    
    transform=transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize((0.1307,), (0.3081,))
                                 ])
    trainset_new = subMNIST(root=path + 'data', train=True, transform=transform, download=True, k=size)
    trainset_new.train_data = new_train_data.clone()
    trainset_new.train_labels = new_train_labels.clone()   
    
    pickle.dump(trainset_new, open(path + "trainset_new_"+ "zoom_10"+"X.p", "wb" ))
        
        
    
    
