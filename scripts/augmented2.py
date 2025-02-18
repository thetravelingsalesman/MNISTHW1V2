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


def rotate(image, degree):
    return ndimage.rotate(image, degree, reshape=False)

def rotation(image, biggestAngle=30):
    degree = random.uniform(-biggestAngle, biggestAngle)
    return ndimage.rotate(image, degree, reshape=False)


def randomCrop(image):
    if np.random.random()<.5:
        part = image[4: 28, 4: 28]
        return scipy.misc.imresize(part, 1.2)
    else:
        part = image[0: 24, 0: 24]
        return scipy.misc.imresize(part, 1.2)




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
    


def makeTransformedImage(image, kindOfSymmetry):
    if kindOfSymmetry == 'rotation':
        return rotation(image)
    if kindOfSymmetry == 'crop':
        return randomCrop(image)
    if kindOfSymmetry == 'elastic':
        return elastic_transform(image, 34, 4)
    if kindOfSymmetry == 'gaussian':
        return gaus_filter(image, sigma=1.5)



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
    
    



        
    # symmetries = {'rotation':10,'crop':0,'elastic':0,'gaussian':0}
    
    symmetries = {'rotation':10,'elastic':0,'gaussian':0}

    for whichSymnetry in symmetries.keys(): # for every kind of symnetry
        symmetries = {'rotation':0,'crop':0,'elastic':0,'gaussian':0}   
        symmetries[whichSymnetry] = 10
        image = data_images[0]
        new_images = []
        new_labels = []

        #perform all transformations on every image
        for i in xrange(data_images.shape[0]):
            created_images = []
            image = data_images[i]
            created_images.append(rotate(image, 0)) # add original image

            for key in symmetries.keys():  # for every kind of symmetry add images
                for i  in range(symmetries[key]): # to it this many times
                    created_images.append(makeTransformedImage(image, key))


            new_images.extend(np.array(created_images)) #
            new_labels.extend([target[i]] * len(created_images)) # match appropriate number of labels
        

        size = len(new_images)
        print 'new size:', size
        print  whichSymnetry
        new_train_data = np.array(new_images)
        new_train_labels = np.array(new_labels)
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
        
        pickle.dump(trainset_new, open(path + "trainset_new_"+whichSymnetry+ "_size_"+ str(size/3000) +"X.p", "wb" ))
            
            
        
        
