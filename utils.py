from __future__ import print_function, absolute_import, division
import os, glob, sys
from skimage.io import imread, imshow, imsave
from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean
import numpy as np


def upd_print(str):
    sys.stdout.write('\r')
    sys.stdout.write(str)
    sys.stdout.flush()

def getData(num_tests, start, type):
    """
    Subfunction that loads data filenames.
    """
    if 'CITYSCAPES_DATASET' in os.environ:
        cityscapesPath = os.environ['CITYSCAPES_DATASET']
    else:
        cityscapesPath = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..')
    searchAnnotated = os.path.join(cityscapesPath, "gtFine", type, "*", "*_gt*_labelTrain*")
    searchRaw = os.path.join(cityscapesPath, "leftImg8bit", type, "*", "*.png")

    
    filesAnnotated =glob.glob(searchAnnotated)

    filesRaw=glob.glob(searchRaw)
    filesAnnotated.sort()
    filesRaw.sort()
    

    if not filesAnnotated:
        filesAnnotated = np.array([1])
    return filesAnnotated[start:start+num_tests], filesRaw[start:start+num_tests]

def UpscaleImg(img,scale, dims):
    """
    
    Upscales image by $scale times. dims should be 1 if image is of shape WxHx3 and 0 otherwise

    """
    if dims:
        new_img = np.zeros((img.shape[0]*scale,img.shape[1]*scale,3))
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                new_img[i*scale:(i+1)*scale,j*scale:(j+1)*scale,:]=img[i,j,:]
    else:
        new_img = np.zeros((img.shape[0] * scale, img.shape[1] * scale))
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                new_img[i * scale:(i + 1) * scale, j * scale:(j + 1) * scale] = img[i, j]
    return new_img

def ImportImages(folder):
   """
   Imports all images from $folder
   """
    search = os.path.join(folder, "*.png")
    files = glob.glob(search)
    files.sort()
    input = []

    z = 0
    for i in range(len(files)):
        z += 1
        if z % 100 == 0:
            print('loaded files input - ', z)

        file = files[i]
        img = imread(file)
        input.append(img)

    X = np.array(input)
    return X
def importRandomBatch(num, type, scale = 0):
    """
    Same as importBatch, but images are chosen at random
    """
    X_files, y_files = getData(3000, 0, type)
    r = np.random.choice(len(X_files), num)

    y1_files = np.array(X_files)
    X1_files = np.array(y_files)
    
    y_files = y1_files[r]
    X_files = X1_files[r]

    X_input = []
    y_input = []
    
    filenames = []
    z = 0
    
    for i in range(X_files.shape[0]):



        X_file = X_files[i]
        filenames.append(X_file[:-16])
        X_img = imread(X_file)

        if (scale != 0):
            X_new = np.zeros((int(X_img.shape[0] / scale), int(X_img.shape[1] / scale), 3))
            k = 0
            for x in X_img[::scale]:
                X_new[k] = x[::scale]
                k += 1
                X_img = X_new
        X_input.append(X_img)
    z = 0
    for i in range(y_files.shape[0]):

        y_file = y_files[i]
        y_img = imread(y_file)
        if (scale != 0):
            y_new = np.zeros((int(y_img.shape[0] / scale), int(y_img.shape[1] / scale)))
            k = 0
            for y in y_img[::scale]:
                y_new[k] = y[::scale]
                k += 1
                y_img = y_new
        y_input.append(y_img)

    X = np.array(X_input)
    y = np.array(y_input)
    if (type == 'val'):
        return X, y, filenames
    return X, y

def importBatch(num_tests, start, verbose, type="train", scale = 0):   
    
     """ Loads batch of data from dataset 
        CITYSCAPES_DATASET environment variable should be initialized or dataset folders should be in the same folder
        Loads $num_tests images, starting from $start. Searches in gtFine/$type folder. """

    y_files, X_files = getData(num_tests,start, type)
   
    X_input = []

   
    filenames = []
    z = 0
    for i in range(len(X_files)):

        z+=1
        if verbose:
            if z % 100 == 0:
                print('loaded files input - ', z)

        X_file = X_files[i]
        filenames.append(X_file[:-16])
        X_img = imread(X_file)

        if (scale != 0):
            X_new = np.zeros((int(X_img.shape[0] / scale), int(X_img.shape[1] / scale),3))
            k = 0
            for x in X_img[::scale]:
                X_new[k]=x[::scale]
                k+=1
                X_img = X_new
        X_input.append(X_img)
    X = np.array(X_input)
    if (type == 'demoVideo'):
        return X
    z = 0
    
    y_input = []
    for i in range(len(y_files)):
        z += 1
        if verbose:
            if z % 100 == 0:
               print('loaded files output - ', z)


        y_file = y_files[i]
        y_img = imread(y_file)
        if (scale != 0):
            y_new = np.zeros((int(y_img.shape[0] / scale), int(y_img.shape[1] / scale)))
            k = 0
            for y in y_img[::scale]:
                y_new[k] = y[::scale]
                k += 1
                y_img = y_new
        y_input.append(y_img)

    

    y = np.array(y_input)

    if (type=='val'):
        return X,y, filenames

    return X, y


def colorImage(input_image, output, classes_file, colors_file, saveto):

    """Creates a masked image showing segmentation results, saves image to $saveto file
        Input: input_image of shape WxHx3, output - image of shape WxH """
    
    CLASSES = open(classes_file).read().strip().split("\n")
    COLORS = open(colors_file).read().strip().split("\n")
    COLORS = [np.array(c.split(",")).astype("int") for c in COLORS]
    COLORS = np.array(COLORS, dtype="uint8")    
    image = input_image    
    classMap = output
    mask = COLORS[classMap]
    output = ((1 * image) + (1 * mask)).astype("uint8")
    imshow(output)
    imsave(saveto, output)
