
# import OS module
import os
#import opencv
import cv2
import shutil
import numpy as np
from matplotlib import pyplot as plt

 
# Get the list of all files and directories
pathNormal = "./train/NORMAL"
pathPneumonia='./train/PNEUMONIA'

pathTestNormal="./test/NORMAL"
pathTestPneumonia="./test/PNEUMONIA"

dir_list = os.listdir(pathNormal)
dir_list_Pneumonia=os.listdir(pathPneumonia)

dir_list_test = os.listdir(pathTestNormal)
dir_list_Pneumonia_test=os.listdir(pathTestPneumonia)

#train data
#Normal
dirnameOtsuNormal = 'testProcessed/Otsu/train/NORMAL'
dirNameAdaptiveNormal ='testProcessed/Adaptive/train/NORMAL'
dirNameHPFAdaptiveNormal='testProcessed/HPFAdaptive/train/NORMAL'
dirNameHPFOtsuNormal='testProcessed/HPFOtsu/train/NORMAL'
dirNameLPFAdaptiveNormal='testProcessed/LPFAdaptive/train/NORMAL'

#Pneumonic
dirnameOtsuPNEUMONIA= 'testProcessed/Otsu/train/PNEUMONIA'
dirNameAdaptivePNEUMONIA ='testProcessed/Adaptive/train/PNEUMONIA'
dirNameHPFAdaptivePNEUMONIA='testProcessed/HPFAdaptive/train/PNEUMONIA'
dirNameHPFOtsuPNEUMONIA='testProcessed/HPFOtsu/train/PNEUMONIA'
dirNameLPFAdaptivePNEUMONIA='testProcessed/LPFAdaptive/train/PNEUMONIA'

#test data
#Normal
dirnameOtsuNormalTest = 'testProcessed/Otsu/test/NORMAL'
dirNameAdaptiveNormalTest ='testProcessed/Adaptive/test/NORMAL'
dirNameHPFAdaptiveNormalTest='testProcessed/HPFAdaptive/test/NORMAL'
dirNameHPFOtsuNormalTest='testProcessed/HPFOtsu/test/NORMAL'
dirNameLPFAdaptiveNormalTest='testProcessed/LPFAdaptive/test/NORMAL'

#Pneumonic
dirnameOtsuPNEUMONIATest= 'testProcessed/Otsu/test/PNEUMONIA'
dirNameAdaptivePNEUMONIATest ='testProcessed/Adaptive/test/PNEUMONIA'
dirNameHPFAdaptivePNEUMONIATest='testProcessed/HPFAdaptive/test/PNEUMONIA'
dirNameHPFOtsuPNEUMONIATest='testProcessed/HPFOtsu/test/PNEUMONIA'
dirNameLPFAdaptivePNEUMONIATest='testProcessed/LPFAdaptive/test/PNEUMONIA'

if os.path.exists(dirnameOtsuNormal):
    shutil.rmtree(dirnameOtsuNormal)
os.makedirs(dirnameOtsuNormal)

if os.path.exists(dirNameAdaptiveNormal):
    shutil.rmtree(dirNameAdaptiveNormal)
os.makedirs(dirNameAdaptiveNormal)

if os.path.exists(dirNameHPFAdaptiveNormal):
    shutil.rmtree(dirNameHPFAdaptiveNormal)
os.makedirs(dirNameHPFAdaptiveNormal)

if os.path.exists(dirNameHPFOtsuNormal):
    shutil.rmtree(dirNameHPFOtsuNormal)
os.makedirs(dirNameHPFOtsuNormal)

if os.path.exists(dirNameLPFAdaptiveNormal):
    shutil.rmtree(dirNameLPFAdaptiveNormal)
os.makedirs(dirNameLPFAdaptiveNormal)



if os.path.exists(dirnameOtsuNormalTest):
    shutil.rmtree(dirnameOtsuNormalTest)
os.makedirs(dirnameOtsuNormalTest)

if os.path.exists(dirNameAdaptiveNormalTest):
    shutil.rmtree(dirNameAdaptiveNormalTest)
os.makedirs(dirNameAdaptiveNormalTest)

if os.path.exists(dirNameHPFAdaptiveNormalTest):
    shutil.rmtree(dirNameHPFAdaptiveNormalTest)
os.makedirs(dirNameHPFAdaptiveNormalTest)

if os.path.exists(dirNameHPFOtsuNormalTest):
    shutil.rmtree(dirNameHPFOtsuNormalTest)
os.makedirs(dirNameHPFOtsuNormalTest)

if os.path.exists(dirNameLPFAdaptiveNormalTest):
    shutil.rmtree(dirNameLPFAdaptiveNormalTest)
os.makedirs(dirNameLPFAdaptiveNormalTest)



#making pneumonia folder
if os.path.exists(dirnameOtsuPNEUMONIA):
    shutil.rmtree(dirnameOtsuPNEUMONIA)
os.makedirs(dirnameOtsuPNEUMONIA)

if os.path.exists(dirNameAdaptivePNEUMONIA):
    shutil.rmtree(dirNameAdaptivePNEUMONIA)
os.makedirs(dirNameAdaptivePNEUMONIA)

if os.path.exists(dirNameHPFAdaptivePNEUMONIA):
    shutil.rmtree(dirNameHPFAdaptivePNEUMONIA)
os.makedirs(dirNameHPFAdaptivePNEUMONIA)

if os.path.exists(dirNameHPFOtsuPNEUMONIA):
    shutil.rmtree(dirNameHPFOtsuPNEUMONIA)
os.makedirs(dirNameHPFOtsuPNEUMONIA)

if os.path.exists(dirNameLPFAdaptivePNEUMONIA):
    shutil.rmtree(dirNameLPFAdaptivePNEUMONIA)
os.makedirs(dirNameLPFAdaptivePNEUMONIA)








if os.path.exists(dirnameOtsuPNEUMONIATest):
    shutil.rmtree(dirnameOtsuPNEUMONIATest)
os.makedirs(dirnameOtsuPNEUMONIATest)

if os.path.exists(dirNameAdaptivePNEUMONIATest):
    shutil.rmtree(dirNameAdaptivePNEUMONIATest)
os.makedirs(dirNameAdaptivePNEUMONIATest)

if os.path.exists(dirNameHPFAdaptivePNEUMONIATest):
    shutil.rmtree(dirNameHPFAdaptivePNEUMONIATest)
os.makedirs(dirNameHPFAdaptivePNEUMONIATest)

if os.path.exists(dirNameHPFOtsuPNEUMONIATest):
    shutil.rmtree(dirNameHPFOtsuPNEUMONIATest)
os.makedirs(dirNameHPFOtsuPNEUMONIATest)

if os.path.exists(dirNameLPFAdaptivePNEUMONIATest):
    shutil.rmtree(dirNameLPFAdaptivePNEUMONIATest)
os.makedirs(dirNameLPFAdaptivePNEUMONIATest)



def otsuThreshold(image):
    # Otsu's thresholding
    ret2,th2 = cv2.threshold(image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # print('ret2', ret2)
    # print('th2 ',th2)
    return th2
def adaptiveThreshold(image):
    th3 = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,11,2)
    return th3
def hpfAdaptiveThreshFunction(img):
    #kernel for high pass filter
    kernel = np.array([[-3.0, -1.0, 3.0], 
                   [-10.0, 8.0, 10.0],
                   [-3.0, -1.0, 3.0]])

    kernel = kernel/(np.sum(kernel) if np.sum(kernel)!=0 else 1)

    # Clache(takes care of over-amplification of the contrast)
    clahe=cv2.createCLAHE(clipLimit=40)
    imgClahe=clahe.apply(img)

    #High pass filter
    hpfImage= cv2.filter2D(imgClahe,-1,kernel)

    # Adaptive Thresh
    threshImage = cv2.adaptiveThreshold(hpfImage,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,11,2)

    return threshImage
def hpfOtsuThreshFunction(img):
    #kernel for high pass filter
    kernel = np.array([[-3.0, -1.0, 3.0], 
                    [-10.0, 8.0, 10.0],
                    [-3.0, -1.0, 3.0]])

    kernel = kernel/(np.sum(kernel) if np.sum(kernel)!=0 else 1)

    # Clache(takes care of over-amplification of the contrast)
    clahe=cv2.createCLAHE(clipLimit=40)
    imgClahe=clahe.apply(img)

    #High pass filter
    hpfImage= cv2.filter2D(imgClahe,-1,kernel)

    # Otsu Thresh
    ret2,threshImage = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    return threshImage
def lpfAdaptiveThreshFunction(img):
    #kernel for high pass filter
    kernel = np.array([[1, 1, 1, 1, 1], 
                   [1, 1, 1, 1, 1], 
                   [1, 1, 1, 1, 1], 
                   [1, 1, 1, 1, 1], 
                   [1, 1, 1, 1, 1]])     

    kernel = kernel/(np.sum(kernel) if np.sum(kernel)!=0 else 1)

    # Clache(takes care of over-amplification of the contrast)
    clahe=cv2.createCLAHE(clipLimit=40)
    imgClahe=clahe.apply(img)

    #High pass filter
    lpfImage= cv2.filter2D(imgClahe,-1,kernel)

    # Adaptive Thresh
    threshImage = cv2.adaptiveThreshold(lpfImage,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,11,2)

    return threshImage
print("Files and directories in '", pathNormal, "' :")
 
# prints all files
print(dir_list)
# first process all files in 
for x in dir_list:
    if x.endswith(".jpeg"):
        # Prints only text file present in My Folder
        # print(x)
        image = cv2.imread(os.path.join(pathNormal, x), cv2.IMREAD_GRAYSCALE)
        #print(image)
        transformedImgOtsu=otsuThreshold(image)
        cv2.imwrite(os.path.join(dirnameOtsuNormal,x), transformedImgOtsu)
        
        transformedImgAdaptive=adaptiveThreshold(image)
        cv2.imwrite(os.path.join(dirNameAdaptiveNormal,x), transformedImgAdaptive)

        hpfAdaptive=hpfAdaptiveThreshFunction(image)
        cv2.imwrite(os.path.join(dirNameHPFAdaptiveNormal,x), hpfAdaptive)
        
        hpfOtsu=hpfOtsuThreshFunction((image))
        cv2.imwrite(os.path.join(dirNameHPFOtsuNormal,x), hpfOtsu)
        
        lpfAdaptive=lpfAdaptiveThreshFunction((image))
        cv2.imwrite(os.path.join(dirNameLPFAdaptiveNormal,x), lpfAdaptive)



for x in dir_list_Pneumonia:
    if x.endswith(".jpeg"):
        # Prints only text file present in My Folder
        # print(x)
        image = cv2.imread(os.path.join(pathPneumonia, x), cv2.IMREAD_GRAYSCALE)
        #print(image)
        transformedImgOtsu=otsuThreshold(image)
        cv2.imwrite(os.path.join(dirnameOtsuPNEUMONIA,x), transformedImgOtsu)
        
        transformedImgAdaptive=adaptiveThreshold(image)
        cv2.imwrite(os.path.join(dirNameAdaptivePNEUMONIA,x), transformedImgAdaptive)

        hpfAdaptive=hpfAdaptiveThreshFunction(image)
        cv2.imwrite(os.path.join(dirNameHPFAdaptivePNEUMONIA,x), hpfAdaptive)
        
        hpfOtsu=hpfOtsuThreshFunction((image))
        cv2.imwrite(os.path.join(dirNameHPFOtsuPNEUMONIA,x), hpfOtsu)
        
        lpfAdaptive=lpfAdaptiveThreshFunction((image))
        cv2.imwrite(os.path.join(dirNameLPFAdaptivePNEUMONIA,x), lpfAdaptive)












for x in dir_list_test:
    if x.endswith(".jpeg"):
        # Prints only text file present in My Folder
        # print(x)
        image = cv2.imread(os.path.join(pathTestNormal, x), cv2.IMREAD_GRAYSCALE)
        #print(image)
        transformedImgOtsu=otsuThreshold(image)
        cv2.imwrite(os.path.join(dirnameOtsuNormalTest,x), transformedImgOtsu)
        
        transformedImgAdaptive=adaptiveThreshold(image)
        cv2.imwrite(os.path.join(dirNameAdaptiveNormalTest,x), transformedImgAdaptive)

        hpfAdaptive=hpfAdaptiveThreshFunction(image)
        cv2.imwrite(os.path.join(dirNameHPFAdaptiveNormalTest,x), hpfAdaptive)
        
        hpfOtsu=hpfOtsuThreshFunction((image))
        cv2.imwrite(os.path.join(dirNameHPFOtsuNormalTest,x), hpfOtsu)
        
        lpfAdaptive=lpfAdaptiveThreshFunction((image))
        cv2.imwrite(os.path.join(dirNameLPFAdaptiveNormalTest,x), lpfAdaptive)



for x in dir_list_Pneumonia_test:
    if x.endswith(".jpeg"):
        # Prints only text file present in My Folder
        # print(x)
        image = cv2.imread(os.path.join(pathTestPneumonia, x), cv2.IMREAD_GRAYSCALE)
        #print(image)
        transformedImgOtsu=otsuThreshold(image)
        cv2.imwrite(os.path.join(dirnameOtsuPNEUMONIATest,x), transformedImgOtsu)
        
        transformedImgAdaptive=adaptiveThreshold(image)
        cv2.imwrite(os.path.join(dirNameAdaptivePNEUMONIATest,x), transformedImgAdaptive)

        hpfAdaptive=hpfAdaptiveThreshFunction(image)
        cv2.imwrite(os.path.join(dirNameHPFAdaptivePNEUMONIATest,x), hpfAdaptive)
        
        hpfOtsu=hpfOtsuThreshFunction((image))
        cv2.imwrite(os.path.join(dirNameHPFOtsuPNEUMONIATest,x), hpfOtsu)
        
        lpfAdaptive=lpfAdaptiveThreshFunction((image))
        cv2.imwrite(os.path.join(dirNameLPFAdaptivePNEUMONIATest,x), lpfAdaptive)

