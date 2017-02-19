# -*- coding: utf-8 -*-
"""
Data preparation file for Udacity Behaviorial Cloning Project
Takes as input JPG files and CSV of steering angles
Outputs a pickle file

@author: DWolf
"""

import pandas
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import PIL
from PIL import Image


# Load dataset
# The input file has a column for image name and a column for steering angle

# Create labels array
df = pandas.read_csv('data.csv')
y = df['angle'].as_matrix()

filenames = df['filename'].as_matrix()


# Create feature array
print("Reading in images...")
for i in range(0,len(y)):
    filename = filenames[i]
    # Just to show progress, print a count every 1000 images loaded
    if i/300 == round(i/300):
        print(i)
    img = Image.open('IMG/' + filename)
    # Resize to 100 x 200 pixels
    img = img.resize((200, 100), PIL.Image.ANTIALIAS)
    # Crop the top 34 pixels of each image
    img = img.crop((0, 34, 200, 84))
#    plt.imshow(img)
#    plt.show()
#    img.save('pictures/' + 'resizedcropped.jpg')
    img = np.asarray(img)
    if i == 0:
        X = np.array([img])
    else:
        X = np.concatenate([X, [img]])
print("Finished reading images.")
#print(array)
print("Number of labels:",len(y))
print("Feature array shape:",X.shape)

#img=mpimg.imread(filename)
#plt.imshow(img)
#plt.show()

# Save the data to a pickle file
with open('pkldata.p', 'wb') as save_features:
    pickle.dump({ 'features': X , 'labels': y }, save_features)


