import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from os import listdir

model_dir = "./saved_models/"

im_dir = model_dir + 'img/C100P3S0resnet34_e19/'

num_prototypes = 3

files = listdir(im_dir)[::-1]
images = [Image.open(im_dir + f) for f in files]

fig, axes = plt.subplots(nrows=3, ncols=num_prototypes, figsize=(20,20))
for idx, image in enumerate(images):
    row = idx // num_prototypes
    col = idx %  num_prototypes
    axes[row, col].axis("off")
    axes[row, col].imshow(image)
plt.subplots_adjust(wspace=.05, hspace=.05)
plt.show()
