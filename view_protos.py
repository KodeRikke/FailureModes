import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from os import listdir

model_dir = "C:/users/ohrai/desktop/IAI/saved_models/"
num_prototypes = 15

files = listdir(model_dir + 'img/C100P15S2resnet34_E9I19push0.7266/')[::-1]
images = [Image.open(model_dir + 'img/C100P15S2resnet34_E9I19push0.7266/' + f) for f in files]

fig, axes = plt.subplots(nrows=4, ncols=num_prototypes, figsize=(20,20))
for idx, image in enumerate(images):
    row = idx // num_prototypes
    col = idx %  num_prototypes
    axes[row, col].axis("off")
    axes[row, col].imshow(image)
plt.subplots_adjust(wspace=.05, hspace=.05)
plt.show()
