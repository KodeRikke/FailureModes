import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np

num_prototypes = 5
names = num_prototypes * ["prototype-img"] + num_prototypes * ["prototype-img-original_with_self_act"] + num_prototypes * ["prototype-img-original"]
protos = 3 * ['75', '76', '77', '78', '79'] # prototype ids
name = model_dir + "img/C100P5S3resnet34_E9I19push0.6945/" # location of prototype images 

fig = plt.figure(figsize=(12, 12))
grid = ImageGrid(fig, 111, nrows_ncols=(3, num_prototypes), axes_pad=0.1)

for ax, i, j in zip(grid, protos, names):
    im = plt.imread(name + j + i + ".png")
    h, v = np.shape(im)[0], np.shape(im)[1]
    if v != h:
        h = h * 10 / v
    else: 
        h = 10
    ax.set_axis_off()
    ax.imshow(im, extent = [0, 10, 0, h])
plt.show()
