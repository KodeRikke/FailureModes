import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np

num_prototypes = 10
names = num_prototypes * ["prototype-img"] + num_prototypes * ["prototype-img-original_with_self_act"] + num_prototypes * ["prototype-img-original"]
#protos = 3 * ["225", "226", "227", "228", "229", "230", "231", "232", "233", "234", "235", "236", "237", "238", "239"] # prototype ids
#protos = 3 * ["45", "46", "47"]
protos = 3 * ["150", "151", "152", "153", "154", "155", "156", "157", "158", "159"]
name = "pushes/img5/C100P10S3E0resnet34/class/" # location of prototype images

fig = plt.figure(figsize=(12, 12))
grid = ImageGrid(fig, 111, nrows_ncols=(3, num_prototypes), axes_pad=0.1)

for ax, i, j in zip(grid, protos, names):
    ax.set_axis_off()
    im = plt.imread(name + j + i + ".png")
    h, v = np.shape(im)[0], np.shape(im)[1]
    if v > h: 
        placeholder = 10/v
        v = v*placeholder 
        h = h*placeholder 
    elif h > v: 
        placeholder = 10/h
        v = v*placeholder 
        h = h*placeholder
    else: 
        v = 10 
        h = 10
    ax.imshow(im, extent = [0, v, 0, h])

plt.show()
