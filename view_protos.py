import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np

names = 5 * ["prototype-img"] + 5 * ["prototype-img-original_with_self_act"] + 5 * ["prototype-img-original"]
protos = 3 * ["75", "76", "77", "78", "79"] # prototype ids
name = model_dir + "img/C100P5S0resnet34_E10I5push0.6934/" # location of prototype images

fig = plt.figure(figsize=(12, 12))
grid = ImageGrid(fig, 111, nrows_ncols=(3, 5), axes_pad=0.1)

for ax, i, j in zip(grid, protos, names):
    ax.set_axis_off()
    ax.imshow(plt.imread(name + j + i + ".png"), extent = [0, 10, 0, 10])

plt.show()
