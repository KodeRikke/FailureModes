from PIL import Image
import Augmentor
import os
import pandas as pd

def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path)

path = "" # path containing CUB_200_2011

makedir(path + 'datasets/cub200_cropped/test_cropped/')
makedir(path + 'datasets/cub200_cropped/train_cropped/')

df_image = pd.read_csv(path + "CUB_200_2011/images.txt", delimiter = " ", names = ["id", "path"])
df_bounding = pd.read_csv(path + "CUB_200_2011/bounding_boxes.txt", delimiter = " ", names = ["id", "x", "y", "w", "h"])
df_train_test_split = pd.read_csv(path + "CUB_200_2011/train_test_split.txt", delimiter = " ", names = ["id","split"])
df = pd.merge(pd.merge(df_image, df_bounding) , df_train_test_split)

for index, row in df.iterrows():
    id, x, y, w, h, split = str(row["id"]), float(row["x"]), float(row["y"]), float(row["w"]), float(row["h"]), float(row["split"]) # 1 train 0 test
    img_path = str(row["path"])
    im = Image.open(path + "CUB_200_2011/images/" + row["path"])
    img_dir, img_name = img_path.split("/")
    img = im.crop((x, y, x + w, y + h))
    if split == 0:
        makedir(path + 'ProtoPNet/datasets/cub200_cropped/test_cropped/' + img_dir)
        img.save(path + 'ProtoPNet/datasets/cub200_cropped/test_cropped/' + img_path[:-4] + '.jpg', 'JPEG')
    else:
        makedir(path + 'ProtoPNet/datasets/cub200_cropped/train_cropped/' + img_dir)
        img.save(path + 'ProtoPNet/datasets/cub200_cropped/train_cropped/' + img_path[:-4] + '.jpg', 'JPEG')

dir = path + 'ProtoPNet/datasets/cub200_cropped/train_cropped/'
target_dir = path + 'ProtoPNet/datasets/cub200_cropped/train_cropped_augmented/'
makedir(target_dir)
folders = [os.path.join(dir, folder) for folder in next(os.walk(dir))[1]]
target_folders = [os.path.join(target_dir, folder) for folder in next(os.walk(dir))[1]]

for i in range(len(folders)):
    fd = folders[i]
    tfd = target_folders[i]
    # rotation
    p = Augmentor.Pipeline(source_directory=fd, output_directory=tfd)
    p.rotate(probability=1, max_left_rotation=15, max_right_rotation=15)
    p.flip_left_right(probability=0.5)
    for i in range(10):
        p.process()
    del p
    # skew
    p = Augmentor.Pipeline(source_directory=fd, output_directory=tfd)
    p.skew(probability=1, magnitude=0.2)  # max 45 degrees
    p.flip_left_right(probability=0.5)
    for i in range(10):
        p.process()
    del p
    # shear
    p = Augmentor.Pipeline(source_directory=fd, output_directory=tfd)
    p.shear(probability=1, max_shear_left=10, max_shear_right=10)
    p.flip_left_right(probability=0.5)
    for i in range(10):
        p.process()
    del p
    # random_distortion
    p = Augmentor.Pipeline(source_directory=fd, output_directory=tfd)
    p.random_distortion(probability=1.0, grid_width=10, grid_height=10, magnitude=5)
    p.flip_left_right(probability=0.5)
    for i in range(10):
       p.process()
    del p
