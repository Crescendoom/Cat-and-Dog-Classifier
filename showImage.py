# Code block for showing of image
import pandas as pd
import os
from glob import glob
from matplotlib import pyplot as plt

base_dir = "./Dataset"
train_dir = os.path.join(base_dir, "training_set")
val_dir = os.path.join(base_dir, "validation_set")

cat_files = glob(os.path.join(train_dir, "cats", "*.jpg"))
dog_files = glob(os.path.join(train_dir, "dogs", "*.jpg"))
img_mpl = plt.imread(cat_files[11])

def showCatImage(cat_files):
    if len(cat_files) == 0:
        print(f"No cat images found in {os.path.join(train_dir, 'cats')}")
        exit(1)
   
    plt.figure(figsize=(6, 6))
    plt.imshow(img_mpl)
    plt.title(f"Cat Image: {os.path.basename(cat_files[0])}")
    plt.axis('off')
    plt.show()

def showPixelValuesOfCatImage():
    pd.Series(img_mpl.flatten()).plot(kind='hist', bins=50, title='Distribution of Pixel Values')
    plt.show()

# UNCOMMENT THIS IF YOU WANT TO RUN
# showCatImage(cat_files)
# showPixelValuesOfCatImage()
