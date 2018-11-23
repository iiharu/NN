# coding: utf-8
import os
import shutil
import sys
import tarfile
import urllib.request


CLASSES = ["Tulip",
           "Snowdrop",
           "LilyValley",
           "Bluebell",
           "Crocus",
           "Iris",
           "Tigerlily",
           "Daffodil",
           "Fritillary",
           "Sunflower",
           "Daisy",
           "ColtsFoot",
           "Dandelion",
           "Cowslip",
           "Buttercup",
           "Windflower",
           "Pansy"]


def download(url, filename):
    filepath = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), filename)
    if not os.path.exists(filepath):
        urllib.request.urlretrieve(url, filename=filepath)

    return filepath


def extract(filepath):
    with tarfile.open(filepath, mode='r') as tar:
        dirpath = os.path.dirname(os.path.abspath(filepath))
        tar.extractall(path=dirpath)

    dirpath = os.path.join(dirpath, 'jpg')
    return dirpath


def move(dirpath):

    target = os.path.dirname(dirpath)

    if os.path.exists(os.path.join(target, 'files.txt')):
        print("Dataset is already prepared.")
        return
    shutil.move(os.path.join(dirpath, 'files.txt'), target)

    traindir = os.path.join(target, 'train')
    testdir = os.path.join(target, 'test')
    os.mkdir(traindir)
    os.mkdir(testdir)

    for i, classname in enumerate(CLASSES):
        class_traindir = os.path.join(traindir, classname)
        class_testdir = os.path.join(testdir, classname)
        os.mkdir(class_traindir)
        os.mkdir(class_testdir)

        for j in range(80):
            filepath = os.path.join(
                dirpath, "image_" + str(i * 80 + j + 1).zfill(4) + ".jpg")
            if j < 60:
                shutil.move(filepath, class_traindir)
            else:
                shutil.move(filepath, class_testdir)


def clean(dirpath):
    shutil.rmtree(dirpath)


if __name__ == '__main__':
    print("Downloading...")
    filepath = download(url="http://www.robots.ox.ac.uk/~vgg/data/flowers/17/17flowers.tgz",
                        filename="17flowers.tgz")
    print("Extracting...")
    dirpath = extract(filepath)
    print("Moving...")
    move(dirpath)
    print("Cleaning...")
    clean(dirpath)
