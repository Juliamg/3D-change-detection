import os
import random
import sys

IGNORE_FILES = ['.DS_Store']
random.seed(30)

### Run this script for splitting data into train and test set ###

def split_train_test(data_folder, split_fraction: float):

    test_path = os.path.join(os.getcwd(), "test")
    train_path = data_folder

    if not os.path.isdir(train_path):
        os.mkdir(test_path)

    for dir in os.listdir(train_path): #list of subjects
        if dir in IGNORE_FILES:
            continue

        if not os.path.isdir(os.path.join(test_path, dir)): # create folder for each subject in test folder
            os.mkdir(os.path.join(test_path, dir))

        subject_path = os.path.join(train_path, dir)
        l = len(os.listdir(subject_path))

        for i in range(1, l): # haven't tested this part
            if i % int(len(os.listdir(subject_path)) * (1 - split_fraction)) == 0:
                k = random.choice(os.listdir(os.path.join(train_path, dir)))
                os.rename(os.path.join(train_path, dir, k),
                          os.path.join(test_path, dir, k))

if __name__=="__main__":
    split_train_test(sys.argv[1], float(sys.argv[2]))