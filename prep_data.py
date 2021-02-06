import numpy as np
import os
import cv2
import random
import argparse
import time

from sklearn.decomposition import PCA


IGNORE_FILES = ['.DS_Store']
prep_dataset = True

def normalize_data_column(img_matrix):
    normalized_img = img_matrix / np.sqrt(np.sum(img_matrix ** 2))
    return normalized_img

def pca_dim_reduction(train_imgs, test_imgs, n_features):
    img_matrix_train = train_imgs.transpose()
    img_matrix_test = test_imgs.transpose()
    pca = PCA(n_components=n_features, svd_solver='randomized', whiten=True).fit(img_matrix_train)
    pca.fit(img_matrix_train)
    resized_matrix_train = pca.transform(img_matrix_train)
    resized_matrix_test = pca.transform(img_matrix_test)
    resized_matrix_train = resized_matrix_train.transpose()
    resized_matrix_test = resized_matrix_test.transpose()

    return resized_matrix_train, resized_matrix_test

def prep_train_test(train_path, test_path, options: dict):
    init_data_matrix = True
    TrainSet = {}
    class_label_train = []
    TestSet = {}
    class_label_test = []
    test_file = []

    dims = options['dims'] #either a tuple for downsampling or an integer for pca

    for folder in os.listdir(train_path):
        init_class_matrix = True
        if folder in IGNORE_FILES:
            continue
        class_folder = train_path + folder + os.sep
        class_vector = os.listdir(class_folder)

        for img_file in class_vector:
            if img_file in IGNORE_FILES:
                continue
            class_label_train.append(folder)
            img_path = train_path + folder + os.sep + img_file # absolute path to image
            X_orig = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if options['feature_selection'] == 'downsampling':
                X = cv2.resize(X_orig, dims, interpolation = cv2.INTER_AREA)

            else:
                X = X_orig

            X = X.reshape(-1, 1)
            X = normalize_data_column(X)

            if init_class_matrix:
                D_c = X # initialize data matrix
                init_class_matrix = False
            else:
                D_c = np.hstack((D_c, X))

        if init_data_matrix: # this will run the first time
            D_train = D_c
            init_data_matrix = False
        else:
            D_train = np.hstack((D_train, D_c))

    # Now process test data
    init_data_matrix = True

    for folder in os.listdir(test_path):
        init_class_matrix = True
        if folder in IGNORE_FILES:
            continue
        class_folder = test_path + folder + os.sep
        class_vector = os.listdir(class_folder)

        for img_file in class_vector:
            if img_file in IGNORE_FILES:
                continue
            class_label_test.append(folder)
            img_path = test_path + folder + os.sep + img_file  # absolute path to image
            test_file.append(img_path)
            X_orig = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if options['feature_selection'] == 'downsampling':
                X = cv2.resize(X_orig, dims, interpolation=cv2.INTER_AREA)

            else:
                X = X_orig

            X = X.reshape(-1, 1)
            X = normalize_data_column(X)

            if init_class_matrix:
                D_c = X  # initialize data matrix
                init_class_matrix = False
            else:
                D_c = np.hstack((D_c, X))

        if init_data_matrix:  # this will run the first time
            D_test = D_c
            init_data_matrix = False
        else:
            D_test = np.hstack((D_test, D_c))

    if options['feature_selection'] == 'pca':
        D_train, D_test = pca_dim_reduction(D_train, D_test, dims)

    TrainSet['X'] = D_train
    TrainSet['y'] = np.array(class_label_train)
    TestSet['X'] = D_test
    TestSet['y'] = np.array(class_label_test)
    TestSet['files'] = test_file

    return TrainSet, TestSet