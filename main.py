import prep_data
import src_algorithm

def main():
    """

    Main entrypoint

    """

    train_path = '/Users/juliagraham/IT/MasterThesis2021/dmd-and-cs-for-3D-object-detection/train/'
    test_path = '/Users/juliagraham/IT/MasterThesis2021/dmd-and-cs-for-3D-object-detection/test/'
    shape = (7,7)

    ### Feature selection option ###
    #options = {'feature_selection': 'downsampling', 'dims': shape} # feature selection can be wither pca (eigenfaces) or downsampling
    options = {'feature_selection': 'pca', 'dims': 40}

    TrainSet, TestSet = prep_data.prep_train_test(train_path, test_path, options)

    print(TrainSet['X'].shape, TestSet['X'].shape)

    ### Parameters for src algorithm ###
    sigma = 0.001
    threshold = 0.1

    print(f"Running SRC classifier with threshold: {threshold}, and feature selection: {options['feature_selection']}")

    rec_rate, failed = src_algorithm.src_algorithm(TrainSet, TestSet, sigma, threshold)

    #for f in failed:
    #    print(f)

if __name__ == '__main__':
    main()