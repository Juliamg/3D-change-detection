import prep_data
import src_algorithm

def main():
    """

    Main entrypoint

    """
    # TODO: find best pca dims & find false positives and try with a better suited dataset?

    train_path = '/Users/juliagraham/IT/MasterThesis2021/dmd-and-cs-for-3D-object-detection/train2/'
    test_path = '/Users/juliagraham/IT/MasterThesis2021/dmd-and-cs-for-3D-object-detection/testReal/'
    shape = (10,10)
    n_features = 30

    ### Feature selection option ###
    #options = {'feature_selection': 'downsampling', 'dims': shape} # feature selection can be wither pca (eigenfaces) or downsampling
    options = {'feature_selection': 'pca', 'dims': 40}
    #options = {'feature_selection': 'random', 'dims': n_features}

    TrainSet, TestSet = prep_data.prep_train_test(train_path, test_path, options)

    print(TrainSet['X'].shape, TestSet['X'].shape)

    ### Parameters for src algorithm ###
    sigma = 0.001
    threshold = 0.5

    print(f"Running SRC classifier with threshold: {threshold}, and feature selection: {options['feature_selection']}")

    rec_rate, results = src_algorithm.src_algorithm(TrainSet, TestSet, sigma, threshold)

    print("False positives: ", results['false_pos'])

    #for f in results['failed_imgs']:
    #    print(f)

if __name__ == '__main__':
    main()