import numpy as np
import seaborn as sns
import cvxpy as cvx
import matplotlib.pyplot as plt

# Calculates the SCI index of the input image and deems it valid if coefficients are sparse according to user threshold
def accept_image(class_norms, num_classes, x_opt, threshold: int) -> bool:
    SCI = (num_classes*max(class_norms)/np.linalg.norm(x_opt)-1)/(num_classes-1)
    print("SCI score: ", SCI)
    return SCI >= threshold

# Returns predicted class label according to the class with minimum residual
def calc_class_residuals(y, D, x_opt, train_labels, classes, num_classes):
    residuals = np.zeros((num_classes))
    class_norms = []

    for j in range(num_classes):
        idx = np.where(classes[j] == train_labels)
        last_index = np.size(idx) - 1
        class_vec = np.zeros(x_opt.shape)

        residuals[j] = np.linalg.norm(y - D[:, idx[0][0]:idx[0][last_index] + 1].dot(x_opt[idx]))
        class_vec[idx] = x_opt[idx]
        class_norms.append(np.linalg.norm(class_vec)) # To be used for calculating SCI index

    return residuals, class_norms


def src_algorithm(TrainSet, TestSet, sigma, threshold):
    classes = np.unique(TrainSet['y'])
    num_classes = len(classes)
    num_test_samples = len(list(TestSet['y']))
    identity = []
    failed_imgs = []

    for i in range(num_test_samples):
        y = TestSet['X'][:,i]
        D = TrainSet['X'] # dictionary matrix
        m, n = D.shape

        # do l1 optimization
        x = cvx.Variable(n)
        objective = cvx.Minimize(cvx.norm(x, 1))
        #constraints = [D@x == y]
        z = D @ x - y
        constraints = [cvx.norm(z, 2) <= sigma]
        prob = cvx.Problem(objective, constraints)
        prob.solve(solver=cvx.ECOS) # runs economy optimizer by default
        x_opt = np.array(x.value).squeeze()

        if len(x_opt) is None:
            print("Infeasible solution for ")
            continue

        #    pred_label = class_res_to_label(y, D, x_opt, TrainSet['y'], num_classes)
        class_residuals, class_norms = calc_class_residuals(y, D, x_opt, TrainSet['y'], classes, num_classes)
        if accept_image(class_norms, num_classes, x_opt, threshold):
            pred_label = classes[np.argmin(class_residuals)]
        else:
            pred_label = None

        print("RECOGNIZED AS: ", pred_label, "TRUE: ", TestSet['y'][i], "\nFilename: ",
        TestSet['files'][i])
        identity.append(pred_label)

        x_ax = np.arange(len(x_opt))
        true = TestSet['y'][i]

        plt.figure(figsize=(5, 3))
        plt.title(f'Coefficients vector for: {true}, Predicted: {pred_label}')
        markerline, stemlines, baseline = plt.stem(x_ax, x_opt)
        plt.setp(stemlines, 'linewidth', 0.7)
        plt.setp(markerline, markersize=2.5)
        plt.setp(markerline, 'linewidth', 1)
        plt.pause(3)
        plt.close()

        # graph = sns.barplot(x=classes, y=residuals)
        # plt.title('Residuals of each class')
        # graph.axhline(thresh_certainty, color='r', label='threshold')
        # graph.axhline(certainty, color='g', label='certainty')
        #
        # plt.show(block=False)
        # plt.legend()
        # plt.pause(2)
        # plt.close()

    ### Calculate accuracy ###
    correct_num = [i for i in range(len(identity)) if identity[i] == TestSet['y'][i]]
    rec_rate = len(correct_num)/num_test_samples * 100
    print(f"Predicted correctly: {len(correct_num)} out of {np.size(TestSet['y'])} with recognition rate: {rec_rate} %")

    return rec_rate, failed_imgs