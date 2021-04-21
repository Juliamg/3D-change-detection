import numpy as np
import seaborn as sns
import cvxpy as cvx
from PIL import Image
import matplotlib.pyplot as plt

# Calculates the SCI index of the input image and deems it valid if coefficients are sparse according to user threshold
def accept_image(class_norms, num_classes, x_opt, threshold: int):
    SCI = (num_classes*max(class_norms)/np.linalg.norm(x_opt)-1)/(num_classes-1)
    print("SCI score: ", SCI)
    return SCI >= threshold, SCI

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
    results = {}
    results['coeff_vecs'] = []
    results['sci_scores'] = []
    results['pred'] = []
    results['residuals'] = []
    results['failed_imgs'] = []

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
        results['coeff_vecs'].append(x_opt)

        try:
            len(x_opt)
        except:
            print("Infeasible solution.....")
            breakpoint()
            continue

        class_residuals, class_norms = calc_class_residuals(y, D, x_opt, TrainSet['y'], classes, num_classes)
        accept, SCI = accept_image(class_norms, num_classes, x_opt, threshold)
        print("------------ New prediction ------------")
        if accept:
            pred_label = classes[np.argmin(class_residuals)]
        else:
            pred_label = None
            print("Failed img: ", TestSet['files'][i])
            #img = Image.open(TestSet['files'][i])
            #img.show()
            results['failed_imgs'].append(TestSet['files'][i])

        results['sci_scores'].append(SCI)
        results['pred'].append(pred_label)
        results['residuals'].append(class_residuals)
        print("RECOGNIZED AS: ", pred_label, "TRUE: ", TestSet['y'][i])
        identity.append(pred_label)

    ### Calculate accuracy ###
    correct_num = [i for i in range(len(identity)) if identity[i] == TestSet['y'][i]]
    false_pos = [i for i in range(len(identity)) if identity[i] != TestSet['y'][i] and identity[i] is not None]
    results['false_pos'] = len(false_pos)
    rec_rate = len(correct_num)/num_test_samples * 100

    print(f"Predicted correctly: {len(correct_num)} out of {np.size(TestSet['y'])} with recognition rate: {rec_rate} %")


    return rec_rate, results