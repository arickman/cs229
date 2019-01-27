import numpy as np
import util

from p01b_logreg import LogisticRegression

# Character to replace with sub-problem letter in plot_path/pred_path
WILDCARD = 'X'


def main(train_path, valid_path, test_path, pred_path):
    """Problem 2: Logistic regression for incomplete, positive-only labels.

    Run under the following conditions:
        1. on y-labels,
        2. on t-labels,
        3. on t-labels with correction factor alpha.

    Args:
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    pred_path_c = pred_path.replace(WILDCARD, 'c')
    pred_path_d = pred_path.replace(WILDCARD, 'd')
    pred_path_e = pred_path.replace(WILDCARD, 'e')

    # *** START CODE HERE ***
    # Part (c): Train and test on true labels
    #make x train out of the first two columns, and t train out of the fourth
    logreg = LogisticRegression()
    x_train, t_train = util.load_dataset(train_path, label_col='t',add_intercept=True)
    logreg.fit(x_train, t_train)
    x_test, y_test = util.load_dataset(test_path, label_col='t', add_intercept=True) 
    y_test.reshape((len(y_test),1))
    predictiont = logreg.predict(x_test)
    np.savetxt(pred_path_c, predictiont, delimiter=',')
    #plot
    x_test, t_test = util.load_dataset(test_path, label_col='t', add_intercept=True) 
    #util.plot(x_test, t_test, logreg.theta, "output/prob2e_c.png")
    # Part (d): Train on y-labels and test on true labels
    x_train, y_train = util.load_dataset(train_path, label_col='y',add_intercept=True)
    y_train.reshape((len(y_train),1))
    logreg.fit(x_train, y_train)
    x_test, y_test = util.load_dataset(test_path, label_col='t', add_intercept=True) 
    y_test.reshape((len(y_test),1))
    predictiony = logreg.predict(x_test)
    np.savetxt(pred_path_d, predictiony, delimiter=',')
    #plot
    x_test, t_test = util.load_dataset(test_path, label_col='t', add_intercept=True) 
    #util.plot(x_test, t_test, logreg.theta, "output/prob2e_d.png")
    # Part (e): Apply correction factor using validation set and test on true labels
    x_valid,y_valid = util.load_dataset(valid_path, label_col='y', add_intercept=True)
    total_length = len(x_valid)
    alpha = 0
    n = len(x_valid[0])
    true_length = 0
    for i in range(total_length):
        if y_valid[i] == 1:
            true_length += 1
            x_valid_i = np.matrix(x_valid[i].reshape((n,1))) #vector
            dp = np.sum(x_valid_i.T * logreg.theta)
            alpha  += (1 + np.exp(-dp))**(-1)
    alpha = (true_length)**(-1)*alpha
    print(alpha)

    x_test, t_test = util.load_dataset(test_path, label_col='t', add_intercept=True) 
    y_test.reshape((len(y_test),1))
    predictiony = (alpha)**(-1)*logreg.predict(x_test)
    np.savetxt(pred_path_e, predictiony, delimiter=',')


    #plot
    x_test, t_test = util.load_dataset(test_path, label_col='t', add_intercept=True) 
    #util.plot(x_test, t_test, logreg.theta, "output/prob2e_e.png", correction = alpha)
    # Plot and use np.savetxt to save outputs

    # *** END CODER HERE
