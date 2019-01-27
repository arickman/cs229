import matplotlib.pyplot as plt
import numpy as np
import util

from p05b_lwr import LocallyWeightedLinearRegression


def main(tau_values, train_path, valid_path, test_path, pred_path):
    """Problem 5(b): Tune the bandwidth paramater tau for LWR.

    Args:
        tau_values: List of tau values to try.
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)
    x_valid, y_valid = util.load_dataset(valid_path, add_intercept=False)

    # *** START CODE HERE ***
    # # Search tau_values for the best tau (lowest MSE on the validation set)
    mse = 10000000
    tau_opt = 0
    for tau in tau_values:
        lw = LocallyWeightedLinearRegression(tau)
        lw.fit(x_train, y_train)
        prediction = lw.predict(x_valid)
        sum = 0
        for i in range(len(prediction)):
            sum += (prediction[i] - y_valid.T[i])**(2)
        mse_new = (len(prediction))**(-1) * sum
        if (mse_new < mse): 
            mse = mse_new
            tau_opt = tau
        # Plot data
        #plt.scatter(x_train, y_train.T, marker='x', color = 'blue')
        #plt.scatter(x_valid, prediction, marker='o', color = 'red')
        #plt.show()
    print(tau_opt)
    print(mse)
    #have optimal tau now, so can do fit,predict with that tau.
    # Fit a LWR model with the best tau value
    lw = LocallyWeightedLinearRegression(tau_opt)
    lw.fit(x_train, y_train)
    # Run on the test set to get the MSE value
    x_test, y_test = util.load_dataset(test_path, add_intercept=False)
    prediction = lw.predict(x_test)
    sum = 0
    for i in range(len(prediction)):
        sum += (prediction[i] - y_test.T[i])**(2)
        mse_final = (len(prediction))**(-1) * sum
    print(mse_final)
    # Save test set predictions to pred_path
    np.savetxt(pred_path, prediction, delimiter=',')
    # *** END CODE HERE ***
