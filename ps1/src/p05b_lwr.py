import matplotlib.pyplot as plt
import numpy as np
import util

from linear_model import LinearModel


def main(tau, train_path, eval_path):
    """Problem 5(b): Locally weighted regression (LWR)

    Args:
        tau: Bandwidth parameter for LWR.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
    """ 
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    # *** START CODE HERE ***
    lw = LocallyWeightedLinearRegression(tau)
    # Fit a LWR model
    lw.fit(x_train, y_train)
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=False) 
    y_eval.reshape((len(y_eval),1))
    prediction = lw.predict(x_eval)
    sum = 0
    for i in range(len(prediction)):
        sum += (prediction[i] - y_eval.T[i])**(2)
    mse = (len(prediction))**(-1) * sum
    print(mse)
    # Get MSE value on the validation set
    # Plot validation predictions on top of training set
    # No need to save anything
    # Plot data
    # plt.scatter(x_train, y_train.T, marker='x', color = 'blue')
    # plt.scatter(x_eval, prediction, marker='o', color = 'red')
    # plt.show()
    # *** END CODE HERE ***


class LocallyWeightedLinearRegression(LinearModel):
    def __init__(self, tau):
        super(LocallyWeightedLinearRegression, self).__init__()
        self.tau = tau
        self.x = None
        self.y = None

    def fit(self, x, y):
        """Fit LWR by saving the training set.

        """
        # *** START CODE HERE ***
        self.x = x #training x
        self.y = y #training y
        # *** END CODE HERE ***

    def predict(self, x):
        """Make predictions given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        #Will input the training set
        #First need to create a W for each point in the training set, from the x in the fit function, self.x
        train_points = self.x
        m_train = len(self.x)
        n_train = len(self.x[0])
        m_test = len(x)
        n_test = len(x[0])
        output = np.zeros((m_test,1))
        for k in range(m_test): #one matrix for each point in the test set
            x_k = np.matrix(x[k]) #vector, test set point xk
            #now we create a W matrix for each point in the training set
            W = np.zeros((m_train,m_train))
            for i in range(m_train): #all points in the training point
                x_i = train_points[i] #gives a vector 
                diff = (np.linalg.norm(x_i - x_k, ord = 2))**2
                w_i = 0.5*np.exp(-diff/(2 * self.tau**2))
                W[i][i] = w_i
            #now we have a W for this training point, so we can get a theta vector.
            self.x = np.matrix(self.x)
            self.y = np.matrix(self.y)
            W = np.matrix(W)
            self.theta = np.linalg.inv(self.x.T * W * self.x) * self.x.T * W * self.y.T
            hyp = np.sum(x_k.T * self.theta)
            output[k] = hyp
        return output
        # *** END CODE HERE ***
