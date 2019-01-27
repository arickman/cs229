import numpy as np
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(e): Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    # *** START CODE HERE ***
    # Train a GDA classifier
    gda = GDA(LinearModel)
    gda.fit(x_train, y_train)
    x_eval,y_eval = util.load_dataset(eval_path, add_intercept=True) #why add intercept?
    y_eval.reshape((len(y_eval),1))
    prediction = gda.predict(x_eval)
    np.savetxt(pred_path, prediction, delimiter=',')
    # Plot decision boundary on top of validation set
    # if (eval_path == '../data/ds1_valid.csv'):
    #     util.plot(x_eval, y_eval, gda.theta, "output/DS1_prob1_e.png")
    # elif(eval_path == '../data/ds2_valid.csv'):
    #     util.plot(x_eval, y_eval, gda.theta, "output/DS2_prob1_e.png")
    # *** END CODE HERE ***


class GDA(LinearModel):
    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).

        Returns:
            theta: GDA model parameters.
        """
        # *** START CODE HERE ***
        # Find phi, mu_0, mu_1, and sigma
        m = len(x)
        n = len(x[0])
        #Use numpy matrices instead!!!!!!

        phi = 0
        for i in range(m):
            if y[i] == 1: phi = phi + 1
        phi = m**(-1)*phi

        mu_0 = np.zeros((n,1))
        for i in range(m):
            if y[i] == 0: mu_0 = mu_0 + np.matrix(x[i].reshape((n,1))) #vector
        sum = 0
        for i in range(m):
            if y[i] == 0: sum += 1
        mu_0 = (sum)**(-1)* mu_0


        mu_1 = np.zeros((n,1))
        for i in range(m):
            if y[i] == 1: mu_1 = mu_1 + np.matrix(x[i].reshape((n,1))) #vector
        sum = 0
        for i in range(m):
            if y[i] == 1: sum += 1
        mu_1 = (sum)**(-1)* mu_1


        sigma = np.zeros((n,n))
        for i in range(m):
            if y[i] ==1: sigma += (np.matrix(x[i].reshape((n,1))) - mu_1)*(np.matrix(x[i].reshape((n,1))) - mu_1).T
            else: sigma += (np.matrix(x[i].reshape((n,1))) - mu_0)*(np.matrix(x[i].reshape((n,1))) - mu_0).T
        sigma = m**(-1)*sigma


        # Write theta in terms of the parameters
        theta_0 = 0.5 * (mu_0.T * np.linalg.inv(sigma) * mu_0 - mu_1.T * np.linalg.inv(sigma) * mu_1) - np.log((1-phi)/phi)
        self.theta = (mu_1.T * np.linalg.inv(sigma) - mu_0.T * np.linalg.inv(sigma)).T
        self.theta = np.matrix(np.insert(self.theta, 0, theta_0))
        self.theta = self.theta.T


        # print("phi")
        # print(phi)
        # print("mu")
        # print(mu_0)
        # print(np.shape(mu_0))
        # print("sigma")
        # print(sigma)
        # print(np.shape(sigma))
        # print("theta")
        # print(self.theta)
        # print(np.shape(self.theta))
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.
  
        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        m = len(x)
        n = len(x[0])
        output = np.zeros((m,1))
        for k in range(m):
            x_k = np.matrix(x[k].reshape((n,1))) #vector
            dp = np.sum(self.theta.T * x_k)
            output[k] = (1 + np.exp(-dp))**(-1)
        return output
        # *** END CODE HERE
