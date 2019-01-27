import numpy as np
import util

from linear_model import LinearModel


def main(lr, train_path, eval_path, pred_path):
    """Problem 3(d): Poisson regression with gradient ascent.

    Args:
        lr: Learning rate for gradient ascent.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    # *** START CODE HERE ***
    # Fit a Poisson Regression model
    preg = PoissonRegression(step_size = lr)
    # Train a logistic regression classifier
    preg.fit(x_train, y_train)
    # Run on the validation set, and use np.savetxt to save outputs to pred_path
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=False) 
    y_eval.reshape((len(y_eval),1))
    prediction = preg.predict(x_eval)
    np.savetxt(pred_path, prediction, delimiter=',')
    # *** END CODE HERE ***


class PoissonRegression(LinearModel):

    def fit(self, x, y):
        """Run gradient ascent to maximize likelihood for Poisson regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***



        m = len(x)
        n = len(x[0])
        #self.step_size = lr
        #lr = self.step_size
        self.theta = np.zeros((n,1))
        iter = 0
        while (iter < self.max_iter):
            iter += 1
            #loop through training set
            theta_new = np.matrix(np.zeros((n,1)))
            theta_new.reshape((n,1))
            update = 0
            for i in range(m):
                x_i = np.matrix(x[i].reshape((n,1))) #vector
                dp_i = np.sum(self.theta.T * x_i)
                g_i = np.exp(dp_i)
                diff = y[i] - g_i
                factor = self.step_size * diff
                #create theta
                update += factor * x_i
            update = (m)**(-1)*update
            theta_new = self.theta + update
            if (np.linalg.norm((theta_new - self.theta), ord=1) < self.eps): 
                self.theta = theta_new
                break
            else: self.theta = theta_new
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Floating-point prediction for each input, shape (m,).
        """
        # *** START CODE HERE ***
        #given an x, predict the expectation value of y, which will be lambda. 
        #lambda = exp(dp)
        m = len(x)
        n = len(x[0])
        output = np.zeros((m,1))
        for k in range(m):
            x_k = np.matrix(x[k].reshape((n,1))) #vector
            dp = np.sum(x_k.T * self.theta)
            output[k] = np.exp(dp)
        return output
        # *** END CODE HERE ***
