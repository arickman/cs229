import numpy as np
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(b): Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    logreg = LogisticRegression(LinearModel)
    # Train a logistic regression classifier
    logreg.fit(x_train, y_train)
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True) 
    y_eval.reshape((len(y_eval),1))
    prediction = logreg.predict(x_eval)
    np.savetxt(pred_path, prediction, delimiter=',')
    # Plot decision boundary on top of validation set
    # if (eval_path == '../data/ds1_valid.csv'):
    #     util.plot(x_eval, y_eval, logreg.theta, "output/DS1_prob1_b.png")
    # elif(eval_path == '../data/ds2_valid.csv'):
    #     util.plot(x_eval, y_eval, logreg.theta, "output/DS2_prob1_b.png")
    # Use np.savetxt to save predictions on eval set to pred_path
    # *** END CODE HERE ***


class LogisticRegression(LinearModel):
    """Logistic regression with Newton's Method as the solver."""

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        # m = len(y)
        # n = len(x[0])
        # #Construct H and the gradient of J and then update theta in each iteration of k. 
        # self.theta = np.zeros((n,1))
        # while(True):
        #     #given a theta, calculate H and grad, then perform the update separately. 
        #     H = np.zeros((n,n)) #initialize both to zeros
        #     grad = np.zeros((n,1))
        #     for k in range(m):
        #         x_k = np.array(x[k].reshape((n,1))) #vector
        #         y_k = y[k] #scalar


        #         dp = np.asscalar(np.dot(x_k, self.theta))

        #         g_k = (1 + np.exp(-dp))**(-1) #scalar 
        #         #H(i,j) = 1/m sum from 1 to m (xi * xj)g*1-g , H is (nxn)
        #         for i in range(n):
        #             grad[i] = grad[i] + x_k[i]*(g_k - y_k)
        #             for j in range(n):
        #                 H[i][j] = H[i][j] + x_k[i]*x_k[j]*g_k*(1-g_k)
        #     H = m**(-1)*H
        #     grad = m**(-1)*grad
        #     #Now we have the Hession and the gradient of J for theta_0. 
        #     theta_new = self.theta - (np.linalg.inv(H))*grad
        #     if (np.linalg.norm((theta_new - self.theta), ord=1) < self.eps): 
        #         self.theta = theta_new
        #         break
        #     else: self.theta = theta_new
        # *** END CODE HERE ***

        # *** START CODE HERE ***
        m = len(y)
        n = len(x[0])
        #Construct H and the gradient of J and then update theta in each iteration of k. 
        self.theta = np.zeros((n,1))
        while(True):
            #given a theta, calculate H and grad, then perform the update separately. 
            H = np.zeros((n,n),dtype=int)
            grad = np.zeros((n,1),dtype=int)
            for k in range(m):
                #in each iteration add another matrix to H and another vector to grad. 
                x_k = np.matrix(x[k].reshape((n,1))) #vector
                y_k = y[k] #scalar
                dp = np.sum(self.theta.T * x_k)
                g_k = (1 + np.exp(-dp))**(-1) #scalar 
                #H(i,j) = 1/m sum from 1 to m (xi * xj)g*1-g , H is (nxn)
                grad = grad + (g_k - y_k)* x_k
                H = H + g_k*(1-g_k)* np.dot(x_k,x_k.T)
            H = m**(-1)*H
            grad = m**(-1)*grad
            #Now we have the Hession and the gradient of J for given theta. 
            theta_new = self.theta - (np.linalg.inv(H))*grad
            if (np.linalg.norm((theta_new - self.theta), ord=1) < self.eps): 
                self.theta = theta_new
                break
            else: self.theta = theta_new
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
            dp = np.sum(x_k.T * self.theta)
            output[k] = (1 + np.exp(-dp))**(-1)
        return output
        # *** END CODE HERE ***
