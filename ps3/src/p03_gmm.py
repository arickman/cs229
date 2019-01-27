import matplotlib.pyplot as plt
import numpy as np
import os
import random
from collections import defaultdict

PLOT_COLORS = ['red', 'green', 'blue', 'orange']  # Colors for your plots
K = 4           # Number of Gaussians in the mixture model
NUM_TRIALS = 3  # Number of trials to run (can be adjusted for debugging)
UNLABELED = -1  # Cluster label for unlabeled data points (do not change)


def main(is_semi_supervised, trial_num):
    """Problem 3: EM for Gaussian Mixture Models (unsupervised and semi-supervised)"""
    print('Running {} EM algorithm...'
          .format('semi-supervised' if is_semi_supervised else 'unsupervised'))
 
    # Load dataset
    train_path = os.path.join('..', 'data', 'ds3_train.csv')
    x, z = load_gmm_dataset(train_path)
    x_tilde = None

    if is_semi_supervised:
        # Split into labeled and unlabeled examples
        labeled_idxs = (z != UNLABELED).squeeze()
        x_tilde = x[labeled_idxs, :]   # Labeled examples
        z = z[labeled_idxs, :]         # Corresponding labels
        x = x[~labeled_idxs, :]        # Unlabeled examples

    # *** START CODE HERE ***
    m = len(x)
    # (1) Initialize mu and sigma by splitting the m data points uniformly at random
    # into K groups, then calculating the sample mean and covariance for each group
    random_groups = defaultdict(list)
    for row in x:
        group = random.randint(0,K - 1)
        random_groups[group].append(row)

    mu = []
    sigma = []
    for key in random_groups:
        group = random_groups[key] #this needs to be a list of n vectors
        mu.append(np.mean(group, axis=0).tolist())
        my_mean = np.mean(group, axis=0).tolist()
        cov_sum = np.zeros((len(x[0]),len(x[0])))
        for elem in group:
            p = elem - my_mean
            cov_sum += np.outer(p,p.T)
        sigma.append((cov_sum/(len(group) - 1)).reshape((2,2)))
    np.array(sigma)
    np.array(mu)

    # (2) Initialize phi to place equal probability on each Gaussian
    # phi should be a numpy array of shape (K,)
    phi = np.ones((K,))
    prob = 1/K
    phi *= prob

    # (3) Initialize the w values to place equal probability on each Gaussian
    # w should be a numpy array of shape (m, K)
    w = np.ones((len(x), K))
    w *= 1/K

    # print(mu[0])
    # print(np.shape(mu[0]))
    # print(len(x[0]))
    #print(np.shape(mu))
    #print(np.shape(sigma))
    #print(phi)
    #print(w)
    # *** END CODE HERE ***

    if is_semi_supervised:
        w = run_semi_supervised_em(x, x_tilde, z, w, phi, mu, sigma)
    else:
        w = run_em(x, w, phi, mu, sigma)

    # Plot your predictions
    z_pred = np.zeros(m)
    if w is not None:  # Just a placeholder for the starter code
        for i in range(m):
            z_pred[i] = np.argmax(w[i])

    plot_gmm_preds(x, z_pred, is_semi_supervised, plot_id=trial_num)


def run_em(x, w, phi, mu, sigma):
    """Problem 3(d): EM Algorithm (unsupervised).

    See inline comments for instructions.

    Args:
        x: Design matrix of shape (m, n).
        w: Initial weight matrix of shape (m, k).
        phi: Initial mixture prior, of shape (k,).
        mu: Initial cluster means, list of k arrays of shape (n,).
        sigma: Initial cluster covariances, list of k arrays of shape (n, n).

    Returns:
        Updated weight matrix of shape (m, k) resulting from EM algorithm.
        More specifically, w[i, j] should contain the probability of
        example x^(i) belonging to the j-th Gaussian in the mixture.
    """
    # No need to change any of these parameters
    eps = 1e-3  # Convergence threshold
    max_iter = 1000

    # Stop when the absolute change in log-likelihood is < eps
    # See below for explanation of the convergence criterion
    it = 0
    ll = prev_ll = None
    while it < max_iter and (prev_ll is None or np.abs(ll - prev_ll) >= eps):
        pass  # Just a placeholder for the starter code
        # *** START CODE HERE
        # (1) E-step: Update your estimates in w
        prev_phi = phi
        prev_mu = mu
        prev_sigma = sigma
        prev_ll = ll
        if it > 0:
            for i in range(len(x)):
                #calculate the sum so as to not get confused later
                denom_i = 0
                for j in range(K):
                    q = (x[i] - mu[j])
                    A = np.linalg.inv(sigma[j])
                    prod1 = np.matmul(q.T,A)
                    prod2 = np.matmul(prod1,q)
                    denom_i += np.exp(-0.5*prod2)*phi[j]/(np.linalg.det(sigma[j]))
                for j in range(len(w[0])):
                    q = (x[i] - mu[j])
                    A = np.linalg.inv(sigma[j])
                    prod1 = np.matmul(q.T,A)
                    prod2 = np.matmul(prod1,q)
                    w[i,j] = (np.exp(-0.5*prod2)*phi[j]/((np.linalg.det(sigma[j]))))/denom_i
            
        # (2) M-step: Update the model parameters phi, mu, and sigma
        phi = np.zeros((len(phi),1))
        for i in range(len(w)):
            add = np.array(w[i].T)
            add = add.reshape((4,1))
            phi += add #should give a k length column vector
        phi /= len(w)


        mu = np.matmul(w.T,x)
        #now each row want to divide by corresponding m*phi[j]
        for i in range(K):
            for j in range(len(x[0])):
                mu[i,j] /= len(w)*phi[i] # is k by n

        #print(mu)


        for j in range(len(phi)):
            #Now create a matrix for each j
            mat_sum = np.zeros((len(x[0]),len(x[0])))
            denom = 0
            for i in range(len(w)):
                q = (x[i] - prev_mu[j])
                mat_sum += w[i,j]*np.outer(q,q.T)
                denom += w[i,j]
            sigma[j] = mat_sum/denom


        # (3) Compute the log-likelihood of the data to check for convergence.
        ll = 0
        for i in range(len(w)):
            for j in range(K):
                q = (x[i] - mu[j])
                A = np.linalg.inv(sigma[j])
                prod1 = np.matmul(q.T,A)
                prod2 = np.matmul(prod1,q)
                ll += (-0.5)*prod2 - (0.5)*(len(x[0]))*np.log(2*3.14159) - 0.5*np.log(np.linalg.det(sigma[j])) + np.log(phi[j])

        it += 1
        # print("prev_ll")
        # print(prev_ll)
        # print("ll")
        # print(ll)
        # By log-likelihood, we mean `ll = sum_x[log(sum_z[p(x|z) * p(z)])]`.
        # We define convergence by the first iteration where abs(ll - prev_ll) < eps.
        # Hint: For debugging, recall part (a). We showed that ll should be monotonically increasing.
        # *** END CODE HERE ***
    print(it)
    return w


def run_semi_supervised_em(x, x_tilde, z, w, phi, mu, sigma):
    """Problem 3(e): Semi-Supervised EM Algorithm.

    See inline comments for instructions.

    Args:
        x: Design matrix of unlabeled examples of shape (m, n).
        x_tilde: Design matrix of labeled examples of shape (m_tilde, n).
        z: Array of labels of shape (m_tilde, 1).
        w: Initial weight matrix of shape (m, k).
        phi: Initial mixture prior, of shape (k,).
        mu: Initial cluster means, list of k arrays of shape (n,).
        sigma: Initial cluster covariances, list of k arrays of shape (n, n).

    Returns:
        Updated weight matrix of shape (m, k) resulting from semi-supervised EM algorithm.
        More specifically, w[i, j] should contain the probability of
        example x^(i) belonging to the j-th Gaussian in the mixture.
    """
    # No need to change any of these parameters
    alpha = 20  # Weight for the labeled examples
    eps = 1e-3   # Convergence threshold
    max_iter = 1000

    # Stop when the absolute change in log-likelihood is < eps
    # See below for explanation of the convergence criterion
    it = 0
    ll = prev_ll = None
    while it < max_iter and (prev_ll is None or np.abs(ll - prev_ll) >= eps):
        pass  # Just a placeholder for the starter code
        # *** START CODE HERE ***
        # (1) E-step: Update your estimates in w
        prev_ll = ll
        prev_phi = phi
        prev_mu = mu
        prev_sigma = sigma
        if it > 0:
            #calculate w now
            for i in range(len(x)):
                #calculate the sum so as to not get confused later
                denom_i = 0
                for j in range(K):
                    q = (x[i] - mu[j])
                    A = np.linalg.inv(sigma[j])
                    prod1 = np.matmul(q.T,A)
                    prod2 = np.matmul(prod1,q)
                    denom_i += np.exp(-0.5*prod2)*phi[j]/(np.linalg.det(sigma[j]))
                for j in range(len(w[0])):
                    q = (x[i] - mu[j])
                    A = np.linalg.inv(sigma[j])
                    prod1 = np.matmul(q.T,A)
                    prod2 = np.matmul(prod1,q)
                    w[i,j] = (np.exp(-0.5*prod2)*phi[j]/((np.linalg.det(sigma[j]))))/denom_i

        # print(w)
        # print(it)

        # (2) M-step: Update the model parameters phi, mu, and sigma
        phi = np.zeros((len(phi),1))
        for i in range(len(w)):
            add = np.array(w[i].T)
            add = add.reshape((4,1))
            phi += add #should give a k length column vector

        #Now change each phi_j and then divide by the constants at the very end
        for j in range(K):
            for i in range(len(x_tilde)):
                if (int(z[i]) == j) : phi[j] += alpha

        phi /= (len(x) + alpha*len(x_tilde))

        #print(phi) barely change after a few iterations


        mu = np.matmul(w.T,x)
        for l in range(K):
            for i in range(len(x_tilde)):
                if (int(z[i]) == l): mu[l] += alpha*x_tilde[i]


        denom1 = np.zeros((K,1))
        for i in range(len(x)):
            for l in range(K):
                denom1[l] += w[i,l]

        denom2 = np.zeros((K,1))
        for i in range(len(x_tilde)):
            for l in range(K):
                if int(z[i]) == l: denom2[l] += alpha

        denom = denom1 + denom2

        for i in range(len(x[0])):
            for j in range(K):
                mu[j,i] /= denom[j]

        #print(mu) barely change

        for j in range(K):
            #Now create a matrix for each j
            mat_sum = np.zeros((len(x[0]),len(x[0])))
            for i in range(len(x)): 
                q = (x[i] - prev_mu[j])
                mat_sum += w[i,j]*np.outer(q,q.T)
            sigma[j] = mat_sum


        for l in range(K):
            mat_sum_squiggle = np.zeros((len(x[0]),len(x[0])))
            for i in range(len(x_tilde)):
                if (int(z[i]) == l):
                    q_squiggle = (x_tilde[i] - prev_mu[l])
                    mat_sum_squiggle += alpha*np.outer(q_squiggle, q_squiggle.T)
            sigma[l] += mat_sum_squiggle



        denom1 = np.zeros((K,1))
        for i in range(len(x)):
            for l in range(K):
                denom1[l] += w[i,l]

        #print(denom1[2])

        denom2 = np.zeros((K,1))
        for i in range(len(x_tilde)):
            for l in range(K):
                if int(z[i]) == l: denom2[l] += alpha


        denom = denom1 + denom2

        for j in range(K):
            sigma[j] = sigma[j]/denom[j]  
            

        # (3) Compute the log-likelihood of the data to check for convergence.
        ll_unsup = 0
        for i in range(len(w)):
            for j in range(K):
                q = (x[i] - mu[j])
                A = np.linalg.inv(sigma[j])
                prod1 = np.matmul(q.T,A)
                prod2 = np.matmul(prod1,q)
                ll_unsup += (-0.5)*prod2 - (0.5)*(len(x[0]))*np.log(2*3.14159) - 0.5*np.log(np.linalg.det(sigma[j])) + np.log(phi[j])

        ll_sup = 0
        for i in range(len(x_tilde)):
            q = x_tilde[i] - mu[int(z[i])]
            B = np.linalg.inv(sigma[int(z[i])])
            prod1 = np.matmul(q.T,B)
            prod2 = np.matmul(prod1,q)
            ll_sup += (-0.5)*prod2 - (0.5)*(len(x_tilde[0]))*np.log(2*3.14159) - 0.5*np.log(np.linalg.det(sigma[int(z[i])])) + np.log(phi[int(z[i])])


        ll = ll_unsup + alpha* ll_sup

        it += 1
        # print("prev_ll")
        # print(prev_ll)
        # print("ll")
        # print(ll)
        # Hint: Make sure to include alpha in your calculation of ll.
        # Hint: For debugging, recall part (a). We showed that ll should be monotonically increasing.
        # *** END CODE HERE ***
    print(it)
    return w


# *** START CODE HERE ***
# Helper functions
# *** END CODE HERE ***


def plot_gmm_preds(x, z, with_supervision, plot_id):
    """Plot GMM predictions on a 2D dataset `x` with labels `z`.

    Write to the output directory, including `plot_id`
    in the name, and appending 'ss' if the GMM had supervision.

    NOTE: You do not need to edit this function.
    """
    plt.figure(figsize=(12, 8))
    plt.title('{} GMM Predictions'.format('Semi-supervised' if with_supervision else 'Unsupervised'))
    plt.xlabel('x_1')
    plt.ylabel('x_2')

    for x_1, x_2, z_ in zip(x[:, 0], x[:, 1], z):
        color = 'gray' if z_ < 0 else PLOT_COLORS[int(z_)]
        alpha = 0.25 if z_ < 0 else 0.75
        plt.scatter(x_1, x_2, marker='.', c=color, alpha=alpha)

    file_name = 'p03_pred{}_{}.png'.format('_ss' if with_supervision else '', plot_id)
    save_path = os.path.join('output', file_name)
    plt.savefig(save_path)


def load_gmm_dataset(csv_path):
    """Load dataset for Gaussian Mixture Model (problem 3).

    Args:
         csv_path: Path to CSV file containing dataset.

    Returns:
        x: NumPy array shape (m, n)
        z: NumPy array shape (m, 1)

    NOTE: You do not need to edit this function.
    """

    # Load headers
    with open(csv_path, 'r') as csv_fh:
        headers = csv_fh.readline().strip().split(',')

    # Load features and labels
    x_cols = [i for i in range(len(headers)) if headers[i].startswith('x')]
    z_cols = [i for i in range(len(headers)) if headers[i] == 'z']

    x = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=x_cols, dtype=float)
    z = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=z_cols, dtype=float)

    if z.ndim == 1:
        z = np.expand_dims(z, axis=-1)

    return x, z


if __name__ == '__main__':
    np.random.seed(229)
    # Run NUM_TRIALS trials to see how different initializations
    # affect the final predictions with and without supervision
    for t in range(NUM_TRIALS):
        main(is_semi_supervised=False, trial_num=t)

        # *** START CODE HERE ***
        # Once you've implemented the semi-supervised version,
        # uncomment the following line.
        # You do not need to add any other lines in this code block.
        main(is_semi_supervised=True, trial_num=t)
        # *** END CODE HERE ***
