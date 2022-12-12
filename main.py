import numpy as np
from matplotlib import pyplot as plt
from utils import BayesianLinearRegression, polynomial_basis_functions, load_prior


def log_evidence(model: BayesianLinearRegression, X, y):
    """
    Calculate the log-evidence of some data under a given Bayesian linear regression model
    :param model: the BLR model whose evidence should be calculated
    :param X: the observed x values
    :param y: the observed responses (y values)
    :return: the log-evidence of the model on the observed data
    """
    # extract the variables of the prior distribution
    mu = model.mu
    sig = model.cov
    n = model.sig

    # extract the variables of the posterior distribution
    model.fit(X, y)
    map = model.fit_mu
    map_cov = model.fit_cov

    # calculate the log-evidence
    H = model.h(X)
    N = y.shape[0]
    p = H.shape[1]

    log_evidence_calc = 0.5 * np.log(np.linalg.det(map_cov) / np.linalg.det(sig)) \
                        - 0.5 * ((map - mu).T @ np.linalg.inv(sig) @ (map - mu)
                                 + (1/n)*(np.linalg.norm(y - H@map))
                                 + N*np.log(n)) \
                        - (p/2)*np.log(2*np.pi)
    return log_evidence_calc


def main():
    # ------------------------------------------------------ section 2.1
    # set up the response functions
    f1 = lambda x: x**2 - 1
    f2 = lambda x: -x**4 + 3*x**2 + 50*np.sin(x/6)
    f3 = lambda x: .5*x**6 - .75*x**4 + 2.75*x**2
    f4 = lambda x: 5 / (1 + np.exp(-4*x)) - (x - 2 > 0)*x
    f5 = lambda x: np.cos(x*4) + 4*np.abs(x - 2)
    functions = [f1, f2, f3, f4, f5]
    x = np.linspace(-3, 3, 500)

    # set up model parameters
    degrees = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    noise_var = .25
    alpha = 5

    # go over each response function and polynomial basis function
    for i, f in enumerate(functions):
        y = f(x) + np.sqrt(noise_var) * np.random.randn(len(x))
        evidences = []
        for j, d in enumerate(degrees):
            # set up model parameters
            pbf = polynomial_basis_functions(d)
            mean, cov = np.zeros(d + 1), np.eye(d + 1) * alpha

            # calculate evidence
            ev = log_evidence(BayesianLinearRegression(mean, cov, noise_var, pbf), x, y)
            # <your code here>
            evidences.append(ev)

        # plot evidence versus degree and predicted fit
        # <your code here>
        best_model_deg = degrees[np.argmax(evidences)]
        worse_model_deg = degrees[np.argmin(evidences)]

        # Plot evidence as a function of the degree
        plt.figure()
        plt.plot(degrees, evidences, lw=2)
        plt.title(f'Function {i+1},'
                  f' best model d={best_model_deg},'
                  f' worse model d={worse_model_deg}')
        plt.xlabel('Degree')
        plt.ylabel('Log-Evidence')
        plt.show()

        # Plot bayesian regression of the best and worse models according to evidence
        # TODO: Check, not sure if we need to predict and plot predictions or not
        plt.figure()
        for d in [best_model_deg, worse_model_deg]:
            pbf = polynomial_basis_functions(d)
            mean, cov = np.zeros(d + 1), np.eye(d + 1) * alpha
            blr = BayesianLinearRegression(mean, cov, noise_var, pbf)
            blr.fit(x, f(x))
            pred = blr.predict(x)
            std = blr.predict_std(x)

            plt.fill_between(x, pred - std, pred + std,
                             alpha=.5, label='confidence interval')
            label = 'best model' if d == best_model_deg else 'worse model'
            plt.plot(x, pred, lw=2, label=label)
        plt.scatter(x, f(x), c='blue', s=10, alpha=0.2)
        plt.legend()
        plt.show()

    # # ------------------------------------------------------ section 2.2
    # # load relevant data
    # nov16 = np.load('nov162020.npy')
    # hours = np.arange(0, 24, .5)
    # train = nov16[:len(nov16) // 2]
    # hours_train = hours[:len(nov16) // 2]
    #
    # # load prior parameters and set up basis functions
    # mu, cov = load_prior()
    # pbf = polynomial_basis_functions(7)
    #
    # noise_vars = np.linspace(.05, 2, 100)
    # evs = np.zeros(noise_vars.shape)
    # for i, n in enumerate(noise_vars):
    #     # calculate the evidence
    #     mdl = BayesianLinearRegression(mu, cov, n, pbf)
    #     ev = log_evidence(mdl, hours_train, train)
    #     # <your code here>
    #
    # # plot log-evidence versus amount of sample noise
    # # <your code here>


if __name__ == '__main__':
    main()



