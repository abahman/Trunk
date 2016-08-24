import scipy.optimize as optimize
import numpy as np

X = np.array([[1.020626, 1.013055], [0.989094, 1.059343]])
freq = 13.574380165289256
x_0 = [1., 1.]

def objective(b):
    def foo(r_log, freq):
        mu, sd = r_log.mean(), r_log.std()
        sd += 0.5 / freq
        return mu / sd * np.sqrt(freq)

    print(b)
    return -foo(np.log(np.maximum(np.dot(X - 1, b) + 1, 0.2)), freq=freq)

cons = ({'type': 'ineq', 'fun': lambda b: 2. - sum(b)},)
res = optimize.minimize(objective, x_0, bounds=[(0., 2.)]*len(x_0), constraints=cons, method='slsqp')
print(res)