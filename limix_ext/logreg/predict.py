import numpy as np


def predict(y, covariate_train, G_train, covariate_test, G_test):
    from sklearn.linear_model import LogisticRegression

    G_train = np.hstack([covariate_train, G_train])
    G_test = np.hstack([covariate_test, G_test])

    lr = LogisticRegression()
    model = lr.fit(G_train, y)
    p = model.predict_proba(G_test)

    prob = dict()
    prob[0] = p[:,0]
    prob[1] = p[:,1]
    return prob
