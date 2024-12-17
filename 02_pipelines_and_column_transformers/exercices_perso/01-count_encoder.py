### Réponse a la question 1  : creation d'un count encoder : 

from collections import Counter
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class CountEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X): 
        n_features = X.shape[1]
        # l'idee c'est que pour chacune des lines on compte
        # le nombre de fois qu'apparait une des valeurs 
        Counter_per_feature = []
        for k in range(0, n_features ) : 
                Counter_per_feature.append(Counter(X[:, k]))
                # on a ajouté les dictionnaires pour chque feature

        self.counters_ = Counter_per_feature
        return self

    def transform(self, X) :
        X_t = X.copy()
        for x, counter in zip(X_t.T, self.counters_):
            # Uses numpy broadcasting
            idx = np.nonzero(list(counter.keys()) == x[:, None])[1]
            x[:] = np.asarray(list(counter.values()))[idx]
        return X_t


#### TEST / 

X = np.array([
    [0, 2],
    [1, 3],
    [1, 1],
    [1, 1],
])
ce = CountEncoder()
print(ce.fit_transform(X))

