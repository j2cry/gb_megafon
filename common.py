import ctypes
from sklearn.utils import parallel_backend
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV


def trim_memory() -> int:
     libc = ctypes.CDLL("libc.so.6")
     return libc.malloc_trim(0)


def select_and_sort(df, mask, by):
    return df[mask].sort_values(by)


def cv_fit(estimator, X, y, *, cv, pipe=None, scorer=f1_score):
    """ Fit estimator with cross-validation
    :param estimator: estimator to be fitted
    :param X: prepared train data
    :param y: true values
    :param cv: cross-validator
    :param pipe: featuring pipeline. It is applied separately to train and valid data
    :param scorer: scoring function
    """
    metrics = []
    models = []

    for train_index, valid_index in cv.split(y):
        # push train/valid dataframes to the cluster
        train_df = X.iloc[train_index]
        valid_df = X.iloc[valid_index]
        # fit and apply featuring pipeline
        if pipe is not None:
            featuring = pipe.fit(train_df, y)
            X_train = featuring.transform(train_df)
            X_valid = featuring.transform(valid_df)
        else:
            X_train = train_df
            X_valid = valid_df
        # exctract targets and push them to the cluster
        y_train = y.iloc[train_index]
        y_valid = y.iloc[valid_index]

        # with parallel_backend('threading'):
        #     model = estimator.fit(X_train, y_train)
        model = estimator.fit(X_train, y_train)

        # predicts & metrics
        prediction = model.predict(X_valid)
        score = f1_score(y_valid, prediction, average='macro')
        # append step result
        models.append(model)
        metrics.append(score)
    
    avg = sum(metrics) / len(metrics)
    return avg, metrics, models


def cv_compare(estimators, X, y, *, grids, cv, pipe=None, scorer=f1_score, logger=None):
    """ Fit estimators with GridSearch best params and compare results on CV fit
    :param estimator: list of estimator to be fitted
    :param X: prepared train data
    :param y: true values
    :param grids: parameters collections list (for each estimator) for GridSearch
    :param cv: cross-validator
    :param pipe: featuring pipeline. It is applied separately to train and valid data
    :param scorer: scoring function
    :param logger: use given logger (INFO messages only)
    """
    # init scorer for GridSearch
    def grid_scorer(estimator, X_test, y_test):
        pred = estimator.predict(X_test)
        return scorer(y_test, pred)
    
    result = {
        'model': [],
        'GS score': [],
        'CV score': [],
    }
    # GridSearch for selected estimator
    models = []
    for est, params in zip(estimators, grids):
        gscv = GridSearchCV(est, params, cv=cv, scoring=grid_scorer)
        gscv.fit(X, y)

        # CV fit with best params
        avg, _, _ = cv_fit(est, X, y, cv=cv, scorer=scorer)

        # collect results
        result['model'].append(est.__class__.__name__)
        result['GS score'].append(gscv.best_score_)
        result['CV score'].append(avg)
        models.append(gscv.best_estimator_)
        logger.info(f'GridSearch: model={est.__class__.__name__}, params={gscv.best_params_}, GS score={gscv.best_score_}, CV score={avg}')
    return result, models
