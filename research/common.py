import ctypes
from sklearn.utils import parallel_backend
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline


def trim_memory() -> int:
     libc = ctypes.CDLL("libc.so.6")
     return libc.malloc_trim(0)


def select_and_sort(df, mask, by):
    return df[mask].sort_values(by)


def cv_fit(estimator, X, y, *, cv, pipe=None, scorer=f1_score, logger=None, prefix='[FIT]'):
    """ Fit estimator with cross-validation
    :param estimator: estimator to be fitted
    :param X: prepared train data
    :param y: true values
    :param cv: cross-validator
    :param pipe: featuring pipeline. It is applied separately to train and valid data
    :param scorer: scoring function
    :param logger: logging context
    """
    metrics = []
    models = []

    for train_index, valid_index in cv.split(y):
        # apply featuring pipeline
        X_train = pipe.fit_transform(X.iloc[train_index], y) if pipe is not None else X.iloc[train_index]
        X_valid = pipe.transform(X.iloc[valid_index]) if pipe is not None else X.iloc[valid_index]
        # exctract targets and push them to the cluster
        y_train = y.iloc[train_index]
        y_valid = y.iloc[valid_index]

        # with parallel_backend('threading'):
        #     model = estimator.fit(X_train, y_train)
        model = estimator.fit(X_train, y_train)

        # metrics
        score = scorer(y_valid, model.predict(X_valid))
        # append step result
        models.append(model)
        metrics.append(score)

    avg = sum(metrics) / len(metrics)
    msg = f'{prefix} {estimator.__class__.__name__}: {avg}'
    if logger is not None:
        logger.info(msg)
    print(msg)
    return avg, metrics, models


def whole_fit(estimator, X, y, *, pipe=None, scorer=f1_score, logger=None, prefix='[FIT]'):
    """ Fit estimator on whole data
    :param estimator: estimator to be fitted
    :param X: prepared train data
    :param y: true values
    :param pipe: featuring pipeline. It is applied separately to train and valid data
    :param scorer: scoring function
    :param logger: logging context
    """
    # extend pipeline if it is specified
    model = make_pipeline(pipe, estimator) if pipe is not None else estimator
    # fit model
    model.fit(X, y)
    score = scorer(y, model.predict(X))
    msg = f'{prefix} {estimator.__class__.__name__}: {score}'
    if logger is not None:
        logger.info(msg)
    print(msg)
    return score, model


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
        'WH score': [],
    }
    # GridSearch for selected estimator
    models = []
    for est, params in zip(estimators, grids):
        gscv = GridSearchCV(est, params, cv=cv, scoring=grid_scorer)
        gscv.fit(X, y)
        est.set_params(**gscv.best_params_)

        # CV fit with best params
        avg, _, _ = cv_fit(est, X, y, cv=cv, pipe=pipe, scorer=scorer, logger=logger, prefix='[GridSearchCV] [CVFIT]')
        score, _ = whole_fit(est, X, y, pipe=pipe, scorer=scorer, logger=logger, prefix='[GridSearchCV] [WHOLE]')

        # collect results
        result['model'].append(est.__class__.__name__)
        result['GS score'].append(gscv.best_score_)
        result['CV score'].append(avg)
        result['WH score'].append(score)
        models.append(gscv.best_estimator_)
        if logger is not None:
            logger.info(f'[GridSearchCV] {est.__class__.__name__} params={gscv.best_params_}')
    return result, models
