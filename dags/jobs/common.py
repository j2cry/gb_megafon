def load_model_config(model_params_path, fit_params_path):
    """ Load model and fit parameters """
    import json
    import configparser

    # read model parameters
    config = configparser.ConfigParser()
    config.read(model_params_path)
    fit_params = json.load(open(fit_params_path, 'r'))
    return config, fit_params


def data_load(train_path, features_path, bound_date):
    """ Load train and features data """
    import pandas as pd
    import datetime as dt

    # read train data
    train_data = pd.read_csv(train_path).drop('Unnamed: 0', axis=1)
    # extract required train data
    used_mask = train_data['buy_time'] >= dt.datetime.fromisoformat(bound_date).timestamp()
    train_data = train_data[used_mask]
    # read compressed features
    features = pd.read_csv(features_path)
    return train_data, features


def get_preparer(features):
    """ Model pipeline """    
    from  dags.jobs.transformers import Merger, TimeDifference, Clusterer, PurchaseRatio, ColumnsCorrector, BasicFiller
    from sklearn.pipeline import make_pipeline

    return make_pipeline(
        Merger(features, method='backward', fillna='mean'),
        TimeDifference('feats_time', 'train_time'),
        Clusterer(['0', '1', '2'], n_clusters=8, random_state=13),
        PurchaseRatio(by=['cluster']),
        ColumnsCorrector('drop', ['id', 'train_time', 'feats_time']),
        BasicFiller(strategy='mean', apply_on_fly=True),
    )


def get_estimator(**fit_params):
    """ Model estimator """
    from lightgbm import LGBMClassifier
    return LGBMClassifier(**fit_params)


def folds(n_folds):
    """ Folds for CV """
    from sklearn.model_selection import KFold
    return KFold(n_splits=n_folds, shuffle=True, random_state=29)

    
def scorer(estimator, X_test, y_test):
    """ Default metric function """
    from sklearn.metrics import f1_score
    pred = estimator.predict(X_test)
    return f1_score(y_test, pred, average='macro')
    

# ================================================================================================
def run_spark_executor(ram):
    """ Run local Spark executor
        THIS IS DEVELOPMENT/DEBUG FEATURE ONLY
    """
    # TODO по-хорошему, надо собрать и запустить Spark Master и отправлять на него, а не вот-это-вот-все
    from pyspark import SparkConf
    from pyspark.sql import SparkSession

    config = SparkConf().setAll([
        ('spark.driver.memory', ram),
    ])
    return SparkSession.builder.config(conf=config).getOrCreate()


def compress_features(paths):
    """ Compress features with PCA """
    import ast
    import configparser    
    from pyspark.ml.feature import PCA, VectorAssembler
    from pyspark.sql.functions import col
    from pyspark.ml.functions import vector_to_array

    spark = run_spark_executor('12g')       # get Spark context

    # read parameters
    config = configparser.ConfigParser()
    config.read(paths['model_params'])
    drop_feats = ast.literal_eval(config['FEATURES']['drop'])

    # read raw features
    feats = spark.read.csv(paths['raw_features'], sep='\t', header=True, inferSchema=True).drop('Unnamed: 0', *drop_feats)
    columns = feats.columns[3:]

    # apply PCA
    assembler = VectorAssembler(inputCols=columns, outputCol='features')
    assembled = assembler.transform(feats)
    pca = PCA(k=3, inputCol='features', outputCol='compressed').fit(assembled)
    compressed = pca.transform(assembled)

    # collect features
    features = compressed.withColumn('f', vector_to_array('compressed'))\
                         .select(['id', 'buy_time'] + [col('f')[i].alias(f'{i}') for i in range(3)])

    # save
    features.repartition(1).write.mode('overwrite').csv('.compressed', header=True, sep=',')


# ===============================================================================
def search_params_job(paths):
    """ GridSearch over given parameters """
    import logging
    import json
    from dags.jobs import common     # STILL MODULE NOT FOUND EXCEPTION 
    from sklearn.utils.class_weight import compute_class_weight
    from sklearn.model_selection import GridSearchCV

    # load grid
    grid = json.load(open(paths['grid'], 'r'))
    # prepare model
    config, fit_params = common.load_model_config(paths['model_params'], paths['fit_params'])
    data, features = common.data_load(paths['train'], paths['pca_features'], config['MODEL']['bound_date'])
    preparer = common.get_preparer(features)
    target = data['target']
    prepared_data = preparer.fit_transform(data.drop('target', axis=1), target)

    # calc class weights
    if config['FIT_PARAMS']['adaptive_class_balance'] == 'True':
        fit_params['class_weight'] = dict(enumerate(compute_class_weight('balanced', classes=[0, 1], y=target)))
    n_folds = int(config['MODEL']['n_folds'])
    
    # GridSearch
    est = common.get_estimator(**fit_params)
    gscv = GridSearchCV(est, grid, cv=common.folds(n_folds), scoring=common.scorer)
    gscv.fit(prepared_data, target)

    # save best params
    if config['FIT_PARAMS']['update_on_cv'] == 'True':
        fit_params.update(gscv.best_params_)
        json.dump(fit_params, open(paths['fit_params'], 'w'))
    # logs
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('airflow.task')
    logger.info(f'[GS] Best metric on {n_folds} folds = {gscv.best_score_}; model = {est.__class__.__name__}; parameters = {gscv.best_params_}\n')


def fit_model_job(paths):
    """ Fit and export model """
    import logging
    import cloudpickle
    from dags.jobs import common
    from sklearn.utils.class_weight import compute_class_weight
    from sklearn.pipeline import make_pipeline

    # prepare model
    config, fit_params = common.load_model_config(paths['model_params'], paths['fit_params'])
    data, features = common.data_load(paths['train'], paths['pca_features'], config['MODEL']['bound_date'])
    preparer = common.get_preparer(features)
    target = data['target']    

    # calc class weights
    if config['FIT_PARAMS']['adaptive_class_balance'] == 'True':
        fit_params['class_weight'] = dict(enumerate(compute_class_weight('balanced', classes=[0, 1], y=target)))
    
    # fit model and calc metric
    model = make_pipeline(preparer, common.get_estimator(**fit_params))
    model.fit(data.drop('target', axis=1), target)
    metric = common.scorer(model, data.drop('target', axis=1), target)
    # save model
    cloudpickle.dump(model, open(paths['export'], 'wb'))     # сохраняются и пути импорта библиотек!
    # logs
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('airflow.task')
    logger.info(f'[FIT] metric on whole train data = {metric}; model = {model[-1].__class__.__name__}; parameters = {model[-1].get_params()}\n')


def cross_validate_job(paths):
    """ Cross validation with current parameters """
    import logging
    from dags.jobs import common
    from sklearn.utils.class_weight import compute_class_weight
    from sklearn.pipeline import make_pipeline

    # prepare model
    config, fit_params = common.load_model_config(paths['model_params'], paths['fit_params'])
    data, features = common.data_load(paths['train'], paths['pca_features'], config['MODEL']['bound_date'])
    preparer = common.get_preparer(features)
    target = data['target']
    prepared_data = preparer.fit_transform(data.drop('target', axis=1), target)

    # calc class weights
    if config['FIT_PARAMS']['adaptive_class_balance'] == 'True':
        fit_params['class_weight'] = dict(enumerate(compute_class_weight('balanced', classes=[0, 1], y=target)))

    metrics = []
    models = []
    n_folds = int(config['MODEL']['n_folds'])
    for train_index, valid_index in common.folds(n_folds).split(target):
        # extract data
        X_train = prepared_data.iloc[train_index]
        X_valid = prepared_data.iloc[valid_index]
        # exctract target
        y_train = target.iloc[train_index]
        y_valid = target.iloc[valid_index]

        # fit, predict & score
        model = common.get_estimator(**fit_params).fit(X_train, y_train)        
        score = common.scorer(model, X_valid, y_valid)
        # append step result
        models.append(model)
        metrics.append(score)
    
    avg = sum(metrics) / len(metrics)
    # logs
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('airflow.task')
    logger.info(f'[CV] Avg. metric on {n_folds} folds = {avg}; model = {model.__class__.__name__}; parameters = {model.get_params()}\n')
