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


def compress_features(features_path, model_params_path, target_path):
    """ Compress features with PCA
    :param spark: Spark context
    :param features_path: path to the features file
    :param drop_feats: columns to drop before processing
    :param target_path: path to the result file
    """
    import ast
    import configparser    
    from pyspark.ml.feature import PCA, VectorAssembler
    from pyspark.sql.functions import col
    from pyspark.ml.functions import vector_to_array

    spark = run_spark_executor('12g')       # get Spark context

    # read parameters
    config = configparser.ConfigParser()
    config.read(model_params_path)
    drop_feats = ast.literal_eval(config['FEATURES']['drop'])

    # read features
    feats = spark.read.csv(features_path, sep='\t', header=True, inferSchema=True).drop('Unnamed: 0', *drop_feats)
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
    features.repartition(1).write.mode('overwrite').csv(target_path, header=True, sep=',')


def search_params(grid_path, train_path, features_path):
    """ GridSearch over given parameters
    :param grid_path: path to the .json file with parameters collection
    :param train_path: path to the train data file
    :param features_path: path to the features file
    """
    import json
    import pandas as pd


def fit_model(jobs_path, train_path, features_path, model_params_path, fit_params_path, export_path):
    """ Fit and export model """
    import sys
    sys.path.append(jobs_path)
    import json
    import cloudpickle
    import configparser
    import pandas as pd
    import datetime as dt
    from sklearn.utils.class_weight import compute_class_weight
    from sklearn.pipeline import make_pipeline
    from transformers import Merger, TimeDifference, Clusterer, ColumnsCorrector
    # from lightgbm import LGBMClassifier       # OSError: libgomp.so.1 not found
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import f1_score

    # read model parameters
    config = configparser.ConfigParser()
    config.read(model_params_path)
    bound_date = config['MODEL']['bound_date']
    fit_params = json.load(open(fit_params_path, 'r'))

    # read train data
    train_data = pd.read_csv(train_path).drop('Unnamed: 0', axis=1)
    # extract required train data
    used_mask = train_data['buy_time'] >= dt.datetime.fromisoformat(bound_date).timestamp()
    train_data = train_data[used_mask]
    target = train_data['target']

    # read compressed features
    features = pd.read_csv(features_path)

    # calc class weights
    class_weights = dict(enumerate(compute_class_weight('balanced', classes=[0, 1], y=target)))

    # build featuring pipeline
    pipeline = make_pipeline(
        Merger(features, method='backward', fillna='nearest'),
        TimeDifference('feats_time', 'train_time'),
        # Clusterer(['0', '1', '2'], n_clusters=8, random_state=13),
        ColumnsCorrector('drop', ['id', 'train_time', 'feats_time']),        
        # LGBMClassifier(random_state=17, class_weight='balanced', n_jobs=-1, **fit_params)
        RandomForestClassifier(random_state=17, class_weight=class_weights, n_jobs=-1, **fit_params)
    )
    # fit model
    pipeline.fit(train_data.drop('target', axis=1), target)

    metric = f1_score(target, pipeline.predict(train_data.drop('target', axis=1)), average='macro')
    with open(export_path + '.metric', 'w') as f:
        f.write(str(metric))
    # save model
    cloudpickle.dump(pipeline, open(export_path, 'wb'))     # сохраняются пути импорта библиотек! и по какой-то причине колоссальное отличие метрики
    # import dill
    # dill.dump(pipeline, open(export_path + '.dill', 'wb'))


def cv_fit(jobs_path, train_path, features_path, model_params_path, fit_params_path):
    """ DEBUG CROSS-FIT """
    import sys
    sys.path.append(jobs_path)
    import json
    import configparser
    import pandas as pd
    import datetime as dt
    from functools import partial
    from sklearn.utils.class_weight import compute_class_weight
    from sklearn.model_selection import KFold
    from sklearn.pipeline import make_pipeline
    from transformers import Merger, TimeDifference, Clusterer, ColumnsCorrector
    # from lightgbm import LGBMClassifier       # OSError: libgomp.so.1 not found
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import f1_score

    # read model parameters
    config = configparser.ConfigParser()
    config.read(model_params_path)
    bound_date = config['MODEL']['bound_date']
    n_folds = int(config['MODEL']['n_folds'])
    fit_params = json.load(open(fit_params_path, 'r'))

    # read train data
    train_data = pd.read_csv(train_path).drop('Unnamed: 0', axis=1)
    # extract required train data
    used_mask = train_data['buy_time'] >= dt.datetime.fromisoformat(bound_date).timestamp()
    train_data = train_data[used_mask]
    target = train_data['target']

    # read compressed features
    features = pd.read_csv(features_path)

    # calc class weights
    class_weights = dict(enumerate(compute_class_weight('balanced', classes=[0, 1], y=target)))
    folds = KFold(n_splits=n_folds, shuffle=True, random_state=29)
    # f1_macro = partial(f1_score, average='macro')

    # build featuring pipeline
    pipeline = make_pipeline(
        Merger(features, method='backward', fillna='nearest'),
        TimeDifference('feats_time', 'train_time'),
        # Clusterer(['0', '1', '2'], n_clusters=8, random_state=13),
        ColumnsCorrector('drop', ['id', 'train_time', 'feats_time']),        
        # LGBMClassifier(random_state=17, class_weight='balanced', n_jobs=-1, **fit_params)
        # RandomForestClassifier(random_state=17, class_weight=class_weights, n_jobs=-1, **fit_params)
    )
    train_data = pipeline.fit_transform(train_data.drop('target', axis=1))
    metrics = []
    models = []

    for train_index, valid_index in folds.split(target):
        # extract data
        X_train = train_data.iloc[train_index]
        X_valid = train_data.iloc[valid_index]
        # exctract target
        y_train = target.iloc[train_index]
        y_valid = target.iloc[valid_index]

        model = RandomForestClassifier(random_state=17, class_weight=class_weights, n_jobs=-1, **fit_params).fit(X_train, y_train)

        # predicts & metrics
        prediction = model.predict(X_valid)
        score = f1_score(y_valid, prediction, average='macro')
        # append step result
        models.append(model)
        metrics.append(score)
    
    avg = sum(metrics) / len(metrics)
    with open('/opt/airflow/data/debug.log', 'w') as f:
        f.write(str(avg))
