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


def compress_features(features_path, drop_feats, target_path):
    """ Compress features with PCA
    :param spark: Spark context
    :param features_path: path to the features file
    :param drop_feats: columns to drop before processing
    :param target_path: path to the result file
    """
    from pyspark.ml.feature import PCA, VectorAssembler
    from pyspark.sql.functions import col
    from pyspark.ml.functions import vector_to_array

    spark = run_spark_executor('12g')       # get Spark context

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


def fit_model(train_path, features_path, **kwargs):
    pass
