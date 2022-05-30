import sys
import pathlib
from airflow import DAG
# from airflow.decorators import task
# from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from airflow.operators.python import PythonOperator, PythonVirtualenvOperator
from airflow.operators.bash import BashOperator
from airflow.sensors.filesystem import FileSensor
from datetime import datetime


GROUP_NAME = 'megafon'      # name of DAG group
args = {
    'owner': 'fragarie',
    'start_date': datetime(2022, 5, 25),
    'provide_context': True,
}
# parse paths
base_path = pathlib.Path().cwd()
path = {
    # 'dag':  base_path.joinpath('dags', dag.dag_id),
    # 'jobs': dag_path.joinpath('dags', dag.dag_id, 'jobs'),
    # 'data': base_path.joinpath('data', dag.dag_id),     # shared data folder
    'dag': base_path.joinpath('dags', GROUP_NAME),  # dag root
    'jobs': base_path.joinpath('dags', GROUP_NAME, 'jobs'),  # jobs folder
    'data': base_path.joinpath('data', GROUP_NAME),  # shared data folder
}
path['input_features'] = path['data'].joinpath('features.csv')  # source features
path['output_features'] = path['data'].joinpath('compressed_features.csv')
path['temp'] = path['data'].joinpath('.temp')  # temporary filename


with DAG('prepare_features', description='Geekbrains+Megafon DataScience course (prepare features)',
         schedule_interval=None, catchup=False, default_args=args) as dag:
    # refresh imports
    sys.path.append(path['dag'].as_posix())
    from jobs.source import compress_features, fit_model

    # tasks
    feats_waiting = FileSensor(
        task_id='waiting_for_features',
        filepath=path['input_features'].as_posix(),
        fs_conn_id='fs_default'
    )

    # compress = SparkSubmitOperator(
    #     task_id='compress_features',
    #     conn_id='spark_default',
    #     application=f"{path['jobs'].joinpath('pca_compress.py').as_posix()} "
    #                 f"{path['input_features'].as_posix()} "     # input filename
    #                 f"{path['temp'].as_posix()}",               # output filename
    # )

    compress = PythonOperator(
        task_id='compress_features_with_PCA',
        python_callable=compress_features,
        op_args=[path['input_features'].as_posix(),     # features_path
                 ['75', '81', '85', '139', '203'],      # drop_feats
                 path['temp'].as_posix(),               # target_path
                 ]
    )

    move_feats = BashOperator(
        task_id='move_features',
        bash_command=f"mv {path['temp'].as_posix() + '/*.csv'} {path['output_features'].as_posix()};"
                     f"rm -rf {path['temp'].as_posix()}"
    )

    # tasks order
    feats_waiting >> compress >> move_feats
