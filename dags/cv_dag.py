import sys
import pathlib
sys.path.append(pathlib.Path().joinpath('dags', 'megafon').as_posix())

import common
from airflow import DAG
from airflow.sensors.filesystem import FileSensor
from airflow.operators.python import PythonVirtualenvOperator, PythonOperator


with DAG('DEBUG_CV_FIT', description='Geekbrains+Megafon DataScience course (search for best parameters)',
         schedule_interval=None, catchup=False, default_args=common.args) as dag:
    # refresh imports
    sys.path.append(common.path['dag'].as_posix())
    from jobs.source import cv_fit

    # tasks
    train_waiting = FileSensor(
        task_id='waiting_for_train_data',
        filepath=common.path['train'].as_posix(),
        fs_conn_id='fs_default'
    )

    pca_feats_waiting = FileSensor(
        task_id='waiting_for_pca_features',
        filepath=common.path['pca_features'].as_posix(),
        fs_conn_id='fs_default'
    )

    model_parameters_waiting = FileSensor(
        task_id='waiting_for_model_parameters',
        filepath=common.path['model_params'].as_posix(),
        fs_conn_id='fs_default'
    )

    fit_parameters_waiting = FileSensor(
        task_id='waiting_for_fit_parameters',
        filepath=common.path['fit_params'].as_posix(),
        fs_conn_id='fs_default'
    )

    fit_model = PythonVirtualenvOperator(
        system_site_packages=False,
        requirements=['numpy==1.21.6', 'pandas==1.4.2', 'scikit-learn==1.0.2', 'cloudpickle==2.1.0'],
        # requirements=['scikit-learn==1.0.2', 'lightgbm'],        # OSError: libgomp.so.1 not found when importing LGBMClassifier
        python_version='3.9',
        task_id='fit_model',
        python_callable=cv_fit,
        op_args=[common.path['jobs'].as_posix(),
                 common.path['train'].as_posix(),
                 common.path['pca_features'].as_posix(),
                 common.path['model_params'].as_posix(),
                 common.path['fit_params'].as_posix(),
                 ]
    )

    # tasks
    [train_waiting, pca_feats_waiting, model_parameters_waiting, fit_parameters_waiting] >> fit_model
