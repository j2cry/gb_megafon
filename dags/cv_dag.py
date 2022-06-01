import sys
import pathlib
sys.path.append(pathlib.Path().joinpath('dags', 'megafon').as_posix())

import settings
from airflow import DAG
from airflow.sensors.filesystem import FileSensor
from airflow.operators.python import PythonVirtualenvOperator, PythonOperator


with DAG('DEBUG_CV_FIT', description='Geekbrains+Megafon DataScience course (search for best parameters)',
         schedule_interval=None, catchup=False, default_args=settings.args) as dag:
    # refresh imports
    sys.path.append(settings.path['dag'].as_posix())
    from jobs.common import cv_fit_job

    # tasks
    train_waiting = FileSensor(
        task_id='waiting_for_train_data',
        filepath=settings.path['train'].as_posix(),
        fs_conn_id='fs_default'
    )

    pca_feats_waiting = FileSensor(
        task_id='waiting_for_pca_features',
        filepath=settings.path['pca_features'].as_posix(),
        fs_conn_id='fs_default'
    )

    model_parameters_waiting = FileSensor(
        task_id='waiting_for_model_parameters',
        filepath=settings.path['model_params'].as_posix(),
        fs_conn_id='fs_default'
    )

    fit_parameters_waiting = FileSensor(
        task_id='waiting_for_fit_parameters',
        filepath=settings.path['fit_params'].as_posix(),
        fs_conn_id='fs_default'
    )

    fit_model = PythonVirtualenvOperator(
        system_site_packages=False,
        requirements=['numpy==1.21.6', 'pandas==1.4.2', 'scikit-learn==1.0.2'],
        # requirements=['scikit-learn==1.0.2', 'lightgbm'],        # OSError: libgomp.so.1 not found when importing LGBMClassifier
        python_version='3.9',
        task_id='fit_model',
        python_callable=cv_fit_job,
        op_args=[settings.path['jobs'].as_posix(),
                 settings.path['train'].as_posix(),
                 settings.path['pca_features'].as_posix(),
                 settings.path['model_params'].as_posix(),
                 settings.path['fit_params'].as_posix(),
                 ]
    )

    # tasks
    [train_waiting, pca_feats_waiting, model_parameters_waiting, fit_parameters_waiting] >> fit_model
