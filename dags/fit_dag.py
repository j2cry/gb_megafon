import settings
from airflow import DAG
from airflow.operators.python import PythonVirtualenvOperator


with DAG('fit_model', description='Geekbrains+Megafon DataScience course (fit model)',
         schedule_interval=None, catchup=False, default_args=settings.args) as dag:
    import task
    from dags.jobs.common import fit_model_job

    # tasks
    fit_model = PythonVirtualenvOperator(
        system_site_packages=False,
        requirements=['numpy==1.21.6', 'pandas==1.4.2', 'scikit-learn==1.0.2', 'cloudpickle==2.1.0', 'lightgbm==3.3.2',
                      'packages/telecom_transformers-1.0.0-py3-none-any.whl'
                      ],
        python_version='3.9',
        task_id='fit_model',
        python_callable=fit_model_job,
        op_args=[settings.paths, ]
    )

    # tasks
    [task.train_waiting(), task.pca_feats_waiting(), task.model_parameters_waiting(), task.fit_parameters_waiting()] \
        >> fit_model
