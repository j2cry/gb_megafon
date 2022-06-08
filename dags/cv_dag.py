import settings
from airflow import DAG
from airflow.operators.python import PythonVirtualenvOperator


with DAG('cross_validate', description='Geekbrains+Megafon DataScience course (search for best parameters)',
         schedule_interval=None, catchup=False, default_args=settings.args) as dag:
    import task
    from dags.jobs.common import cross_validate_job

    # tasks
    validate_model = PythonVirtualenvOperator(
        system_site_packages=False,
        requirements=['numpy==1.21.6', 'pandas==1.4.2', 'scikit-learn==1.0.2', 'lightgbm==3.3.2',
                      'packages/telecom_transformers-1.0.0-py3-none-any.whl'
                      ],
        python_version='3.9',
        task_id='fit_model',
        python_callable=cross_validate_job,
        op_args=[settings.paths,]
    )

    # tasks
    [task.train_waiting(), task.pca_feats_waiting(), task.model_parameters_waiting(), task.fit_parameters_waiting()] \
        >> validate_model
