import settings
from airflow import DAG
from airflow.operators.python import PythonVirtualenvOperator


with DAG('grid_search', description='Geekbrains+Megafon DataScience course (search for best parameters)',
         schedule_interval=None, catchup=False, default_args=settings.args) as dag:
    import task
    from dags.jobs.common import search_params_job

    # tasks
    grid_search = PythonVirtualenvOperator(
        system_site_packages=False,
        requirements=['numpy==1.21.6', 'pandas==1.4.2', 'scikit-learn==1.0.2',
                      'lightgbm==3.3.2'
                      ],
        python_version='3.9',
        task_id='grid_search',
        python_callable=search_params_job,
        op_args=[settings.paths, ]
    )

    # tasks
    [task.grid_waiting(), task.train_waiting(), task.pca_feats_waiting(), task.model_parameters_waiting(), 
    task.fit_parameters_waiting()] \
        >> grid_search
