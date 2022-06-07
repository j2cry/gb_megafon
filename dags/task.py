import settings
from airflow.sensors.filesystem import FileSensor

    
# tasks
def raw_feats_waiting():
    return FileSensor(
        task_id='waiting_for_raw_features',
        filepath=settings.paths['raw_features'],
        fs_conn_id='fs_default'
    )


def pca_feats_waiting():
    return FileSensor(
        task_id='waiting_for_pca_features',
        filepath=settings.paths['pca_features'],
        fs_conn_id='fs_default'
    )


def train_waiting():
    return FileSensor(
        task_id='waiting_for_train_data',
        filepath=settings.paths['train'],
        fs_conn_id='fs_default'
    )



def model_parameters_waiting():
    return FileSensor(
        task_id='waiting_for_model_parameters',
        filepath=settings.paths['model_params'],
        fs_conn_id='fs_default'
    )


def fit_parameters_waiting():
    return FileSensor(
        task_id='waiting_for_fit_parameters',
        filepath=settings.paths['fit_params'],
        fs_conn_id='fs_default'
    )


def grid_waiting():
    return FileSensor(
        task_id='waiting_for_grid_parameters',
        filepath=settings.paths['grid'],
        fs_conn_id='fs_default'
    )
