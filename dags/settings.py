import pathlib
from datetime import datetime


GROUP_NAME = 'megafon'      # name of DAG group

# basic parameters
args = {
    'owner': 'fragarie',
    'start_date': datetime(2022, 5, 25),
    'provide_context': True,
}

# parse paths
base_path = pathlib.Path().cwd()
path = {
    'dag': base_path.joinpath('dags', GROUP_NAME),  # dag root
    'jobs': base_path.joinpath('dags', GROUP_NAME, 'jobs'),  # jobs folder
    'data': base_path.joinpath('data', GROUP_NAME),  # shared data folder
}
path['train'] = path['data'].joinpath('data_train.csv')                     # source train data file
path['pca_features'] = path['data'].joinpath('compressed_features.csv')     # features file
path['raw_features'] = path['data'].joinpath('features.csv')                # raw features file
path['temp'] = path['data'].joinpath('.temp')                               # temporary filename
path['model_params'] = path['data'].joinpath('parameters.conf')             # model parameters file
path['fit_params'] = path['data'].joinpath('fit_params.json')               # fit parameters file
path['export'] = path['data'].joinpath('model.pkl')                         # export model file
path['grid'] = path['data'].joinpath('grid_params.json')                           # parameters grid for GridSearchCV
