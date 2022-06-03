import pathlib
from datetime import datetime


# basic parameters
args = {
    'owner': 'fragarie',
    'start_date': datetime(2022, 5, 25),
    'provide_context': True,
}

# parse paths
data_path = pathlib.Path().cwd().joinpath('data')                       # shared data folder
paths = {
    'train': data_path.joinpath('data_train.csv').as_posix(),                      # source train data file
    'pca_features': data_path.joinpath('compressed_features.csv').as_posix(),      # features file
    'raw_features': data_path.joinpath('features.csv').as_posix(),                 # raw features file
    'model_params': data_path.joinpath('parameters.conf').as_posix(),              # model parameters file
    'fit_params': data_path.joinpath('fit_params.json').as_posix(),                # fit parameters file
    'export': data_path.joinpath('model.pkl').as_posix(),                          # export model file
    'grid': data_path.joinpath('grid_params.json').as_posix(),                     # parameters grid for GridSearchCV
}
