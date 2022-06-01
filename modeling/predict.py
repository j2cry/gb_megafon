import argparse
import pathlib
import pickle
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument('-f', dest='file', help='input data .csv file', required=True)
parser.add_argument('-m', dest='model', help='model .pkl file', default='model.pkl')
parser.add_argument('-p', dest='proba', help='predict proba', action='store_true')
args = parser.parse_args()

# read
model = pickle.load(open(args.model, 'rb'))
data = pd.read_csv(args.file, usecols=['id', 'vas_id', 'buy_time'])
# predict
data['target'] = model.predict_proba(data).T[1] if args.proba else model.predict(data)
# write
data.to_csv('predict_' + pathlib.Path(args.file).name, index=False)
print('done.')
