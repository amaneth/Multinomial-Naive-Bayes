from config import args
import argparse
import pandas as pd
import nltk
import pickle

from utils import train_test_split
from model import MultinomialNB
from train import fit
from test import test

nltk.download('stopwords')
nltk.download('wordnet')

metadata = args.metadata
k= args.k
train_size= args.train_size


parser = argparse.ArgumentParser()
parser.add_argument('--k', help='k is the laplacian smooting constant', default=k, type=float)
parser.add_argument('--dataset', help='path to the dataset', default=metadata, type=str)
parser.add_argument('--trainsize', help='the train data size', default=train_size, type=float)
parser.add_argument('--savemodel', help='save the model as file', default=True, type=bool)
arguments = vars(parser.parse_args())
k= arguments['k']
path= arguments['dataset']
train_size= arguments['trainsize']
save_model= arguments['savemodel']

data= pd.read_csv(path) 
X_train, y_train, X_test, y_test= train_test_split(data, train_size)
naive= MultinomialNB(k)
fit(naive, X_train, y_train)
print("Training is done.")
if save_model:
    with open('./model/model.pkl','wb') as handle:
        pickle.dump(naive, handle, protocol=pickle.HIGHEST_PROTOCOL)
test(naive, X_test, y_test)

