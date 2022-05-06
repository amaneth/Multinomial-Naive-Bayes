from config import args
import argparse
import pandas as pd
import nltk

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
parser.add_argument('-k', '--k', help='k is the laplacian smooting constant', default=k, type=float)
parser.add_argument('-p', '--dataset', help='path to the dataset', default=metadata, type=str)
parser.add_argument('-t', '--trainsize', help='the train data size', default=train_size, type=float)
arguments = vars(parser.parse_args())
k= arguments['k']
path= arguments['dataset']
train_size= arguments['trainsize']

data= pd.read_csv(path)[:1000] 
X_train, y_train, X_test, y_test= train_test_split(data, train_size)
naive= MultinomialNB(k)
fit(naive, X_train, y_train)
print("Training is done.")
test(naive, X_test, y_test)

