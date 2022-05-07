import numpy as np
import argparse
import pickle

def test(model,text):
    #returns the prediction of categories for a test document
    X_new= model.transform(text)
    log_probablity= X_new[..., np.newaxis]*np.log(model.likelihood[np.newaxis,...])
    cat_log_probablity= log_probablity.sum(axis=1)

    category_probablity = np.log(model.prior) + cat_log_probablity
    prediction=np.argmax(category_probablity,axis=1)
    print("Prediction:", model.labels[prediction[0]]) 

parser = argparse.ArgumentParser()
parser.add_argument('--text', help='the text to be predicted', required=True, type=str)
arguments= vars(parser.parse_args())
text=arguments['text']
with open('./model/model.pkl', 'rb') as handle:
    model=pickle.load(handle)

test(model, [text])


