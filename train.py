
import numpy as np

def fit(model, X, y):
    #fit the the train data to the model
    documents = model.tokenize(X, train=True)
    classes=y.unique()
    num_classes=len(classes)
    labels = y.map(lambda x: np.where(classes==x)[0][0])

    counts = model.count_words(documents, labels)
    model.prior_probablity(labels)
    model.likelihood_probablity(counts)
