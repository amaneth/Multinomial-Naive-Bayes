import numpy as np
import pandas as pd
import nltk
import re
from nltk.corpus import stopwords

class MultinomialNB():
    def __init__(self,k):
        self.k=k
        self.vocublary={}
        self.id=0
        self.prior=None
        self.likelihood=None
        self.labels=None

    def lemmantize(self, word):
        stemmer = SnowballStemmer(language = 'english')
        lemmantizer = WordNetLemmatizer()
        stemized= stemmer.stem(word,) 
        lemmantized= lemmantizer.lemmatize(stemized,)
        return lemmantized

    def addto_vocublary(self,word):
        self.vocublary[word]=self.id
        self.id+=1

    def tokenize(self, documents, train=True, lemmantize=False):
        stop_words= set(stopwords.words('english')) 
        tokenized_docs=[]
        for document in documents:
            tokenized_doc=[]
            words =re.sub('[^A-Za-z ]+', '', document).split()
            for word in words:
                word =word.lower()
                if word not in stop_words:
                    if lemmantize:
                        word=lemmantize(word)
                    if train:
                        if word not in self.vocublary:
                            self.addto_vocublary(word)
                    tokenized_doc.append(word)  
            tokenized_docs.append(tokenized_doc)
        return tokenized_docs 

    def count_words(self, documents, labels):
        """
          args:
          documents: array of tokenized documents
          labels: categories of the documents
          returns: 
            counts : a maxtrix that represents the frequency of each word in each category
        """
        num_classes=len(labels.unique())
        counts = np.zeros((len(self.vocublary), num_classes))
        for document, label in zip(documents,labels):
            for word in document:
                if word in self.vocublary.keys():
                    counts[self.vocublary[word]][label] += 1
         
        return counts

    def transform(self, documents):
        """
        args: 
            documents: a pandas series of documents
        returns:
            sparse: a matrix of occurence of each word in each categore for each document
        """
        sparse= np.zeros((len(documents), len(self.vocublary)))
        documents= self.tokenize(documents, train=False)
        for i, document in enumerate(documents):
            for word in document:
                if word in self.vocublary.keys():
                    sparse[i][self.vocublary[word]]=1
        return sparse




    def prior_probablity(self, labels):
        #returns the prior probablity of each class
        num_classes=len(labels.unique())
        prior_probabilities = np.zeros(num_classes)
        total_documents = len(labels)

        for cat in range(num_classes):
            prior_probabilities[cat] = labels.value_counts()[cat]/total_documents

        self.prior=prior_probabilities


    def likelihood_probablity(self, counts):
        """
        args:
            counts: the count maxtrix of frequency of each word in the document
        """
        num_classes= counts.shape[1]
        vocabulary_size = len(self.vocublary)
        word_probablities=np.zeros((vocabulary_size, num_classes))
        category_count=np.sum(counts,axis=0)

        category_count= category_count+ self.k*vocabulary_size
        counts = counts+self.k
        word_probablities=counts/category_count
        self.likelihood=word_probablities
