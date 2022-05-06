import numpy as np
def test(model,X,y,batch_size=500):
    #returns the prediction of categories for a test document
    batches=(len(X)//batch_size)
    if (len(X)%batch_size)!=0:
        batches=batches+1
    predictions=[]
    for batch in range(batches):
        X_new=X[batch*batch_size:(batch+1)*(batch_size)]
        X_new= model.transform(X_new)
        log_probablity= X_new[..., np.newaxis]*np.log(model.likelihood[np.newaxis,...])
        cat_log_probablity= log_probablity.sum(axis=1)

        category_probablity = np.log(model.prior) + cat_log_probablity
        prediction=np.argmax(category_probablity,axis=1)
        predictions.extend(prediction)
    accuracy=(np.sum(y == predictions)/len(y))*100
    print(f"The accuracy is:{accuracy:.2f}%") 
