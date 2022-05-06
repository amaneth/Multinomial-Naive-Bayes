

def train_test_split(data, train_size):
    size = len(data)
    train_size = int(train_size * size)
    X_train, y_train = data['review'][:train_size], data['sentiment'][:train_size]
    X_test, y_test = data['review'][train_size:], data['sentiment'][train_size:]
    cats = data['sentiment'].unique().tolist()
    y_test = y_test.apply(lambda x : cats.index(x))
    return X_train, y_train, X_test, y_test
