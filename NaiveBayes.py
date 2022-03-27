from pandas import read_csv
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB

data = read_csv('./Data/OccupationData.csv', header=0, index_col=0, parse_dates=True).squeeze("columns")
values = data.values
# split data into inputs and outputs
X, y = values[:, :-1], values[:, -1]

x=0
for arr in X:
    dt = arr[0]
    date = dt.split()[0].replace('-','')
    time = dt.split()[1].replace(':','')
    dt = date + '.' + time
    dt = float(dt)
    X[x,0] = dt
    x += 1

print(X)

# split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=1)

#predict
gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train.astype('int')).predict(X_test)
missed_points = (y_test != y_pred).sum()
accuracy = ((X_test.shape[0]-missed_points)/X_test.shape[0])*100
print("Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0], missed_points))
print("Accuracy: %.2f%%" % (accuracy))

print()

# make a naive prediction
def naive_prediction(X_test, value):
    return [value for x in range(len(X_test))]
 
# evaluate skill of predicting each class value
for value in [0, 1]:
    # forecast
    y_hat = naive_prediction(X_test, value)
    # evaluate
    score = accuracy_score(y_test.tolist(), y_hat)
    # summarize
    print('Naive=%d score=%.3f' % (value, score))