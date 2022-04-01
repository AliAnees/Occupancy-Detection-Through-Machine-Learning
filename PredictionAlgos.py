#Importing Libraries
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix
from timeit import default_timer as timer
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

#Function for plotting confusion matrix
def plotCM(y_test, pred, title):
    plt.rcParams["font.family"] = "Times New Roman"
    ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=["Occupied", "Empty"], cmap=plt.cm.Blues)
    plt.title(title, fontweight='bold', fontsize=16)
    plt.xlabel("Predicted", fontweight='bold', fontsize=14)
    plt.ylabel("True", fontweight='bold', fontsize=14)
    plt.show()

totalStart = timer()
#Reading data from csv
data = read_csv('./Data/OccupationData.csv', header=0, index_col=0, parse_dates=True).squeeze("columns")
values = data.values

#Splitting data into inputs and outputs
X, y = values[:, 1:-1], values[:, -1]

#Converting date to float
"""
x=0
for arr in X:
    dt = arr[0]
    date = dt.split()[0].replace('-','')
    time = dt.split()[1].replace(':','')
    dt = date + '.' + time
    dt = float(dt)
    X[x,0] = dt
    x += 1
"""
#Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=1)

#Converting y values to readable integers
y_train = y_train.astype('int')
y_test = y_test.astype('int')

#Naive Bayes Prediction
timerStartNB = timer()
gnb = GaussianNB()
gnb.fit(X_train, y_train)
predNB = gnb.predict(X_test)

print(confusion_matrix(y_test, predNB))
print(classification_report(y_test, predNB, digits=4))

print("Time for Naive Bayes Prediction: %.2fs\n" % (timer() - timerStartNB))

#Random Forest Prediction
timerStartRF = timer()
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
predRF = rf.predict(X_test)

print(confusion_matrix(y_test, predRF))
print(classification_report(y_test, predRF, digits=4))

print("Time for Random Forest Prediction: %.2fs\n" % (timer() - timerStartRF))

#Support Vector Machine Prediction
timerStartSVC = timer()
svc = svm.SVC()
svc.fit(X_train, y_train)
predSVC = svc.predict(X_test)

print(confusion_matrix(y_test, predSVC))
print(classification_report(y_test, predSVC, digits=4))

print("Time for Support Vector Machine Prediction: %.2fs\n" % (timer() - timerStartSVC))

print("Total time: %.2fs" %(timer() - totalStart)) #Getting total runtime of algorithms

#Plotting Confusion Matrices
plotCM(y_test, predNB, "Naive Bayes")
plotCM(y_test, predRF, "Random Forest")
plotCM(y_test, predSVC, "Support Vector Machine")