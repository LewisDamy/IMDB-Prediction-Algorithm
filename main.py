from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn import metrics
from scipy import stats
import pandas as pd
import warnings
import time

# Read data from IMDb.csv file
data = pd.read_csv('mylib/IMDb.csv')
print(data)
data.drop('id', axis=1)

# list all the useful columns from IMDb file
useful_columns = [
    'castTotalLikes',
    'directorLikes',
    'actor1likes',
    'movieLikes',
    'fbPosters',
    'year',
    'duration',
    'genre',
    'contentRating',
    'criticReviews',
    'userReviews',
    'userVotes',
    'rating'
]

# save the useful data
data = data[useful_columns]

# remove null values from data
data = data.dropna()

# grab the columns where data is > or <= than 1990
column1 = data[data.year > 1990]
column2 = data[data.year <= 1990]


# right data to IMDb_likes_review and grab from IMDB_new
data.to_csv('mylib/IMDb_likes_review.csv', sep=',')
data = pd.read_csv('mylib/IMDB_new.csv', sep=',')

# remove id
data = data.drop('id', axis=1)

# print in histograms the distribution of the data
pd.DataFrame.hist(data, figsize=[15, 15])
# plt.show()

################### LINEAR REGRESSION MODEL ###################

# loading x value to be used in the model
x = data.drop(['rating', 'movieLikes', 'directorLikes', 'genre', 'castTotalLikes', 'actor1likes'], axis=1)

# get the ratings column from the data for the test model
y = data.rating

# train the data as using 40% as test
X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0, test_size=0.4)


start = time.time()
# run the data with Ordinary Least Squares regression(OLS) passing the training
linear_model_regression = sm.OLS(y_train, X_train)
result = linear_model_regression.fit()
end = time.time()
# print the summary of the results received

with open('OLS_result.txt', 'w') as f:
    f.write(result.summary().as_text())


# Calculate the total prediction and accuracy of Linear Regression
X_accuracy = pd.DataFrame(result.predict(X_test))
X_accuracy_round = X_accuracy.round(1)
Y_accuracy = pd.DataFrame(y_test)
Y_accuracy = Y_accuracy.reset_index(drop=True)
X_accuracy_round['rating'] = Y_accuracy
X_accuracy_round.columns = ['pred', 'actual']
X_accuracy_round['difference'] = round(abs(X_accuracy_round.pred - X_accuracy_round.actual), 2)


print(f"Total Predictions: {X_accuracy_round.difference.count()}")
print(f"Accuracy of Linear Regression:")
print((X_accuracy_round.difference < 1.1).sum()/X_accuracy_round.difference.count())
print(f"Time execution: {end - start} sec")


################### K-NEAREST NEIGHBOUR MODEL ###################

knn_data = data

rate = [0.0, 2.5, 5.0, 7.5, 10.0]
grade = ['VeryBad', 'Bad', 'Good', 'VeryGood']

knn_data['grades'] = pd.cut(knn_data.rating, rate, labels=grade)
print(knn_data)

x = knn_data.drop(['rating', 'grades'], axis=1)
y = knn_data.grades

X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0, test_size=0.4)

k_range = range(10, 70)
score = []

start = time.time()
for i in k_range:
    warnings.filterwarnings("ignore")
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    prediction = knn.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, prediction)
    score.append(accuracy)
end = time.time()

final_score = 0
for i in range(len(score)):
    final_score += score[i]

final_score = final_score/len(score)

print(f"accuracy: {final_score} | time execution:{end - start} sec")

plt.plot(k_range, score)
plt.xlabel('Value of K')
plt.ylabel('Accuracy')
# plt.show()



