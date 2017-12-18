
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
import pickle
fake_data = pd.read_csv('https://s3.amazonaws.com/assets.datacamp.com/production/course_3629/fake_or_real_news.csv',
                        low_memory=True
                        )
#pickle.dump(fake_data, open('data_frame.pkl', 'wb'))
fake_data.head()
response = fake_data.label
count_vectorizer = CountVectorizer(stop_words="english")
X_train, X_test, Y_train, Y_test = train_test_split(fake_data["title"],
                                                    response,
                                                    test_size=0.33,
                                                    random_state=53)
count_train = count_vectorizer.fit_transform(X_train)
#print(count_vectorizer.vocabulary_)

count_test = count_vectorizer.transform(X_test)
pickle.dump(count_vectorizer, open('count_vector.pkl', 'wb'))
nb_classifier = MultinomialNB()
#nb_classifier = RandomForestClassifier(n_estimators = 100)
nb_classifier.fit(count_train, Y_train)
pred = nb_classifier.predict(count_test)
score = metrics.accuracy_score(Y_test, pred)
cm = metrics.confusion_matrix(Y_test,
                              pred,
                              labels=["FAKE",  "REAL"]
                             )
pickle.dump(nb_classifier, open('model.pkl', 'wb'))
#pickle.dump(score, open('accuracy.pkl', 'wb'))
print(score)
print(cm)




