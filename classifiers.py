from sklearn import linear_model, metrics
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


class ClassifierController:
    def classify(self, dataframe, pipeline, classifier):
        # Create a series to store the labels
        y = dataframe.genre

        # Create training and test sets
        x_train, x_test, y_train, y_test = train_test_split(dataframe.lyrics, y, test_size=0.3)

        # Transform the training data
        count_train = pipeline.fit_transform(x_train)

        # Transform the test data
        count_test = pipeline.transform(x_test)

        # Fit the classifier to the training data
        classifier.fit(count_train, y_train)

        # Create the predicted tags
        pred = classifier.predict(count_test)

        # Calculate the accuracy score
        score = metrics.accuracy_score(y_test, pred)
        return score


nb_classifier = MultinomialNB()
ln_classifier = linear_model.LogisticRegression()
classifier_controller = ClassifierController()
