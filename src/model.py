from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


class Model:
    def __init__(self, x, y) -> None:
        self.x = x
        self.y = y

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.x, self.y, test_size=0.2, random_state=42
        )

        self.model = LogisticRegression(random_state=6, max_iter=500)
        self.model.fit(self.x_train, self.y_train)

    def get_input_prediction(self, matrix_input):
        return self.model.predict_proba(matrix_input)

    def get_test_prediction(self):
        return self.model.predict(self.x_test)

    def get_performance_report(self, pred):
        return classification_report(self.y_test, pred)
