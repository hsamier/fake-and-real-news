from flask import Flask,render_template,request
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    df = pd.read_csv('news.csv')
    X = df['combine_text']
    y = df['is_True']
    cv = CountVectorizer()
    X = cv.fit_transform(X)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    clf.score(X_test, y_test)
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = cv.transform(data).toarray()
        pred = clf.predict(vect)
        return render_template('home.html', prediction=pred)
    else:
        return render_template('home.html', prediction="Something went wrong")

if __name__ == '__main__':
    app.run(debug=True)