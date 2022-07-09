#Importing Essential Libraries
from flask import Flask,render_template,request
import pickle
#Load Multinomial Navie Bayes and Tf-IDF Vectorizer from disk
filename = 'movie-reviews-sentiment-model-multinomialNB.pkl'
classifer = pickle.load(open(filename,'rb'))
tfid = pickle.load(open('tfidf-transform.pkl','rb'))

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = tfid.transform(data)
        my_prediction = classifer.predict(vect)
        return render_template('result.html',prediction=my_prediction)
    
if __name__ == '__main__':
 app.run(debug=True)
