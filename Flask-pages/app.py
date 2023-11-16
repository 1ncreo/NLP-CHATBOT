from flask import Flask, render_template, request
import pickle 
import numpy as np

model = pickle.load(open('data.pkl','rb'))

app = Flask(__name__)

@app.route('/')
def man():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def home():
    data1 = request.form['a']
    
   # arr = np.array([[data1]])
   # pred = model.predict(arr)
    
    jarvis = model.predict(data1)
    return render_template('index.html', data= jarvis)

if __name__ == "__main__":
    app.run(debug=True)
