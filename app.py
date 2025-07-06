from flask import Flask, render_template, request
import sqlite3
from detect import Start
from live import Start_live
import os
import cv2
import numpy as np

connection = sqlite3.connect('user_data.db')
cursor = connection.cursor()

command = """CREATE TABLE IF NOT EXISTS user(name TEXT, password TEXT, mobile TEXT, email TEXT)"""
cursor.execute(command)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/home')
def home():
    return render_template('userlog.html')

@app.route('/userlog', methods=['GET', 'POST'])
def userlog():
    if request.method == 'POST':
        connection = sqlite3.connect('user_data.db')
        cursor = connection.cursor()

        name = request.form['name']
        password = request.form['password']

        query = "SELECT name, password FROM user WHERE name = ? AND password = ?"
        cursor.execute(query, (name, password))

        result = cursor.fetchall()

        if len(result) == 0:
            return render_template('index.html', msg='Sorry, Incorrect Credentials Provided, Try Again')
        else:
            return render_template('userlog.html')

    return render_template('index.html')

@app.route('/userreg', methods=['GET', 'POST'])
def userreg():
    if request.method == 'POST':
        connection = sqlite3.connect('user_data.db')
        cursor = connection.cursor()

        name = request.form['name']
        password = request.form['password']
        mobile = request.form['phone']
        email = request.form['email']

        print(name, mobile, email, password)

        command = """CREATE TABLE IF NOT EXISTS user(name TEXT, password TEXT, mobile TEXT, email TEXT)"""
        cursor.execute(command)

        cursor.execute("INSERT INTO user VALUES (?, ?, ?, ?)", (name, password, mobile, email))
        connection.commit()

        return render_template('index.html', msg='Successfully Registered')

    return render_template('index.html')

@app.route('/analyse', methods=['GET', 'POST'])
def analyse():
    if request.method == 'POST':
        image = request.form['img']
        PT = request.form['pt']

        path = f'static/test/{PT.split(".")[0]}/{image}'
        ImageDisplay = f"http://127.0.0.1:5000/{path}"

        if not os.path.exists(path):
            return render_template('userlog.html', msg="Error: Image not found!")

        Start(path, PT)
        fram = cv2.imread(path)

        if fram is None:
            return render_template('userlog.html', msg="Error: Could not read the image!")

        # Noise removal techniques
        median_filtered = cv2.medianBlur(fram, 5)
        gaussian_filtered = cv2.GaussianBlur(fram, (5, 5), 0)
        bilateral_filtered = cv2.bilateralFilter(fram, 9, 75, 75)

        cv2.imwrite('static/median_filtered.jpg', median_filtered)
        cv2.imwrite('static/gaussian_filtered.jpg', gaussian_filtered)
        cv2.imwrite('static/bilateral_filtered.jpg', bilateral_filtered)

        # Convert to grayscale for some operations
        gray = cv2.cvtColor(fram, cv2.COLOR_BGR2GRAY)

        # Thresholding
        _, threshold2 = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
        cv2.imwrite('static/threshold.jpg', threshold2)

        # Sharpening Kernel
        kernel_sharpening = np.array([[-1, -1, -1],
                                      [-1,  9, -1],
                                      [-1, -1, -1]])
        sharpened = cv2.filter2D(fram, -1, kernel_sharpening)
        cv2.imwrite('static/sharpened.jpg', sharpened)

        # Edge Detection (Canny requires grayscale)
        edges = cv2.Canny(gray, 100, 200)
        cv2.imwrite('static/edges.jpg', edges)

        return render_template('userlog.html', 
                               ImageDisplay=ImageDisplay, 
                               ImageDisplay1="http://127.0.0.1:5000/static/result/out.jpg",
                               ImageDisplay2="http://127.0.0.1:5000/static/median_filtered.jpg",
                               ImageDisplay3="http://127.0.0.1:5000/static/gaussian_filtered.jpg",
                               ImageDisplay4="http://127.0.0.1:5000/static/bilateral_filtered.jpg",
                               ImageDisplay5="http://127.0.0.1:5000/static/threshold.jpg",
                               ImageDisplay6="http://127.0.0.1:5000/static/sharpened.jpg",
                               ImageDisplay7="http://127.0.0.1:5000/static/edges.jpg")

    return render_template('userlog.html')

@app.route('/live', methods=['GET', 'POST'])
def live():
    if request.method == 'POST':
        PT = request.form['ptlive']
        Start_live(PT)
        return render_template('userlog.html')
    return render_template('userlog.html')

@app.route('/logout')
def logout():
    return render_template('index.html')

@app.route('/soldier')
def soldier():
    return render_template('soldier.html')

@app.route('/createdata', methods=['GET', 'POST'])
def createdata():
    if request.method == 'POST':
        solname = request.form['solname']
        from dataset import create_dataset
        create_dataset(solname)
        return render_template('soldier.html')
    return render_template('soldier.html')

@app.route('/recognise')
def recognise():
    os.system("python recognition.py")
    return render_template('soldier.html')

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
