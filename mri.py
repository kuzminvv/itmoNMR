

from flask import Flask
from flask import render_template
from flask import request
import matplotlib.pyplot as plt
import numpy as np

app = Flask(__name__)

@app.route('/hello')
def helloIndex():
	return str(5*5)

@app.route('/start')
def template():
    return render_template('index.html')

@app.route('/run', methods=['GET', 'POST'])
def response():
	'''
    Обработчик, который принимает данные от пользователя,
     и возвращает обновлённую с ними
    object = request.form.get("object")
    gradient = request.form.get("gradient")
	'''
	if request.method == 'POST':
		object = request.form.get('object')
		gradient = request.form.get('gradient')
		return "<h1>The object value is: {}</h1><h1>The gradient value is: {}</h1>".format(object, gradient)

app.run(host='0.0.0.0', port=5000)
