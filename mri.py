from flask import Flask, render_template, request
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

		gradient = request.form.get('gradient')
		object = request.form.get('object')

		file = open("static_data/"+object+".csv")

		t2 = np.loadtxt(file, delimiter=",")

		w = 20
		gamma = 2

		def whirlpool_matrix(n=9):
		    mat = np.array([[0]*n]*n)
		    value = 0
		    i, j = -1, n - 1
		    for offset in range(n, 0, -1):
		        value -= 1
		        i += 1
		        mat[i][j] = value
		    for m in range(n - 1, 0, -1):
		        for offset in range(m, 0, -1):
		            value -= 1
		            j += (-1) ** (m + n)
		            mat[i][j] = value
		        for offset in range(m, 0, -1):
		            value -= 1
		            i += (-1) ** (m + n)
		            mat[i][j] = value
		    return mat + n*n

		mat = whirlpool_matrix(40)
		Maa = mat/160

		kx_new = np.linspace(-5., 5., 40)
		ky_new = np.linspace(-5., 5., 40)

		new_t = np.linspace(0., 10., 1600)

		def M_new(i, j):
		    acc = 0
		    kx_za = kx_new[i]
		    ky_za = ky_new[j]
		    t = Maa[i,j]
		    for x in (range(t2.shape[0])):
		        for y in range(t2.shape[1]):
		            if t2[x,y] != 0:
		                acc += np.exp(-(gamma * (((x - 140) * kx_za) + ((y-112) * ky_za)))*1j) * np.exp(-t/t2[x,y])
		    return acc.real, acc.imag

		res_new = np.zeros((40,40,2))

		for i in (range(40)):
		    for j in range(40):
		        res_new[i,j,0], res_new[i,j,1] = 1,1 #M_new(i,j)

		C = np.zeros((40,40),dtype=np.complex_)
		C = res_new[:,:,0] + res_new[:,:,1]*1j
		a = (np.fft.ifft2(C))
		plt.imshow((abs(a)))
		plt.savefig('saved_figure.png')


		return render_template('index.html', array_size=np.shape(t2)[0], object=object, gradient=gradient)
		return "<h1>The object value is: {}</h1><h1>The gradient value is: {}</h1><h1>Array size is{}</h1>".format(object, gradient, np.shape(t2)[0])



app.run(host='0.0.0.0', port=5000)
