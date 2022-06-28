import base64
from flask import Flask, render_template, request
import matplotlib.pyplot as plt
import numpy as np
# import scipy
from scipy.integrate import simps
from numpy import inf
import io


app = Flask(__name__)

@app.route('/hello')
def helloIndex():
	return str(5*5)

@app.route('/start')
def template():
	return render_template('index.html')

@app.route('/run_0_img', methods=['GET', 'POST'])
def response_0_img():
	'''
	Обработчик, для генерации нулевой картинки при изменении данных формы на лету
	'''
	if request.method == 'POST':

		gradient = request.form.get('gradient')
		object = request.form.get('object')
		grad_k_max = int(request.form.get('k_max'))*0.01
		dk_scale = int(request.form.get('dk'))
		dt = int(request.form.get('dt'))

		file = open("static_data/"+object+".csv")

		t2 = np.loadtxt(file, delimiter=",")

		def rectangular_grid(t1,t2,resolution):
			FOV = t1.shape
			x = np.linspace(-round(FOV[0]*resolution/2,1), round(FOV[0]*resolution/2,1)-resolution, FOV[0])
			y = np.linspace(-round(FOV[1]*resolution/2,1), round(FOV[1]*resolution/2,1)-resolution, FOV[1])
			return x, y


		# k-space
		def k_space(FOV, resolution):
			k_fov = 1/resolution #1/mm
			dk_x = 1/FOV[0]/resolution #1/mm
			dk_y = 1/FOV[1]/resolution #1/mm
			N = round(k_fov/dk_x)
			k_map = np.zeros(shape=(N,N), dtype=complex)
			return k_map


		def snake_grad(FOV1, resolution, grad_k_max, dk_scale):
			k_map = k_space(FOV1, resolution)

			k_fov_x = 1/resolution #1/mm
			k_fov_y = 1/resolution #1/mm

			dk_x = 1/FOV1[0]/resolution #1/mm
			dk_y = 1/FOV1[1]/resolution #1/mm

			Gx = 200*1e-6 #T/mm
			Gy = 200*1e-6 #T/mm
			gamma = 42.57*1e6 # Hz/T
			k_x_max = k_fov_x/2 #1/mm
			k_y_max = k_fov_y/2 #1/mm

			t1_x = grad_k_max * k_x_max * 2 * np.pi / gamma / Gx #sec
			t1_y = grad_k_max * k_y_max * 2 * np.pi / gamma / Gy #sec
			t2_x = grad_k_max * 2 * k_x_max * 2 * np.pi / gamma / Gx #sec
			t3_y = dk_scale * dk_y * 2 * np.pi / gamma / Gy #sec

			t1_x = round(t1_x*1e6) #mu sec
			t1_y = round(t1_y*1e6) #mu sec
			t2_x = round(t2_x*1e6) #mu sec
			t3_y = round(t3_y*1e6) #mu sec

			# dt = round(dk_x * 2 * np.pi / gamma / Gx *1e6) #mu sec

			GTx = -np.ones(shape=(t1_x,))
			GTy = np.ones(shape=(t1_y,))

			#change range number
			for i in range(round(FOV1[0]*grad_k_max/dk_scale)+1):
			# for i in range(32):
				if i % 2 != 0:
					sign = -1
				else:
					sign = 1
				add_x = sign * np.ones(shape=(t2_x,))
				add_y = np.zeros(shape=(t2_x,))
				GTx = np.concatenate((GTx,add_x), axis=0)
				GTy = np.concatenate((GTy,add_y), axis=0)

				add_x = np.zeros(shape=(t3_y,))
				add_y = -np.ones(shape=(t3_y,))
				GTx = np.concatenate((GTx,add_x), axis=0)
				GTy = np.concatenate((GTy,add_y), axis=0)

			return GTx*Gx, GTy*Gy

		def circles_grad(FOV1, resolution, grad_k_max, dk_scale):
			pass

		def spiral_grad(FOV1, resolution, grad_k_max, dk_scale):
			pass

		def find_coord(k_i, k_j, FOV, resolution):
			k_x_max = 1/resolution/2 #1/mm
			k_y_max = 1/resolution/2 #1/mm


			kx = np.linspace(-k_x_max, k_x_max-(1/FOV1[0]), FOV1[0]) + 1/FOV1[0]
			ky = np.linspace(-k_y_max, k_y_max-(1/FOV1[1]), FOV1[1]) + 1/FOV1[1]
			ii = np.argmin(abs(kx - k_i))
			jj = np.argmin(abs(ky - k_j))

			return ii, jj


		resolution = 0.8 #mm
		gamma = 42.57*1e6 # Hz/T

		if (max(t2.shape)-min(t2.shape)) != 0:
			max_sh = max(t2.shape)
			min_sh = min(t2.shape)
			delta = max_sh - min_sh
			if t2.shape[0] > t2.shape[1]:
				zer = np.zeros(shape=(max_sh,delta//2))
				t2_test = np.concatenate((zer, t2, zer), axis=1)
			else:
				zer = np.zeros(shape=(delta//2,max_sh))
				t2_test = np.concatenate((zer, t2, zer), axis=0)
		else:
			t2_test = t2

		FOV1 = t2_test.shape

		#Градиенты
		if gradient == 'snake':
			GTx, GTy = snake_grad(t2_test.shape, resolution, grad_k_max, dk_scale)
		elif gradient == 'circles':
			GTx, GTy = circles_grad(t2_test.shape, resolution, grad_k_max, dk_scale)
		else:
			GTx, GTy = spiral_grad(t2_test.shape, resolution, grad_k_max, dk_scale)

		N = round(GTx.shape[0])
		t = np.linspace(0,N,N)

		traj = k_space(FOV1, resolution)

		for k in range(1,N,dt):

			#EPI
			kx_coord = simps(GTx[:k]) * gamma / 2 / np.pi/1e6
			ky_coord = simps(GTy[:k]) * gamma / 2 / np.pi/1e6
			k_i, k_j = find_coord(kx_coord, ky_coord, FOV1, resolution)
			traj[k_i, k_j] = 1+0*1j

		plt.imshow(abs(traj))
		buf = io.BytesIO()
		plt.savefig(buf, format='png')
		buf.seek(0)
		img0 = base64.b64encode(buf.getvalue()).decode()
		# plt.savefig('k_map_figure.png')


		return {
			'img0': img0

		}


@app.route('/run_new', methods=['GET', 'POST'])
def response_new():
	'''
	Обработчик, который принимает данные от пользователя,
	 и возвращает обновлённую с ними
	object = request.form.get("object")
	gradient = request.form.get("gradient")
	'''

	if request.method == 'POST':

		gradient = request.form.get('gradient')
		object = request.form.get('object')
		grad_k_max = int(request.form.get('k_max'))*0.01
		dk_scale = int(request.form.get('dk'))
		dt = int(request.form.get('dt'))

		file = open("static_data/"+object+".csv")

		t2 = np.loadtxt(file, delimiter=",")

		def rectangular_grid(t1,t2,resolution):
			FOV = t1.shape
			x = np.linspace(-round(FOV[0]*resolution/2,1), round(FOV[0]*resolution/2,1)-resolution, FOV[0])
			y = np.linspace(-round(FOV[1]*resolution/2,1), round(FOV[1]*resolution/2,1)-resolution, FOV[1])
			return x, y

		def signal(M, t2, resolution=0.8):
			x, y = rectangular_grid(t2,t2,resolution)
			M_re = simps(simps(M.real,y),x)
			M_im = simps(simps(M.imag,y),x)
			return M_re+1j*M_im


		# k-space
		def k_space(FOV, resolution):
			k_fov = 1/resolution #1/mm
			dk_x = 1/FOV[0]/resolution #1/mm
			dk_y = 1/FOV[1]/resolution #1/mm
			N = round(k_fov/dk_x)
			k_map = np.zeros(shape=(N,N), dtype=complex)
			return k_map


		def snake_grad(FOV1, resolution, grad_k_max, dk_scale):
			k_map = k_space(FOV1, resolution)

			k_fov_x = 1/resolution #1/mm
			k_fov_y = 1/resolution #1/mm

			dk_x = 1/FOV1[0]/resolution #1/mm
			dk_y = 1/FOV1[1]/resolution #1/mm

			Gx = 200*1e-6 #T/mm
			Gy = 200*1e-6 #T/mm
			gamma = 42.57*1e6 # Hz/T
			k_x_max = k_fov_x/2 #1/mm
			k_y_max = k_fov_y/2 #1/mm

			t1_x = grad_k_max * k_x_max * 2 * np.pi / gamma / Gx #sec
			t1_y = grad_k_max * k_y_max * 2 * np.pi / gamma / Gy #sec
			t2_x = grad_k_max * 2 * k_x_max * 2 * np.pi / gamma / Gx #sec

			# t1_x = k_x_max/10 * 2 * np.pi / gamma / Gx #sec
			# t1_y = k_y_max/10 * 2 * np.pi / gamma / Gy #sec
			# t2_x = 2 * k_x_max/10 * 2 * np.pi / gamma / Gx #sec

			t3_y = dk_scale * dk_y * 2 * np.pi / gamma / Gy #sec

			t1_x = round(t1_x*1e6) #mu sec
			t1_y = round(t1_y*1e6) #mu sec
			t2_x = round(t2_x*1e6) #mu sec
			t3_y = round(t3_y*1e6) #mu sec

			# dt = round(dk_x * 2 * np.pi / gamma / Gx *1e6) #mu sec

			GTx = -np.ones(shape=(t1_x,))
			GTy = np.ones(shape=(t1_y,))

			#change range number
			for i in range(round(FOV1[0]*grad_k_max/dk_scale)+1):
			# for i in range(32):
				if i % 2 != 0:
					sign = -1
				else:
					sign = 1
				add_x = sign * np.ones(shape=(t2_x,))
				add_y = np.zeros(shape=(t2_x,))
				GTx = np.concatenate((GTx,add_x), axis=0)
				GTy = np.concatenate((GTy,add_y), axis=0)

				add_x = np.zeros(shape=(t3_y,))
				add_y = -np.ones(shape=(t3_y,))
				GTx = np.concatenate((GTx,add_x), axis=0)
				GTy = np.concatenate((GTy,add_y), axis=0)

			return GTx*Gx, GTy*Gy

		def circles_grad(FOV1, resolution, grad_k_max, dk_scale):
			pass

		def spiral_grad(FOV1, resolution, grad_k_max, dk_scale):
			pass

		def find_coord(k_i, k_j, FOV, resolution):
			k_x_max = 1/resolution/2 #1/mm
			k_y_max = 1/resolution/2 #1/mm


			kx = np.linspace(-k_x_max, k_x_max-(1/FOV1[0]), FOV1[0]) + 1/FOV1[0]
			ky = np.linspace(-k_y_max, k_y_max-(1/FOV1[1]), FOV1[1]) + 1/FOV1[1]
			ii = np.argmin(abs(kx - k_i))
			jj = np.argmin(abs(ky - k_j))

			return ii, jj


		resolution = 0.8 #mm
		gamma = 42.57*1e6 # Hz/T

		if (max(t2.shape)-min(t2.shape)) != 0:
			max_sh = max(t2.shape)
			min_sh = min(t2.shape)
			delta = max_sh - min_sh
			if t2.shape[0] > t2.shape[1]:
				zer = np.zeros(shape=(max_sh,delta//2))
				t2_test = np.concatenate((zer, t2, zer), axis=1)
			else:
				zer = np.zeros(shape=(delta//2,max_sh))
				t2_test = np.concatenate((zer, t2, zer), axis=0)
		else:
			t2_test = t2

		FOV1 = t2_test.shape

		# Начаальные условия
		Mt = np.zeros(shape=t2_test.shape)
		for i in range(t2_test.shape[0]):
			for j in range(t2_test.shape[1]):
				if t2_test[i,j]!=0:
					Mt[i,j]=1


		#Градиенты
		if gradient == 'snake':
			GTx, GTy = snake_grad(t2_test.shape, resolution, grad_k_max, dk_scale)
		elif gradient == 'circles':
			GTx, GTy = circles_grad(t2_test.shape, resolution, grad_k_max, dk_scale)
		else:
			GTx, GTy = spiral_grad(t2_test.shape, resolution, grad_k_max, dk_scale)

		N = round(GTx.shape[0])
		t = np.linspace(0,N,N)

		xx, yy = rectangular_grid(t2_test,t2_test,resolution)
		xv, yv = np.meshgrid(xx,yy)
		rect = xv + 1j*yv
		Mt_evo = np.zeros(shape=(Mt.shape[0],Mt.shape[1]),dtype=complex)
		k_map = k_space(FOV1, resolution)

		for k in range(1,N,dt):

			#EPI
			t_int = k
			kx_coord = simps(GTx[:k]) * gamma / 2 / np.pi/1e6
			ky_coord = simps(GTy[:k]) * gamma / 2 / np.pi/1e6
			k_i, k_j = find_coord(kx_coord, ky_coord, FOV1, resolution)

			Mt_evo = Mt * np.exp(-t[k]*1e-6/t2_test) * np.exp(-1j*2*np.pi*(kx_coord*rect.imag+ky_coord*rect.real))

			k_map[k_i,k_j] = signal(Mt_evo, t2_test)

		#image reconstruction from k-space
		k_map_fft = np.fft.fftshift(k_map)
		k_map_ifft = np.fft.ifft2(k_map_fft)
		k_map_fft = np.fft.fftshift(k_map_ifft)


		plt.imshow(abs(k_map))
		# plt.savefig('t_map_figure.png')
		buf = io.BytesIO()
		plt.savefig(buf, format='png')
		buf.seek(0)
		img1 = base64.b64encode(buf.getvalue()).decode()


		plt.imshow(abs(k_map_fft))
		# plt.savefig('t_map_figure.png')
		buf = io.BytesIO()
		plt.savefig(buf, format='png')
		buf.seek(0)
		img2 = base64.b64encode(buf.getvalue()).decode()



		return {
			'img1': img1,
			'img2': img2

		}

		# return render_template('index.html', object=object, gradient=gradient)

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

		file = open("static_data/"+str(object)+".csv")

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
