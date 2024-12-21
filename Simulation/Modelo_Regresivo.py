import numpy as np
from scipy.signal import lfilter
import matplotlib.pyplot as plt 

np.random.seed(42)
n_samples=5000#Numero de muestras
order=4 #Orden del AR
phi=[0.9, -0.5, 0.3, -0.1] #Coeficientes del AR
sigma=0.5 #Varianza del ruido 

#Generar ruido blanco
epsilon=np.random.normal(0, sigma, n_samples)

#Generar ruido AR
ar_coeffs=[1]+[-coef for coef in phi]   
ar_signal=lfilter([1], ar_coeffs, epsilon)

plt.plot(ar_signal,label='Ruido AR')
plt.title('Ruido AR')
plt.xlabel('Tiempo')
plt.ylabel('Amplitud')
plt.legend()
plt.show()
