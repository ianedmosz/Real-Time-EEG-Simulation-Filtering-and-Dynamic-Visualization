import numpy as np
from colorednoise import powerlaw_psd_gaussian
import matplotlib.pyplot as plt


beta=1.0
n_samples = 1000   

fractal_noise = powerlaw_psd_gaussian(beta, n_samples)

plt.plot(fractal_noise,label='Ruido Fractal')
plt.title('Ruido Fractal')
plt.xlabel('Tiempo') 
plt.ylabel('Amplitud')
plt.legend()
plt.show()
