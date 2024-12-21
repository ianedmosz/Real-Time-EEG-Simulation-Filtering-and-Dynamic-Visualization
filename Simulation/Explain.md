# Explanation

## **Model_Regresivo.py**
The Model_Regresive.py is a code of the Autoregressive Mode that Captures the temporal dynamics of the signal.

### 1. Autoregressive Model (AR)
**General Formula**  
The AR model is described as:

$$ S(t) = \sum_{i=1}^p \phi_i S(t - i) + \epsilon(t) $$

- **p**: Order of the model (number of lags).  
- **$$\phi_i$$**: Coefficients of the AR model.  
- **$$\epsilon(t)$$**: Gaussian white noise.  

## **ruid_fractal.py**
Adds the characteristic $$1/𝑓$$ noise typical of EEG signals.

### 2. Fractal Noise

Fractal noise $$\frac{1}{f}$$ follows a power spectral density proportional to $$\frac{1}{f^\beta}$$, where:

$$
S(f) \propto \frac{1}{f^\beta}
$$

- **$$(f\)$$**: Frequency.  
- **$$(\beta\)$$**: Power-law exponent that controls the "color" of the noise (e.g., white noise, pink noise, etc.).

## **RF_AR.py**
Is a combination of all tree models. 

 **Sinusoidal Components:**  
   $$S_{sin}(t) = \sum_{k} A_k \sin(2\pi f_k t + \phi_k)$$

   - $$( A_k \)$$: Amplitude of the \( k \)-th band.  
   - $$( f_k \)$$: Central frequency of the \( k \)-th band.  
   - $$( \phi_k \)$$: Random phase of the \( k \)-th band.
     
### Combined Formula

The simulated EEG signal is the sum of the three components:

$$
S_{combined}(t) = S_{AR}(t) + N_{fractal}(t) + S_{sin}(t)
$$


##**Result of the Simulated EEG:**

### Comparison of Simulated and Real EEG Signals

![Comparison of Simulated and Real EEG Signals](simulated_vs_real_eeg_signal.png)


