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
Adds the characteristic $$ 1/𝑓$$ noise typical of EEG signals.
