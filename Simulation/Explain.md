# Explanation

## **Model_Regresivo.py**

### 2. Autoregressive Model (AR)

**General Formula**  
The AR model is described as:

$$ S(t) = \sum_{i=1}^p \phi_i S(t - i) + \epsilon(t) $$

- **p**: Order of the model (number of lags).  
- **$$\phi_i$$**: Coefficients of the AR model.  
- **$$\epsilon(t)$$**: Gaussian white noise.  
