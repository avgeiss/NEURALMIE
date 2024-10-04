import numpy as np
from scipy.special import erf, erfinv
from tensorflow.keras.models import load_model
from bulk_optics import mass_efficiency
ann = load_model('./coreshell.h5',compile=False)
RHO = 1E3

def rayleigh_approx(wavelength, mu, sigma, m_real, m_imag, rho):
    
    #called by 'ann_bulk_optics' in cases where the Rayleigh approximation can be used instead of the neural network
    
    mu_x = 2*np.pi*mu/wavelength    #geometric mean radius of the size distribution expressed as a size parameter    
    m = m_real + 1j*m_imag          #complex refractive index
    
    #calculate the bounds of integration as a size parameter
    integration_bounds = np.exp(erfinv(2*np.array([.0005,0.9995])-1)*np.log(sigma)*np.sqrt(2))*mu
    [x1,x2] = 2*np.pi*integration_bounds/wavelength
    
    #scattering calculation:
    lns = np.log(sigma)
    int_top = erf((6*lns**2 - np.log(x2/mu_x))/(lns*2**0.5)) - erf((6*lns**2 - np.log(x1/mu_x))/(lns*2**0.5))
    int_bot = erf((3*lns**2 - np.log(x2/mu_x))/(lns*2**0.5)) - erf((3*lns**2 - np.log(x1/mu_x))/(lns*2**0.5))
    coef = (4*np.pi*mu_x**3) * np.exp(13.5*lns**2) * np.abs((m**2-1)/(m**2+2))**2
    ks = (coef*int_top/int_bot)/(wavelength*rho)
    
    #absorption calculation:
    ka = 6*np.pi*((m**2-1)/(m**2+2)).imag/(wavelength*rho)
    
    #get the mass extinction coefficient and the single scattering albedo:
    ke = ks+ka
    ssa = ks/ke
    
    return ke, ssa, 0

def ann_bulk_optics(wavelength, mu, sigma, m_real, m_imag, m_real_core, m_imag_core, f, rho=RHO):
    
    #inputs are:
    #wavelength: wavelength of light (center of band) (m)
    #mu: geometric mean radius of the size distrubution (m)
    #sigma: geometric standard deviation of the size distribution (prescribed in MAM)
    #m_real and m_imag: real and imaginary components of the refractive index of the shell (both positive)
    #m_real_core, m_imag_core: real and imaginary components of the refractive index of the core (both positive)
    #f: fraction of the particle radius of the core
    #rho: the density of the aerosol species (kg m^-3), (for testing I am using RHO = 1E3 for everything, this input isn't used by the neural network)
    
    mu_x = 2*np.pi*mu/wavelength    #geometric mean radius of the size distribution expressed as a size parameter    
    
    #when the particles are small enough use a Rayleigh approximation instead of the ANN:
    if mu_x*np.exp(np.sqrt(2) * np.log(sigma) * erfinv(0.999)) <= 0.1: #(99.9% of the particles have size parameter less than 0.1)
        m_real = m_real_core*f**3 + m_real*(1-f**3)
        m_imag = m_imag_core*f**3 + m_imag*(1-f**3)
        return rayleigh_approx(wavelength, mu, sigma, m_real, m_imag, rho)
    
    #scale the dimensional inputs for use by the neural network
    mu_x = (np.log(mu_x) + 1.5)/3.6     
    sigma = 2*sigma-4
    m_real = 2*m_real-4
    m_imag = (np.log(m_imag) + 9)/5
    m_real_core = 2*m_real_core-4
    m_imag_core = (np.log(m_imag_core) + 9)/5
    f = f*4-2
    
    #apply the NN
    outputs = ann(np.array([mu_x,sigma,m_real,m_imag,m_real_core,m_imag_core,f])[None,:]).numpy().squeeze()
    
    #scale the outputs:
    ke = np.exp(outputs[0])/(wavelength*rho)    #the mass extinction coefficient (m^2 kg^-1)
    ssa = 1/(1+np.exp(-outputs[1]))             #the single scattering albedo (dimensionless)
    g = 1/(1+np.exp(-outputs[2]))               #the asymmetry parameter (dimensionless)
    
    #scattering and absorption can be calculated like this:
    #ks = ke*ssa  #the mass scattering coefficient (m^2kg^-1)
    #ka = ke-ks   #the mass absorption coefficient (m^2kg^-1)
    
    return ke, ssa, g

#this calls a function to compute the solution using Mie code and numerical integration:
def mie_bulk_optics(inputs):
    ke_rho, ks_rho, g = mass_efficiency(*inputs)
    return ke_rho/RHO, ks_rho/ke_rho, g

#test cases that produce a range of ks and ka values:
print('ANN outputs should be within about 1% of Mie code (see paper) \n')
    
inputs = [2.072e-05, 1.867e-06, 2.565e+00, 2.726e+00, 4.977e-06, 1.541e+00, 2.045e-03, 0.2]
print('Mie code:    ', mie_bulk_optics(inputs))
print('ANN Outputs: ', ann_bulk_optics(*inputs),'\n')

inputs = [4.2e-07, 1.414e-08, 2.307, 1.231, 7.709e-01, 2.275, 2.738e-03, 0.85]
print('Mie code:    ', mie_bulk_optics(inputs))
print('ANN Outputs: ', ann_bulk_optics(*inputs),'\n')

inputs =  [2.376e-07, 1.681e-08, 1.838, 2.275, 2.738e-03, 2.565e+00, 2.726e-4, 0.95]
print('Mie code:    ', mie_bulk_optics(inputs))
print('ANN Outputs: ', ann_bulk_optics(*inputs),'\n')

inputs = [5.306e-04, 4.420e-06, 1.664, 2.281, 1.264e-06, 1.3, 0.1, 0.1]
print('Mie code:    ', mie_bulk_optics(inputs))
print('ANN Outputs: ', ann_bulk_optics(*inputs),'\n')

#test cases using the Rayleigh approximation
print('Cases using the Rayleigh approximation (this will be less accurate for core-shell):\n')
inputs = [1.085e-04, 6.545e-08, 1.356, 2.467, 1.293e-03, 2.275, 2.738e-03, 0.05]
print('Mie code:    ', mie_bulk_optics(inputs))
print('ANN Outputs: ', ann_bulk_optics(*inputs),'\n')

inputs = [4.274e-06, 6.254e-09, 1.649, 2.730, 1.648e-04, 1.231, 7.745e-02, 0.8]
print('Mie code:    ', mie_bulk_optics(inputs))
print('ANN Outputs: ', ann_bulk_optics(*inputs),'\n')

'''
Expected output:
ANN outputs should be within about 1% of Mie code (see paper) 

Mie code:     (162.1302266770385, 0.9996692070776021, 0.36228318089696215)
ANN Outputs:  (162.55949454878285, 0.9994967193171056, 0.36263933886492605) 

Mie code:     (17614.50875842452, 0.5588602548842743, 0.47281065031467623)
ANN Outputs:  (17560.944103059315, 0.5589373098427436, 0.4730276759214842) 

Mie code:     (38929.42966432101, 0.997645698440115, 0.30941471627079425)
ANN Outputs:  (38935.05443226208, 0.9976111161675837, 0.30885781847015564) 

Mie code:     (0.025027849355997795, 0.9627966993796945, 0.011135903567231062)
ANN Outputs:  (0.025051913781971292, 0.9629013692227235, 0.011243319582558135) 

Cases using the Rayleigh approximation (this will be less accurate for core-shell):

Mie code:     (0.0508713960151816, 0.00016063577013797906, 1.4706035688541834e-05)
ANN Outputs:  (0.05086887281606239, 0.0001606394207435787, 0) 

Mie code:     (48.20767657500021, 0.00021835539406120597, 0.00039716741870908235)
ANN Outputs:  (60.26607442778232, 0.00016836331283013583, 0) 
'''