import numpy as np
from scipy.special import erf, erfinv
from tensorflow.keras.models import load_model
from bulk_optics import mass_efficiency
ann = load_model('./sphere.h5',compile=False)
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

def ann_bulk_optics(wavelength, mu, sigma, m_real, m_imag, rho=RHO):
    
    #inputs are:
    #wavelength: wavelength of light (center of band) (m)
    #mu: geometric mean radius of the size distrubution (m)
    #sigma: geometric standard deviation of the size distribution (prescribed in MAM)
    #m_real and m_imag: real and imaginary components of the refractive index (both positive)
    #rho: the density of the aerosol species (kg m^-3), (for testing I am using RHO = 1E3 for everything, this input isn't used by the neural network)
    
    mu_x = 2*np.pi*mu/wavelength    #geometric mean radius of the size distribution expressed as a size parameter    
    
    #when the particles are small enough use a Rayleigh approximation instead of the ANN:
    if mu_x*np.exp(np.sqrt(2) * np.log(sigma) * erfinv(0.999)) <= 0.1: #(99.9% of the particles have size parameter less than 0.1)
        return rayleigh_approx(wavelength, mu, sigma, m_real, m_imag, rho)
    
    #scale the dimensional inputs for use by the neural network
    mu_x = (np.log(mu_x) + 1.5)/3.6     
    sigma = 2*sigma-4
    m_real = 2*m_real-4
    m_imag = (np.log(m_imag) + 9)/5
    
    #apply the NN
    outputs = ann(np.array([mu_x,sigma,m_real,m_imag])[None,:]).numpy().squeeze()
    
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
    
inputs = [2.072e-05, 1.867e-06, 2.565e+00, 2.726e+00, 4.977e-06]
print('Mie code:    ', mie_bulk_optics(inputs))
print('ANN Outputs: ', ann_bulk_optics(*inputs),'\n')

inputs = [4.2e-07, 1.414e-08, 2.307, 1.231, 7.709e-01]
print('Mie code:    ', mie_bulk_optics(inputs))
print('ANN Outputs: ', ann_bulk_optics(*inputs),'\n')

inputs =  [2.376e-07, 1.681e-08, 1.838, 2.275, 2.738e-03]
print('Mie code:    ', mie_bulk_optics(inputs))
print('ANN Outputs: ', ann_bulk_optics(*inputs),'\n')

inputs = [5.306e-04, 4.420e-06, 1.664, 2.281, 1.264e-06]
print('Mie code:    ', mie_bulk_optics(inputs))
print('ANN Outputs: ', ann_bulk_optics(*inputs),'\n')

#test cases using the Rayleigh approximation
print('Cases using the Rayleigh approximation:\n')
inputs = [1.085e-04, 6.545e-08, 1.356, 2.467, 1.293e-03]
print('Mie code:    ', mie_bulk_optics(inputs))
print('ANN Outputs: ', ann_bulk_optics(*inputs),'\n')

inputs = [4.274e-06, 6.254e-09, 1.649, 2.730, 1.648e-04]
print('Mie code:    ', mie_bulk_optics(inputs))
print('ANN Outputs: ', ann_bulk_optics(*inputs),'\n')


'''
Expected output:

ANN outputs should be within about 1% of Mie code (see paper) 

Mie code:     (162.24251805289094, 0.9999107993257795, 0.35062561238918993)
ANN Outputs:  (162.4925256235719, 0.9999040872823134, 0.3496360368434989) 

Mie code:     (17041.47712245558, 0.22927182541418883, 0.46123146770070467)
ANN Outputs:  (17031.456175304593, 0.22857165278242655, 0.4609948733150081) 

Mie code:     (34233.94281880267, 0.9891886826552949, 0.3730802403558004)
ANN Outputs:  (34244.305356985795, 0.9891149819424135, 0.3729642399101323) 

Mie code:     (0.024131849871891375, 0.9994970384772996, 0.011130771800989018)
ANN Outputs:  (0.0241331673970413, 0.9994851901145024, 0.011107804140884163) 

Cases using the Rayleigh approximation:

Mie code:     (0.05086312786119721, 0.0001606646903998998, 1.470590792156103e-05)
ANN Outputs:  (0.050860774283457734, 0.00016066777491833314, 0) 

Mie code:     (0.15325712048850026, 0.12979524831314324, 0.00040129915507320415)
ANN Outputs:  (0.15310873914425868, 0.12982299762436167, 0) 
'''