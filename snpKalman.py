import numpy as np
import matplotlib.pyplot as plt
import filterpy.stats as stats
from collections import namedtuple
import pandas as pd

def predict(pos, movement):
    return gaussian(pos.mean + movement.mean, pos.var + movement.var)
def gaussian_multiply(g1, g2):
    mean = (g1.var * g2.mean + g2.var * g1.mean) / (g1.var + g2.var)
    variance = (g1.var * g2.var) / (g1.var + g2.var)
    return gaussian(mean, variance)
def update(prior, likelihood):
    posterior = gaussian_multiply(likelihood, prior)
    return posterior
gaussian = namedtuple('Gaussian', ['mean', 'var'])
gaussian.__repr__ = lambda s: '𝒩(μ={:.3f}, 𝜎²={:.3f})'.format(s[0], s[1])

date=[]
price=[]
data = pd.read_csv("snpData.csv") 
date=data.iloc[:,0]
price=data.iloc[:,5]
date=date.tolist()
price=price.tolist()

np.random.seed(13)

process_var = 5. 
sensor_var = 3. 
x = gaussian(800., 10.**2)  # dog's position, N(0, 20**2)
velocity = 10
dt = 1. # time step in seconds
process_model = gaussian(velocity*dt, process_var) # displacement to add to x


for i in range (0,len(price)):
    prior = predict(x, process_model)
    likelihood = gaussian(price[i], sensor_var)
    x = update(prior, likelihood)
    print()
    print('actual final position: {:10.3f}'.format(price[i]))
    print('final estimate:        {:10.3f}'.format(x.mean))

