import numpy as np

def sigmoid(v):
    return 1 / (1 + np.exp(-v))

w1 = np.random.randn(3, 3) * 0.1 #multipe *0,1 pour eviter que le gradient devient proche de 0 (eviter les poids trop grands)
b1=np.random.randn(3,1)*0.1 
# w1 et b1 est les poids et biais (couche entree  3 neurones)
