import numpy as np

def sigmoid(v):
    return 1 / (1 + np.exp(-v))

w1 = np.random.randn(3, 3) * 0.1 #multipe *0,1 pour eviter que le gradient devient proche de 0 (eviter les poids trop grands)
b1=np.random.randn(3,1)*0.1 
# w1 et b1 est les poids et biais (couche entree  3 neurones)
w2=np.random.randn(2,3)*0,1
b2=np.random.randn(2,1)*0,1
#entre la couch cachée1 (3neurones )et cachée2(2 neurones)
w3=np.random.randn(1,2)*0.1 
b3=np.random.randn(1,1)*0.1
#entre la couche cachée2 et sortie ( 1 neurones )
print(f"w1= {w1} b1= {b1}")
print(f"w2={w2} et b2={b2}")
print(f"w3= {w3} b3={b3}")



