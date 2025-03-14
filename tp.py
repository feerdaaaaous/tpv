import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

w1 = np.random.randn(3, 3) * 0.1 #multipe *0,1 pour eviter que le gradient devient proche de 0 (eviter les poids trop grands)
b1=np.random.randn(3,1)*0.1 
# w1 et b1 est les poids et biais (couche entree  3 neurones)
w2=np.random.randn(2,3)*0.1
b2=np.random.randn(2,1)*0.1
#entre la couch cachée1 (3neurones )et cachée2(2 neurones)
w3=np.random.randn(1,2)*0.1 
b3=np.random.randn(1,1)*0.1
#entre la couche cachée2 et sortie ( 1 neurones )
print(f"w1= \n {w1} \n b1= \n {b1}\n")
print(f"w2=\n {w2}\n b2= \n {b2}\n")
print(f"w3=\n {w3}\n b3= \n {b3}\n")

#decalaration des variable (les entrées 3 )
vars=np.array([[0.5],[0.2],[-0.3]])
#x=no.random.randn(3,1)

#propagation vers l'avant 
c1=np.dot(w1,vars)+b1
s1=sigmoid(c1) #matrice 3*1
c2=np.dot(w2,s1)+b2
s2=sigmoid(c2)#matrice 2*1
c3=np.dot(w3,s2)+b3
s=sigmoid(c3)
print(f"sortie = {s}")






