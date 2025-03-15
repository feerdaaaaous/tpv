import numpy as np

class MLP:
    def __init__(self):
        #multipe *0,1 pour eviter que le fonctions d'activation devient presque 0 (eviter les poids  trop grands ou trop petit)
        self.w1 = np.random.randn(3, 3) * 0.1
        self.b1 = np.random.randn(3, 1) * 0.1  
        # w1 et b1 est les poids et biais (couche entree  3 neurones)
        self.w2 = np.random.randn(2, 3) * 0.1
        self.b2 = np.random.randn(2, 1) * 0.1
        #entre la couch cachée1 (3neurones )et cachée2(2 neurones)
        self.w3 = np.random.randn(1, 2) * 0.1
        self.b3 = np.random.randn(1, 1) * 0.1
        #entre la couche cachée2 et sortie ( 1 neurones )
    def afficher_parametres(self):
        print(f"w1 = \n{self.w1}\n")
        print(f"b1 = \n{self.b1}\n")
        print(f"w2 = \n{self.b1}\n")
        print(f"b2 = \n{self.b1}\n")
        print(f"w3 = \n{self.b1}\n")
        print(f"b3 = \n{self.b1}\n")
    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))

    def propa_vers_avant(self, vars):
        self.c1 = np.dot(self.w1, vars) + self.b1
        self.s1 = self.sigmoid(self.c1)#matrice 3*1
        self.c2 = np.dot(self.w2, self.s1) + self.b2
        self.s2 = self.sigmoid(self.c2)#matrice 2*1
        self.c3 = np.dot(self.w3, self.s2) + self.b3
        self.s = self.sigmoid(self.c3)
        
        return self.s 
    
    def retropropagation(self,vars):
        sc=self.propa_vers_avant(vars)
        
reseau = MLP()
reseau.afficher_parametres()
vars = np.array([[0.5], [0.2], [-0.3]])#decalaration des variable (les entrées 3 )
sortie_finale = reseau.propa_vers_avant(vars)
print(f"sortie finale du réseau :\n{sortie_finale}")



































"""
print(f"w1= \n {self.w1} \n b1= \n {self.b1}\n")
print(f"w2=\n {self.w2}\n b2= \n {self.b2}\n")
print(f"w3=\n {self.w3}\n b3= \n {self.b3}\n")
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
"""





