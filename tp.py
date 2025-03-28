import numpy as np
np.set_printoptions(threshold=np.inf, floatmode='unique', suppress=True)#this is for display the data full float numbers


class MLP:
    def __init__(self):
        self.alpha=0.1
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
        print(f"w2 = \n{self.w2}\n")
        print(f"b2 = \n{self.b2}\n")
        print(f"w3 = \n{self.w3}\n")
        print(f"b3 = \n{self.b3}\n")
    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))
    def df_sigmoid(self, x):
        return x * (1 - x)
    def propa_vers_avant(self, vals):
        self.c1 = np.dot(self.w1, vals) + self.b1
        self.s1 = self.sigmoid(self.c1)#matrice 3*1
        self.c2 = np.dot(self.w2, self.s1) + self.b2
        self.s2 = self.sigmoid(self.c2)#matrice 2*1
        self.c3 = np.dot(self.w3, self.s2) + self.b3
        self.s = self.sigmoid(self.c3)

        return self.s 
    def retropropagation(self,x,s_reel,s_calculer):
        error=s_reel-s_calculer
     #calcule des gradients et miss a jour les poids entre la couche sortie et la couche cachee2
        delta_s=error*self.df_sigmoid(s_calculer)
        delta_w3=np.dot(delta_s,self.s2.T) #.T is for converts the 2,1 vector to 1,2 row vector 
        delta_b3=delta_s

        self.w3+=self.alpha*delta_w3
        self.b3+=self.alpha*delta_b3
     #calcule des gradients et miss a jour les poids entre la couche cachee2 et la couche cachee1
        delta_s2=np.dot(self.w3.T,delta_s)*self.df_sigmoid(self.s2)
        delta_w2=np.dot(delta_s2,self.s1.T)
        delta_b2=delta_s2

        self.w2+=self.alpha*delta_w2
        self.b2+=self.alpha*delta_b2
     #calcule des gradients et miss a jour les poids entre la couche cachee1 et lentree
        delta_s1=np.dot(self.w2.T,delta_s2)*self.df_sigmoid(self.s1)
        delta_w1=np.dot(delta_s1,x.T)
        delta_b1=delta_s1
        
        self.w1+=self.alpha*delta_w1
        self.b1+=self.alpha*delta_b1
    

        
   
        

reseau = MLP()
reseau.afficher_parametres()

#chargement des données
x=[]
srx=[]
with open ("data.txt","r") as file:
   for l in file:
      l=l.strip()
      if l:
       val =[float (num) for num in l.split()]# reading each entrees 
       x.append(val[:-1])
       srx.append(val[-1])

x=np.array(x,dtype=np.float64)
srx=np.array(srx)

#application de la fonction propagation avant 
scx=[]
for i,exemple in enumerate(x):
    xcolumn=np.array(exemple).reshape(-1,1)#convertir to column vector 
    sortie=reseau.propa_vers_avant(xcolumn)
    scx.append(sortie)
    print(f"\n exemple {i+1} => {exemple}")
    print(f"sortie attendue = {srx[i]}")
    print(f"sortie calcule = {sortie}")
#application de la fonction backpropagation 
epsilon=0.001
maxiter=100
for _ in range (maxiter):
    nbr_x=np.arange(len(x))
    np.random.shuffle(nbr_x)
    x=x[nbr_x]
    srx=srx[nbr_x]
    scx=[scx[i] for i in nbr_x]
    for i,exemple in enumerate(x):
        xcolumn=np.array(exemple).reshape(-1,1)
        reseau.retropropagation(xcolumn,srx[i],scx[i])
print("apres entrainement ")
for i,exemple in enumerate(x):
    xcolumn=np.array(exemple).reshape(-1,1)
    sortie=reseau.propa_vers_avant(xcolumn)
    print(f"\n exemple {i+1} => {exemple}")
    print(f"sortie attendue = {srx[i]}")
    print(f"sortie calcule = {sortie}")












































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





