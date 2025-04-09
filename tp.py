import numpy as np
np.set_printoptions(threshold=np.inf, floatmode='unique', suppress=True)#this is for display the data full float numbers


class MLP:
    def __init__(self):
        self.alpha=0.001
        #change to xavier initialization pour maintenir la meme variance des activation entre les diff couches 
        self.w1 = np.random.randn(3, 3) * np.sqrt(2. / (3 + 3)) # 3+3 mean 3 entree et 3 couche c1 
        self.b1 = np.zeros((3, 1))
        # w1 et b1 est les poids et biais (couche entree  3 neurones)
        self.w2 = np.random.randn(2, 3) * np.sqrt(2. / (3 + 2))#3+2 mean 3 couche c1 et 2 couche c2 
        self.b2 = np.zeros((2, 1))
        #entre la couch cachée1 (3neurones )et cachée2(2 neurones)
        self.w3 = np.random.randn(1, 2) * np.sqrt(2. / (2 + 1))#2+1 mean 2 for couche c2 et la couche sortie 1 
        self.b3 = np.zeros((1, 1))
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
    
    def retropropagation(self,x,s_reel,prediction,iteration=10):
        
        
        for i in range(iteration):
            s_calculer=self.propa_vers_avant(x)
            error=np.sum((s_reel-s_calculer)**2)

            #calcule des gradients et miss a jour les poids entre la couche sortie et la couche cachee2
            delta_s=(s_calculer - s_reel)*self.df_sigmoid(s_calculer)
            delta_w3=np.dot(delta_s,self.s2.T) #.T is for converts the 2,1 vector to 1,2 row vector 
            delta_b3=delta_s
            #calcule des gradients et miss a jour les poids entre la couche cachee2 et la couche cachee1
            delta_s2=np.dot(self.w3.T,delta_s)*self.df_sigmoid(self.s2)
            delta_w2=np.dot(delta_s2,self.s1.T)
            delta_b2=delta_s2
            #calcule des gradients et miss a jour les poids entre la couche cachee1 et lentree
            delta_s1=np.dot(self.w2.T,delta_s2)*self.df_sigmoid(self.s1)
            delta_w1=np.dot(delta_s1,x.T)
            delta_b1=delta_s1
           
            #miss a jour les poids et biases
            self.w3-=self.alpha*delta_w3
            self.b3-=self.alpha*delta_b3
            self.w2-=self.alpha*delta_w2
            self.b2-=self.alpha*delta_b2
            self.w1-=self.alpha*delta_w1
            self.b1-=self.alpha*delta_b1

            
            
            s_calculer=self.propa_vers_avant(x)
            error=np.mean(np.square(s_reel - s_calculer))
            i+=1
            if error < 1e-4:
                print(f"exemple end early in iteration{i+1}")
                break
            
        #return the updated 
        final_pred=self.propa_vers_avant(x)[0][0]
        print(f"end with iteration")
        return prediction,final_pred
    """
    def entrainement(self,x,srx,maxiter=10,epsilon=1e-6):
        last_error=float('inf') #start with a large nbr of error 
        # before training il faut calculer les sortie avant la retropropagation
        sorties_avant=[]
        for exemple in x:
            xcolumn=np.array(exemple).reshape(-1,1)
            sortie=reseau.propa_vers_avant(xcolumn)
            sorties_avant.append(sortie)
      
        for _ in range (maxiter):
            print(f"~~~~~~~~~~~~~~iteration {_+1}~~~~~~~~~~~~~~")
            total_error=0
            nbr_x=np.arange(len(x))
            np.random.shuffle(nbr_x)
            
            xshuffle=x[nbr_x]
            srxshuffle=srx[nbr_x]
            sortie_avantshuffle= [sorties_avant[i] for i in nbr_x]
            for i,exemple in enumerate(xshuffle):
                xcolumn=np.array(exemple).reshape(-1,1)
                sortie_avant=sortie_avantshuffle[i]
                sortie_apres=reseau.retropropagation(xcolumn,srxshuffle[i],sortie_avant)
                exemple_error=np.sum(np.abs(srxshuffle[i]-sortie_apres))
                total_error+=exemple_error
                
                print(f"\nexemple {nbr_x[i]} => {exemple}")
                print(f"sortie attendue = {srxshuffle[i]}")
                print(f"sortie avant entraînement = {sortie_avant}")  # scx[i] was the output before training
                print(f"sortie après entraînement = {sortie_apres}")
                
            if abs(last_error-total_error)<epsilon:
                print(f"stopping earrly at iteration {_+1} with error = {total_error}")
                break
            last_error=total_error
            print(f"iteration {_+1} total error = {total_error}")
    """
    def entrainement(self,x,srx,maxiter=50):
        predicions_initail={}
        predictions_courrent={}
        for i,exemple in enumerate(x):
            xcolumn = np.array(exemple).reshape(-1, 1)
            pred = self.propa_vers_avant(xcolumn)[0][0]#we did [0][0]par ce que la sortie est une matrice [[valeur]]
            predicions_initail[i] = pred
            predictions_courrent[i] = pred
        for j in range (maxiter):
            print(f"~~~~~~~~~~~~~~iteration {j+1}~~~~~~~~~~~~~~")
            total_error=0
            #shuffle les exemples
            nbr_x=np.arange(len(x))
            np.random.shuffle(nbr_x)
            for ex in nbr_x:
                exemple =x[ex]
                sortie=srx[ex]

                xcolumn=np.array(exemple).reshape(-1,1)
                pred_before=predictions_courrent[ex]#pour voir le pred before start every iteration ( affichage )
                _,new_pred=self.retropropagation(xcolumn,sortie,pred,iteration=20)

                predictions_courrent[ex]=new_pred #stocker new prediction de chaque exemple a chaque iteration 
                #calcule error total 
                error=abs(sortie-new_pred)
                total_error+=error
                #affichage des calcules de chaque exemple 
                print(f"\nExemple {ex}: {exemple}")
                print(f"Sortie attendue: {sortie}")
                print(f"Sortie initiale: {predicions_initail[ex]:.6f}")
                print(f"Sortie avant mise à jour: {pred_before:.6f}")
                print(f"Sortie après mise à jour: {new_pred:.6f}")
            error_moy=total_error/len(x)
            print(f"\nEpoch {j+1} - Erreur moyenne: {error_moy:.6f}")

            #condition de covergence
            if error_moy<0.1:
               print(f"covergence attient a l'iteration {j+1}")
               break 
        #resultat final 
        print("\n~~~~ Résultats finaux ~~~~")
        for i,exemple in enumerate(x):
           print(f"Exemple {i}: {exemple}, Cible: {srx[i]}, Prédiction: {predictions_courrent[i]:.6f}")   
        

            

if __name__=="__main__":
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
reseau.entrainement(x,srx)
"""
#application de la fonction propagation avant 
scx=[]
for i,exemple in enumerate(x):
    xcolumn=np.array(exemple).reshape(-1,1)#convertir to column vector 
    sortie=reseau.propa_vers_avant(xcolumn)
    scx.append(sortie)
    print(f"\n exemple {i+1} => {exemple}")
    print(f"sortie attendue = {srx[i]}")
    print(f"sortie calcule = {sortie}")
"""









"""
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





