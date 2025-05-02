import numpy as np
import pickle
np.set_printoptions(threshold=np.inf, floatmode='unique', suppress=True)#this is for display the data full float numbers


class MLP:
    def __init__(self):
        self.alpha=0.5    
        self.w1 = np.random.randn(3, 3) 
        self.b1 = np.zeros((3, 1))
        # w1 et b1 est les poids et biais (couche entree  3 neurones)
        self.w2 = np.random.randn(2, 3) 
        self.b2 = np.zeros((2, 1))
        #entre la couch cachée1 (3neurones )et cachée2(2 neurones)
        self.w3 = np.random.randn(1, 2) 
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
    
    def retropropagation(self,x,s_reel,prediction,iteration=1000):
        
        
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
    
    def entrainement(self,x,srx,maxiter=500):
        predictions_initail={}
        predictions_courrent={}
        for i,exemple in enumerate(x):
            xcolumn = np.array(exemple).reshape(-1, 1)
            pred = self.propa_vers_avant(xcolumn)[0][0]#we did [0][0]par ce que la sortie est une matrice [[valeur]]
            predictions_initail[i] = pred
            predictions_courrent[i] = pred


        #initialiser du meilleur modele
        meilleur_erreur=float('inf')
        meilleur_param={
            'w1':self.w1.copy(),
            'b1':self.b1.copy(),
            'w2':self.w2.copy(),
            'b2':self.b2.copy(),
            'w3':self.w3.copy(),
            'b3':self.b3.copy(),   
        }

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
                _,new_pred=self.retropropagation(xcolumn,sortie,pred,iteration=1000)

                predictions_courrent[ex]=new_pred #stocker new prediction de chaque exemple a chaque iteration 
                #calcule error total 
                error=abs(sortie-new_pred)
                total_error+=error
                #affichage des calcules de chaque exemple 
                print(f"\nexemple {ex}: {exemple}")
                print(f"sortie attendue: {sortie}")
                print(f"sortie initiale: {predictions_initail[ex]:.6f}")
                print(f"sortie avant mise à jour: {pred_before:.6f}")
                print(f"sortie après mise à jour: {new_pred:.6f}")
            error_moy=total_error/len(x)
            print(f"\niteration{j+1} - erreur moyenne: {error_moy:.6f}")
            #sauvegarder le meilleur modéle 
            if error_moy < meilleur_erreur:
                meilleur_erreur=error_moy
                meilleur_param={
                  'w1':self.w1.copy(),
                  'b1':self.b1.copy(),
                  'w2':self.w2.copy(),
                  'b2':self.b2.copy(),
                  'w3':self.w3.copy(),
                  'b3':self.b3.copy(),   
                }
            #condition de covergence
            if error_moy<0.1:
               print(f"covergence attient a l'iteration {j+1}")
               break 
        #save the best model
        self.w1= meilleur_param['w1']
        self.b1= meilleur_param['b1']
        self.w2= meilleur_param['w2']
        self.b2= meilleur_param['b2']
        self.w3= meilleur_param['w3']
        self.b3= meilleur_param['b3']
        #resultat final 
        print("\n~~~~ résultats finaux ~~~~")
        for i,exemple in enumerate(x):
           print(f"exemple {i}: {exemple}, sortie attendu: {srx[i]}  ,prédiction initial : {predictions_initail[i]} , prédiction : {predictions_courrent[i]:.6f}")   
        
    def savewandb (self,mon_fichier="poidsetbiais.pkl"):
        poids_biais={
            'w1':reseau.w1,
            'b1':reseau.b1,
            'w2':reseau.w2,
            'b2':reseau.b2,
            'w3':reseau.w3,
            'b3':reseau.b3,
        }
        with open (mon_fichier,'wb') as f:
            pickle.dump(poids_biais,f)
        with open(mon_fichier, 'rb') as file:
            data = pickle.load(file)
        print(data)  

    def validation (self,x,srx):
        correcte_pred=0
        for i,exemple in enumerate(x):
            xcolumn = np.array(exemple).reshape(-1,1)
            pred=self.propa_vers_avant(xcolumn)[0][0]
            reel=srx[i]
            print(f"exemple {i+1} -> {exemple} -> sortie attendue ={reel}-> prediction ={pred:.6f}")
            if np.isclose(pred,reel,atol=0.1):
                correcte_pred+=1
                print(f"~~~~~~~~~~correcte prediction de exemple {i+1}")
            else:
                print(f"~~~~~~~~~~incorrecte prediction de exemple {i+1}")
            
        pourcentage=correcte_pred/len(x)*100
        print(f"peurcentage = {pourcentage}%")
        print(f"nbr exemple correcte {correcte_pred}")


            

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
#reseau.savewandb("poidsetbiais.pkl")
xv=[]
srxv=[]
with open ( "validation.txt","r") as file:
    for l in file:
        l=l.strip()
        if l:
            val=[float(num) for num in l.split()]
            xv.append(val[:-1])
            srxv.append(val[-1])
xv=np.array(xv,dtype=np.float64)
srxv=np.array(srxv)
reseau.validation(xv,srxv)




























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













































