import numpy as np 

class MLPXOR:
    def __init__(self):
        self.alpha=0.5 
        # w and b couche entree and cachéé
        self.w1 = np.random.randn(2, 2) 
        self.b1 = np.zeros((2, 1))
        # w and b couche cachée and sortie 
        self.w2 = np.random.randn(1, 2) 
        self.b2 = np.zeros((1, 1))
    def sigmoid(self,x):
         return 1 / (1 + np.exp(-x))
    def df_sigmoid(self, x):
        return x * (1 - x)   
    
    def propa_vers_avant(self, vals):
        self.c1 = np.dot(self.w1, vals) + self.b1
        self.s1 = self.sigmoid(self.c1)#matrice 2*1
        self.c2 = np.dot(self.w2, self.s1) + self.b2
        self.s = self.sigmoid(self.c2)#matrice 1*1
        return self.s 
    def retropropagation(self,x,s_reel,prediction,iteration=150):
        
        
        for i in range(iteration):
            s_calculer=self.propa_vers_avant(x)
            error=np.sum((s_reel-s_calculer)**2)

            #calcule des gradients et miss a jour les poids entre la couche sortie et la couche cachee
            delta_s=(s_calculer - s_reel)*self.df_sigmoid(s_calculer)
            delta_w2=np.dot(delta_s,self.s1.T) #.T is for converts the 2,1 vector to 1,2 row vector 
            delta_b2=delta_s
            
            #calcule des gradients et miss a jour les poids entre la couche cachee et lentree
            delta_s1=np.dot(self.w2.T,delta_s)*self.df_sigmoid(self.s1)
            delta_w1=np.dot(delta_s1,x.T)
            delta_b1=delta_s1
           
            #miss a jour les poids et biases
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

    def entrainement(self,X,Y,maxiter=1000):
        for i in range (maxiter):
            total_error=0
            for x,y_reel in zip(X,Y):
                x=x.reshape(-1,1)
                y_reel=np.array([[y_reel]])
                prediction=self.propa_vers_avant(x)
                self.retropropagation(x,y_reel,prediction,iteration=150)
                y_pred=self.propa_vers_avant(x)
                error=abs(y_reel-y_pred)
                total_error+= error
                
            
            if total_error < 0.01:
                print(f"\n>>> convergence atteinte à l'itération {i+1}")
                break
        # Résultats finaux
        print("\n~~~~ résultats finaux ~~~~")
        for i ,(x,y_true) in enumerate(zip(X,Y)):
            x=x.reshape(-1,1)
            y_pred=self.propa_vers_avant(x)[0][0]
            print(f"exemple {i}: entrée: {x.ravel()}, sortie attendue: {y_true}, Prédiction finale: {y_pred:.6f}")

    def tester(self, x ,y):
        for i,(xi,yi) in enumerate(zip(x,y)):
            xi=xi.reshape(-1,1)
            prediction=self.propa_vers_avant(xi)[0][0]
            print(f"input{i+1}:{xi.ravel()} sortie attendue = {yi} prediction {prediction}")
            
if __name__=="__main__":
   
    X=np.array([[0,0],[0,1],[1,0],[1,1]],dtype=np.float64) 
    Y=np.array([0,1,1,0],dtype=np.float64)
    reseau=MLPXOR()
    reseau.entrainement(X,Y)
    reseau.tester(X,Y)