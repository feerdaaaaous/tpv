import numpy as np 
import pickle
class MLP:
    def __init__(self):
       self.alpha=0.5
       self.w1=None
       self.b1=None
       self.w2=None
       self.b2=None
       self.w3=None
       self.b3=None
    def sigmoid(self,x):
        return 1/ (1+ np.exp(-x))
    def propa_vers_avant(self,vals):
        vals=np.array(vals).reshape(-1,1)
        self.c1 = np.dot(self.w1, vals) + self.b1
        self.s1 = self.sigmoid(self.c1)
        self.c2 = np.dot(self.w2, self.s1) + self.b2
        self.s2 = self.sigmoid(self.c2)
        self.c3 = np.dot(self.w3, self.s2) + self.b3
        self.s = self.sigmoid(self.c3)
        if np.isclose(self.s,1,atol=0.1):
            return 1
        if np.isclose(self.s,0,atol=0.1):
            return 0
        return 0 
        
def load_params(mlp,file):
    with open(file,'rb') as f:
        data=pickle.load(f)
        mlp.w1=data['w1']
        mlp.b1=data['b1']
        mlp.w2=data['w2']
        mlp.b2=data['b2']
        mlp.w3=data['w3']
        mlp.b3=data['b3']
def main():
    mlp=MLP()
    load_params(mlp,"poidsetbiaisff.pkl")
    results=[]
    with open("data.txt","r")as file :
        for l in file:
            l=l.strip()
            if l:
                val=[float(num) for num in l.split()]
                input = val [:3]
                output=mlp.propa_vers_avant(input)
                r=' '.join(map(str,input))+f' {output}\n'
                results.append(r)
               
    with open("data.txt","w") as file :
        file.writelines(results)
if __name__ == "__main__":
    main()
