import pickle
import json
import numpy as np


__data_columns=None
__knn=None
__species=None

def result(sl,sw,pl,pw):
    arr=np.zeros(4)
    arr[0]=sl
    arr[1]=sw
    arr[2]=pl
    arr[3]=pw
    out=__knn.predict([arr])
    st=""
    if out =='Iris-versicolor':
        st=__species[77]
    elif out =='Iris-virginica':
        st=__species[140]
    else:
        st=__species[1]
    return st
def load_saved_artifacts():
    print("loading saved artifacts...start")

    global __data_columns
    global  __knn
    global __species
    

    with open("./artifacts/iriscolumns.json", "r") as f:
        __data_columns = json.load(f)['data_columns']
        
    if __knn is None:
        with open('./artifacts/IRIS (1).pickle', 'rb') as f:
            __knn = pickle.load(f)
    
    with open("./artifacts/species.josn", "r") as f:
        __species= json.load(f)['target_names']
        
        



if __name__ == '__main__':
    load_saved_artifacts()
    print(result(5.1,3.5,1.4,0.2))
    print(result(6.2, 3.4, 5.4, 2.3))
        