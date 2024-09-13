import numpy as np
import pandas as pd




def readFile(filename):
    data=pd.read_csv(filename)
   
    #print (data.iloc[0,0])
    return data







if __name__=="__main__":
    filename='C:\\Users\\Jack\\Desktop\\Deep learning\\a1 perceptron\\diabetes.csv'
    readFile(filename)
    