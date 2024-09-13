import numpy as np
import pandas as pd





def readFile(filename):
    data=pd.read_csv(filename)
    dataMatrix=[]
    for index, row in data.iterrows():
        dataMatrix.append(row.tolist())
   
   
    #print (dataMatrix[0])
    return dataMatrix

def productsum(input, weight):
    sum=0
    for i in range(0,8):
        sum=sum+(input[i]*weight[i])
        #print(sum)
    return sum

def perceptron(data):
    idexNumber=0
    b=0
    f=0
    c=0
    weight=[1,1,1,1,1,1,1,1]
    for index in range(0,768):
    #print(data)
        inputLayer=data[index][:8]
    #print(inputLayer)
        TestResult=data[index][-1:]
    #print(result)
        result=productsum(inputLayer,weight)+b
        #print(result)
        b,f,c=trainThreshold(TestResult,345,result,b,c,f)
        #print (b)
    print (c/769)
  

def activation(result,threshold):
    if (result > threshold):
        #print(result)
        #print(threshold)
        y=[1.0]
    else:
        y=[0.0]
    #print(f"Result: {result}, Threshold: {threshold}")
    #print(y)
    return y

def trainThreshold(TestResult,threshold,result,b,c,f):
    idk=activation(result,threshold)
    if (idk!=TestResult):
        #print(type(TestResult))
        #print(f"idk: {idk}, testresult: {TestResult}")
        b=threshold-result
        f=f+1
        #print(f)
    else:
        c=c+1
        #print(f)
    return b,f,c




        








if __name__=="__main__":
    filename='C:\\Users\\Jack\\Desktop\\Deep learning\\a1 perceptron\\diabetes.csv'
    data=readFile(filename)
    perceptron(data)

    