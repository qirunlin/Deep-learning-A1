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
    inputN=normalize(input)
    for i in range(0,8):
        sum=sum+(inputN[i]*weight[i])
        #print(sum)
    return sum
idexNumber=0
givenResult=[]
def perceptron(data,weight,predyy,LearningRate):
    global idexNumber
    global givenResult
    global global_c
    b=0
    f=0
    c=0
    test=0    
    for index in range(0,768):
        
        test=test+1
        #print(data)
        
        inputLayer=data[index][:8]
        #print(inputLayer)
     
        TestResult=data[index][-1:]
        givenResult=givenResult+TestResult
        #print(result)
        if (idexNumber>0):
       
            for j in range(0,8):
                #print(predyy[index])
                #print(givenResult)
                #print(test)
                weight[j]=weight[j]-LearningRate*(predyy[index]-givenResult[index])*inputLayer[j]
                b=b-LearningRate*(predyy[index]-givenResult[index])
                #print(b)


        result=productsum(inputLayer,weight)+b
            #print(result)
        f,c,predy=trainThreshold(TestResult,345,result,c,f)
    idexNumber+=1
   
        
    print (global_c/798)
    global_c=0
    return predy
def normalize(inputs):
    """
    Normalize the input values using Min-Max Scaling.
    
    Args:
    inputs (list): List of input values to normalize.
    
    Returns:
    list: Normalized input values.
    """
    inputs = np.array(inputs)
    min_val = np.min(inputs)
    max_val = np.max(inputs)
    
    # Normalize using Min-Max Scaling
    normalized = (inputs - min_val) / (max_val - min_val)
    return normalized.tolist()

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
predy=[]
global_c=0
def trainThreshold(TestResult,threshold,result,c,f):
    global global_c
    global predy
    
    idk=activation(result,threshold)
    predy=predy+idk
    #print(predy)
    if (idk!=TestResult):
        #print(type(TestResult))
        #print(f"idk: {idk}, testresult: {TestResult}")
     
        f=f+1
        #print(f)
    else:
        c=c+1
        global_c=global_c+1
    return f,c,predy

def training(data,LearningRate,epoche,weight):
    predy=[]
    for i in range(0,epoche):
       predy=perceptron(data,weight,predy,LearningRate)
       #print(predy)

     





        








if __name__=="__main__":
    filename='C:\\Users\\Jack\\Desktop\\Deep learning\\a1 perceptron\\diabetes.csv'
    data=readFile(filename)
    weight=np.random.rand(8)
    training(data,0.001,10,weight)

    