import numpy as np
import pandas as pd
import math as math
def readFile(filename):
    data=pd.read_csv(filename)
    dataMatrix=[]
    Xi=[]
    Yi=[]
    for index, row in data.iterrows():
        dataMatrix.append(row.tolist())
    
    for i in range(768):
        
        
        #print(data)
        
        Xi.append(dataMatrix[i][:8])
        #print(inputLayer)
     
        TestResult=dataMatrix[i][-1:]
        Yi.append(TestResult)
   
    #print (dataMatrix[0])
    return Xi ,Yi
class Neuron:
    bias=0
    def __init__(self, Xi, Yi,OutPut,NumberofX):
        self.weight=np.random.rand(NumberofX)
        self.Xi=Xi
        self.Yi=Yi
        self.output=OutPut
        self.NumberofX=NumberofX
    
def normalize(Xi):

    inputs = np.array(Xi)
    min_val = np.min(Xi)
    max_val = np.max(Xi)
    if min_val!=max_val:
        normalized = (inputs - min_val) / (max_val - min_val)
        return normalized.tolist()
    else:
        return Xi
       

    
    # Normalize using Min-Max Scaling
        
    
    
def productsumPlusBias(Xi, weight,bias):
    sum=0
    NormalizedXi=normalize(Xi)
    for i in range(len(weight)):
        #print(NormalizedXi[i])
        sum=sum+(NormalizedXi[i]*weight[i])
        #print(sum)
    return sum+bias
def CrossEntropy(z,Yi):
    loss=0
    for i in range(len(z)):
        logit=1/(1+math.exp(z[i]*-1))
        loss=loss-((Yi[i][0]*math.log(logit))+((1-Yi[i][0])*math.log(1-logit)))
    print(loss)

def activation(result):
    #print(f"idk1: {result}, result testresult1: {threshold}")
    if (result >0):
        #print(result)
        #print(threshold)
        predictedY=1.0
    else:
        predictedY=0.0
    #print(f"Result: {result}, Threshold: {threshold}")
    #print(y)
    return predictedY

def accuracy(output,Yi):
    correct=0
    for i in range(len(output)):
        if output[i][0]==Yi[i][0]:
            correct+=1
    print(correct/len(output))


def perceptron(Neuron):
    PredictedY=[]
    RawYArray=[]
    
    for i in range (len(Neuron.Xi)):
        RawY=productsumPlusBias(Neuron.Xi[i],Neuron.weight,Neuron.bias)
        RawYArray.append(RawY)
        PredictedY.append([activation(RawY)])
    #print(PredictedY)
    if Neuron.output==True:
        CrossEntropy(RawYArray,Neuron.Yi)
        accuracy(PredictedY,Neuron.Yi)


    return PredictedY


def TrainWeightBias(predictedY,Neuron,LearningRate):
    for i in range (len(predictedY)):
        for j in range(Neuron.NumberofX):
            Neuron.weight[j]=Neuron.weight[j]-LearningRate*(predictedY[i][0]-Neuron.Yi[i][0])*Neuron.Xi[i][j]
            Neuron.bias=Neuron.bias-LearningRate*(predictedY[i][0]-Neuron.Yi[i][0])

def FinalFuction(Xi,Yi,HiddenSize,NumberofX,LearningRate,epoch,MLP):
    ArrayOfNeuron=[]
    PredictedYHidden=[]
    PredictedYOutPut=[]
    for i in range(HiddenSize):
        ArrayOfNeuron.append(Neuron(Xi,Yi,False,NumberofX))
    for x in range(epoch):
        for i in range(HiddenSize):
            if i==0:
                PredictedYHidden=perceptron(ArrayOfNeuron[i])
            else:
                PredictedYHidden = [row1 + row2 for row1, row2 in zip(PredictedYHidden, perceptron(ArrayOfNeuron[i]))]
        if MLP==True or HiddenSize>1: 
            ArrayOfNeuron.append(Neuron(PredictedYHidden,Yi,True,HiddenSize))
            #print(ArrayOfNeuron[-1].Xi)
            PredictedYOutPut=perceptron(ArrayOfNeuron[-1])
            
        for i in range(len(ArrayOfNeuron)):
            TrainWeightBias(PredictedYOutPut,ArrayOfNeuron[i],LearningRate)

    
if __name__=="__main__":
    filename='C:\\Users\\Jack\\Desktop\\Deep learning\\a1 perceptron\\diabetes.csv'
    Xi, Yi=readFile(filename)
    #TestNeuron=Neuron(Xi,Yi,False,8)
    #TestNeuron2=Neuron(Xi,Yi,False,8)
    #print(perceptron(TestNeuron))
    #print(TestNeuron.Xi)
    #TrainWeightBias(perceptron(TestNeuron2),TestNeuron2,0.01)
    FinalFuction(Xi,Yi,2,8,0.001,10,True)