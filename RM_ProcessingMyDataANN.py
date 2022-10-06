"""
-----------------------------------------------------
   Name:     RM_ProcessingMyDataANN
   Author:   Russell Mendenhall
   Date:     Fri Sep  9 18:03:20 2022
   Language: Python
   Purpose: Predict buying habits based off MyData.txt for each person
-----------------------------------------------------
 ChangeLog:
   Who:      Russell Mendenhall          
   Date:     Fri Sep  9 18:03:20 2022
   Desc.:    This is the original file.
--------------------------------------------------------"""
from KMEANS import *
from math import inf, e
from datetime import date, timedelta, datetime, datetime
import pylab
from random import random
import matplotlib.pyplot as plt


def AmericaDate(date_list):
    #Change the date format from Day Month Year to American style of Month Day Year
    AmeriDateList=[]
    for item in date_list:
        AmeriDateList.append([item[1],item[0],item[2]])
    return AmeriDateList
def interpolate(n):
    return ((1/9)*(n-1))
def binary(n,bits):
    binaryList=[]
    quotient = n
    while quotient > 0:
        r1 = quotient%2
        quotient=quotient//2
        binaryList=[r1]+binaryList
    while len(binaryList) < bits:
        binaryList=[0]+binaryList
    return binaryList

def binaryToDecimal(binary):
    string=''
    for bit in binary:
        string=string+str(bit)
    binary = int(string)
    decimal, i, n = 0, 0, 0
    while(binary != 0):
        dec = binary % 10
        decimal = decimal + dec * pow(2, i)
        binary = binary//10
        i += 1
    return decimal

def ConvertToWD(Day,Month,Year,string=False):
        WeekDay={0:'Monday',1:'Tuesday',2:'Wednesday',3:'Thursday',4:'Friday',5:'Saturday',6:'Sunday'}
        Date=datetime(Year,Month,Day)
        if string:
            return WeekDay[Date.weekday()]
        else:
            return Date.weekday()        

def date_range_list(start_date, end_date):
    # Return list of dates between start_date(Date Object) and end_date(Date Object) (inclusive).
    date_list = []
    curr_date = start_date
    while curr_date <= end_date:
        
        date_list.append([curr_date.day,curr_date.month,curr_date.year])
        curr_date += timedelta(days=1)
    return date_list

class Person:
    def __init__(self,title):
        self.Title=str(title)
        self.Data=[]
        self.LastBought=[0,0]
        self.Avg=0
        
        
    def AddData(self,date,amount):
        date=[date[1],date[0],date[2]]
        self.Data.append((date,amount))
    
    def FindAvg(self):
        
        for couple in self.Data:
            self.Avg+=int(float(couple[1]))
        self.Avg=round(self.Avg/len(self.Data))
    def FillBlankDates(self):
        self.FindAvg()
        #Input American Style Date List
        #If I wanted to add in any missing dates with no information
        x=self.Data[0][0]
        y=self.Data[-1][0]
        start_date = datetime(x[2],x[0],x[1])
        stop_date = datetime(y[2], y[0], y[1])
        date_list = date_range_list(start_date, stop_date)
        
        #Change the date format from Day Month Year to American style of Month Day Year
        datesList=AmericaDate(date_list)
        
        
        #Break link between input list
        dateList=[]
        oldDateList=[]
        for date in datesList:
            oldDateList.append(date)
            dateList.append(date)
            
        newData=[]
        for date in self.Data:
            day=date[0][1]
            month=date[0][0]
            year=date[0][2]
            SearchForDate=[month,day,year]
            
            if SearchForDate in dateList:
                dateList.remove(SearchForDate)
        
        
        
        for date in oldDateList:
            if date in dateList:
                newData.append((date,0))
            else:
                for couple in self.Data:
                    if couple[0] == date:
                        newData.append((date,couple[1]))
                        
        self.Data=newData
        
    def OutputPatterns(self):
        self.FillBlankDates()
        patterns=[]
        itemline=0
        Bought=False
        #Look at each item in data
        for item in self.Data:
            #Exclude first line as person is always buying first
            if itemline==0:
                self.LastBought=[1,int(float(item[1]))]
                itemline+=1
                
                
            else:
                #Make InputPattern plus Target for pattern data then input into patterns
                patternData=[]
                
                #Make InputPatterns
                BinaryList=binary(self.LastBought[0],5)
                for bit in BinaryList:
                    patternData.append(bit)
                date=datetime(item[0][2],item[0][0],item[0][1])
                date=date.weekday()
                BinaryList=binary(date,3)
                for bit in BinaryList:
                    patternData.append(bit)
                BinaryList=binary(item[1],4)
                for bit in BinaryList:
                    patternData.append(bit)
                
                #Make Targets
                if float(item[1])>0:
                    
                    self.LastBought=[0,item[1]]
                    Bought=binary(item[1],4)
                    
                else:
                    self.LastBought[0]+=1
                    Bought=binary(0,4)
    
                patterns.append([patternData,Bought])
                
        return patterns
        



    


#Objective: Try to predict what day is a On or Off day for each person

#Sort my data based off each person
Me=Person('Me')
Dave=Person('Dave')
infile=open('MyData.txt','r')
linenum=0
for line in infile:
    if linenum<1:
        linenum+=1
    else:
        linelist=line[:-1].split(',')
        person1=linelist[1]
        amount=round(int(float(linelist[2])))
        date=linelist[0].split('-')
        #Convert Given day to a day of the week
        DateData=(int(date[1]),int(date[0]), int(date[2]))
        if person1=='Dave':
            Dave.AddData(DateData,amount)
        elif person1=='Me':
            Me.AddData(DateData,amount)
infile.close()


def sigma(x):
    return 1/(1+e**(-x))

class PerceptronANN:
    """A variation on Perceptron. No training method. Sigmoid activation"""
    
    def __init__(self,inputNum):
        self.inputNum = inputNum
        self.weightList=[]
       
    
    def getWeights(self):
        return self.weightList
    
     
    def assignWeights(self):
        """Weights assigned with random numbers in range -0.05 to 0,5"""
        for i in range(self.inputNum):
            self.weightList.append(0.1*random()-0.05)
        
    def output(self,inputs):
        """inputs is assumed to be a list of floats"""
        net = 0
        for i in range(self.inputNum):
            net=net+inputs[i]*self.weightList[i]
            
        if net < -700:
            net=-700
            
        return sigma(net)
    
    
    
        
class ANN:
    """Artificial neural network: nIn inputs, nHidden hidden units, nOut output units"""
    def __init__(self, nIn,nHidden,nOut):
        self.nIn = nIn
        self.nOut = nOut
        self.nHidden = nHidden
    def build(self):
        self.hiddenLayer=[]
        for i in range(self.nHidden):
            h = PerceptronANN(self.nIn)
            h.assignWeights()
            self.hiddenLayer.append(h)
        self.outerLayer = []
        for i in range(self.nOut):
            o = PerceptronANN(self.nHidden)
            o.assignWeights()
            self.outerLayer.append(o)
    
    def netOut(self,inputs):
        """Assumes inputs is an nIn list of floats"""
        self.hiddenOutputs = []
        for i in range(self.nHidden):
            self.hiddenOutputs.append(self.hiddenLayer[i].output(inputs))
        self.outOutputs = []
        for j in range(self.nOut):
            self.outOutputs.append(self.outerLayer[j].output(self.hiddenOutputs))

    def outUnitError(self, inputs, target):
        self.netOut(inputs)
        self.deltaOut = []
        for i in range(self.nOut):
            self.deltaOut.append(self.outOutputs[i]*(1-self.outOutputs[i])*(target[i]-self.outOutputs[i]))
            
    def hiddenUnitError(self):
        self.deltaHidden = []
        for i in range(self.nHidden):
            bigSum = 0
            for j in range(self.nOut):
                bigSum = bigSum + self.hiddenLayer[i].weightList[j]*self.deltaOut[j]
            self.deltaHidden.append(self.hiddenOutputs[i]*(1-self.hiddenOutputs[i])*bigSum)
    
    def adjustHiddenWeights(self,eta,inputs):
        for i in range(self.nIn):
            for j in range(self.nHidden):
                self.hiddenLayer[j].weightList[i]= self.hiddenLayer[j].weightList[i]+eta*self.deltaHidden[j]*inputs[i]
    
    def adjustOuterWeights(self, eta):
        for i in range(self.nHidden):
            for j in range(self.nOut):
                self.outerLayer[j].weightList[i]= self.outerLayer[j].weightList[i]+eta*self.deltaOut[j]*self.hiddenOutputs[i]
    
    def Test(self,LastBoughtDays,LastBoughtAmount,DayOfWeek,string=False,):
        """Test the most recent ANN. Day of week is in digits where monday is 0 and sunday is 6."""
        BinaryList=binary(LastBoughtDays,5)
        AllBinaryList=[]
        for bit in BinaryList:
            AllBinaryList.append(bit)
        BinaryList=binary(LastBoughtAmount,4)
        for bit in BinaryList:
            AllBinaryList.append(bit)
        BinaryList=binary(DayOfWeek,3)
        for bit in BinaryList:
            AllBinaryList.append(bit)
            
        
        self.netOut(AllBinaryList)
        if string:
            print(AllBinaryList)
            for o in self.outOutputs:
                print('{0:0.4f}'.format(round(o)))    
        else:
            return  self.outOutputs
        




def ModularEtaANN(patterns,startETA=5,EndETA=5,string=False):
    """A method to better optimize the 2 eta areas of my ANN made with input patterns. Returns a list =[[trials,startETA],[trailsAtETAChange,endETA],TrainedANN]"""
    q=ANN(12,8,4)
    q.build()
    eta = startETA
    trials=0
    Error=inf
    etaChange=False
    out=False
    if string==1:
        string=True
        out=True
    elif string==False:
        out=True
    while Error > 0.2:
        trials+=1
        Error = 0
        
        for j in range(len(patterns)):
            inputs=patterns[j][0]
            target = patterns[j][1]
            q.outUnitError(inputs,target)
            for i in range(len(q.outOutputs)):
                x = abs(target[i]-q.outOutputs[i])
                
            Error=Error+x
            q.hiddenUnitError()
            q.adjustHiddenWeights(eta, inputs)
            q.adjustOuterWeights(eta)
        Error = 0.5*Error
        if (round(Error*100,0))<21 and etaChange==False:
            eta=EndETA
            etaChange=[trials,Error]
        
        if string:
            print('Current Error Rate: {0}%'.format(round(Error,4)*100))
    if string:
        print('After ',trials,' trials it did it.',etaChange,'dope')
        
    if out:
        return [[trials,startETA],etaChange,q]


#x=ModularEtaANN(Me.OutputPatterns(),string=1)

def PredictFuture(ANN,startdate,enddate,Person,Wait=2,string=True):
    """Predict a future persons buying habits using Modular ann"""
    #Create blank date set of future dates
    dateSet=date_range_list( datetime(startdate[2],startdate[0],startdate[1]),datetime(enddate[2],enddate[0],enddate[1]))
    #Change input date to American:Month Day Year to Day month year
    dateSet=AmericaDate(dateSet)
    date=dateSet[0]
    #Make first date a buy date with the person's avg amount as the amount bought
    #info=[LastDayBought,Person's Avg, An int 0-6 for the day of the week]
    info=[0,Person.Avg,ConvertToWD(date[1], date[0], date[2])]
    Predictions=[]
    
    
    #Force date set into ANN
    positive=0
    Iterations=0
    for d in dateSet[1:]:
        info[2]=ConvertToWD(d[1], d[0], d[2])
        RawGuess=ANN.Test(info[0],info[1],info[2])
        info[0]+=1
        guess=[]
        for bit in RawGuess:
            guess.append(round(bit))
        
        guess=binaryToDecimal(guess)
        if string:
            print(guess)
        if guess>0:
            Predictions.append([Iterations,'True',guess])
            positive+=1
            
        else:
            Predictions.append([Iterations,'False'])
        Iterations+=1
        if positive==Wait:
            info[0]=Wait
            positive=0
    if string:
        for guess in Predictions:
            print(guess)
    else:
        return Predictions
    
#PredictFuture([9,18,2022],[9,30,2022],Me,Wait=1)
def TestMultiple(Range,startETA=10,endETA=8):
    Sums=0
    for x in range(Range):    
        x=ModularEtaANN(pattern,startETA,endETA)
        Sums+=x[0][0] 
        
    print('The Avg amount of trials is {0} out of {1} tries'.format(Sums/Range,Range))
    
#TestMultiple(50)
"""

#-Here is a test of the ANN to learn patterns
#Inputs will be Days since bought, Previous amount bought, DayOfWeek, 
#The learning task is to determine if a day is true or false
#patterns is a list sorted [[Inputs][Target]]
#n = int(input('Dave or you? 1 for Dave 0 for You.\n' ))
n=0
if n==1:
    patterns=Dave.OutputPatterns()
elif n==0:
    patterns=Me.OutputPatterns()
    
x=ModularEtaANN(patterns)[-1]
"""
"""
#test of learning
FalseGuesses=0
TrueGuesses=0
q=ModularEtaANN(patterns,string=1)[-1]
for p in patterns:
    q.netOut(p[0])
    guess=[]
    for o in q.outOutputs:
        guess.append(round(o))
        
    target=p[1]
    print('{0} vs ACTUAL: {1}'.format(binaryToDecimal(guess),binaryToDecimal(target)))
    if target==guess:
        TrueGuesses+=1
    elif target != guess:
        FalseGuesses+=1
            
print('True:',TrueGuesses,'  ','False:',FalseGuesses)
"""
#Dates in a list AMERICAN
Info=PredictFuture(x, [10,2,2022], [12,2,2022], Dave,Wait=1,string=False)
def GraphPredictions(info):
    """Graph all the informations using matplotlib"""
    #Some code I jacked from my big data class Lol
    Days=[]
    Amount=[]
    for item in info:
        Days.append(item[0])
        if len(item)>2:
            Amount.append(item[2])
        else:
            Amount.append(0)
        
        
    Orders=Amount
    
    # display to console the results from the query (detail of what is in each array):
    print("Number of days ", Days)
    print("Total Amount per day",Orders)

    # visulize data using Matplotlib:
    plt.plot(Days, Orders)
    plt.ylim(0, max(Orders)+2)
    plt.xlim(min(Days),max(Days))
    plt.xlabel("Days over time")
    plt.ylabel("Stuff Per Day")
    plt.title("Why I do not trust my AI. Stuff per Day over time")
    plt.show()
    
GraphPredictions(Info)


