import FoldsInfo
import random

class Fold :
    indices = []
    numInstancesPerClass = []
    indicesByClass = []
    numOfInstancesInFold = 0 
    distinctValMappings = {} 
    typeOfFold = None

    def __init__(self, numOfClasses, typeOfFold) :
        self.indices = [] 
        self.numInstancesPerClass = self.init_list(numOfClasses, 0)       
        self.indicesByClass = self.init_list(numOfClasses , None)
        i = 0
        while i < numOfClasses:
            self.numInstancesPerClass[i] = 0 
            self.indicesByClass[i] = []
            i += 1
            
        self.typeOfFold = typeOfFold 
    
    
    
    def init_list(size, element):
        list_to_return = []
        i = 0
        while i< size:
            list_to_return.append(element)
            i+=1
        return list_to_return


    
    # Used to gnerate a sub-fold by randomly sampling a predefined number of samples from the fold. In the case
    # of distinct values, the the number of samples referred to the distinct values and all the associated indices
    # will be added.
    def generateSubFold(self, numOfSamples, randomSeed) :
        subFold = Fold(self.numInstancesPerClass.length, self.typeOfFold) 

        #determine how many instances of each class needs to be added
        requiredNumOfSamplesPerClass = self.getRequiredNumberOfInstancesPerClass(numOfSamples) 

        #now we need to randomly select the samples
        random.seed(randomSeed)
        for i in range(0, len(self.numInstancesPerClass), 1):
            if (len(self.distinctValMappings) == 0) :
                selectedIndicesPerClass = [] 
                while (len(selectedIndicesPerClass) < requiredNumOfSamplesPerClass[i]) :
                    instanceIndex = self.indicesByClass[i].get(random.randint(0, len(self.indicesByClass[i]))) 
                    if (not instanceIndex in selectedIndicesPerClass) :
                        selectedIndicesPerClass.append(instanceIndex) 
                        subFold.addInstance(instanceIndex, i) 
             
            else :
                keySetValues = self.init_list(len(self.distinctValMappings.keys()), "")
                counter = 0 
                for key in self.distinctValMappings.keys():
                    keySetValues[counter] = key 
                    counter+=1
                
                selectedIndicesPerClass = [] 
                while (len(selectedIndicesPerClass) < requiredNumOfSamplesPerClass[i]) :
                    distictValKey = keySetValues[random.randint(0,len(keySetValues))] 
                    if (not distictValKey in selectedIndicesPerClass and self.distinctValMappings.get(distictValKey)[0] in self.indicesByClass[i]):
                        selectedIndicesPerClass.append(distictValKey) 
                        subFold.addDistinctValuesBatch(distictValKey, self.distinctValMappings.get(distictValKey),i) 

        return subFold 
    
                
    def getRequiredNumberOfInstancesPerClass(self, numOfSamples) :
        numOfInstancesPerClass = self.init_list(len(self.numInstancesPerClass),0.0)

        #If there are no distinct values, the problem is simple
        if (len(self.distinctValMappings) == 0) :
            for i in range(0, len(numOfInstancesPerClass), 1):
                numOfInstancesPerClass[i] = (float(self.numInstancesPerClass[i])/ float(sum(self.numInstancesPerClass))) * numOfSamples 

        else :
            #We need to find the number of DISTINCT VALUES per class
            for item in self.distinctValMappings.keys():
                index = self.distinctValMappings.get(item)[0]
                for i in range(0, len(self.indicesByClass), 1):
                    if index in self.indicesByClass[i] :
                        numOfInstancesPerClass[i]+=1 
                        break 
            for i in range(0, len(numOfInstancesPerClass), 1):
                numOfInstancesPerClass[i] = (numOfInstancesPerClass[i]/sum(numOfInstancesPerClass))* numOfSamples 
   
        return numOfInstancesPerClass 
   

   
     # Receives a list of indices which all have the same distinct value. The function adds all the indices
     # to the fold and updates the map that keeps track of the relations among them
    def addDistinctValuesBatch(self, key, indices, instancesClass) :
        self.distinctValMappings[key] = indices 
        for i in indices:
            self.addInstance(i, self.instancesClass) 

        
  
     # Sets the distinct values and indices of this fold
    def setDistinctValMappings(self, distinctValMappings) :
        self.distinctValMappings = distinctValMappings 
    

  
     # Adds an instance to the fold and updates the counter and indices list
    def addInstance(self, index, classIdx):
        self.indices.add(index) 
        self.indicesByClass[classIdx].add(index) 
        self.numInstancesPerClass[classIdx] += 1 
        self.numOfInstancesInFold += 1 
    

  
     # Returns the distinct val mappings required for validaion and the generation of the Weka-compatible
     # dataset used in the acutal classification.
    def getDistinctValMappings(self) :
        return self.distinctValMappings 
    

    
     # Returns the indices of the instances that belong to a specific class
    def getIndicesPerClass(self, classIdx) :
        return self.indicesByClass[classIdx] 
    


    def getIndicesPerClass(self) :
        return self.indicesByClass 
    


    def setIndicesPerClass(self, indicesByClass) :
        self.indicesByClass = indicesByClass 
    

    
     # Returns the overall number of instances in the fold, regardless of class
    
    def getNumOfInstancesInFold(self) :
        return self.numOfInstancesInFold 
    

    
     # Sets the overall number of instnaces in the fold, regardless of class
     
    def setNumOfInstancesInFold(self, numOfInstancesInFold) :
        self.numOfInstancesInFold = numOfInstancesInFold  

    
     # Gets the number of instances of a certain class in the fold
    def getNumOfInstancesPerClass(self, classIdx) :
        return self.numInstancesPerClass[classIdx] 
    


    def getInstancesClassDistribution(self) :
        return self.numInstancesPerClass 
    

    def setInstancesClassDistribution(self, numInstancesPerClass) :
            self.numInstancesPerClass = numInstancesPerClass
    

    
     # Returns all the indices in the fold
    def getIndices(self) :
        return self.indices 
    


     # Sets the indices of the fold
    def setIndices(self, indices) :
        self.indices = indices 
    

    def getTypeOfFold(self) :
        return self.typeOfFold 
    

   
     # Returns true if this fold needs to be used as the test fold
  
    def isTestFold(self) :
        return self.typeOfFold == "Test"

   
     # Used to define a fold as test
    def setIsTestFold(self, isTest) :
        self.typeOfFold = FoldsInfo.foldType.Test 
    

    
     # Returns true if this fold needs to be used as the train fold
   
    def isTrainFold(self) :
        return self.typeOfFold == "Train"
    
    
     # Used to define a fold as train
    def setIsTrainFold(self, isTest) :
        self.typeOfFold = "Train" 
    

   
     # Returns true if this fold needs to be used as the validation fold
 
    def isValidationFold(self) :
        return self.typeOfFold == "Validation"

    
     # Used to define a fold as validation
   
    def setIsValidationFold(self, isTest) :
        self.typeOfFold = "Validation" 
    