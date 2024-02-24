import math


class FoldsInfo :

    _numOfTrainingFolds = 0
    _numOfValidationFolds = 0
    _numOfTestFolds = 0
    _trainingSetPercentage = 0.0
    _numOfTrainingInstances = 0
    _validationSetPercentage = 0.0
    _numOfValidationInstances = 0
    _testSetPercentage = 0.0
    _numOfTestSetInstances = 0
    _maintainClassRatio = False
    _assignRemainingInstancesToFold = ""




    # @param numOfValidationFolds can only be 0 or 1
    # @param numOfTestFolds can only be 0 or 1
    # @param assignRemainingInstancesToFold The fold to which the "leftover" instances (those that are left after the partitioning to the various folds is done) are assigned to
     
    def __init__(self, numOfTrainingFolds, numOfValidationFolds, numOfTestFolds, trainingSetPercentage, numOfTraningInstances, validationSetPercentage,
                     numOfValidationSetInstances, testSetPercentage, numOfTestSetInstances, maintainClassRatio, assignRemainingInstancesToFold)   :
        self._numOfTrainingFolds = numOfTrainingFolds 
        self._numOfValidationFolds = numOfValidationFolds 
        self._numOfTestFolds = numOfTestFolds 
        self._trainingSetPercentage = trainingSetPercentage 
        self._numOfTrainingInstances = numOfTraningInstances 
        self._validationSetPercentage = validationSetPercentage 
        self._numOfValidationInstances = numOfValidationSetInstances 
        self._maintainClassRatio = maintainClassRatio 
        self._testSetPercentage = testSetPercentage 
        self._numOfTestSetInstances = numOfTestSetInstances 
        self._assignRemainingInstancesToFold = assignRemainingInstancesToFold 

        if ((self._numOfValidationFolds > 1 or self._numOfValidationFolds < 0) or (self._numOfTestFolds > 1 or self._numOfTestFolds <0))  :
            raise Exception("There can only be 0 or 1 validation and test folds") 
         
     

    def getNumOfTrainingFolds(self)  :
        return self._numOfTrainingFolds 
     

    def getNumOfValidationFolds(self)  :
        return self._numOfValidationFolds 
     

    def getNumOfTestFolds(self)  :
        return self._numOfTestFolds 
     

    def getTraninigSetPercentage(self)  :
        return self._trainingSetPercentage 
     

    def getNumOfTrainingInstances(self)  :
        return self._numOfTrainingInstances 
     

    def getValidationSetPercentage(self)  :
        return self._validationSetPercentage 
     

    def getNumOfValidationInstances(self)  :
        return self._numOfValidationInstances 
     

    def getTestSetPercentage(self)  :
        return self._testSetPercentage 
     

    def getNumOfTestInstances(self)  :
        return self._numOfTestSetInstances 
     

    def getMaintainClassRatio(self)  :
        return self._maintainClassRatio 
     

    def getAssignRemainingInstancesToFold(self) :
        return self._assignRemainingInstancesToFold 
     


    
     # Returns the number of folds of each type
    def getNumOfFolds(self) :
        mapToReturn = {}
        mapToReturn["Train"] = self._numOfTrainingFolds 
        mapToReturn["Validation"] = self._numOfValidationFolds
        mapToReturn["Test"] = self._numOfTestFolds
        return mapToReturn 
     

    
     # Returns a map with the actual number of instance PER FOLD for each type of fold
    def getNumOfInstacesPerFold(self, numOfDatasetInstances):
        numOfTrainingFoldInstances = math.max( math.max(0, self._numOfTrainingInstances), math.max(0, (int(self._trainingSetPercentage*numOfDatasetInstances/self._numOfTrainingFolds))))

        numOfValidationFoldInstances = 0 
        if (self._numOfValidationFolds > 0)  :
            numOfValidationFoldInstances = math.max( math.max(0, self._numOfValidationInstances), math.max(0, int(self._validationSetPercentage*numOfDatasetInstances))) 
         

        numOfTestFoldInstances = 0 
        if (self._numOfTestFolds > 0)  :
            numOfTestFoldInstances = math.max( math.max(0, self._numOfTestSetInstances), math.max(0, int(self._testSetPercentage*numOfDatasetInstances))) 
         

        totalNumOfAllocatedInstences = (numOfTrainingFoldInstances * self._numOfTrainingFolds) + numOfValidationFoldInstances + numOfTestFoldInstances 

        # If we over or under 1% of the instances then throw an exception
        if (totalNumOfAllocatedInstences < 0.99*numOfDatasetInstances or totalNumOfAllocatedInstences > 1.01*numOfDatasetInstances)  :
            raise Exception("instance allocation is incorrect. Please re-check") 
         

        instancesAllocationMap = {} 
        instancesAllocationMap["Train"] =  numOfTrainingFoldInstances 
        instancesAllocationMap["Validation"] = numOfValidationFoldInstances 
        instancesAllocationMap["Test"] = numOfTestFoldInstances 
        return instancesAllocationMap 
     

 
    
     # Used to determine the number of instances PER FOLD for each type of fold while also returning the NUMBER OF INSTANCES PER CLASS
     # (if the class ratio does not need to be maintained, the getNumOfInstacesPerFold can be called directly).
    def getNumOfInstancesPerFoldPerClass(self, itemIndicesByClass, numOfDatasetInstances) :
        #start by getting the total number of instances per fold
        totalNumOfInstancesPerFold = self.getNumOfInstacesPerFold(numOfDatasetInstances) 

        #perform the conversion to percentages
        classPercentages = {} 
        for i in range(0, len(itemIndicesByClass), 1):
            classPercentages[i] = (float(len(itemIndicesByClass.get(i)))/numOfDatasetInstances) 
         

        mapToReturn = {}

        for fold in totalNumOfInstancesPerFold.keys()  :
            mapToReturn[fold] = {}
            for i in range(0, len(itemIndicesByClass), 1):
                mapToReturn.get(fold)[i] = int(round(totalNumOfInstancesPerFold.get(fold)*classPercentages.get(i))) 

        return mapToReturn 
     
 