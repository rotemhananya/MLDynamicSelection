import Fold
import DiscreteColumn
import NumericColumn
import DateColumn
import StringColumn
import Column
import ColumnInfo
import FoldsInfo

import Instances
import DenseInstance
import File
import Attribute
import ArffSaver

import random
import copy


class Dataset :
    columns = []
    numOfInstancesPerColumn = 0

    folds = []
    indices = []
    indicesOfTrainingFolds = []
    indicesOfValidationFolds = []
    indicesOfTestFolds = []
    trainingIndicesByClass = []
    validationIndicesByClass = []

    numOfTrainingInstancesPerClass = []
    numOfValidationInstancesPerClass = []
    numOfTestInstancesPerClass = []

    numOfTrainingRows = 0
    numOfValidationRows = 0
    numOfTestRows = 0

    targetColumnIndex = 0
    name = ""

    distinctValColumns = []
    distinctValueCompliantColumns = []
    trainFoldDistinctValMappings = {}
    validationFoldDistinctValMappings = {}
    testFoldDistinctValMappings = {}
    trainFoldsDistinctValRepresentatives = []
    testFoldsDistinctValRepresentatives = []

    #Defines the maximal number of distinct values a discrete attribute can have in order to be included in the ARFF file
    maxNumOFDiscreteValuesForInstancesObject = 0


    #used in all the operations which require a random variable. A fixed seed enables us to recreate experiments.
     
    randomSeed = 0
    
    def init_list(size, element):
        list_to_return = []
        i = 0
        while i< size:
            list_to_return.append(element)
            i+=1
        return list_to_return


    def __init__(self, columns, folds, targetClassIdx, name, numOfInstancesPerColumn, distinctValColumns, randomSeed, maxNumOfValsPerDiscreteAttribtue) :
        self.randomSeed = randomSeed 
        self.columns = columns 
        self.numOfInstancesPerColumn = numOfInstancesPerColumn 
        self.folds = folds 
        self.targetColumnIndex = targetClassIdx 
        self.name = name 
        self.maxNumOFDiscreteValuesForInstancesObject = maxNumOfValsPerDiscreteAttribtue 
        if distinctValColumns != None:
            self.distinctValColumns = distinctValColumns 
            trainFoldDistinctValMappings = {} 
            testFoldDistinctValMappings = {}
        

        self.indices = []
        self.indicesOfTrainingFolds = []
        self.indicesOfTestFolds = []
        self.trainingIndicesByClass = self.init_list(len(folds.get(0).getInstancesClassDistribution()), 0) 

        self.numOfTrainingInstancesPerClass = self.init_list(len(folds.get(0).getInstancesClassDistribution()),0) 
        self.numOfTestInstancesPerClass = self.init_list(len(folds.get(0).getInstancesClassDistribution()),0) 
        
        for fold in folds :
            self.indices+=(fold.getIndices()) 
            if fold.getTypeOfFold() == "Train":
                self.indicesOfTrainingFolds += fold.getIndices()
                
                classIdx = 0
                while classIdx < len(fold.getInstancesClassDistribution()):
                    numOfInstance = fold.getNumOfInstancesPerClass(classIdx) 
                    self.numOfTrainingInstancesPerClass[classIdx] += numOfInstance 
                    self.numOfTrainingRows += numOfInstance
                    classIdx+=1

                i = 0
                while (i<len(folds.get(0).getInstancesClassDistribution())):
                    if self.trainingIndicesByClass[i] == None:
                         self.trainingIndicesByClass[i] = []
                    
                    self.trainingIndicesByClass[i] += fold.getIndicesPerClass(i)
                    i+=1
            

                #Add all the distint values of the fold to the dataset object
                trainFoldDistinctValMappings.update(fold.getDistinctValMappings()) 
      
            if fold.getTypeOfFold() == "Validation":
                 self.indicesOfValidationFolds += fold.getIndices() 
                 
                 classIdx = 0
                 while classIdx < len(fold.getInstancesClassDistribution()) :
                     numOfInstance = fold.getNumOfInstancesPerClass(classIdx) 
                     self.numOfValidationInstancesPerClass[classIdx] += numOfInstance 
                     self.numOfValidationRows += numOfInstance 
                     classIdx+=1
                     
                     i = 0
                     while i < len(folds.get(0).getInstancesClassDistribution()):
                         if ( self.validationIndicesByClass[i] == None) :
                                 self.validationIndicesByClass[i] = []
                    
                         self.validationIndicesByClass[i] += fold.getIndicesPerClass(i)
                
                         i+=1
                         

                 #Add all the distint values of the fold to the dataset object
                 self.validationFoldDistinctValMappings.update(fold.getDistinctValMappings()) 
            


            if (fold.getTypeOfFold() == "Test") :
                 self.indicesOfTestFolds += fold.getIndices()
                 classIdx = 0
                 while classIdx < len(fold.getInstancesClassDistribution()) :
                    numOfInstance = fold.getNumOfInstancesPerClass(classIdx) 
                    self.numOfTestInstancesPerClass[classIdx] += numOfInstance 
                    self.numOfTestRows += numOfInstance 
                    classIdx+=1

                 #Add all the distint values of the fold to the dataset object
                 testFoldDistinctValMappings.update(fold.getDistinctValMappings()) 


        #Now that we are done processing the indices, we select one "representative" for each distinct value
        trainFoldsDistinctValRepresentatives = []
        for  key in trainFoldDistinctValMappings.keys():
            index = trainFoldDistinctValMappings.get(key)[0] 
            trainFoldsDistinctValRepresentatives.append(index) 
        
        testFoldsDistinctValRepresentatives = []
        for  key in testFoldDistinctValMappings.keys():
            index = testFoldDistinctValMappings.get(key)[0] 
            testFoldsDistinctValRepresentatives.append(index) 
        

        #finally, we sort the indices so that they will correspond with the order of the values in the columns
        self.indices.sort()
        self.indicesOfTrainingFolds.sort()
        self.indicesOfTestFolds.sort()
        self.trainFoldsDistinctValRepresentatives.sort()
        self.testFoldsDistinctValRepresentatives.sort() 
        i = 0
        while i < len(self.trainingIndicesByClass):
            self.trainingIndicesByClass[i].sort()
            i+=1

        for ci in columns:
            if self.isColumnDistinctValuesCompatibe(ci):
                self.distinctValueCompliantColumns.append(ci) 


    
     # Recieved another dataset that needs to be added to the current dataset as a test set
    
    def AttachExternalTestFold(self, testSet) :

        numOfRowsInBaseDataset =  self.numOfTrainingRows +  self.numOfTestRows 

        #If an existing fold is defined as test, change it to train
        for fold in self.folds:
            if (fold.isTestFold()) :
                fold.setIsTestFold(False) 
            

        newTestFold = Fold(self.numOfTrainingInstancesPerClass.length, "Test") 
        if (testSet.getDistinctValueColumns() == None or len(testSet.getDistinctValueColumns()) == 0) :
            i = 0
            while i < testSet.numOfTrainingRows + testSet.numOfTestRows  :     
                newTestFold.addInstance( self.numOfTrainingRows +  self.numOfTestRows + i, testSet.getTargetClassColumn().getColumn().getValue(i)) 
                i+=1
            
        else :
            for testSetFold in testSet.folds:
                for sources in testSetFold.getDistinctValMappings().keys():
                    firstItemIndexInBatch = testSetFold.getDistinctValMappings().get(sources)[0]
                    groupClass = testSet.getTargetClassColumn().getColumn().getValue(firstItemIndexInBatch) 
                    newIndices = []
                    for val in testSetFold.getDistinctValMappings().get(sources):
                        newIndices.append(val + numOfRowsInBaseDataset) 
                    
                    newTestFold.addDistinctValuesBatch(sources, newIndices, groupClass) 

        self.folds.append(newTestFold) 

        #change the folds indices
        self.indices += testSet.getIndices() 
        self.indicesOfTrainingFolds+=(self.indicesOfTestFolds) 

        self.indicesOfTestFolds = testSet.getIndicesOfTestInstances() 
        self.indicesOfTestFolds+=(testSet.getIndicesOfTrainingInstances()) 

        #update the total size of the joined datast
        numOfTrainingRows = len(self.indicesOfTrainingFolds) 
        numOfTestRows = len(self.indicesOfTestFolds) 

        #now we need to update every aspect of the current Dataset object
        self.numOfInstancesPerColumn += testSet.getNumOfInstancesPerColumn() 

        #the current division of train/test needs to be discarded. All items need to be transferred to the training
        i = 0
        while i<len(self.numOfTrainingInstancesPerClass):
            self.numOfTrainingInstancesPerClass[i] += self.numOfTestInstancesPerClass[i] 
            self.numOfTestInstancesPerClass[i] = testSet.getNumOfRowsPerClassInTrainingSet()[i] + testSet.getNumOfRowsPerClassInTestSet()[i] 
            i+=1

        #If the dataset has distinct values then we need to modify additional parameters
        #move all the distinct values in the test (of the training dataset) to the train
        if (self.distinctValColumns != None) :
            self.AttachExternalDatasetDistinctValues(testSet) 
        


        #Finally, the task pf updating the column objects
        i = 0  
        while i < len(self.columns) :
            currentColumn = self.columns.get(i).getColumn() 
            tempList = []  
            tempList.append(i) 
            testSetColumn = testSet.getColumns(tempList).get(0).getColumn() 


            if currentColumn.getType() == "Discrete":
                discreteReplacementColumn = DiscreteColumn(self.numOfInstancesPerColumn, currentColumn.getNumOfPossibleValues()) 
                self.populateJoinedColumnValues(discreteReplacementColumn, currentColumn, testSetColumn) 
                self.columns.get(i).setColumn(discreteReplacementColumn)

            elif currentColumn.getType() == "Numeric":
                numericReplacementColumn = NumericColumn(self.numOfInstancesPerColumn) 
                self.populateJoinedColumnValues(numericReplacementColumn, currentColumn, testSetColumn) 
                self.columns.get(i).setColumn(numericReplacementColumn)
            elif currentColumn.getType() == "Data":
                dateReplacementColumn = DateColumn(self.numOfInstancesPerColumn, currentColumn.getDateFomat()) 
                self.populateJoinedColumnValues(dateReplacementColumn, currentColumn, testSetColumn) 
                self.columns.get(i).setColumn(dateReplacementColumn)
            elif currentColumn.getType() == "String":
                stringReplacementColumn = StringColumn(self.numOfInstancesPerColumn) 
                self.populateJoinedColumnValues(stringReplacementColumn, currentColumn, testSetColumn) 
                self.columns.get(i).setColumn(stringReplacementColumn) 
            else:
                raise Exception("unidentified column type")
                
            i+=1



    def AttachExternalDatasetDistinctValues(self, testSet) :
        self.trainFoldDistinctValMappings.clear() 
        self.testFoldDistinctValMappings.clear() 
        self.trainFoldsDistinctValRepresentatives.clear() 
        self.testFoldsDistinctValRepresentatives.clear() 

        for fold in self.folds:
            #Add all the distint values of the fold to the dataset object
            if not fold.isTestFold():
                self.trainFoldDistinctValMappings.update(fold.getDistinctValMappings()) 
            else :
                self.testFoldDistinctValMappings.update(fold.getDistinctValMappings()) 

        #Now that we are done processing the indices, we select one "representative" for each distinct value
        self.trainFoldsDistinctValRepresentatives.clear() 
        self.testFoldsDistinctValRepresentatives.clear() 
        self.trainFoldsDistinctValRepresentatives = []
        for key in self.trainFoldDistinctValMappings.keys():
            index = self.trainFoldDistinctValMappings.get(key)[0] 
            self.trainFoldsDistinctValRepresentatives.append(index) 
        
        testFoldsDistinctValRepresentatives = []
        for key in self.testFoldDistinctValMappings.keys():
            index = self.testFoldDistinctValMappings.get(key)[0] 
            testFoldsDistinctValRepresentatives.append(index) 

        

    def populateJoinedColumnValues(self, newColumn, currentColumn, testSetColumn) :
        
        j = 0
        while j < self.numOfTrainingRows :
            newColumn.setValue(j,currentColumn.getValue(j)) 
            j+=1
            

        j = self.numOfTrainingRows
        while j<self.numOfTrainingRows+self.numOfTestRows:
            newColumn.setValue(j,testSetColumn.getValue(j-self.numOfTrainingRows)) 
            j+=1


  
    # Returns the required size of each
   
    def getNumOfInstancesPerColumn(self) :
        return  self.numOfInstancesPerColumn 
    

  
     # Gets the indeices of the instances assigned to the training folds
   
    def getIndicesOfTrainingInstances(self) :
        return self.indicesOfTrainingFolds 
    


    # Gets the idices of the instances assigned to the test folds

    def getIndicesOfTestInstances(self) :
        return self.indicesOfTestFolds 

    
    # Returns the indices of the samples allocated to this dataset
   
    def getIndices(self) :
        return self.indices 
    

    # Returns the number of samples in this dataset (both training and test)

    def getNumberOfRows(self) :
        return  self.numOfTrainingRows +  self.numOfTestRows +  self.numOfValidationRows 
    

   
    # Returns the name of the dataset

    def getName(self) :
        return  self.name 
    

    
    # Returns the total number of lines in the training dataset
    def getNumOfTrainingDatasetRows(self) :
        return self.numOfTrainingRows 
    

    
    # Returns the total number of lines in the test dataset
    def getNumOfTestDatasetRows(self) :
        return self.numOfTestRows 
    

   
    # Returns the number of classes in the dataset
    
    def getNumOfClasses(self) :
        return len(self.numOfTrainingInstancesPerClass )
    


    # Returns the number of samples that belong to each class in the training set
    def getNumOfRowsPerClassInTrainingSet(self) :
        return self.numOfTrainingInstancesPerClass 
    

    # Returns the number of samples that belong to each class in the test set

    def getNumOfRowsPerClassInTestSet(self) :
        return self.numOfTestInstancesPerClass 
    

    def getTrainingIndicesByClass(self) :
        return  self.trainingIndicesByClass 
    

    def getFolds(self) :
        return  self.folds 
    

   
    # Returns the index of the class with the least number of instances
    
    def getMinorityClassIndex(self) :
        currentIdx = -1 
        numOfInstances = 10000000000000000000000000000000000000000
        i = 0
        while i < len(self.numOfTrainingInstancesPerClass):
            if (self.numOfTrainingInstancesPerClass[i]< numOfInstances) :
                numOfInstances = self.numOfTrainingInstancesPerClass[i] 
                currentIdx = i 
            i+=1
 
        return currentIdx 
  
    
  
    # Returns speofic columns from the dataset
    
    def getColumns(self, columnIndices) :
        columnsList = []
        for columnIndex in columnIndices :
            columnsList.append(self.columns.get(columnIndex))
        
        return columnsList 
    

    def addColumn(self, column) :
         self.columns.append(column) 
    

    
     # Returns all the colums of the dataset object
     # @param includeTargetColumn whether the target column should also be returned
    def getAllColumns(self, includeTargetColumn) :
        columnsList = [] 
        for  column in self.columns :
            if ((not column in self.distinctValColumns) and (not column.isTargetClass() or includeTargetColumn)) :
                columnsList.add(column) 
      
        return columnsList 
    

   
     # Returns all the columns of a specified type
     # @param includeTargetColumn whether the target column should also be returned if it meets the criterion
    def getAllColumnsOfType (self,columnType, includeTargetColumn) :
        columnsToReturn = []
        for  ci in self.columns:
            if (ci.getColumn().getType() == columnType) :
                if (ci == self.getTargetClassColumn()) :
                    if (includeTargetColumn) :
                        columnsToReturn.append(ci) 
                else :
                    columnsToReturn.append(ci) 
        return columnsToReturn 
    

   
   # Returns the target class column
    
    def getTargetClassColumn(self) :
        return self.columns.get(self.targetColumnIndex) 
   

    
    # Returns the columns used to create the distinct value of the instances
 
    def getDistinctValueColumns(self) :
        return  self.distinctValColumns 


   
    # Samples a predefined number of samples from the dataset (while maintaining the ratio)
    # and generates a Weka Instances object.
    # IMPORTANT:this function is written so that it can only be applied on the training set, because
    # the classification model is trained on it. The test set is meant to be used as a whole.

    def generateSetWithSampling(self, numOfSamples, randomSeed):
        numOfRequiredIntancesPerClass = self.init_list(len(self.numOfTrainingInstancesPerClass),0.0)

        #Start by getting the number of items we need from each class
        i = 0
        while i < len(numOfRequiredIntancesPerClass) :
            numOfRequiredIntancesPerClass[i] = numOfSamples * (self.numOfTrainingInstancesPerClass[i]/self.numOfTrainingRows) 
            i+=1
    
        #Now we extract the subset for each class
        random.seed(randomSeed)
        #Random rnd = new Random(randomSeed) 
        subsetIndicesList = []
        i = 0
        while i < len(numOfRequiredIntancesPerClass):
            assignedItemsFromClass = 0 
            while assignedItemsFromClass < numOfRequiredIntancesPerClass[i]:
                pos = random.randint(0, len(self.trainingIndicesByClass[i])) 
                index = self.trainingIndicesByClass[i].get(pos) 
                if not index in subsetIndicesList:
                    subsetIndicesList.append(index) 
                    assignedItemsFromClass+=1 
            
            i+=1
            

        #get all the attributes that need to be included in the set
        attributes = []
        self.getAttributesListForClassifier(attributes) 
        finalSet = Instances("trainingSet", attributes, 0) 
        dataMatrix = self.getDataMatrixByIndices(subsetIndicesList) 
        i = 0
        while i < len(dataMatrix[0]):
            arr = self.init_list(len(dataMatrix[0]), 0.0)
            j = 0
            while j < len(dataMatrix):
                arr[j] = dataMatrix[j][i] 
                j+=1
            di = DenseInstance(1.0, arr) 
            finalSet.add(i, di)
            i+=1
        
        finalSet.setClassIndex(self.targetColumnIndex-self.getNumberOfDateStringAndDistinctColumns()) 
        return finalSet 
    


     # Used to obtain either the training or test set of the dataset
     # @param foldType the type of fold from which we want to extract the data
    
    def generateSet(self, foldType,  instanceIndices) :
        attributes = [] 

        #get all the attributes that need to be included in the set
        self.getAttributesListForClassifier(attributes) 

        #Create an empty set of instances and populate with the instances
        dataMatrix = None 
        finalSet =  None 
        if (foldType == "Train"):
            finalSet = Instances("trainingSet", attributes, 0) 
            dataMatrix = self.getTrainingDataMatrix(instanceIndices) 
        
        if (foldType == "Validation") :
            finalSet = Instances("testSet", attributes, 0) 
            dataMatrix = self.getValidationDataMatrix(instanceIndices) 
        
        if (foldType == "Test") :
            finalSet = Instances("testSet", attributes, 0) 
            dataMatrix = self.getTestDataMatrix(instanceIndices) 
        

        i = 0
        while i < len(dataMatrix[0]):
            arr = self.init_list(len(dataMatrix), 0.0)
            j = 0
            while j < len(dataMatrix):
                arr[j] = dataMatrix[j][i]
                j+=1

            di = DenseInstance(1.0, arr) 
            finalSet.add(i, di) 
            i+=1
            
        finalSet.setClassIndex(self.targetColumnIndex-self.getNumberOfDateStringAndDistinctColumnsBeforeTargetClass()) 

        return finalSet 
    

    
     # Iterates over all the columns of the dataset object and returns those that can be
     # included in the Instances object that will be fed to Weka (i.e. excluding the Date
     # and String columns)
    
    def getAttributesListForClassifier(self, attributes) :
        
        i = 0
        while (i < len(self.columns)):
            
            currentColumn = self.columns.get(i) 
    
            #The dataset will not include the distinct value columns, if they exist
            if currentColumn in self.distinctValColumns:
                continue 
            
            att = None 
            checkType = currentColumn.getColumn().getType()
            if (checkType == "Numeric"):
                att = Attribute(str(i),i) 
            elif (checkType == "Discrete"):
                values = []
                numOfDiscreteValues = currentColumn.getColumn().getNumOfPossibleValues() 
                #if the number of distinct values exceeds the maximal amount, skip it
                if (numOfDiscreteValues <= self.maxNumOFDiscreteValuesForInstancesObject) :
                    j = 0
                    while j < self.numOfDiscreteValues:
                        values.append(str(j))
                        j+=1

                    att = Attribute(str(i), values, i) 
                     
            elif (checkType == "String"):
                #Most classifiers can't handle Strings. Currently we don't include them in the dataset
                x = 5
            elif (checkType == "Date"):
                #Currently we don't include them in the dataset. We don't have a way of handling "raw" dates
                x = 5
            else:
                raise Exception("unsupported column type")
           
    
            if (att != None) :
                attributes.append(att) 
    
            
            i+=1
        

                
   
     # Returns the number of columns in the dataset which are either String or Date
    
    def getNumberOfDateStringAndDistinctColumns(self) :
        numOfColumns = 0 
        for ci in self.columns :
            if (ci.getColumn().getType() == "Date" or ci.getColumn().getType() == "String" or ci in self.distinctValColumns) :
                numOfColumns+=1
            
            if (ci.getColumn().getType() == "Discrete" and ci.getColumn().getNumOfPossibleValues() >  self.maxNumOFDiscreteValuesForInstancesObject) :
                numOfColumns+=1 
            
        return numOfColumns 
    

    
     # In cases where the target class is not the last attribute, we need to determine its new index.
    
    def getNumberOfDateStringAndDistinctColumnsBeforeTargetClass(self) :
        numOfColumns = 0 
        for ci in self.columns :
            if (ci == self.columns.get(self.targetColumnIndex)) :
                return numOfColumns 
            
            if (ci.getColumn().getType() == "Date" or ci.getColumn().getType() == "String" or
                    ci in self.distinctValColumns) :
                numOfColumns+=1 
            
            if (ci.getColumn().getType() == "Discrete" and (ci.getColumn()).getNumOfPossibleValues() >  self.maxNumOFDiscreteValuesForInstancesObject) :
                numOfColumns+=1 
       
        return numOfColumns 
    

   
     # Returns a two-dimensional, Weka-friendly array. The array contains only the lines whose indices
     # were provided
    
    def getDataMatrixByIndices(self, indicesList) :
        #we distinct val column(s) is not included in the matrix
        #init data matrix
        data = self.init_list(len(self.columns) - ( len(self.distinctValColumns) + self.getNumberOfDateStringAndDistinctColumns()), None)
        i = 0
        while i < len(data):
            data[i] = self.init_list(len(indicesList), 0.0)
            i+=1
            
        skippedColumnsCounter = 0 
        
        col = 0
        while col < len(self.columns):
            #if this is a distinct val column or if the column is a raw string
            if self.shouldColumnBeExludedFromDataMatrix(col) :
                skippedColumnsCounter+=1
                continue 
            
            rowCounter = 0 
            isNumericColumn = self.columns.get(col).getColumn().getType() == "Numeric" 

            for row in indicesList :
                if (isNumericColumn) :
                    data[col-skippedColumnsCounter][rowCounter] = float(self.columns.get(col).getColumn().getValue(row))
                else :
                    data[col-skippedColumnsCounter][rowCounter] = int(self.columns.get(col).getColumn().getValue(row))
                
                rowCounter+=1 
            
            col+=1
     
        return data 
    

   
    # Used to return the target class values for a list of intances.
    def getInstancesClassByIndex(self, instanceIndices) :
        targetColumn = self.getTargetClassColumn().getColumn() 
        mapToReturn = {} 
        for index in instanceIndices:
            mapToReturn[index]= int(targetColumn.getValue(index)) 
        
        return mapToReturn 
    

    
    # Returns the training set instances in a Weka-friendly, two-dimentsional array format
  
    def getTrainingDataMatrix(self, indices) :
        if (self.trainFoldDistinctValMappings != None and len(self.trainFoldDistinctValMappings) > 0) :
            raise Exception("need to address this scenario") 

        if (indices != None) :
            indicesToRunOn = indices 
        
        else :
            indicesToRunOn = self.indicesOfTrainingFolds 
        
        data = self.init_list(len(self.columns),None)
        i = 0
        while i < len(data):
            data[i] = self.init_list(len(indicesToRunOn),0.0)
            i+=1
            
        skippedColumnsCounter = 0 
        col = 0
        while col < len(self.columns):
            
            #if this is a distinct val column or if the column is a raw string
            if ( self.shouldColumnBeExludedFromDataMatrix(col)) :
                skippedColumnsCounter+=1 
                continue 
            
            rowCounter = 0 
            isNumericColumn = self.columns.get(col).getColumn().getType() == "Numeric" 

            for row in indicesToRunOn :
                if (isNumericColumn) :
                    data[col-skippedColumnsCounter][rowCounter] = float(self.columns.get(col).getColumn().getValue(row) ) 
                else :
                    data[col-skippedColumnsCounter][rowCounter] = int(self.columns.get(col).getColumn().getValue(row)) 
                
                rowCounter+=1 
            
            col+=1
        
        return data 
    


    # Returns the training set instances in a Weka-friendly, two-dimentsional array format
    def getValidationDataMatrix(self, indices):
        if (self.trainFoldDistinctValMappings != None and len(self.trainFoldDistinctValMappings) > 0):
            return self.getValidationDataMatrixWithDistinctVals() 

        if (indices != None) :
            indicesToRunOn = indices 
        
        else :
            indicesToRunOn = self.indicesOfValidationFolds 
         
        
        data = self.init_list(len(self.columns) - (self.getNumberOfDateStringAndDistinctColumns()),None)
        i = 0
        while i < len(data):
            data[i] = self.init_list(len(indicesToRunOn),0.0)
            i+=1
            
        skippedColumnsCounter = 0 
        
        col = 0
        while col < len(self.columns):
                        #if this is a distinct val column or if the column is a raw string
            if ( self.shouldColumnBeExludedFromDataMatrix(col)) :
                skippedColumnsCounter+=1 
                continue 
            
            rowCounter = 0 
            isNumericColumn = self.columns.get(col).getColumn().getType() == "Numeric" 

            for row in indicesToRunOn :
                if (isNumericColumn) :
                    data[col-skippedColumnsCounter][rowCounter] = float(self.columns.get(col).getColumn().getValue(row))
                else :
                    data[col-skippedColumnsCounter][rowCounter] = int(self.columns.get(col).getColumn().getValue(row))
                
                rowCounter+=1

            col+=1

        return data 
    

   
    # Returns the test set instances in a Weka-friendly, two-dimentsional array format

    def getTestDataMatrix( self, indices) :
        if (self.testFoldDistinctValMappings != None and len(self.testFoldDistinctValMappings) > 0):
            return self.getTestDataMatrixWithDistinctVals() 

        if (indices != None) :
            indicesToRunOn = indices 
        
        else :
            indicesToRunOn = self.indicesOfTestFolds 
        
        
         
        data = self.init_list(len(self.columns) - (self.getNumberOfDateStringAndDistinctColumns()),None)
        i = 0
        while i < len(data):
            data[i] = self.init_list(len(indicesToRunOn),0.0)
            i+=1
            
        skippedColumnsCounter = 0
        
        col = 0
        while col < len(self.columns) :

            if ( self.shouldColumnBeExludedFromDataMatrix(col)) :
                skippedColumnsCounter+=1 
                continue 
            
            rowCounter = 0 
            isNumericColumn = self.columns.get(col).getColumn().getType() == "Numeric" 
            
            for row in indicesToRunOn :
                if (isNumericColumn) :
                    data[col-skippedColumnsCounter][rowCounter] = float(self.columns.get(col).getColumn().getValue(row)) 
                
                else :
                    data[col-skippedColumnsCounter][rowCounter] = int(self.columns.get(col).getColumn().getValue(row)) 
                
                rowCounter+=1 
                
            col+=1
        
        return data 
    


    # Identical to the getTrainingDataMatrix function, but returns a matrix containing a single index for
    # each distinct value combination
 
    def getTrainingDataMatrixWithDistinctVals(self) :
        data = self.init_list(len(self.columns) - (self.getNumberOfDateStringAndDistinctColumns()),None)
        i = 0
        while i < len(data):
            data[i] = self.init_list(len(self.trainFoldDistinctValMappings),0.0)
            i+=1

        skippedColumnsCounter = 0 
        col = 0
        while col < len(self.columns):
            #if this is a distinct val column or if the column is a raw string
            if ( self.shouldColumnBeExludedFromDataMatrix(col)) :
                skippedColumnsCounter+=1 
                continue 
            
            rowCounter = 0 
            isNumericColumn = self.columns.get(col).getColumn().getType() == "Numeric" 

            for key in self.trainFoldDistinctValMappings.keys():
                #now we take a single representative from this group
                index = self.trainFoldDistinctValMappings.get(key)[0] 
                if (isNumericColumn) :
                    data[col-skippedColumnsCounter][rowCounter] = float(self.columns.get(col).getColumn().getValue(index))
                else :
                    data[col-skippedColumnsCounter][rowCounter] = int(self.columns.get(col).getColumn().getValue(index))
        
                rowCounter+=1 
            
            col+=1       
        
        return data 
    


    # Returns the distinct value mappings of each instance in the training set (for each instance, we get
    # a list of all the instances with the same distinct value)

    def getTrainFoldDistinctValMappings(self) :
        return self.trainFoldDistinctValMappings  


    # Returns the distinct value mappings of each instance in the test set (for each instance, we get
    # a list of all the instances with the same distinct value)

    def getTestFoldDistinctValMappings(self) :
        return self.testFoldDistinctValMappings 
    



    def shouldColumnBeExludedFromDataMatrix(self, columnIndex) :

        if ( self.columns.get(columnIndex)) in self.distinctValColumns:
            return True 
        if self.columns.get(columnIndex).getColumn().getType() == "String":
            return True 
        if self.columns.get(columnIndex).getColumn().getType() == "Date":
            return True 
        if self.columns.get(columnIndex).getColumn().getType() == "Discrete" and self.columns.get(columnIndex).getColumn().getNumOfPossibleValues() >  self.maxNumOFDiscreteValuesForInstancesObject:
            return True 
        return False 
    

    def getValidationDataMatrixWithDistinctVals() :
        raise Exception("NotImplementedException") 
    
    
     # Identical to the getTrainingDataMatrix function, but returns a matrix containing a single index for
     # each distinct value combination
   
    def getTestDataMatrixWithDistinctVals(self) :
        
        data = self.init_list(len(self.columns) - (self.getNumberOfDateStringAndDistinctColumns()),None)
        i = 0
        while i < len(data):
            data[i] = self.init_list(len(self.testFoldDistinctValMappings),0.0)
            i+=1

        skippedColumnsCounter = 0 
        col = 0
        while (col < len (self.columns)) :
            if ( self.shouldColumnBeExludedFromDataMatrix(col)) :
                skippedColumnsCounter+=1 
                continue 
            
            rowCounter = 0 
            isNumericColumn = self.columns.get(col).getColumn().getType() == "Numeric" 
            for  key in self.testFoldDistinctValMappings.keys():
                #now we take a single representative from this group
                index = self.testFoldDistinctValMappings.get(key)[0] 
                if (isNumericColumn) :
                    data[col-skippedColumnsCounter][rowCounter] = float(self.columns.get(col).getColumn().getValue(index))
                else :
                    data[col-skippedColumnsCounter][rowCounter] = int(self.columns.get(col).getColumn().getValue(index)) 
                
                rowCounter+=1 
            
            col +=1
     
        return data 
    


     # Partitions the training folds into a set of LOO folds. One of the training folds is designated as "test",
     # while the remaining folds are used for training. All possible combinations are returned.
   
    def GenerateTrainingSetSubFolds(self) :
        #first, get all the training folds in the current dataset

        trainingFolds = [] 
        for fold in self.folds :
            if (not fold.isTestFold()) :
                trainingFolds.append(fold) 
            
        
        trainingDatasets = []
        i = 0
        while i < len(trainingFolds):
            newFoldsList = [] 
            j = 0
            while j < len (trainingFolds):
                
                currentFold = trainingFolds.get(j) 
                #if i==j, then this is the test fold
                if i == j:
                    dataType = "Test"
                else:
                    dataType = "Train"
                newFold = Fold(self.getNumOfClasses(),dataType) 
                newFold.setIndices(currentFold.getIndices()) 
                newFold.setNumOfInstancesInFold(currentFold.getNumOfInstancesInFold()) 
                newFold.setInstancesClassDistribution(currentFold.getInstancesClassDistribution()) 
                newFold.setIndicesPerClass(currentFold.getIndicesPerClass()) 
                newFold.setDistinctValMappings(currentFold.getDistinctValMappings()) 
                newFoldsList.append(newFold) 
                
                j+=1
         
            #now that we have the folds, we can generate the Dataset object
            subDataset = Dataset( self.columns, newFoldsList, self.targetColumnIndex,  self.name,  self.numOfInstancesPerColumn,  self.distinctValColumns,  self.randomSeed,  self.maxNumOFDiscreteValuesForInstancesObject) 
            trainingDatasets.append(subDataset) 
        
            i+=1

        return trainingDatasets 
    

   
     # Determines whether the values of the a column adhere to the distinct value requirements.
     # These columns will be used fot the initial candiate features generation
  
    def isColumnDistinctValuesCompatibe(self, ci) :
        if ci.isTargetClass() or (ci in self.distinctValColumns) or ci.getColumn().getType() == "Date"  or ci.getColumn().getType() == "String":
            return False 

        try :
            distinctValsDict = {} 
            valuesMap = {} 
            i = 0
            while i < len(self.indices) :
                j = self.indices.get(i) 
                sourceValues = map(lambda c : c.getColumn().getValue(j), self.distinctValColumns) 
                if (not sourceValues in distinctValsDict) :
                    distinctValsDict[sourceValues] = ci.getColumn().getValue(j) 
                else :
                    if (distinctValsDict.get(sourceValues) != ci.getColumn().getValue(j)):
                        return False 
                
                i+=1

        except:
            raise Exception("Error in isColumnDistinctValuesCompatibe") 
        
        return True 
    


     # Generates a new Dataset object which points to a subset of the columns in the original dataset. The
     # target class attribute is always added (if only the target class is returned then this function becomes
     # the equivalent of emptyReplica()
 
    def replicateDatasetByColumnIndices(self, indices) :
        dataset = Dataset() 

        #We need to create a new columns object and just reference the same objects
        dataset.columns = []
        indices.sort() 
        for index in indices:
            dataset.columns.append(self.columns.get(index)) 
        
        if not self.getTargetClassColumn() in dataset.columns:
            dataset.addColumn(self.getTargetClassColumn()) 
        
        dataset.targetColumnIndex = len(dataset.columns)-1 

        dataset.numOfInstancesPerColumn =  self.numOfInstancesPerColumn 
        dataset.indices =  self.indices 
        dataset.folds =  self.folds 
        dataset.indicesOfTrainingFolds =  self.indicesOfTrainingFolds 
        dataset.indicesOfValidationFolds =  self.indicesOfValidationFolds 
        dataset.indicesOfTestFolds =  self.indicesOfTestFolds 
        dataset.numOfTrainingInstancesPerClass =  self.numOfTrainingInstancesPerClass 
        dataset.numOfValidationInstancesPerClass =  self.numOfValidationInstancesPerClass 
        dataset.numOfTestInstancesPerClass =  self.numOfTestInstancesPerClass 
        dataset.numOfTrainingRows =  self.numOfTrainingRows 
        dataset.numOfValidationRows =  self.numOfValidationRows 
        dataset.numOfTestRows =  self.numOfTestRows 
        dataset.name =  self.name 
        dataset.distinctValColumns =  self.distinctValColumns 
        dataset.trainingIndicesByClass =  self.trainingIndicesByClass 
        dataset.trainFoldDistinctValMappings =  self.trainFoldDistinctValMappings 
        dataset.validationIndicesByClass =  self.validationIndicesByClass 
        dataset.validationFoldDistinctValMappings =  self.validationFoldDistinctValMappings 
        dataset.testFoldDistinctValMappings =  self.testFoldDistinctValMappings 
        dataset.trainFoldsDistinctValRepresentatives =  self.trainFoldsDistinctValRepresentatives 
        dataset.testFoldsDistinctValRepresentatives =  self.testFoldsDistinctValRepresentatives 
        dataset.distinctValueCompliantColumns =  self.distinctValueCompliantColumns 
        dataset.maxNumOFDiscreteValuesForInstancesObject =  self.maxNumOFDiscreteValuesForInstancesObject 

        return dataset 
    

   
     # Creates an exact replica of the dataset, except for the fact that it creates a new List of columns
     # instead of referencing to the existing list. This enables the addition of columns to this object without
     # adding them to the original.
  
    def replicateDataset(self) :
        dataset = Dataset() 

        #We need to create a new columns object and just reference the same objects
        dataset.columns = [] 
        for  ci in self.columns:
            dataset.columns.append(ci) 

        dataset.numOfInstancesPerColumn =  self.numOfInstancesPerColumn 
        dataset.indices =  self.indices 
        dataset.folds =  self.folds 
        dataset.indicesOfTrainingFolds =  self.indicesOfTrainingFolds 
        dataset.indicesOfTestFolds =  self.indicesOfTestFolds 
        dataset.numOfTrainingInstancesPerClass =  self.numOfTrainingInstancesPerClass 
        dataset.numOfTestInstancesPerClass =  self.numOfTestInstancesPerClass 
        dataset.numOfTrainingRows =  self.numOfTrainingRows 
        dataset.numOfTestRows =  self.numOfTestRows 
        dataset.targetColumnIndex =  self.targetColumnIndex 
        dataset.name =  self.name 
        dataset.distinctValColumns =  self.distinctValColumns 
        dataset.trainingIndicesByClass =  self.trainingIndicesByClass 
        dataset.trainFoldDistinctValMappings =  self.trainFoldDistinctValMappings 
        dataset.testFoldDistinctValMappings =  self.testFoldDistinctValMappings 
        dataset.trainFoldsDistinctValRepresentatives =  self.trainFoldsDistinctValRepresentatives 
        dataset.testFoldsDistinctValRepresentatives =  self.testFoldsDistinctValRepresentatives 
        dataset.distinctValueCompliantColumns =  self.distinctValueCompliantColumns 
        dataset.maxNumOFDiscreteValuesForInstancesObject =  self.maxNumOFDiscreteValuesForInstancesObject 

        return dataset 
    

    
     # Creates an exact replica of the dataset, except for the fact that it creates a new List of columns
     # instead of referencing to the existing list. This enables the addition of columns to this object without
     # adding them to the original.
  
    def replicateDatasetDeep(self) :
        dataset = Dataset() 

        #We need to create a new columns object and just reference the same objects
        dataset.columns = []
        for  ci in self.columns:
            dataset.columns.append(ci) 
        
        dataset.numOfInstancesPerColumn =  self.numOfInstancesPerColumn 
        dataset.indices = copy.deepcopy(self.indices) 
        dataset.folds =  copy.deepcopy(self.folds) 
        dataset.indicesOfTrainingFolds = copy.deepcopy(self.indicesOfTrainingFolds) 
        dataset.indicesOfTestFolds = copy.deepcopy( self.indicesOfTestFolds)
             
        dataset.numOfTrainingInstancesPerClass = copy.deepcopy(self.numOfTrainingInstancesPerClass)
  
        dataset.numOfTestInstancesPerClass = copy.deepcopy(self.numOfTestInstancesPerClass)

        dataset.numOfTrainingRows =  self.numOfTrainingRows 
        dataset.numOfTestRows =  self.numOfTestRows 
        dataset.targetColumnIndex =  self.targetColumnIndex 
        dataset.name =  self.name 
        dataset.distinctValColumns = copy.deepcopy( self.distinctValColumns) 
        dataset.trainingIndicesByClass = copy.deepcopy(self.trainingIndicesByClass)

        dataset.trainFoldDistinctValMappings = copy.deepcopy( self.trainFoldDistinctValMappings) 
        dataset.testFoldDistinctValMappings = copy.deepcopy( self.testFoldDistinctValMappings) 
        dataset.trainFoldsDistinctValRepresentatives = copy.deepcopy( self.trainFoldsDistinctValRepresentatives) 
        dataset.testFoldsDistinctValRepresentatives = copy.deepcopy( self.testFoldsDistinctValRepresentatives) 
        dataset.distinctValueCompliantColumns = copy.deepcopy( self.distinctValueCompliantColumns) 
        dataset.maxNumOFDiscreteValuesForInstancesObject =  self.maxNumOFDiscreteValuesForInstancesObject 

        return dataset 
    



    def generateRandomSubDataSet(self, numOfInstancesPerFold, randomSeed) :

        newFoldsList = [] 
        
        for fold in self.folds:
            subFold = fold.generateSubFold(numOfInstancesPerFold, randomSeed) 
            newFoldsList.append(subFold) 
    
        dataset = Dataset( self.columns,newFoldsList, self.targetColumnIndex, self.name+"_subfold", self.numOfInstancesPerColumn, self.distinctValColumns,randomSeed, self.maxNumOFDiscreteValuesForInstancesObject) 
        return dataset 
    


  
     # Creates a replica of the given Dataset object, but without any columns except for the target column and the
     # distinct value columns (if they exist)
    
    def emptyReplica(self) :
        dataset = Dataset() 

        #We need to create a new columns object and just reference the same objects
        dataset.columns = []
        dataset.numOfInstancesPerColumn = self.numOfInstancesPerColumn 
        dataset.indices =  self.indices 
        dataset.indicesOfTrainingFolds = self.indicesOfTrainingFolds 
        dataset.indicesOfTestFolds = self.indicesOfTestFolds 
        dataset.numOfTrainingInstancesPerClass = self.numOfTrainingInstancesPerClass 
        dataset.numOfTestInstancesPerClass =  self.numOfTestInstancesPerClass 
        dataset.numOfTrainingRows = self.numOfTrainingRows 
        dataset.numOfTestRows = self.numOfTestRows 
        dataset.name = self.name 
        dataset.distinctValColumns = self.distinctValColumns 
        dataset.trainingIndicesByClass =  self.trainingIndicesByClass 
        dataset.trainFoldDistinctValMappings =  self.trainFoldDistinctValMappings 
        dataset.testFoldDistinctValMappings =  self.testFoldDistinctValMappings 
        dataset.trainFoldsDistinctValRepresentatives =  self.trainFoldsDistinctValRepresentatives 
        dataset.testFoldsDistinctValRepresentatives =  self.testFoldsDistinctValRepresentatives 
        dataset.distinctValueCompliantColumns =  self.distinctValueCompliantColumns 
        dataset.maxNumOFDiscreteValuesForInstancesObject =  self.maxNumOFDiscreteValuesForInstancesObject 

        #since we only add the target column, it's index in 0
        dataset.targetColumnIndex = 0 
        dataset.columns.append( self.columns.get( self.targetColumnIndex)) 

        #add the distinct value columns to the empty dataset
        for ci in self.distinctValColumns :
            dataset.columns.append(ci)  

        return dataset 
    

   
    # Returns a single indice for each distinct values combination in the training folds
   
    def getTrainFoldsDistinctValRepresentatives(self) :
        return self.trainFoldsDistinctValRepresentatives 
    

 
    # Returns a single indice for each distinct values combination in the test folds
   
    def getTestFoldsDistinctValRepresentatives(self) :
        return self.testFoldsDistinctValRepresentatives 
    

  
     # Returns the list of columns whose values satisfy the constraint of the distinct value
 
    def getDistinctValueCompliantColumns(self) :
        return self.distinctValueCompliantColumns 
    

    def getTrainingFolds(self) :
        listToReturn = []
        for fold in self.folds:
            if (fold.getTypeOfFold() == "Train") :
                listToReturn.append(fold) 
            
        return listToReturn 
    

    def getValidationFolds(self) :
        listToReturn = []
        for fold in self.folds :
            if (fold.getTypeOfFold() == "Validation") :
                listToReturn.append(fold) 
  
        return listToReturn 
    


    def getTestFolds(self) :
        listToReturn = [] 
        for  fold in self.folds :
            if (fold.getTypeOfFold() == "Test") :
                listToReturn.append(fold) 

        return listToReturn 
    

   
    # returns a map with the ratio of each class in the dataset (can be calculated on the training instances or on
    # all of the indices)

    def getClassRatios(self, useAllIndices) :
        
        #numOfTrainingInstancesPerClass 
        #numOfValidationInstancesPerClass 
        #numOfTestInstancesPerClass 
         
        totalNumberOfInstancesPerClass = {} 
        if (useAllIndices) :
            classIndex = 0
            while classIndex < self.getNumOfClasses() :
                if (self.numOfValidationInstancesPerClass != None) :
                    totalNumberOfInstancesPerClass[classIndex] = float(self.numOfTrainingInstancesPerClass[classIndex] +
                            self.numOfValidationInstancesPerClass[classIndex] + self.numOfTestInstancesPerClass[classIndex]) 
                
                else :
                    totalNumberOfInstancesPerClass[classIndex] = float(self.numOfTrainingInstancesPerClass[classIndex] +
                                                                        self.numOfTestInstancesPerClass[classIndex]) 
                
                classIndex+=1
                
        
        else :
            classIndex = 0
            while classIndex < self.getNumOfClasses() :
                totalNumberOfInstancesPerClass[classIndex] = float(self.numOfTrainingInstancesPerClass[classIndex]) 
                classIndex += 1

        mapToReturn = {} 
        classIndex = 0
        while classIndex < self.getNumOfClasses() :
            mapToReturn[classIndex] = totalNumberOfInstancesPerClass.get(classIndex)/int(sum(totalNumberOfInstancesPerClass.values)) 
            classIndex+=1

        return mapToReturn 
    

    
     # Updates the class attribute values of the instances whose indices are provided to the one that is provided to the function (to be used in experimental settings such as co-training).
    def updateInstanceTargetClassValue(self, instanceIndices, newTargetClassValue) :
        for index in instanceIndices :
            self.getTargetClassColumn().getColumn().setValue(index, newTargetClassValue) 

  
    # Returns a (sorted by index) array of target class values, based on the provided indices

    def getTargetClassLabelsByIndex(self, indices) :
        arrayToReturn = self.init_list(len(indices), 0)
        targetClassColumn = self.getTargetClassColumn().getColumn() 
        indices.sort()
        i = 0
        while i < len(indices):
            arrayToReturn[i] = targetClassColumn.getValue(indices.get(i)) 
            i+=1
      
        return arrayToReturn 
    

   
     # Receives an Instances object and writes it into an ARFF file (can be used as a convenient work around for
     # replication and creating subsets of the data).

    def saveInstancesToARFFFile( instances,  path) :

        try :
            s = ArffSaver() 
            s.setInstances(instances) 
            file = File(path)
            s.setFile(file) 
            s.writeBatch() 
            return True 
        
        except: 
            print("Error writing instances to ARFF file. " ) 
            return False 

    
     # Returns the index of the class attribute

    def getTargetColumnIndex(self) :
        return self.targetColumnIndex 
