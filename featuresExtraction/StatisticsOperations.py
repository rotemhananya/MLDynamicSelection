import numpy as np
import math
import ColumnInfo
import EqualRangeDiscretizerUnaryOperator
from scipy import stats
from scipy.stats import ttest_ind
from StatisticOperations import calculateChiSquareTestValues
from StatisticOperations import discretizeNumericColumn
from StatisticOperations import generateChiSuareIntersectionMatrix
from StatisticOperations import calculatePairedTTestValues

class StatisticOperations:




    # The function reveives two lists of features and returns a list of each possible pairs Paired T-Test values
    def calculatePairedTTestValues(list1, list2) :
        tTestValues = []
        for ci1 in list1:
            if ci1.getColumn().getType() != "Numeric":
                 raise Exception("Unable to process non-numeric columns - list 1")
            for ci2 in list2:
                if ci2.getColumn().getType() != "Numeric":
                     raise Exception("Unable to process non-numeric columns - list 2")
                
                testValue = math.abs(ttest_ind(ci1.getColumn().getValues(),ci2.getColumn().getValues()).statistic);
                if not np.isnan(testValue) :
                    tTestValues.append(testValue)

        return tTestValues
    

    def calculatePairedTTestValues(list1, columnInfo) :
        tempList = []
        tempList.append(columnInfo)
        return calculatePairedTTestValues(list1, tempList)
    


     # Calculates the Chi-Square test values among all the possible combonation of elements in the two provided list.
     # Also supports numeric attributes, a discretized versions of which will be used in the calculation.

    def calculateChiSquareTestValues(self, list1, list2, dataset, properties):
        bins = []
        erduo = EqualRangeDiscretizerUnaryOperator(bins, int(properties.numOfDiscretizationBins))
        chiSquareValues = []

        for ci1 in list1:
            if ci1.getColumn().getType() != "Discrete" and ci1.getColumn().getType() != "Numeric":
                 raise Exception("unsupported column type")
            
            for ci2 in list2:
                if ci2.getColumn().getType() != "Discrete" and ci2.getColumn().getType() != "Numeric":
                     raise Exception("unsupported column type")
                
  
                if ci1.getColumn().getType() == "Numeric":
                    tempColumn1 = discretizeNumericColumn(dataset, ci1,erduo, properties)
                
                else :
                    tempColumn1 = ci1
                
                if ci2.getColumn().getType() == "Numeric":
                    tempColumn2 = discretizeNumericColumn(dataset, ci2,erduo, properties)
                
                else:
                    tempColumn2 = ci2
                
                
                chiSquareTestVal , _ = stats.chisquare(
                    self.generateDiscreteAttributesCategoryIntersection(tempColumn1.getColumn(),
                                tempColumn2.getColumn()))

                if not np.isnan(chiSquareTestVal) and np.isfinite(chiSquareTestVal):
                    chiSquareValues.append(chiSquareTestVal)
                
            


        return chiSquareValues
    

    def calculateChiSquareTestValues(list1, columnInfo, dataset, properties):
        tempList = []
        tempList.append(columnInfo)
        return calculateChiSquareTestValues(list1, tempList, dataset, properties)
    

    #Receives a numeric column and returns its discretized version

    def discretizeNumericColumn( dataset, columnInfo, discretizer, properties) :
        if discretizer is None:
            bins = []
            discretizer = EqualRangeDiscretizerUnaryOperator(bins,int(properties.numOfDiscretizationBins))
        
        tempColumnsList = []
        tempColumnsList.append(columnInfo)
        discretizer.processTrainingSet(dataset,tempColumnsList,None)
        discretizedAttribute = discretizer.generate(dataset,tempColumnsList,None,False)
        return discretizedAttribute
    

    # Used to generate the data structure required to conduct the Chi-Square test on two data columns
 
    def generateDiscreteAttributesCategoryIntersection(col1, col2) :

        if (len(col1.getValues()) != len(col2.getValues())) :
             raise Exception("Columns do not have the same number of instances")
        
        return generateChiSuareIntersectionMatrix(col1.getValues(), col1.getNumOfPossibleValues(), col2.getValues(), col2.getNumOfPossibleValues())
    

    def generateChiSuareIntersectionMatrix(col1Values, col1NumOfValues, col2Values, col2NumOfValues):
        intersectionsMatrix = []
        i = 0
        while i < col1NumOfValues:
            intersectionsMatrix_j = []
            j = 0
            while j < col2NumOfValues:
                intersectionsMatrix_j.append(0)
                j+=1    
            intersectionsMatrix.append(intersectionsMatrix_j)
            i+=1
        i = 0
        while i<len(col1Values):
            intersectionsMatrix[col1Values[i]][col2Values[i]]+=1
            i+=1
       
        return intersectionsMatrix
