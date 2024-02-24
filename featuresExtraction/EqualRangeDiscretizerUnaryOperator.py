import ColumnInfo
import Dataset
import DiscreteColumn
import numpy as np
from EqualRangeDiscretizerUnaryOperator import GetBinIndex 

class EqualRangeDiscretizerUnaryOperator :

    upperBoundPerBin = []
    upperBoundPerBin_size = 0

    def __init__(self, upperBoundPerBin1, size):
        self.upperBoundPerBin = upperBoundPerBin1
        self.upperBoundPerBin_size = size
        
        if len(self.upperBoundPerBin) < size:
            i = len(self.upperBoundPerBin)
            while i < size:
                self.upperBoundPerBin.append(0.0)   
                i+=1
    

    def isApplicable(dataset, sourceColumns, targetColumns) :
        if (len(sourceColumns) != 1 or (targetColumns is not None and len(targetColumns) != 0)) :
            return False
        else:
            if (sourceColumns.get(0).getColumn().getType().equals("Numeric")) :
                return True    
        return False
    

    def processTrainingSet(self, dataset, sourceColumns, targetColumns) :
        minVal = 3.402823466E+38 
        maxVal = 1.175494351E-38
        columnInfo = sourceColumns.get(0);
        i = 0
        while ( i<dataset.getNumOfTrainingDatasetRows()):  
            j = dataset.getIndicesOfTrainingInstances().get(i)
            val = columnInfo.getColumn().getValue(i);
            if (not np.isnan(val) and np.isfinite(val)) :
                minVal = min(minVal, val)
                maxVal = max(maxVal, val)  
            else :
                x=5;
            i+=1
            
     
        range_ = (maxVal-minVal)/len(self.upperBoundPerBin)
        currentVal = minVal;
        i=0
        while i<len(self.upperBoundPerBin):
            self.upperBoundPerBin[i] = currentVal + range_
            currentVal += range_
            i+=1


    def generate(self, dataset, sourceColumns, targetColumns, enforceDistinctVal) :
        try :
            column = DiscreteColumn(dataset.getNumOfInstancesPerColumn(), len(self.upperBoundPerBin))
            #this is the number of rows we need to work on - not the size of the vector
            numOfRows = dataset.getNumberOfRows()
            columnInfo = sourceColumns.get(0);
            i = 0
            while (i < numOfRows):   
                if (len(dataset.getIndices()) == i) :
                    x = 5
                
                j = dataset.getIndices().get(i)
                binIndex = GetBinIndex(columnInfo.getColumn().getValue(j))
                column.setValue(j, binIndex)
                i+=1
                
            #now we generate the name of the new attribute
            attString = "EqualRangeDiscretizer("
            attString = attString+columnInfo.getName()
            attString = attString+")"

            return ColumnInfo(column, sourceColumns, targetColumns, None, attString)
  
        except:
            print("error in EqualRangeDiscretizer" )
            return None;


    def GetBinIndex(self, value) :
        i=0
        while (i<len(self.upperBoundPerBin)):
            if (self.upperBoundPerBin[i] > value) :
                return i
            i+=1

        return (len(self.upperBoundPerBin)-1)


    def getType() :
        return "Unary"
    
    def getOutputType(): 
        return "Discrete"

    def requiredInputType():
        return "Numeric"
    

    def getName() :
        return "EqualRangeDiscretizerUnaryOperator"
    

    def getNumOfBins(self):
        return len(self.upperBoundPerBin)
    
    

    # Used to determine whether the values of a column are in accordance with the distinct val requirement.
    def isDistinctValEnforced( dataset, evaluatedColumn) :
        if (len(dataset.getDistinctValueColumns()) == 0) :
            return True
        
        distinctValsDict = {}
        numOfRows = dataset.getNumberOfRows()

        i = 0
        while i < numOfRows:
            j = dataset.getIndices().get(i)
            sourceValues = map(lambda c: c.getColumn().getValue(j),dataset.getDistinctValueColumns())
            
            if (not sourceValues in distinctValsDict) :
                distinctValsDict[sourceValues]= evaluatedColumn.getColumn().getValue(j)
            else :
                if (not distinctValsDict.get(sourceValues) == (evaluatedColumn.getColumn().getValue(j))):
                    return False
                
            i+=1
            
        return True
    
    

