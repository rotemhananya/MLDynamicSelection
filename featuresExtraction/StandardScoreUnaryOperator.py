import Dataset
import ColumnInfo
import NumericColumn
import numpy as np
import statistics
import math
from StandardScoreUnaryOperator import getStandardScore

class StandardScoreUnaryOperator :

    avg = 0
    stdev = 0

    def generate(dataset, sourceColumns, targetColumns, enforceDistinctVal) :
        column = NumericColumn(dataset.getNumOfInstancesPerColumn())
        #this is the number of rows we need to work on - not the size of the vector
        numOfRows = dataset.getNumberOfRows()
        columnInfo = sourceColumns.get(0)
        i = 0
        while( i<numOfRows):
            j = dataset.getIndices().get(i)
            standardScoreVal = getStandardScore(columnInfo.getColumn().getValue(j))
            if (np.isnan(standardScoreVal) or (not np.isfinite(standardScoreVal))) :
                    x = 5
            else :
                column.setValue(j, standardScoreVal)
            
            i+=1

        #now we generate the name of the new attribute
        attString = "StandardScoreUnaryOperator(";
        attString = attString + columnInfo.getName()
        attString = attString + (")")

        return ColumnInfo(column, sourceColumns, targetColumns, None, attString)


    def getStandardScore(self, value) :
        if (self.stdev == 0) :
            return 0  
        return (value - self.avg)/self.stdev
    

    def processTrainingSet(self, dataset, sourceColumns, targetColumns) :
        vals = []
        columnInfo = sourceColumns.get(0)
        
        i = 0
        while (i<dataset.getNumOfTrainingDatasetRows()):
            j = dataset.getIndicesOfTrainingInstances().get(i)
            val = columnInfo.getColumn().getValue(i)        
            if (not np.isnan(val) and np.isfinite(val)) :
                vals.add(val)
            
            i+=1

        tempAvg = statistics.fmean(vals)

        if (tempAvg.isPresent()) :
            self.avg = tempAvg.getAsDouble();
            tempStdev = sum({math.pow(a-self.avg,2) for a in vals})
            self.stdev = math.sqrt(tempStdev/len(vals))
        
        else :
            print("no values in the attribute")
        
    

    def isApplicable(dataset, sourceColumns, targetColumns) :
        if (len(sourceColumns) != 1 or (targetColumns is not None and len(targetColumns) != 0)) :
            return False
        else:
            if (sourceColumns.get(0).getColumn().getType() == "Numeric") :
                return True
        
        return False
    

    def getOutputType() :
        return "Numeric"

    def requiredInputType() :
        return "Numeric"
    
    def getType() :
        return "Unary"
    
    def getName() :
        return "StandardScoreUnaryOperator";
    

    #this is a normalizer, not a discretizer
    def getNumOfBins() :
        return -1;
    
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
    
