class DiscreteColumn:
    
    values = []
    numOfPossibleValues = -1
    size = 0

    def __init__(self, size, numOfPossibleValues):
        self.size = size
        self.numOfPossibleValues = numOfPossibleValues;
    

    def getValue(self, index):
        return self.values[index] 

    def setValue(self, i, obj):
        self.values[i] = obj;
    

    def getType() :
        return "Discrete"

    def getNumOfInstances(self) :
        return len(self.values)
    

    def getNumOfPossibleValues(self) :
        return self.numOfPossibleValues
    

    def setAllValues(self, v) :
        self.values += v;
    

    def getValues(self): 
        return self.values

