

class StringColumn :
    values=[]
    numOfPossibleValues = 0

    def __init__(self, size) :
        i = 0
        while i<size:
            self.values.append("")      
            i+=1
    

    def getValue(self, i) :
        return self.values[i]

    def setValue(self, i, obj) :
        self.values[i] = obj
    

    def getType() :
        return "String"

    def getNumOfInstances(self) :
        return len(self.values)
    

    def getNumOfPossibleValues(self) :
        return self.numOfPossibleValues
    

    def getValues(self) :
        return self.values

    def setAllValues(self, v) :
        self.values = v
    

