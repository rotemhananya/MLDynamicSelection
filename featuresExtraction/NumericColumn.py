
 
class NumericColumn :
    
    
    values = []
    size = 0

    def __init__(self, size) :
        self.size = size
    

    def getValue(self, i) :
        return self.values[i]
    

    def setValue(self, i, obj) :
        self.values[i] = float(obj)


    def getType() :
        return "Numeric"

    def getNumOfInstances(self) :
        return self.size
    

    def setAllValues(self, v) :
        self.values = v
    

    def getValues(self) :
        return self.values
