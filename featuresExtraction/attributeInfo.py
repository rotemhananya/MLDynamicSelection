

class AttributeInfo :
    
    attributeName = ""
    attributeType = ""
    value = -1
    numOfDiscreteValues = -1

    def __init__(self, attName, attType, attValue, numOfValues) :

        self.attributeName = attName
        self.attributeType = attType
        self.value = attValue;
        self.numOfDiscreteValues = numOfValues;
    

    def getAttributeName(self) :
        return self.attributeName
    

    def getAttributeType(self) :
        return self.attributeType
    

    def getValue(self) :
        return self.value
    

    def getNumOfDiscreteValues(self) :
        return self.numOfDiscreteValues
    

    def setValue(self, value) :
        self.value = value
    

