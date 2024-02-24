

class ColumnInfo :
    
    column = None
    sourceColumns = []
    targetColumns = []
    operator = None
    isTargetClass = False
    name = ""

    def __init__(self, column,  sourceColumns, targetColumns, operator,  name) :
        self.column = column
        self.sourceColumns = sourceColumns
        self.targetColumns = targetColumns
        self.operator = operator
        self.name = name
        self.isTargetClass = isTargetClass
    

    def getColumn(self) :
        return self.column
    

    def setColumn( self, column) :
        self.column = column
    

    def SetTargetClassValue(self, isTargetClass1):
        self.isTargetClass = isTargetClass1
    

    def isTargetClass(self):
        return self.isTargetClass
    

    def getSourceColumns(self) :
        return self.sourceColumns
    

    def getTargetColumns(self) :
        return self.targetColumns
    
    
    def getName(self) :
        return self.name
    

   
}