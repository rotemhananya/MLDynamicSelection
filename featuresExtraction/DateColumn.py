import plotly.express as px

class DateColumn :
    values = []
    dateFormat = ""

 
     # This map is required as this type of columns is used for sliding window attribute generation.
     # Since this is the case, we need to be able to sort by it.
     
    indicesByDate = {}
    
    
    def init_list(size, element):
        list_to_return = []
        i = 0
        while i< size:
            list_to_return.append(element)
            i+=1
        return list_to_return

    def __init__(self, size,  dateFormat) :
        self.values = self.init_list(size, None)
        self.dateFormat = dateFormat
    

    def getDateFormat(self) :
        return self.dateFormat
    

    
    def getValue(self, i) :
        return self.values[i]
    

 
    def setValue(self, i,  obj) :
        val = obj
        self.values[i] = val
        if (not val in self.indicesByDate.keys()) :
            new_list = []
            self.indicesByDate[val] = new_list
        
        self.indicesByDate.get(val).append(i)
    

    
    def getNumOfInstances(self) :
        return len(self.values)
    

  
    def getType(self) :
        return self.columnType.Date
    

    def getIndicesByDate(self) :
        return  self.indicesByDate
    

    def getValues(self) :
            return self.values

    def setAllValues(self, v) :
        self.values = v
    

