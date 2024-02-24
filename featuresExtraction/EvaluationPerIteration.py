import plotly.express as px

class EvaluationPerIteraion :

    evaluationPerIterationMap = {}
    lastKey = 0

    def addEvaluationInfo(self, ei,  iterationIndex) :
        self.evaluationPerIterationMap[iterationIndex] = ei
        self.lastKey = iterationIndex
    

    def getLatestEvaluationInfo(self) :
        return self.evaluationPerIterationMap.get(self.lastKey)
    


    # Returns the latest X iterations
    def getLastXEvaluations( numOfEvaluations) :
        return None
    
    
    def getIterationEvaluationInfo( self, iterationIndex):
        return self.evaluationPerIterationMap.get(iterationIndex)
    


    def getEvaluationInfoByIndices(self, indices) :
        mapToReturn = {}
        for index in indices:
            mapToReturn[index] = self.evaluationPerIterationMap.get(index)
        
        return mapToReturn
    

    def getEvaluationInfoByStartAndFinishIndices(self, startIndex, endIndex) :
        mapToReturn = {}
        i = startIndex
        while i<endIndex:
            mapToReturn[i] = self.evaluationPerIterationMap.get(i)
            i+=1

        return mapToReturn
    
    
    def getAllIterations(self) :
        return self.evaluationPerIterationMap
    

    def getAllIterationsScoreDistributions(self) :
        #return evaluationPerIterationMap;
        mapToReturn = {}
        for i in self.evaluationPerIterationMap.values():
            mapToReturn[i] = self.evaluationPerIterationMap.get(i).getScoreDistributions()
        
        return mapToReturn
    
