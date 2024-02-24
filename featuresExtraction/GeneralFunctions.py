import plotly.express as px
from GeneralFunctions import normalizeClassificationResults


class EvaluationAnalysisFunctions :

#/**
#     * combines the results of several classifiers (partitions) by multiplication. After the multiplication
#     * is complete, each confidence score is divided by the ratio of the class in the original labeled dataset
#     * @param evaluationResultsPerPartition
#     * @param numOfClasses
#     * @param classRatios
#     * @return
#     */
    def calculateMultiplicationClassificationResults(evaluationResultsPerPartition,
                                                     numOfClasses,classRatios) :
        resultsToReturn = {}
        for partition in evaluationResultsPerPartition.keys():
            for i in evaluationResultsPerPartition.get(partition).getScoreDistributions().keys():
                if not resultsToReturn.contains_key(i) :
                    double_list = []
                    resultsToReturn[i] = double_list
                  
                j = 0
                while j<numOfClasses:
                    if partition == 0: #if its the first partition we analyze, simply assign the value
                        resultsToReturn.get(i).append(evaluationResultsPerPartition.get(partition).getScoreDistributions().get(i)[j]) 
                      
                    else : #otherwise, multiply
                        resultsToReturn.get(i).append(resultsToReturn.get(i)[j] * evaluationResultsPerPartition.get(partition).getScoreDistributions().get(i)[j] )
                    j+=1
                  
              
          
        for i in evaluationResultsPerPartition.get(0).getScoreDistributions().keys():
            j = 0
            while j < numOfClasses:
                resultsToReturn.get(i).append(resultsToReturn.get(i)[j] / classRatios.get(j)) 
                j+=1
          
        return normalizeClassificationResults(resultsToReturn) 
      

                                                                       
    #/**
    # * Normalizes the classifiaction results for each instance
    # * @param results
    # * @return
    # */
    def normalizeClassificationResults(results) :
        for i in results.keys():
            sum = 0 
            j = 0
            while j < len(results.get(i)):
                sum += results.get(i)[j] 
                j+=1
            j = 0
            while j < len(results.get(i)):
                results.get(i)[j] = results.get(i)[j]/sum 
                j+=1
              
        return results 
      



   # /**
   #  * Combines the results of several classifiers by averaging
   #  * @param evaluationResultsPerPartition
   #  * @param numOfClasses
   #  * @return
   #  */
    def calculateAverageClassificationResults(evaluationResultsPerPartition, numOfClasses) :
        resultsToReturn = {}
        for partition in evaluationResultsPerPartition.keys():
            for i in evaluationResultsPerPartition.get(partition).getScoreDistributions().keys():
                j = 0
                while j < numOfClasses:
                    if (not resultsToReturn.contains_key(i)): 
                        double_list = []
                        resultsToReturn[i] = double_list
                    resultsToReturn.get(i)[j] += evaluationResultsPerPartition.get(partition).getScoreDistributions().get(i)[j] 
                    j+=1
              
          
        #now we normalize
        for i in resultsToReturn.keys():
            j = 0
            while j < numOfClasses:
                 resultsToReturn.get(i)[j] = resultsToReturn.get(i)[j] / numOfClasses 
                 j+=1
          
        return normalizeClassificationResults(resultsToReturn) 
      
