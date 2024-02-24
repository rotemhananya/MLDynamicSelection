# -*- coding: utf-8 -*-
from pyjavaproperties import Properties
import numpy as np
from scipy.stats import ttest_ind, normaltest, skewtest
from sklearn.cluster import KMeans
from collections import Counter
from classifier import Classifier
from sklearn.metrics import accuracy_score
from config import Config
import plotly.express as px
import attributeInfo
from scipy.stats import shapiro
from GeneralFunctions import calculateAverageClassificationResults
from GeneralFunctions import calculateMultiplicationClassificationResults
import StatisticOperations
import statistics
import random 
import configparser
from scipy import stats
from pyjavaproperties import Properties


class ScoreDistributionBasedAttributes:
    
    histogramItervalSize = 0.1
    
    
    # When we look at the previous iterations, we can look at "windows" of varying size. This list specifies the
    # number of iterations back (from the n-1) in each window. The attributes for each are calculated separately
    numOfIterationsBackToAnalyze = [1,3,5,10]
    confidenceScoreThresholds = [0.5001, 0.75, 0.9, 0.95]
    
    generalPartitionPercentageByScoreHistogram = {}  
     
    def getScoreDistributionBasedAttributes(self, unlabeledSamplesDataset, labeledSamplesDataset, currentIterationIndex, 
                                            evaluationResultsPerSetAndInteration, unifiedDatasetEvaulationResults, 
                                            targetClassIndex, type_, properties):

        attributes =  {}
        
        # We create three groups of features:
        # 1) Analyzes the current score distribution of the each partition, the unified set (single classifier) and the averaging/multiplication combinations
        # 2) For each of the cases described in section 1, compare their score distribution to past iterations
        # 3) Compare the score distributions for the various approaches described above (current iteration, past iterations)
        # NOTE: we currently analyze only the score distributions, but we can also extract the information provided by Weka for each evaluation
        
        # method type: reg or td
        # 1) td: for score distribution of the dataset after "selected batch" - by the batch generator
        # 2) reg: for the score distribution of the current dataset
        
        if type_ == "td":
            self.setNumOfIterationsBackToAnalyze([1,2,4,6,11])
        else:
            self.setNumOfIterationsBackToAnalyze([1,3,5,10])
            
        #region Generate the averaging and multiplication score distributions for all iterations
        #TODO: this really needs to be saved in a cache with only incremental updates. For large datasets this can take a long time
        
        averageingScoreDistributionsPerIteration =  {}
        lastKey_averageingScoreDistributionsPerIteration = None
        multiplicationScoreDistributionsPerIteration = {}
        lastKey_multiplicationScoreDistributionsPerIteration = None
        
        
        for i in range (len(evaluationResultsPerSetAndInteration)):
            # start by getting the EvaluationInfo object of the respective iterations for each partition
            partitionsIterationEvaulatioInfos = {}  
            for partitionIndex in evaluationResultsPerSetAndInteration.keys():
                partitionsIterationEvaulatioInfos[partitionIndex] = evaluationResultsPerSetAndInteration.values().get(partitionIndex).getIterationEvaluationInfo(currentIterationIndex)
            
            # next, calculate the averaging score of the iteration
            averagingScoreDistribution = calculateAverageClassificationResults(partitionsIterationEvaulatioInfos, labeledSamplesDataset.getNumOfClasses())
            averageingScoreDistributionsPerIteration[i] = averagingScoreDistribution
            multiplicationScoreDistribution = calculateMultiplicationClassificationResults(partitionsIterationEvaulatioInfos, labeledSamplesDataset.getNumOfClasses(), labeledSamplesDataset.getClassRatios(False))
            multiplicationScoreDistributionsPerIteration[i] = multiplicationScoreDistribution
            
        #endregion
        
        #region Group 1 - Analyzes the current score distribution of the each partition, the unified set (single classifier) and the averaging/multiplication combinations
        #region Get the score distributions of the partitions
        currentScoreDistributionStatistics = {}
        for partitionIndex in evaluationResultsPerSetAndInteration.keys() :
            generalStatisticsAttributes = self.calculateGeneralScoreDistributionStatistics(unlabeledSamplesDataset, labeledSamplesDataset,
                    evaluationResultsPerSetAndInteration.get(partitionIndex).getLatestEvaluationInfo().getScoreDistributions(),targetClassIndex, "partition_" + partitionIndex,properties)
            for  pos in generalStatisticsAttributes.keys() :
                currentScoreDistributionStatistics[len(currentScoreDistributionStatistics.labels)] = generalStatisticsAttributes.get(pos)
            
        #endregion
        
        #region Next we evaluate the current score distribution of the "mixture models" - the unified model (on all features), averaging and multiplication
        #start with the unified set
        unifiedSetGeneralStatisticsAttributes = self.calculateGeneralScoreDistributionStatistics(unlabeledSamplesDataset, labeledSamplesDataset,
                unifiedDatasetEvaulationResults.getLatestEvaluationInfo().getScoreDistributions(),targetClassIndex, "unified", properties)
        for pos in unifiedSetGeneralStatisticsAttributes.keys():
            currentScoreDistributionStatistics[len(currentScoreDistributionStatistics.labels)] = unifiedSetGeneralStatisticsAttributes.get(pos)
          

       #the averaging meta-features
        generalAveragingStatisticsAttributes = self.calculateGeneralScoreDistributionStatistics(unlabeledSamplesDataset, labeledSamplesDataset,
                averageingScoreDistributionsPerIteration.get(averageingScoreDistributionsPerIteration.lastKey()),targetClassIndex, "averaging", properties)
        for  pos in generalAveragingStatisticsAttributes.keys():
            currentScoreDistributionStatistics[len(currentScoreDistributionStatistics.labels)] = generalAveragingStatisticsAttributes.get(pos)
        

        #the multiplication meta-features
        generalMultiplicationStatisticsAttributes = self.calculateGeneralScoreDistributionStatistics(unlabeledSamplesDataset, labeledSamplesDataset,
            multiplicationScoreDistributionsPerIteration.get(multiplicationScoreDistributionsPerIteration.lastKey()),targetClassIndex, "multiplication", properties)
        for pos in generalMultiplicationStatisticsAttributes.keys():
            currentScoreDistributionStatistics[len(currentScoreDistributionStatistics.labels)] = generalMultiplicationStatisticsAttributes.get(pos)
        
        
        attributes.update(currentScoreDistributionStatistics)

        #endregion
        #endregion

        #region Group 2 - For each of the cases described in section 1, compare their score distribution to past iterations
        iterationsBasedStatisticsAttributes = {}

        #region first evaluate each partition separately
        for partitionIndex in evaluationResultsPerSetAndInteration.keys():
            paritionIterationBasedAttributes = self.calculateScoreDistributionStatisticsOverMultipleIterations(currentIterationIndex,
                            evaluationResultsPerSetAndInteration.get(partitionIndex).getAllIterationsScoreDistributions(),targetClassIndex, "partition_" + partitionIndex, properties)
            for pos in paritionIterationBasedAttributes.keys():
                iterationsBasedStatisticsAttributes[len(attributes) + len(iterationsBasedStatisticsAttributes)] = paritionIterationBasedAttributes.get(pos)
            
        #endregion

        #region Analyze the unified set and the averaging and multiplication rsults per iteration
        #next, get the per-iteration statistics of the unified model
        unifiedSetIterationBasedAttributes = self.calculateScoreDistributionStatisticsOverMultipleIterations(currentIterationIndex,
                        unifiedDatasetEvaulationResults.getAllIterationsScoreDistributions(),
                        targetClassIndex, "unified", properties)
        for pos in unifiedSetIterationBasedAttributes.keys():
            iterationsBasedStatisticsAttributes[len(attributes) + len(iterationsBasedStatisticsAttributes)] = unifiedSetIterationBasedAttributes.get(pos)
          

        #now the averaging and multiplication
        averagingIterationBasedAttributes = self.calculateScoreDistributionStatisticsOverMultipleIterations(1,
                        averageingScoreDistributionsPerIteration,targetClassIndex, "averaging", properties)
        for pos in averagingIterationBasedAttributes.keys():
            iterationsBasedStatisticsAttributes[len(attributes) + len(iterationsBasedStatisticsAttributes)] = averagingIterationBasedAttributes.get(pos)
          

        multiplicationIterationBasedAttributes = self.calculateScoreDistributionStatisticsOverMultipleIterations(1,
                        multiplicationScoreDistributionsPerIteration,targetClassIndex,"multiplication", properties)
        for pos in multiplicationIterationBasedAttributes.keys():
            iterationsBasedStatisticsAttributes[len(attributes) + len(iterationsBasedStatisticsAttributes)] = multiplicationIterationBasedAttributes.get(pos)
          
        attributes.update(iterationsBasedStatisticsAttributes) 
        #endregion
        #endregion

        #region Group 3 - Compare the score distributions for the various approaches described above (current iteration, past iterations)
        crossPartitionIterationsBasedStatisticsAttributes =  {}

        #region We now create a single data structure containing all the distributions and then
        # use a pair of loops to evaluate every pair once
        allPartitionsAndDistributions =  {}
        identifierPartitionsMap =  {}

        #insert all the "basic" paritions (this supports more then two partitions,
        # but may case a problem during the meta learning phase)
        for partitionIndex in evaluationResultsPerSetAndInteration.keys():
            allPartitionsAndDistributions[len(allPartitionsAndDistributions)
                                          ] = evaluationResultsPerSetAndInteration.get(partitionIndex).getAllIterationsScoreDistributions()
            identifierPartitionsMap[len(identifierPartitionsMap)] = "partition_" + partitionIndex 
          

        #now the unified set
        allPartitionsAndDistributions[len(allPartitionsAndDistributions)] = unifiedDatasetEvaulationResults.getAllIterationsScoreDistributions() 
        identifierPartitionsMap[len(identifierPartitionsMap)] = "unified"

        #Finally, the averaging and multiplication
        allPartitionsAndDistributions[len(allPartitionsAndDistributions)] = averageingScoreDistributionsPerIteration
        identifierPartitionsMap[len(identifierPartitionsMap)] = "averaging"
        allPartitionsAndDistributions[len(allPartitionsAndDistributions)] = multiplicationScoreDistributionsPerIteration
        identifierPartitionsMap[len(identifierPartitionsMap)] = "multiplication"
        #endregion

        #now we use a pair of loops to analyze every pair of partitions once
        for i in range(len(allPartitionsAndDistributions)-3):
            for j in range(len(allPartitionsAndDistributions)-2):
               if i != j:
                    crossPartitionFeatures = self.calculateScoreDistributionStatisticsOverMultipleSetsAndIterations(
                                    currentIterationIndex
                                    , allPartitionsAndDistributions.get(i),
                                    allPartitionsAndDistributions.get(j)
                                    , targetClassIndex,
                                    "_" + identifierPartitionsMap.get(i) + "_" + identifierPartitionsMap.get(j),
                                    properties) 
                    for key in crossPartitionFeatures.keys():
                        crossPartitionIterationsBasedStatisticsAttributes[
                            len(attributes) + len(crossPartitionIterationsBasedStatisticsAttributes)] = crossPartitionFeatures.get(key)


        attributes.update(crossPartitionIterationsBasedStatisticsAttributes) 
        #endregion


        return attributes 


 

    def calculateScoreDistributionStatisticsOverMultipleSetsAndIterations(
            currentIteration, iterationsEvaluationInfo1, iterationsEvaluationInfo2, targetClassIndex,
            identifier, properties):
        
        
        #The comparison is conducted as follows: we calculate difference statistics on
        # the first, top 5, top 10 etc. Comparisons are only performed for the same time index
        partitionBasedStatisticsAttributes = {}

        #get confidence scores per group
        currentIterationScoreDistGroup1 = iterationsEvaluationInfo1.get(currentIteration) 
        currentIterationScoreDistGroup2 = iterationsEvaluationInfo2.get(currentIteration) 

        #instance delta score on target class
        instanceScoreTargetClassGroup1 = []
        instanceScoreTargetClassGroup2 = [] 
        instanceDeltaScoreTargetClass = []
        deltaBetweenGroups = []
        instanceScoreTargetClass_cnt = 0 
        for ins in currentIterationScoreDistGroup1.keys():
            insScoreGroup1 = currentIterationScoreDistGroup1.get(ins)[targetClassIndex] 
            insScoreGroup2 = currentIterationScoreDistGroup2.get(ins)[targetClassIndex] 
            delta = insScoreGroup1 - insScoreGroup2 
            deltaBetweenGroups.append(delta) 

            instanceScoreTargetClassGroup1.append(insScoreGroup1) 
            instanceScoreTargetClassGroup2.append(insScoreGroup2) 
            instanceDeltaScoreTargetClass.append(delta) 
            instanceScoreTargetClass_cnt+=1
            
        #extract DescriptiveStatistics statistics from instanceDeltaScoreTargetClass
        #stats for distance cross partition
        #max
        maxDeltaScoreDist = attributeInfo
        ("maxDeltaScoreDist_"+identifier+"_iteration_"+currentIteration, "Numeric"
         , max(deltaBetweenGroups), -1) 
        partitionBasedStatisticsAttributes[len(partitionBasedStatisticsAttributes)] = maxDeltaScoreDist 
        #min
        minDeltaScoreDist = attributeInfo
        ("minDeltaScoreDist_"+identifier+"_iteration_"+currentIteration, "Numeric"
         , min(deltaBetweenGroups), -1) 
        partitionBasedStatisticsAttributes[len(partitionBasedStatisticsAttributes)] = minDeltaScoreDist
        #mean
        meanDeltaScoreDist = attributeInfo
        ("meanDeltaScoreDist_"+identifier+"_iteration_"+currentIteration, "Numeric"
         , np.mean(deltaBetweenGroups), -1) 
        partitionBasedStatisticsAttributes[len(partitionBasedStatisticsAttributes)] = meanDeltaScoreDist 
        #std
        stdDeltaScoreDist = attributeInfo
        ("stdDeltaScoreDist_"+identifier+"_iteration_"+currentIteration, "Numeric"
         , statistics.stdev(deltaBetweenGroups), -1) 
        partitionBasedStatisticsAttributes[len(partitionBasedStatisticsAttributes)] = stdDeltaScoreDist
        #p-50
        medianDeltaScoreDist = attributeInfo
        ("medianDeltaScoreDist_"+identifier+"_iteration_"+currentIteration, "Numeric"
         , np.percentile(deltaBetweenGroups,50), -1) 
        partitionBasedStatisticsAttributes[len(partitionBasedStatisticsAttributes)] = medianDeltaScoreDist

        #t-test on scores per group
        TTestStatistic = ttest_ind(instanceScoreTargetClassGroup1,instanceScoreTargetClassGroup2).statistic
        tTest_att = attributeInfo("t_test_"+identifier+"_iteration_"+currentIteration
         , "Numeric", TTestStatistic, -1) 
        partitionBasedStatisticsAttributes[len(partitionBasedStatisticsAttributes)] = tTest_att


        return partitionBasedStatisticsAttributes 
    



    def calculateScoreDistributionStatisticsOverMultipleIterations(self, currentIteration,
             iterationsEvaluationInfo, targetClassIndex, identifier, properties) :

        
        #region Statistics on previous iterations

        #Statistics on the changes in confidence score for each instances compared to previous iterations
        confidenceScoreDifferencesPerSingleIteration = {} 

        #instanceID -> num of iterations backwards -> values
        confidenceScoreDifferencesPerInstance = {} 

        #A histogram of the differences between the current scores histogram and one of the previous iterations' (i.e. the changes in the percentages assigned to each "box") */
        generalPercentageScoresDiffHistogramByIteration = {} 

        #The paired T-Test values for the current iteration and one of the previous iterations
        tTestValueForCurrentAndPreviousIterations = {}

        #Statistics on the Paired T-Test values of the previous X iterations (not including the current)
        previousIterationsTTestStatistics = {}

        #Statistics on the percentage of instances that changed labels given a confidence threshold and number of iterations back
        labelChangePercentageByIterationAndThreshold = {}
        #endregion


        iterationsBasedStatisticsAttributes = {} 

        
        properties = Properties()
        properties.load(open('config.properties'))
        properties.list()
   


        #We operate under the assumption that the generalScoreStats object, which contains the current iteration's confidence scores
        # has already been populated */

        # We begin by loading the deltas of the confidence scores of consecutive iterations into the object.*/
        for i in range(len(iterationsEvaluationInfo)):
            descriptiveStatistics = []
            confidenceScoreDifferencesPerSingleIteration[i] = descriptiveStatistics 
            currentIterationScoreDistribution =  iterationsEvaluationInfo.get(i) 
            previousIterationScoreDistribution = iterationsEvaluationInfo.get(i-1) 

            #Extract the per-iteration information
            for j in currentIterationScoreDistribution.keys():
                delta = currentIterationScoreDistribution.get(j)[targetClassIndex]-previousIterationScoreDistribution.get(j)[targetClassIndex] 
                confidenceScoreDifferencesPerSingleIteration.get(i).append(delta) 
              

            #extract the per-instance information
            for numOfIterationsBack in self.numOfIterationsBackToAnalyze:
                # If the iteration is within the "scope" of the analysis (i.e. the number of iterations back falls within one or more of the ranges */
                if (currentIteration - (i-1)) <= numOfIterationsBack:
                    for j in currentIterationScoreDistribution.keys():
                        if not j in confidenceScoreDifferencesPerInstance.keys():
                            hashmap = {}
                            confidenceScoreDifferencesPerInstance[j] = hashmap 
                          
                        if not numOfIterationsBack in confidenceScoreDifferencesPerInstance.get(j).kesy():
                            descriptiveStatistics = []
                            confidenceScoreDifferencesPerInstance.get(j)[numOfIterationsBack] = descriptiveStatistics 
                          
                        delta = currentIterationScoreDistribution.get(j)[targetClassIndex]-previousIterationScoreDistribution.get(j)[targetClassIndex] 
                        confidenceScoreDifferencesPerInstance.get(j).get(numOfIterationsBack).append(delta) 
                      
                  
              
          

        #now we produce the statistics for a varying number of backwards iterations
        for numOfIterationsBack in self.numOfIterationsBackToAnalyze:
            if currentIteration >= numOfIterationsBack:

                #region start by generating statistics on the ITERATION-LEVEL statistics
                iteraionTempDSMax = []
                iterationTempDSMin = []
                iterationTempDSAvg = [] 
                iterationTempDSStdev = []
                iterationTempDSMedian = [] 
                
                i = currentIteration
                while i > (currentIteration - numOfIterationsBack):
                    try:
                        iteraionTempDSMax.append(max(confidenceScoreDifferencesPerSingleIteration.get(i))) 
                        iterationTempDSMin.append(min(confidenceScoreDifferencesPerSingleIteration.get(i))) 
                        iterationTempDSAvg.append(np.mean(confidenceScoreDifferencesPerSingleIteration.get(i))) 
                        iterationTempDSStdev.append(statistics.stdev(confidenceScoreDifferencesPerSingleIteration.get(i))) 
                        iterationTempDSMedian.append(np.percentile(confidenceScoreDifferencesPerSingleIteration.get(i), 50))           
                    except:
                        continue 
                      
                    i-=1
                  

                # Now we extract the AVG and Stdev of these temp statistics
                maxAvgAtt = attributeInfo(numOfIterationsBack + "iterationsAverageOfMaxDelta" + "_" + identifier, "Numeric", np.mean(iteraionTempDSMax), -1) 
                maxStdevAtt = attributeInfo(numOfIterationsBack + "iterationsStdevOfMaxDelta" + "_" + identifier, "Numeric", statistics.stdev(iteraionTempDSMax), -1) 
                iterationsBasedStatisticsAttributes[len(iterationsBasedStatisticsAttributes)] = maxAvgAtt
                iterationsBasedStatisticsAttributes[len(iterationsBasedStatisticsAttributes)] = maxStdevAtt 

                minAvgAtt = attributeInfo(numOfIterationsBack + "iterationsAverageOfMinDelta" + "_" + identifier, "Numeric", np.mean(iterationTempDSMin), -1) 
                minStdevAtt = attributeInfo(numOfIterationsBack + "iterationsStdevOfMinDelta" + "_" + identifier, "Numeric", statistics.stdev(iterationTempDSMin), -1) 
                iterationsBasedStatisticsAttributes[len(iterationsBasedStatisticsAttributes)] = minAvgAtt
                iterationsBasedStatisticsAttributes[len(iterationsBasedStatisticsAttributes)] = minStdevAtt

                avgAvgAtt = attributeInfo(numOfIterationsBack + "iterationsAverageOfAvgDelta" + "_" + identifier, "Numeric", np.mean(iterationTempDSAvg), -1) 
                avgStdevAtt = attributeInfo(numOfIterationsBack + "iterationsStdevOfAvgDelta" + "_" + identifier, "Numeric", statistics.stdev(iterationTempDSAvg), -1) 
                iterationsBasedStatisticsAttributes[len(iterationsBasedStatisticsAttributes)] = avgAvgAtt 
                iterationsBasedStatisticsAttributes[len(iterationsBasedStatisticsAttributes)] = avgStdevAtt

                stdevAvgAtt = attributeInfo(numOfIterationsBack + "iterationsAverageOfStdevDelta" + "_" + identifier, "Numeric", np.mean(iterationTempDSStdev), -1) 
                stdevStdevAtt = attributeInfo(numOfIterationsBack + "iterationsStdevOfStdevDelta" + "_" + identifier, "Numeric", statistics.stdev(iterationTempDSStdev), -1) 
                iterationsBasedStatisticsAttributes[len(iterationsBasedStatisticsAttributes)] = stdevAvgAtt
                iterationsBasedStatisticsAttributes[len(iterationsBasedStatisticsAttributes)] = stdevStdevAtt

                medianAvgAtt = attributeInfo(numOfIterationsBack + "iterationsAverageOfMedianDelta" + "_" + identifier, "Numeric", np.mean(iterationTempDSMedian), -1) 
                medianStdevAtt = attributeInfo(numOfIterationsBack + "iterationsStdevOfMedianDelta" + "_" + identifier, "Numeric", statistics.stdev(iterationTempDSMedian), -1) 
                iterationsBasedStatisticsAttributes[len(iterationsBasedStatisticsAttributes)] = medianAvgAtt
                iterationsBasedStatisticsAttributes[len(iterationsBasedStatisticsAttributes)] = medianStdevAtt
                #endregion

                #region now we calculate the INSTANCE-LEVEL statistics
                instanceTempDSMax = [] 
                instanceTempDSMin = [] 
                instanceTempDSAvg = []
                instanceTempDSStdev = [] 
                instanceTempDSMedian = [] 

                for instanceID in confidenceScoreDifferencesPerInstance.keys():
                    try:
                        instanceTempDSMax.append(confidenceScoreDifferencesPerInstance.get(instanceID).get(max(numOfIterationsBack))) 
                        instanceTempDSMin.append(confidenceScoreDifferencesPerInstance.get(instanceID).get(min(numOfIterationsBack))) 
                        instanceTempDSAvg.append(confidenceScoreDifferencesPerInstance.get(instanceID).get(np.mean(numOfIterationsBack))) 
                        instanceTempDSStdev.append(confidenceScoreDifferencesPerInstance.get(instanceID).get(statistics.stdev(numOfIterationsBack))) 
                        instanceTempDSMedian.append(np.percentile(confidenceScoreDifferencesPerInstance.get(instanceID).get(numOfIterationsBack), 50)) 
                    except:
                        continue 
                      

                maxAvgPerInstanceAtt = attributeInfo(numOfIterationsBack + "instancesAverageOfMaxDelta" + "_" + identifier, "Numeric", np.mean(instanceTempDSMax), -1) 
                maxStdevPerInstanceAtt = attributeInfo(numOfIterationsBack + "instancesStdevOfMaxDelta" + "_" + identifier, "Numeric", statistics.stdev(instanceTempDSMax), -1) 
                iterationsBasedStatisticsAttributes[len(iterationsBasedStatisticsAttributes)] = maxAvgPerInstanceAtt
                iterationsBasedStatisticsAttributes[len(iterationsBasedStatisticsAttributes)] = maxStdevPerInstanceAtt

                minAvgPerInstanceAtt = attributeInfo(numOfIterationsBack + "instancesAverageOfMinDelta" + "_" + identifier, "Numeric", np.mean(instanceTempDSMin), -1) 
                minStdevPerInstanceAtt = attributeInfo(numOfIterationsBack + "instancesStdevOfMinDelta" + "_" + identifier, "Numeric", statistics.stdev(instanceTempDSMin), -1) 
                iterationsBasedStatisticsAttributes[len(iterationsBasedStatisticsAttributes)] = minAvgPerInstanceAtt
                iterationsBasedStatisticsAttributes[len(iterationsBasedStatisticsAttributes)] = minStdevPerInstanceAtt

                avgAvgPerInstanceAtt = attributeInfo(numOfIterationsBack + "instancesAverageOfAvgDelta" + "_" + identifier, "Numeric", np.mean(instanceTempDSAvg), -1) 
                avgStdevPerInstanceAtt = attributeInfo(numOfIterationsBack + "instancesStdevOfAvgDelta" + "_" + identifier, "Numeric", statistics.stdev(instanceTempDSAvg), -1) 
                iterationsBasedStatisticsAttributes[len(iterationsBasedStatisticsAttributes)] = avgAvgPerInstanceAtt
                iterationsBasedStatisticsAttributes[len(iterationsBasedStatisticsAttributes)] = avgStdevPerInstanceAtt

                stdevAvgPerInstanceAtt = attributeInfo(numOfIterationsBack + "instancesAverageOfStdevDelta" + "_" + identifier, "Numeric", np.mean(instanceTempDSStdev), -1) 
                stdevStdevPerInstanceAtt = attributeInfo(numOfIterationsBack + "instancesStdevOfStdevDelta" + "_" + identifier, "Numeric", statistics.stdev(instanceTempDSStdev), -1) 
                iterationsBasedStatisticsAttributes[len(iterationsBasedStatisticsAttributes)] = stdevAvgPerInstanceAtt
                iterationsBasedStatisticsAttributes[len(iterationsBasedStatisticsAttributes)] = stdevStdevPerInstanceAtt

                medianAvgPerInstanceAtt = attributeInfo(numOfIterationsBack + "instancesAverageOfMedianDelta" + "_" + identifier, "Numeric", np.mean(instanceTempDSMedian), -1) 
                medianStdevPerInstanceAtt = attributeInfo(numOfIterationsBack + "instancesStdevOfMedianDelta" + "_" + identifier, "Numeric", statistics.stdev(instanceTempDSMedian), -1) 
                iterationsBasedStatisticsAttributes[len(iterationsBasedStatisticsAttributes)] = medianAvgPerInstanceAtt
                iterationsBasedStatisticsAttributes[len(iterationsBasedStatisticsAttributes)] = medianStdevPerInstanceAtt
                #endregion
              
            else:
                #if the conditions are not met, just add -1 to all relevant attributes
                #TODO: consider adding a question mark instead
                #region Add -1 instead of the values

                #region Iteration-level values
                maxAvgAtt = attributeInfo(numOfIterationsBack + "iterationsAverageOfMaxDelta" + "_" + identifier, "Numeric", -1.0, -1) 
                maxStdevAtt = attributeInfo(numOfIterationsBack + "iterationsStdevOfMaxDelta" + "_" + identifier, "Numeric", -1.0, -1) 
                iterationsBasedStatisticsAttributes[len(iterationsBasedStatisticsAttributes)] = maxAvgAtt
                iterationsBasedStatisticsAttributes[len(iterationsBasedStatisticsAttributes)] = maxStdevAtt

                minAvgAtt = attributeInfo(numOfIterationsBack + "iterationsAverageOfMinDelta" + "_" + identifier, "Numeric", -1.0, -1) 
                minStdevAtt = attributeInfo(numOfIterationsBack + "iterationsStdevOfMinDelta" + "_" + identifier, "Numeric", -1.0, -1) 
                iterationsBasedStatisticsAttributes[len(iterationsBasedStatisticsAttributes)] = minAvgAtt
                iterationsBasedStatisticsAttributes[len(iterationsBasedStatisticsAttributes)] = minStdevAtt

                avgAvgAtt = attributeInfo(numOfIterationsBack + "iterationsAverageOfAvgDelta" + "_" + identifier, "Numeric", -1.0, -1) 
                avgStdevAtt = attributeInfo(numOfIterationsBack + "iterationsStdevOfAvgDelta" + "_" + identifier, "Numeric", -1.0, -1) 
                iterationsBasedStatisticsAttributes[len(iterationsBasedStatisticsAttributes)] = avgAvgAtt 
                iterationsBasedStatisticsAttributes[len(iterationsBasedStatisticsAttributes)] = avgStdevAtt

                stdevAvgAtt = attributeInfo(numOfIterationsBack + "iterationsAverageOfStdevDelta" + "_" + identifier, "Numeric", -1.0, -1) 
                stdevStdevAtt = attributeInfo(numOfIterationsBack + "iterationsStdevOfStdevDelta" + "_" + identifier, "Numeric", -1.0, -1) 
                iterationsBasedStatisticsAttributes[len(iterationsBasedStatisticsAttributes)] = stdevAvgAtt 
                iterationsBasedStatisticsAttributes[len(iterationsBasedStatisticsAttributes)] = stdevStdevAtt 

                medianAvgAtt = attributeInfo(numOfIterationsBack + "iterationsAverageOfMedianDelta" + "_" + identifier, "Numeric", -1.0, -1) 
                medianStdevAtt = attributeInfo(numOfIterationsBack + "iterationsStdevOfMedianDelta" + "_" + identifier, "Numeric", -1.0, -1) 
                iterationsBasedStatisticsAttributes[len(iterationsBasedStatisticsAttributes)] = medianAvgAtt
                iterationsBasedStatisticsAttributes[len(iterationsBasedStatisticsAttributes)] = medianStdevAtt
                #endregion

                #region Instance-level values
                maxAvgPerInstanceAtt = attributeInfo(numOfIterationsBack + "instancesAverageOfMaxDelta" + "_" + identifier, "Numeric", -1.0, -1) 
                maxStdevPerInstanceAtt = attributeInfo(numOfIterationsBack + "instancesStdevOfMaxDelta" + "_" + identifier, "Numeric", -1.0, -1) 
                iterationsBasedStatisticsAttributes[len(iterationsBasedStatisticsAttributes)] = maxAvgPerInstanceAtt 
                iterationsBasedStatisticsAttributes[len(iterationsBasedStatisticsAttributes)] = maxStdevPerInstanceAtt 

                minAvgPerInstanceAtt = attributeInfo(numOfIterationsBack + "instancesAverageOfMinDelta" + "_" + identifier, "Numeric", -1.0, -1) 
                minStdevPerInstanceAtt = attributeInfo(numOfIterationsBack + "instancesStdevOfMinDelta" + "_" + identifier, "Numeric", -1.0, -1) 
                iterationsBasedStatisticsAttributes[len(iterationsBasedStatisticsAttributes)] = minAvgPerInstanceAtt
                iterationsBasedStatisticsAttributes[len(iterationsBasedStatisticsAttributes)] = minStdevPerInstanceAtt

                avgAvgPerInstanceAtt = attributeInfo(numOfIterationsBack + "instancesAverageOfAvgDelta" + "_" + identifier, "Numeric", -1.0, -1) 
                avgStdevPerInstanceAtt = attributeInfo(numOfIterationsBack + "instancesStdevOfAvgDelta" + "_" + identifier, "Numeric", -1.0, -1) 
                iterationsBasedStatisticsAttributes[len(iterationsBasedStatisticsAttributes)] = avgAvgPerInstanceAtt
                iterationsBasedStatisticsAttributes[len(iterationsBasedStatisticsAttributes)] = avgStdevPerInstanceAtt

                stdevAvgPerInstanceAtt = attributeInfo(numOfIterationsBack + "instancesAverageOfStdevDelta" + "_" + identifier, "Numeric", -1.0, -1) 
                stdevStdevPerInstanceAtt = attributeInfo(numOfIterationsBack + "instancesStdevOfStdevDelta" + "_" + identifier, "Numeric", -1.0, -1) 
                iterationsBasedStatisticsAttributes[len(iterationsBasedStatisticsAttributes)] = stdevAvgPerInstanceAtt
                iterationsBasedStatisticsAttributes[len(iterationsBasedStatisticsAttributes)] = stdevStdevPerInstanceAtt

                medianAvgPerInstanceAtt = attributeInfo(numOfIterationsBack + "instancesAverageOfMedianDelta" + "_" + identifier, "Numeric", -1.0, -1) 
                medianStdevPerInstanceAtt = attributeInfo(numOfIterationsBack + "instancesStdevOfMedianDelta" + "_" + identifier, "Numeric", -1.0, -1) 
                iterationsBasedStatisticsAttributes[len(iterationsBasedStatisticsAttributes)] = medianAvgPerInstanceAtt
                iterationsBasedStatisticsAttributes[len(iterationsBasedStatisticsAttributes)] = medianStdevPerInstanceAtt
                #endregion

                #endregion
              
          


        #now we generate the histograms at different time points and compare
        for numOfIterationsBack in self.numOfIterationsBackToAnalyze:
            if currentIteration >= numOfIterationsBack:
                #region Generate the attbributes representing the histogram changes
                hashmap = {}
                generalPercentageScoresDiffHistogramByIteration[numOfIterationsBack] = hashmap

                #First, calculate the percentage for each histogram, like we did in the "general statistics" section
                i = 0.0
                while i<1.0:
                    generalPercentageScoresDiffHistogramByIteration.get(numOfIterationsBack)[round(i/self.histogramItervalSize)] = 0.0
                    i+=self.histogramItervalSize

                earlierIterationScoreDistribution = iterationsEvaluationInfo.get(currentIteration-numOfIterationsBack) 
                for  i in earlierIterationScoreDistribution.keys():
                    histogramIndex = round(earlierIterationScoreDistribution.get(i)[targetClassIndex]/self.histogramItervalSize) 
                    generalPercentageScoresDiffHistogramByIteration.get(numOfIterationsBack)[histogramIndex] = generalPercentageScoresDiffHistogramByIteration.get(numOfIterationsBack).get(histogramIndex)+1.0 
                  
                for key in generalPercentageScoresDiffHistogramByIteration.get(numOfIterationsBack).keys():
                    histoCellPercentage = generalPercentageScoresDiffHistogramByIteration.get(numOfIterationsBack).get(key)/len(earlierIterationScoreDistribution.keys()) 
                    generalPercentageScoresDiffHistogramByIteration.get(numOfIterationsBack)[key] = histoCellPercentage
                  

                #now, generate the attributes representing the changes in the histogram
                histogramStatistics = []
                for key in generalPercentageScoresDiffHistogramByIteration.get(numOfIterationsBack).keys():
                    delta = self.generalPartitionPercentageByScoreHistogram.get(identifier).get(key) - generalPercentageScoresDiffHistogramByIteration.get(numOfIterationsBack).get(key) 
                    histogramStatistics.append(delta) 
                  
                maxDeltaHistoAtt = attributeInfo("maxDeltaScoreDistHisto_" + numOfIterationsBack + "_IterationsBack" + "_" + identifier, "Numeric", max(histogramStatistics), -1) 
                iterationsBasedStatisticsAttributes[len(iterationsBasedStatisticsAttributes)] = maxDeltaHistoAtt

                minDeltaHistoAtt = attributeInfo("maxDeltaScoreDistHisto_" + numOfIterationsBack + "_IterationsBack" + "_" + identifier, "Numeric", min(histogramStatistics), -1) 
                iterationsBasedStatisticsAttributes[len(iterationsBasedStatisticsAttributes)] = minDeltaHistoAtt

                avgDeltaHistoAtt = attributeInfo("maxDeltaScoreDistHisto_" + numOfIterationsBack + "_IterationsBack" + "_" + identifier, "Numeric", np.mean(histogramStatistics), -1) 
                iterationsBasedStatisticsAttributes[len(iterationsBasedStatisticsAttributes)] = avgDeltaHistoAtt

                stdevDeltaHistoAtt = attributeInfo("maxDeltaScoreDistHisto_" + numOfIterationsBack + "_IterationsBack" + "_" + identifier, "Numeric", statistics.stdev(histogramStatistics), -1) 
                iterationsBasedStatisticsAttributes[len(iterationsBasedStatisticsAttributes)] = stdevDeltaHistoAtt

                medianDeltaHistoAtt = attributeInfo("maxDeltaScoreDistHisto_" + numOfIterationsBack + "_IterationsBack" + "_" + identifier, "Numeric", np.percentile(histogramStatistics, 50), -1) 
                iterationsBasedStatisticsAttributes[len(iterationsBasedStatisticsAttributes)] = medianDeltaHistoAtt
                #endregion

              
            else :
                #region If there are not enough iterations yet, place -1 everywhere
                maxDeltaHistoAtt = attributeInfo("maxDeltaScoreDistHisto_" + numOfIterationsBack + "IterationsBack" + "_" + identifier, "Numeric", -1.0, -1) 
                iterationsBasedStatisticsAttributes[len(iterationsBasedStatisticsAttributes)] = maxDeltaHistoAtt

                minDeltaHistoAtt = attributeInfo("maxDeltaScoreDistHisto_" + numOfIterationsBack + "IterationsBack" + "_" + identifier, "Numeric", -1.0, -1) 
                iterationsBasedStatisticsAttributes[len(iterationsBasedStatisticsAttributes)] = minDeltaHistoAtt

                avgDeltaHistoAtt = attributeInfo("maxDeltaScoreDistHisto_" + numOfIterationsBack + "IterationsBack" + "_" + identifier, "Numeric", -1.0, -1) 
                iterationsBasedStatisticsAttributes[len(iterationsBasedStatisticsAttributes)] = avgDeltaHistoAtt

                stdevDeltaHistoAtt = attributeInfo("maxDeltaScoreDistHisto_" + numOfIterationsBack + "IterationsBack" + "_" + identifier, "Numeric", -1.0, -1) 
                iterationsBasedStatisticsAttributes[len(iterationsBasedStatisticsAttributes)] = stdevDeltaHistoAtt

                medianDeltaHistoAtt = attributeInfo("maxDeltaScoreDistHisto_" + numOfIterationsBack + "IterationsBack" + "_" + identifier, "Numeric", -1.0, -1) 
                iterationsBasedStatisticsAttributes[len(iterationsBasedStatisticsAttributes)] = medianDeltaHistoAtt 
                #endregion
              
          


        #region Calculate the Paired T-Test statistics of the current iteration with previous iterations
        #TO DO: understand what is this structure: iterationsEvaluationInfo -- ASK Gilad
       
        scoreDistributions = iterationsEvaluationInfo.get(currentIteration) 
        currentIterationTargetClassScoreDistributions = []

        #The use of TreeMap rather than HashMap is supposed to ensure that the indices are always read in the same (ascending) order
        counter1 = 0 
        for i in scoreDistributions.keys():
            currentIterationTargetClassScoreDistributions.append(scoreDistributions.get(i)[targetClassIndex] )
            counter1+=1
          

        #now for each iteration we extract its values and calculate the Paired T-Test statistic
        for i in range(len(iterationsEvaluationInfo)):
            tempScoreDistributions = iterationsEvaluationInfo.get(i) 
            tempIterationTargetClassScoreDistributions = [] #CRITICAL: because of the differences in the sizes of the unlabeled sets, we always use the one of the current iteration (which is smallest and shared by all previous iterations)
            #Note: here I use the KeySet of scoreDistributions which belongs to the CURRENT iteration. This should ensure that the values are paired
            counter2 = 0 
            for j in scoreDistributions.keys():
                tempIterationTargetClassScoreDistributions.append(tempScoreDistributions.get(j)[targetClassIndex]) 
                counter2+=1
              
            TTestStatistic = ttest_ind(currentIterationTargetClassScoreDistributions,tempIterationTargetClassScoreDistributions).statistic
            tTestValueForCurrentAndPreviousIterations[i] = TTestStatistic 
          

        for numOfIterationsBack in self.numOfIterationsBackToAnalyze:
            if currentIteration >= numOfIterationsBack:
                descriptiveStatistics = []
                previousIterationsTTestStatistics[numOfIterationsBack] = descriptiveStatistics 
                i=currentIteration-numOfIterationsBack
                while i<currentIteration:
                    previousIterationsTTestStatistics.get(numOfIterationsBack).append(tTestValueForCurrentAndPreviousIterations.get(i)) 
                    i+=1

                #now that we obtained the statistics of all the relevant iterations, we can generate the attributes
                maxTTestStatisticForScoreDistributionAtt = attributeInfo("maxTTestStatisticForScoreDistribution_" + numOfIterationsBack + "IterationsBack"+ "_" + identifier, "Numeric", max(previousIterationsTTestStatistics.get(numOfIterationsBack)), -1) 
                iterationsBasedStatisticsAttributes[len(iterationsBasedStatisticsAttributes)] = maxTTestStatisticForScoreDistributionAtt

                minTTestStatisticForScoreDistributionAtt = attributeInfo("minTTestStatisticForScoreDistribution_" + numOfIterationsBack + "IterationsBack"+ "_" + identifier, "Numeric", min(previousIterationsTTestStatistics.get(numOfIterationsBack)), -1) 
                iterationsBasedStatisticsAttributes[len(iterationsBasedStatisticsAttributes)] = minTTestStatisticForScoreDistributionAtt

                avgTTestStatisticForScoreDistributionAtt = attributeInfo("maxTTestStatisticForScoreDistribution_" + numOfIterationsBack + "IterationsBack"+ "_" + identifier, "Numeric", np.mean(previousIterationsTTestStatistics.get(numOfIterationsBack)), -1) 
                iterationsBasedStatisticsAttributes[len(iterationsBasedStatisticsAttributes)] = avgTTestStatisticForScoreDistributionAtt

                stdevTTestStatisticForScoreDistributionAtt = attributeInfo("stdevTTestStatisticForScoreDistribution_" + numOfIterationsBack + "IterationsBack"+ "_" + identifier, "Numeric", statistics.stdev(previousIterationsTTestStatistics.get(numOfIterationsBack)), -1) 
                iterationsBasedStatisticsAttributes[len(iterationsBasedStatisticsAttributes)] = stdevTTestStatisticForScoreDistributionAtt

                medianTTestStatisticForScoreDistributionAtt = attributeInfo("medianTTestStatisticForScoreDistribution_" + numOfIterationsBack + "IterationsBack"+ "_" + identifier, "Numeric", np.percentile(previousIterationsTTestStatistics.get(numOfIterationsBack), 50), -1) 
                iterationsBasedStatisticsAttributes[len(iterationsBasedStatisticsAttributes)] = medianTTestStatisticForScoreDistributionAtt 

              
            else :
                #region Fill with -1 values if we don't have the required iterations
                maxTTestStatisticForScoreDistributionAtt = attributeInfo("maxTTestStatisticForScoreDistribution_" + numOfIterationsBack + "IterationsBack"+ "_" + identifier, "Numeric", -1.0, -1) 
                iterationsBasedStatisticsAttributes[len(iterationsBasedStatisticsAttributes)] = maxTTestStatisticForScoreDistributionAtt

                minTTestStatisticForScoreDistributionAtt = attributeInfo("minTTestStatisticForScoreDistribution_" + numOfIterationsBack + "IterationsBack"+ "_" + identifier, "Numeric", -1.0, -1) 
                iterationsBasedStatisticsAttributes[len(iterationsBasedStatisticsAttributes)] = maxTTestStatisticForScoreDistributionAtt

                avgTTestStatisticForScoreDistributionAtt = attributeInfo("maxTTestStatisticForScoreDistribution_" + numOfIterationsBack + "IterationsBack"+ "_" + identifier, "Numeric", -1.0, -1) 
                iterationsBasedStatisticsAttributes[len(iterationsBasedStatisticsAttributes)] = maxTTestStatisticForScoreDistributionAtt

                stdevTTestStatisticForScoreDistributionAtt = attributeInfo("stdevTTestStatisticForScoreDistribution_" + numOfIterationsBack + "IterationsBack"+ "_" + identifier, "Numeric", -1.0, -1) 
                iterationsBasedStatisticsAttributes[len(iterationsBasedStatisticsAttributes)] = stdevTTestStatisticForScoreDistributionAtt

                medianTTestStatisticForScoreDistributionAtt = attributeInfo("medianTTestStatisticForScoreDistribution_" + numOfIterationsBack + "IterationsBack"+ "_" + identifier, "Numeric", -1.0, -1) 
                iterationsBasedStatisticsAttributes[len(iterationsBasedStatisticsAttributes)] = medianTTestStatisticForScoreDistributionAtt
                #endregion
              

          
        #endregion


        #region Now we extract the percentage of instances that switched labels from a previous iteration using various thresholds

         # We already have the value of the current iteration from the previous section, in the currentIterationTargetClassScoreDistributions parameter.
        #*  This means that we only need to extract it for the previous iterations. */
        for numOfIterationsBack in self.numOfIterationsBackToAnalyze:
            if currentIteration >= numOfIterationsBack:
                tempScoreDistributions = iterationsEvaluationInfo.get(currentIteration-numOfIterationsBack) 
                tempIterationTargetClassScoreDistributions = []

                #NOTE: here we once again use the indices of the CURRENT iteration to keep everything correlated
                counter2 = 0 
                for i in scoreDistributions.keys():
                    tempIterationTargetClassScoreDistributions.append(tempScoreDistributions.get(i)[targetClassIndex]) 
                    counter2+=1 

                  
                hashmap = {}
                labelChangePercentageByIterationAndThreshold[numOfIterationsBack] = hashmap 
                for threshold in self.confidenceScoreThresholds:
                    counter = 0 
                    for i in range(len(currentIterationTargetClassScoreDistributions)):
                        if ((currentIterationTargetClassScoreDistributions[i] < threshold and
                                tempIterationTargetClassScoreDistributions[i] >= threshold) or
                                (currentIterationTargetClassScoreDistributions[i] >= threshold and
                                        tempIterationTargetClassScoreDistributions[i] < threshold)) :
                            counter+=1 
                          
                      
                    labelChangePercentageByIterationAndThreshold.get(numOfIterationsBack)[threshold] = counter/len(tempIterationTargetClassScoreDistributions) 
                  

                #now that we have extracted the percentages, time to generate the attributes
                for threshold in self.confidenceScoreThresholds:
                    labelPercetageChangeAtt = attributeInfo("labelPercentageChangeFor_" + numOfIterationsBack + "IterationsBack" + "_threshold_" + threshold+ "_" + identifier, "Numeric", labelChangePercentageByIterationAndThreshold.get(numOfIterationsBack).get(threshold), -1) 
                    iterationsBasedStatisticsAttributes[len(iterationsBasedStatisticsAttributes)] = labelPercetageChangeAtt
                  
              
            else :
                #generate attributes with -1 values if we don't have sufficient iterations
                for threshold in self.confidenceScoreThresholds:
                    labelPercetageChangeAtt = attributeInfo("labelPercentageChangeFor_" + numOfIterationsBack + "IterationsBack" + "_threshold_" + threshold+ "_" + identifier, "Numeric", -1.0, -1) 
                    iterationsBasedStatisticsAttributes[len(iterationsBasedStatisticsAttributes)] = labelPercetageChangeAtt
                  
        #endregion

        return iterationsBasedStatisticsAttributes 
      


     #*
     #* Used to generate Group 1 features.
     #* @param unlabeledSamplesDataset
     #* @param labeledSamplesDataset
     #* @param scoreDistributions
     #* @param targetClassIndex
     #* @param properties
     #* @return
     #* @throws Exception
     #*/
    def calculateGeneralScoreDistributionStatistics(self, unlabeledSamplesDataset, labeledSamplesDataset,scoreDistributions,
                                                     targetClassIndex, identifier, properties):
        
         #region Statistics on the current score distribution
        #Random rnd = Random(int(properties.getProperty("randomSeed"))) 

        #Simple general statistics on the overall scores, no additional filtering
        generalScoreStats = []

        #A histogram that contains the percentage of the overall items in each "box". The key is the lowest value of the box
        generalPercentageByScoreHistogram = {}

        #Statistics on whether the score distribution is similar to known distirbutions
        normalDistributionGoodnessOfFitPVAlues = {}
        logNormalDistributionGoodnessOfFitPVAlue = {} 
        uniformDistributionGoodnessOfFitPVAlue = {}

         # For a given confidence score threshold, we partition the instances. Then for each of the Data's attributes, we check correlation between the two groups. We then calculate statistics.
        #Correlation needs to be calculated separately for numeric and discrete attributes. This is done both jointly and separately for numeric and discrete features*/
        allFeatureCorrelationStatsByThreshold = {} 
        numericFeatureCorrelationStatsByThreshold = {}
        discreteFeatureCorrelationStatsByThreshold = {}

        #The "true" imbalance ratio, based on the labeled data available
        #double trainingSetImbalanceRatio 

        #the imbalance ratio according to the current labeling at various thresholds
        imbalanceRatioByConfidenceScoreRatio = {}

        #The ratio of the previous values set and the "real" imbalance ratio
        imbalanceScoreRatioToTrueRatio = {}

        #endregion

        generalStatisticsAttributes = {}

        for i in scoreDistributions.keys():
            generalScoreStats.append(scoreDistributions.get(i)[targetClassIndex]) 
          

        #region General stats
        att0 = attributeInfo("maxConfidenceScore" + "_" + identifier , "Numeric", max(generalScoreStats), -1) 
        att1 = attributeInfo("minConfidenceScore" + "_" + identifier, "Numeric", min(generalScoreStats), -1) 
        att2 = attributeInfo("avgConfidenceScore" + "_" + identifier, "Numeric", np.mean(generalScoreStats), -1) 
        att3 = attributeInfo("stdevConfidenceScore" + "_" + identifier, "Numeric",  statistics.stdev(generalScoreStats), -1) 
        att4 = attributeInfo("medianConfidenceScore" + "_" + identifier, "Numeric",  np.percentile(generalScoreStats, 50), -1) 
        generalStatisticsAttributes[len(generalStatisticsAttributes)] = att0
        generalStatisticsAttributes[len(generalStatisticsAttributes)] = att1
        generalStatisticsAttributes[len(generalStatisticsAttributes)] = att2
        generalStatisticsAttributes[len(generalStatisticsAttributes)] = att3 
        generalStatisticsAttributes[len(generalStatisticsAttributes)] = att4 
        #endregion

        #region Histogram of the percentage of score distributions
        i = 0.0
        while i<1:
            generalPercentageByScoreHistogram[round(i/self.histogramItervalSize)] = 0.0
            i+=self.histogramItervalSize

        for i in scoreDistributions.keys():
            histogramIndex = round(scoreDistributions.get(i)[targetClassIndex]/self.histogramItervalSize) 
            generalPercentageByScoreHistogram[histogramIndex] = generalPercentageByScoreHistogram.get(histogramIndex)+1.0 
          
        for key in generalPercentageByScoreHistogram.keys():
            histoCellPercentage = generalPercentageByScoreHistogram.get(key)/len(scoreDistributions.keys())
            generalPercentageByScoreHistogram[key] = histoCellPercentage
            histoAtt = attributeInfo("generalScoreDistHisto_" + key + "_" + identifier, "Numeric", histoCellPercentage, -1) 
            generalStatisticsAttributes[len(generalStatisticsAttributes)] = histoAtt
          

        #finally, save this to the global objest so that it can be used in Group2 and Group3 calculations
        self.generalPartitionPercentageByScoreHistogram[identifier]= generalPercentageByScoreHistogram
        #endregion

        #region Testing goodness of fit for multiple types of distributions

        #Some of the statistical test cannot be applied on an infinite number of samples (and we don't want to do that,
        #because that would take too long. For this reason, we begin by sampling a subset of instances (if warranted)
        #and use them instead*/

        samplesConfidenceScoreValues = {}

        desiredNumberOfSamplesInSet = 4500 
        if len(scoreDistributions.keys()) < desiredNumberOfSamplesInSet:
            for key in scoreDistributions.keys():
                samplesConfidenceScoreValues[key] = scoreDistributions.get(key)[targetClassIndex] 
              
          
        else :
            scoresListToPull = scoreDistributions.keys()
            random.shuffle(scoresListToPull) 
            scoresListToPullRandomSet = scoresListToPull[0:desiredNumberOfSamplesInSet]
            for pos in scoresListToPullRandomSet :
                samplesConfidenceScoreValues[pos]= scoreDistributions.get(pos)[targetClassIndex] #pos = keyValue
              
          

        pValuesList = [0.01, 0.05, 0.1] 


        #region Normal distribution
        #Used to test normal distribution
        #https:#www.programcreek.com/java-api-examples/index.php?source_dir=datumbox-framework-master/src/main/java/com/datumbox/framework/statistics/nonparametrics/onesample/ShapiroWilk.java
        fdc1 = samplesConfidenceScoreValues.values()
        alpha = 0.05
        for pval in pValuesList:
            isNormallyDistributed = False
            try: 
                _ , p = stats.shapiro(fdc1) 
                if p > alpha:
                    isNormallyDistributed = True
            except:
                isNormallyDistributed = False 
              
            normalDistributionGoodnessOfFitPVAlues[pval]= isNormallyDistributed 
          

        for key in normalDistributionGoodnessOfFitPVAlues.keys():
            isNormal = 0.0 
            if normalDistributionGoodnessOfFitPVAlues.get(key):
                isNormal = 1.0 
              
            normalDistributionAtt = attributeInfo("isNormallyDistirbutedAt" + key + "_" + identifier, "Discrete", isNormal, 2) 
            generalStatisticsAttributes[len(generalStatisticsAttributes)] = normalDistributionAtt
          
        #endregion
        #region Log-normal distribution
        #Used to test logarithmic distribution
        #In order to check whether something is lognormal, we simply need to use ln(x) on the values (taking cate of 0's) and then check for normal distribution
        logNormalSamplesConfidenceScoreValues = {}
        for key in samplesConfidenceScoreValues.keys() :
            logNormalSamplesConfidenceScoreValues[key] = 1 + samplesConfidenceScoreValues.get(key) 
          
        fdc2 = logNormalSamplesConfidenceScoreValues.values()
        for pval in pValuesList:
            isLogNormallyDistributed = False
            try:
                _ , p = stats.shapiro(fdc2) 
                if p > alpha:
                    isLogNormallyDistributed = True
            except:
                isLogNormallyDistributed = False 
              
            logNormalDistributionGoodnessOfFitPVAlue[pval]= isLogNormallyDistributed 
          

        for key in logNormalDistributionGoodnessOfFitPVAlue.keys():
            isLogNormal = 0.0 
            if logNormalDistributionGoodnessOfFitPVAlue.get(key):
                isLogNormal = 1.0 
              
            logNormalDistributionAtt = attributeInfo("isLogNormallyDistirbutedAt" + key + "_" + identifier, "Discrete", isLogNormal, 2) 
            generalStatisticsAttributes[len(generalStatisticsAttributes)] = logNormalDistributionAtt
          
        #endregion

        #region Uniform distribution
        #Used to test whether the scores are distributed uniformly
        #Step 1: we generate a list of values randomly sampled from a uniform [0,1] distribution
        uniformRandomVals = []
        i = 0
        while i<len(samplesConfidenceScoreValues):
            uniformRandomVals.append(random.uniform(0.0, int(properties.randomSeed)))
            i+=1
          

        # Kolmogorov-Smirnov test
        alpha = 0.05
        for pval in pValuesList:
            isUniformlyDistributed = False
            try:
                # uniformRandomVals - the random values
                # samplesConfidenceScoreValues - the values we sampled from the results
                _ , p = stats.kstest(uniformRandomVals,samplesConfidenceScoreValues.values()) 
                if p > alpha:
                    isUniformlyDistributed = True
            except:
                isUniformlyDistributed = False 

              
            uniformDistributionGoodnessOfFitPVAlue[pval]=isUniformlyDistributed 
          

        for key in uniformDistributionGoodnessOfFitPVAlue.keys():
            isUniform = 0.0 
            if uniformDistributionGoodnessOfFitPVAlue.get(key):
                isUniform = 1.0 
              
            uniformDistributionAtt = attributeInfo("isUniformlyDistirbutedAt" + key + "_" + identifier, "Discrete", isUniform, 2) 
            generalStatisticsAttributes[len(generalStatisticsAttributes)] = uniformDistributionAtt
          
        #endregion

        #endregion

        #region Statistical tests on the correlations of features whose instances are partitioned by confidence thresholds


        for threshold in self.confidenceScoreThresholds:
            allFeatureCorrelationStatsByThreshold[threshold] = []
            numericFeatureCorrelationStatsByThreshold[threshold]= [] 
            discreteFeatureCorrelationStatsByThreshold[threshold]= [] 

            #Since we're going to process multiple columns, we need to get the indices of the instances which are above and below the threshold
            indicesOfIndicesOverTheThreshold = {}
            counter = 0 
            for i in scoreDistributions.keys():
                if scoreDistributions.get(i)[targetClassIndex] >= threshold:
                    indicesOfIndicesOverTheThreshold[counter] = True 
                  
                counter+=1 
              

            for ci in unlabeledSamplesDataset.getAllColumns(False):
                try:
                    if (ci.getColumn().getType() == "Numeric") :
                     # We perform the T-Test for all the numeric attributes of the data.
                     #* It is important to note that we can't use paired-t test because the samples are not paired.*/
                        belowThresholdValues =[] 
                        belowThresholdCounter = 0 
                        aboveThresholdValues = []
                        aboveThresholdCounter = 0 
                        values = ci.getColumn().getValues() 
                        i = 0
                        while i<len(values):
                            if indicesOfIndicesOverTheThreshold.contains_key(i) :
                                aboveThresholdValues.append(values[i]) 
                                aboveThresholdCounter+=1 
                              
                            else :
                                belowThresholdValues.append(values[i]) 
                                belowThresholdCounter+=1 
                            i+=1
                          
                        if aboveThresholdCounter > 1:
                            tTestStatistic = ttest_ind(aboveThresholdValues,belowThresholdValues).statistic
                            allFeatureCorrelationStatsByThreshold.get(threshold).append(tTestStatistic) 
                            numericFeatureCorrelationStatsByThreshold.get(threshold).append(tTestStatistic) 
                          

                      
                    if ci.getColumn().getType() == "Discrete":
                        belowThresholdValues = []
                        belowThresholdCounter = 0 
                        aboveThresholdValues = []
                        aboveThresholdCounter = 0 
                        values = ci.getColumn().getValues() 
                        
                        i = 0
                        while i<len(values):
                            if indicesOfIndicesOverTheThreshold.contains_key(i):
                                aboveThresholdValues.append(values[i]) 
                                aboveThresholdCounter+=1 
                              
                            else :
                                belowThresholdValues.append(values[i]) 
                                belowThresholdCounter+=1
                            i+=1
                              
                          

                        so = StatisticOperations() 
                        numOfDiscreteValues = ci.getColumn().getNumOfPossibleValues() 
                        intersectionMatrix = so.generateChiSuareIntersectionMatrix(aboveThresholdValues, numOfDiscreteValues, belowThresholdValues, numOfDiscreteValues) 
                        chiSquareStatistic , _ = stats.chisquare(intersectionMatrix) 
                        allFeatureCorrelationStatsByThreshold.get(threshold).append(chiSquareStatistic) 
                        discreteFeatureCorrelationStatsByThreshold.get(threshold).append(chiSquareStatistic) 
                
                except:
                    continue 
                  
              
          

         #now that have all the test statistics we can generate the attributes for the meta model (I Could have done that in the same loop,
         #* but it's more convenient to separate the two - keeps things clean */
        for threshold in self.confidenceScoreThresholds :
            allFeaturesCorrelationMaxAtt = attributeInfo("allAttributesCorrelationMax_" + threshold + "_" + identifier, "Numeric", max(allFeatureCorrelationStatsByThreshold.get(threshold)), -1) 
            allFeaturesCorrelationMinAtt = attributeInfo("allAttributesCorrelationMin_" + threshold + "_" + identifier, "Numeric", min(allFeatureCorrelationStatsByThreshold.get(threshold)), -1) 
            allFeaturesCorrelationAvgAtt = attributeInfo("allAttributesCorrelationAvg_" + threshold + "_" + identifier, "Numeric", np.mean(allFeatureCorrelationStatsByThreshold.get(threshold)), -1) 
            allFeaturesCorrelationStdevAtt = attributeInfo("allAttributesCorrelationStdev_" + threshold + "_" + identifier, "Numeric", statistics.stdev(allFeatureCorrelationStatsByThreshold.get(threshold)), -1) 
            allFeaturesCorrelationMedianAtt = attributeInfo("allAttributesCorrelationMedian_" + threshold + "_" + identifier, "Numeric", np.percentile(allFeatureCorrelationStatsByThreshold.get(threshold), 50), -1) 
            generalStatisticsAttributes[len(generalStatisticsAttributes)] = allFeaturesCorrelationMaxAtt
            generalStatisticsAttributes[len(generalStatisticsAttributes)] = allFeaturesCorrelationMinAtt
            generalStatisticsAttributes[len(generalStatisticsAttributes)] = allFeaturesCorrelationAvgAtt
            generalStatisticsAttributes[len(generalStatisticsAttributes)] = allFeaturesCorrelationStdevAtt 
            generalStatisticsAttributes[len(generalStatisticsAttributes)] = allFeaturesCorrelationMedianAtt 

            numericFeaturesCorrelationMaxAtt = attributeInfo("numericAttributesCorrelationMax_" + threshold + "_" + identifier, "Numeric", max(numericFeatureCorrelationStatsByThreshold.get(threshold)), -1) 
            numericFeaturesCorrelationMinAtt = attributeInfo("numericAttributesCorrelationMin_" + threshold + "_" + identifier, "Numeric", min(numericFeatureCorrelationStatsByThreshold.get(threshold)), -1) 
            numericFeaturesCorrelationAvgAtt = attributeInfo("numericAttributesCorrelationAvg_" + threshold + "_" + identifier, "Numeric", np.mean(numericFeatureCorrelationStatsByThreshold.get(threshold)), -1) 
            numericFeaturesCorrelationStdevAtt = attributeInfo("numericAttributesCorrelationStdev_" + threshold + "_" + identifier, "Numeric", statistics.stdev(numericFeatureCorrelationStatsByThreshold.get(threshold)), -1) 
            numericFeaturesCorrelationMedianAtt = attributeInfo("numericAttributesCorrelationMedian_" + threshold + "_" + identifier, "Numeric", np.percentile(numericFeatureCorrelationStatsByThreshold.get(threshold), 50), -1) 
            generalStatisticsAttributes[len(generalStatisticsAttributes)] = numericFeaturesCorrelationMaxAtt
            generalStatisticsAttributes[len(generalStatisticsAttributes)] = numericFeaturesCorrelationMinAtt
            generalStatisticsAttributes[len(generalStatisticsAttributes)] = numericFeaturesCorrelationAvgAtt
            generalStatisticsAttributes[len(generalStatisticsAttributes)] = numericFeaturesCorrelationStdevAtt 
            generalStatisticsAttributes[len(generalStatisticsAttributes)] = numericFeaturesCorrelationMedianAtt

            discreteFeaturesCorrelationMaxAtt = attributeInfo("discreteAttributesCorrelationMax_" + threshold + "_" + identifier, "Numeric", max(discreteFeatureCorrelationStatsByThreshold.get(threshold)), -1) 
            discreteFeaturesCorrelationMinAtt = attributeInfo("discreteAttributesCorrelationMin_" + threshold + "_" + identifier, "Numeric", min(discreteFeatureCorrelationStatsByThreshold.get(threshold)), -1) 
            discreteFeaturesCorrelationAvgAtt = attributeInfo("discreteAttributesCorrelationAvg_" + threshold + "_" + identifier, "Numeric", np.mean(discreteFeatureCorrelationStatsByThreshold.get(threshold)), -1) 
            discreteFeaturesCorrelationStdevAtt = attributeInfo("discreteAttributesCorrelationStdev_" + threshold + "_" + identifier, "Numeric", statistics.stdev(discreteFeatureCorrelationStatsByThreshold.get(threshold)), -1) 
            discreteFeaturesCorrelationMedianAtt = attributeInfo("discreteAttributesCorrelationMedian_" + threshold + "_" + identifier, "Numeric", np.percentile(discreteFeatureCorrelationStatsByThreshold.get(threshold), 50), -1) 
            generalStatisticsAttributes[len(generalStatisticsAttributes)] = discreteFeaturesCorrelationMaxAtt
            generalStatisticsAttributes[len(generalStatisticsAttributes)] = discreteFeaturesCorrelationMinAtt
            generalStatisticsAttributes[len(generalStatisticsAttributes)] = discreteFeaturesCorrelationAvgAtt
            generalStatisticsAttributes[len(generalStatisticsAttributes)] = discreteFeaturesCorrelationStdevAtt
            generalStatisticsAttributes[len(generalStatisticsAttributes)] = discreteFeaturesCorrelationMedianAtt
          
        #endregion

        #region Imbalance ratios across different thresholds on the test set
        numOfTargetClassInstancesInTrainingData = 0 
        numOfNonTargetClassInstancesInTrainingData = 0 
        i = 0
        while i<len(labeledSamplesDataset.getNumOfRowsPerClassInTrainingSet()):
        
            if targetClassIndex == i :
                numOfTargetClassInstancesInTrainingData = labeledSamplesDataset.getNumOfRowsPerClassInTrainingSet()[i] 
              
            else :
                numOfNonTargetClassInstancesInTrainingData += labeledSamplesDataset.getNumOfRowsPerClassInTrainingSet()[i] 
            i+=1
          

        trainingSetImbalanceRatio = numOfTargetClassInstancesInTrainingData / numOfNonTargetClassInstancesInTrainingData 
        trainingDataImbalanceRatioAtt = attributeInfo("trainingDataImbalanceRatio_" + "_" + identifier, "Numeric", trainingSetImbalanceRatio, -1) 
        generalStatisticsAttributes[len(generalStatisticsAttributes)] = trainingDataImbalanceRatioAtt

        for threshold in self.confidenceScoreThresholds:
            instancesAboveThreshold = 0 
            instancesBelowThreshold =0 
            for i in scoreDistributions.keys():
                if scoreDistributions.get(i)[targetClassIndex] >= threshold:
                    instancesAboveThreshold+=1 
                  
                else :
                    instancesBelowThreshold+=1 
                  
              
            imbalanceRatio = instancesAboveThreshold/instancesBelowThreshold 
            if (not np.isnan(imbalanceRatio)) and np.isfinite(imbalanceRatio):
                imbalanceRatioByConfidenceScoreRatio[threshold]= imbalanceRatio 
                imbalanceScoreRatioToTrueRatio[threshold] = imbalanceRatio/trainingSetImbalanceRatio
              
            else :
                imbalanceRatioByConfidenceScoreRatio[threshold] = -1.0 
                imbalanceScoreRatioToTrueRatio[threshold] = -1.0
              
            testSetImbalanceRatioByThresholdAtt = attributeInfo("testSetImbalanceRatioByThreshold__"+identifier+"_"+threshold , "Numeric", imbalanceRatioByConfidenceScoreRatio.get(threshold), -1) 
            testAndTrainSetImbalanceRatiosAtt = attributeInfo("testAndTrainSetImbalanceRatios__"+identifier+"_"+threshold, "Numeric", imbalanceScoreRatioToTrueRatio.get(threshold), -1) 
            generalStatisticsAttributes[len(generalStatisticsAttributes)] = testSetImbalanceRatioByThresholdAtt
            generalStatisticsAttributes[len(generalStatisticsAttributes)] = testAndTrainSetImbalanceRatiosAtt 
          


        #endregion

        return generalStatisticsAttributes 
      

    def setNumOfIterationsBackToAnalyze(self, numOfIterationsBackToAnalyze1):
        
        self.numOfIterationsBackToAnalyze = numOfIterationsBackToAnalyze1 
      
  
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        