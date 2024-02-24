
import plotly.express as px
from scipy import stats
import numpy as np
import attributeInfo
import ScoreDistributionBasedAttributes 

class ScoreDistributionBasedAttributesTdBatch:

    def init_list(size, element):
        list_to_return = []
        i = 0
        while i< size:
            list_to_return.append(element)
            i+=1
        return list_to_return

    def getScoreDistributionBasedAttributes(self, datasetAddedBatch, evaluationPerPartition_td, unifiedSetEvaluationResults_td
            , evaluationResultsPerSetAndInterationTree_mainstream, unifiedDatasetEvaulationResults_mainstream
            , labeledToMetaFeatures_td, unlabeledToMetaFeatures_td, current_iteration, targetClass, properties):
        attributes = {}

        #general statistics on td score dist
        currentScoreDistributionStatistics = {}
        for  partitionIndex in evaluationPerPartition_td.keys():
            generalStatisticsAttributes = ScoreDistributionBasedAttributes.calculateGeneralScoreDistributionStatistics(unlabeledToMetaFeatures_td
                    , labeledToMetaFeatures_td, evaluationPerPartition_td.get(partitionIndex).getScoreDistributions(),targetClass,"td_partition_" + partitionIndex, properties) 
            for pos in generalStatisticsAttributes.keys():
                currentScoreDistributionStatistics[len(currentScoreDistributionStatistics)] = generalStatisticsAttributes.get(pos) 
   
        unifiedStatisticsAttributes = ScoreDistributionBasedAttributes.calculateGeneralScoreDistributionStatistics(unlabeledToMetaFeatures_td
                , labeledToMetaFeatures_td, unifiedSetEvaluationResults_td.getScoreDistributions(),targetClass,"td_unified", properties) 
        for  pos in unifiedStatisticsAttributes.keys():
            currentScoreDistributionStatistics[len(currentScoreDistributionStatistics)] = unifiedStatisticsAttributes.get(pos) 
        
        attributes.update(currentScoreDistributionStatistics) 

        #comparison per partition - T test
        for partitionIndex in evaluationPerPartition_td.keys():
            tdPartitionScores  = self.init_list(len(evaluationPerPartition_td.get(partitionIndex).getScoreDistributions()),0.0) 
            tdPartitionScores_cnt = 0 
            for instance in evaluationPerPartition_td.get(partitionIndex).getScoreDistributions().keys():
                tdPartitionScores[tdPartitionScores_cnt] = evaluationPerPartition_td.get(partitionIndex).getScoreDistributions().get(instance)[targetClass] 
                tdPartitionScores_cnt+=1
            
            msPartitionScores = self.init_list(len(evaluationResultsPerSetAndInterationTree_mainstream.get(partitionIndex).getLatestEvaluationInfo().getScoreDistributions()),0.0)
            msPartitionScores_cnt = 0 
            for instance in evaluationResultsPerSetAndInterationTree_mainstream.get(partitionIndex).getLatestEvaluationInfo().getScoreDistributions().keys():
                msPartitionScores[msPartitionScores_cnt] = evaluationResultsPerSetAndInterationTree_mainstream.get(partitionIndex).getLatestEvaluationInfo().getScoreDistributions().get(instance)[targetClass] 
                msPartitionScores_cnt+=1 
            
            TTestStatistic = stats.ttest_ind(tdPartitionScores,msPartitionScores).statistic 
            tTest_att = attributeInfo("td_t_test_partition_"+partitionIndex, "Numeric", TTestStatistic, -1) 
            attributes[len(attributes)] = tTest_att
        
        #comparison on unified
        tdUnufiedScores = self.init_list(len(unifiedSetEvaluationResults_td.getScoreDistributions()), 0.0)
        tdUnufiedScores_cnt = 0 
        for instance in unifiedSetEvaluationResults_td.getScoreDistributions().keys():
            tdUnufiedScores[tdUnufiedScores_cnt] = unifiedSetEvaluationResults_td.getScoreDistributions().get(instance)[targetClass] 
            tdUnufiedScores_cnt+=1 
        
        msUnufiedScores = self.init_list(len(unifiedDatasetEvaulationResults_mainstream.getLatestEvaluationInfo().getScoreDistributions()),0.0)
        msUnufiedScores_cnt = 0 
        for instance in unifiedDatasetEvaulationResults_mainstream.getLatestEvaluationInfo().getScoreDistributions().keys():
            msUnufiedScores[msUnufiedScores_cnt] = unifiedDatasetEvaulationResults_mainstream.getLatestEvaluationInfo().getScoreDistributions().get(instance)[targetClass] 
            msUnufiedScores_cnt+=1 
        
        TTestStatistic = stats.ttest_ind(tdUnufiedScores,msUnufiedScores).statistic 
        tTest_att_uni = attributeInfo("td_t_test_unified", "Numeric", TTestStatistic, -1) 
        attributes[len(attributes)] = tTest_att_uni 

        #fix NaN: convert to -1.0
        for entry in attributes.values():
            ai = entry 
            if ai.getAttributeType() == "Numeric":
                aiVal = ai.getValue() 
                if (np.isnan(aiVal)):
                    ai.setValue(-1.0) 

        return  attributes 
