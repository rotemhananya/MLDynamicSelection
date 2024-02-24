
class Config(object):

  # java config.propeties:
    DatabaseUrl = 'jdbc:mysql://localhost/cotrain?useJDBCCompliantTimezoneShift=true&useLegacyDatetimeCode=false&serverTimezone=UTC&useSSL=false'
    tempDirectory='/Users/guyz/Documents/CoTrainingVerticalEnsemble/tempFiles/'
    inputFilesDirectory='/Users/guyz/Documents/CoTrainingVerticalEnsemble/inputData/'
    DBPassword= 'Bookings1515'
    
    #DatabaseUrl = jdbc:mysql://132.72.23.245/cotrain?useJDBCCompliantTimezoneShift=true&useLegacyDatetimeCode=false&serverTimezone=UTC&useSSL=false
    #tempDirectory=/data/home/zaksg/co-train/tempFiles
    #inputFilesDirectory=/data/home/zaksg/co-train/inputData
    #DBPassword=7ISE!yalla
    #old DBPassword=gw5oV*ielGQk
    
    JDBC_DRIVER='com.mysql.jdbc.Driver'
    DBUser='root'
    
    numOfrandomSeeds=1
    maxNumberOfDiscreteValuesForInclusionInSet=1000
    numOfInstancesToAddPerIterationPerClass=2
    classifier='RandomForest'
    numOfCoTrainingIterations=30
    numOfDiscretizationBins=10
    
    randomSeed=5
    numOfBatchedPerIteration=100
    instancesPerBatch=8
    minNumberOfInstancesPerClassInAbatch=2
    batchSelection='smart'