from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier as RF
from pyspark.ml.feature import StringIndexer, VectorIndexer, VectorAssembler, SQLTransformer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
import numpy as np
import functools
from pyspark.ml.feature import OneHotEncoder
from pyspark import SparkContext
from pyspark.sql import SQLContext

sc = SparkContext.getOrCreate()
sqlContext = SQLContext(sc)

def LearningCurve(df, target):
	
	df_t=df
	string_cols = []
	for (a,b) in df.dtypes:
			if b == 'string' and a != target:
				string_cols.append(a)

	num_cols = [x for x in df.columns if x not in string_cols and x!=target]
	encoded_cols = [x+"_index" for x in string_cols]

	indexers = [StringIndexer(inputCol=column, outputCol=column+"_index").fit(df) for column in string_cols]
	pipeline = Pipeline(stages=indexers)
	df_t = pipeline.fit(df_t).transform(df_t)

	cols_now = num_cols + encoded_cols
	assembler_features = VectorAssembler(inputCols=cols_now, outputCol='features')
	labelIndexer = StringIndexer(inputCol=target, outputCol="label")
	tmp = [assembler_features, labelIndexer]
	pipeline = Pipeline(stages=tmp)
	df_t = pipeline.fit(df_t).transform(df_t)
	df_t.cache()
	trainingData, testData = df_t.randomSplit([0.7, 0.3], seed=0)

	rf = RF(labelCol='label', featuresCol='features',numTrees=200)
	plot_points = []

	#Variable to be adjusted for increment in data%
	step_var = 10

	for i in range(step_var,101,step_var):

		sample_size = (i*trainingData.count())/100
		part_Data=trainingData.rdd.takeSample(False, sample_size, seed=i)
		part_Data=sqlContext.createDataFrame(part_Data)

		model = rf.fit(part_Data)
		evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
		
		#Calculating train error
		transformed = model.transform(part_Data)
		train_accuracy = evaluator.evaluate(transformed)
		train_error = 1 - train_accuracy

		#Calculating test error
		transformed = model.transform(testData)
		test_accuracy = evaluator.evaluate(transformed)
		test_error = 1 - test_accuracy
		
		plot_points.append([i,train_error,test_error])

	return plot_points