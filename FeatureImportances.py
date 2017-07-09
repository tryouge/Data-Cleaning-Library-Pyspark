from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier as RF
from pyspark.ml.feature import StringIndexer, VectorIndexer, VectorAssembler, SQLTransformer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
import numpy as np
import functools
from pyspark.ml.feature import OneHotEncoder

def FeatureImportances(df, target, FeatureImportance_cutoff):
	
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
	trainingData, testData = df_t.randomSplit([0.8, 0.2], seed=0)

	rf = RF(labelCol='label', featuresCol='features',numTrees=200)
	model = rf.fit(trainingData)
	feat_imp = dict()
	vi = model.featureImportances
	no_of_cols = len(cols_now)
	cols_actual = num_cols + string_cols

	for i in range(no_of_cols):
		feat_imp[cols_actual[i]] = vi[i]

	return feat_imp