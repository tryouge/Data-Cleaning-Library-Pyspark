from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier as RF
from pyspark.ml.feature import StringIndexer, VectorIndexer, VectorAssembler, SQLTransformer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
import numpy as np
import functools
from pyspark.ml.feature import OneHotEncoder
from pyspark.ml.clustering import KMeans
from numpy.linalg import norm

def getOutliers(df, opt_k, dist_cutoff):

	#Getting categorical columns
	column_vec_in = []
	for (a,b) in df.dtypes:
		if b=="string":
				column_vec_in.append(a)

	column_vec_out = [x + '_catVec' for x in column_vec_in]
	cols_now = [x for x in df.columns if x not in column_vec_in]
	cols_now += column_vec_out

	indexers = [StringIndexer(inputCol=x, outputCol=x+'_tmp') for x in column_vec_in ]
	encoders = [OneHotEncoder(dropLast=False, inputCol=x+"_tmp", outputCol=y) for x,y in zip(column_vec_in, column_vec_out)]
	tmp = [[i,j] for i,j in zip(indexers, encoders)]
	tmp = [i for sublist in tmp for i in sublist]

	assembler_features = VectorAssembler(inputCols=cols_now, outputCol='features')
	tmp += [assembler_features]

	pipeline = Pipeline(stages=tmp)
	df_t = pipeline.fit(df).transform(df)
	df_t.cache()

	kmeans=KMeans().setK(opt_k).setSeed(0)
	model=kmeans.fit(df_t)

	cluster_centres = model.clusterCenters()
	df_t=model.transform(df_t)

	cluster_nos=df_t.select('prediction').rdd.flatMap(lambda x:x).collect()	
	outliers = []

	row_points = df_t.select('features').rdd.flatMap(lambda x:x).collect()

	#Calculating distances to cluster centres
	for i in range(df.count()):
		a = np.array(row_points[i])
		b = np.array(cluster_centres[cluster_nos[i]])
		if norm(a-b)>dist_cutoff:
			outliers.append(i)


	return outliers

