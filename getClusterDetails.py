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

def getClusterDetails(df):

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

	cluster_plot_points = []

	for i in range(2,51):
		kmeans = KMeans().setK(i).setSeed(i)
		model = kmeans.fit(df_t)
		cluster_plot_points.append((i, model.computeCost(df_t)))

	b = np.array(cluster_plot_points[-1]) - np.array(cluster_plot_points[0])
	b = b / norm(b)

	line_dists = [ norm(np.array(cluster_plot_points[i]) - np.array(cluster_plot_points[0]) - (np.dot(np.array(cluster_plot_points[i])-np.array(cluster_plot_points[0]), np.array(b))*b)) for i in range(1,49)]
	optimum_k = line_dists.index(max(line_dists))+3

	return optimum_k, cluster_plot_points



