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
from functools import reduce
from operator import add
from pyspark.sql.functions import lit, col, create_map
from itertools import chain

def get_cluster_k(df):

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

	return optimum_k


def sum_(*cols):
    return reduce(add, cols, lit(0))

def mergingCatCol(df, cat_col, target_col):

	distinct_num = df.select(cat_col).distinct().count()
	distinct_cutoff = 10

	if distinct_num <= distinct_cutoff:
		print 'Levels are low enough'
		return df

	ldf = df.stat.crosstab(cat_col, target_col)

	target_col_values = ldf.columns[1:]

	cols_to_sum = [col(x) for x in target_col_values]

	ldf = ldf.withColumn("total", (sum_(*cols_to_sum)))

	for x in target_col_values:
		ldf = ldf.withColumn(x, col(x) / col('total'))

	ldf = ldf.drop('total')
	num_cluster = get_cluster_k(ldf)

	tmp = [VectorAssembler(inputCols=target_col_values, outputCol="features")]
	pipeline = Pipeline(stages=tmp)
	ldf = pipeline.fit(ldf).transform(ldf)

	kmeans = KMeans().setK(num_cluster).setSeed(0)
	ldf = kmeans.fit(ldf).transform(ldf)

	return_dict = dict()
	mapping_dict = dict()

	for i in range(num_cluster):
		return_dict[cat_col+'_level_'+str(i)]=[]

	cluster_map = ldf.select(ldf.columns[0], 'prediction').collect()
	
	for x in cluster_map:
		 mapping_dict[x[0]]=cat_col+'_level_'+str(x[1])
		 return_dict[cat_col+'_level_'+str(x[1])].append(x[0])

	mapping_expr=create_map([lit(x) for x in chain(*mapping_dict.items())])
	tdf = df.withColumn(cat_col+'_level', mapping_expr.getItem(col(cat_col)))	

	return tdf, return_dict