import numpy as np
from scipy.stats import kurtosis, skew
from pyspark import SparkContext
from pyspark.sql import SQLContext

sc = SparkContext.getOrCreate()
sqlContext = SQLContext(sc)

def statsCol(df, col_n, d_type):

	dets = []
	dets.append(col_n)
	dets.append(d_type)


	col_vals = df.select(col_n).rdd.flatMap(lambda x:x).collect()
	missing_vals = col_vals.count(None)

	dets.append(len(col_vals) - missing_vals)
	dets.append(missing_vals)

	missing_vals_perc = round(100 * missing_vals/ float(df.count()), 2)
	dets.append(missing_vals_perc)

	no_of_levels = len(set(col_vals))
	dets.append(no_of_levels)

	col_vals = [x for x in col_vals if x!=None]

	if d_type == 'string':
		dets = dets + [None for i in range(10)]

		level_counts = df.groupBy(col_n).count().collect()
		level_counts = [tuple(x) for x in level_counts]

		dets.append(dict(level_counts))

	else:
		dets.append(min(col_vals))
		dets.append(max(col_vals))
		dets.append(float(np.median(col_vals)))
		dets.append(float(round(np.mean(col_vals),2)))
		dets.append(float(round(np.var(col_vals),2)))
		dets.append(float(round(skew(col_vals),2)))
		dets.append(float(round(kurtosis(col_vals),2)))
		dets.append(float(round(np.std(col_vals),2)))
		dets.append(float(np.percentile(col_vals, 25)))
		dets.append(float(np.percentile(col_vals, 75)))
		dets.append(None)

	return tuple(dets)

def sparkstatsCal(df):

	from datetime import datetime
	start = datetime.now()

	stats_cols = ['attr_name', 'var_type', 'record_count', 'no_of_missing_records', 'missing_data_percentage', 'no_of_levels', 'min_value', 'max_value', 'median_value', 'mean_value', 'variance_value', 'skewness_value', 'kurtosis_value', 'standard_deviation_value', 'percentile_25', 'percentile_75', 'level_counts']
	df_rows = []
	for (a,b) in df.dtypes:
		df_rows.append(statsCol(df, a, b))

	df_stat = sqlContext.createDataFrame(df_rows, stats_cols)

	elapsedTimeMin = round((datetime.now() - start).total_seconds()/60,3)

	print 'Time Elapsed: ' + str(elapsedTimeMin) + ' Minutes'
	return df_stat

