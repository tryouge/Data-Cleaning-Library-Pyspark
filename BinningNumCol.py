from pyspark.ml.feature import QuantileDiscretizer
from pyspark.sql.functions import lit, col, create_map
from itertools import chain

def Binning(df, num_col, no_of_buckets):

	for (a,b) in df.dtypes:
			if a==num_col:
				o_dtype = b


	tdf = df.withColumn(num_col, col(num_col).cast('double'))
	qds = QuantileDiscretizer(numBuckets=no_of_buckets, inputCol=num_col, outputCol="bucket_no")
	bucketizer = qds.fit(tdf)
	splits = bucketizer.getSplits()
	tdf = bucketizer.transform(tdf)

	bucket_dict = dict()

	for i in range(no_of_buckets):
		bucket_dict[float(i)] = str(splits[i]) + ' to ' + str(splits[i+1])

	#tdf = tdf.withColumn('bucket_no', col(num_col).cast('string'))

	mapping_expr=create_map([lit(x) for x in chain(*bucket_dict.items())])
	tdf = tdf.withColumn(num_col + '_bucket_range', mapping_expr.getItem(col('bucket_no')))

	tdf = tdf.withColumn(num_col, col(num_col).cast(o_dtype))
	
	return tdf, bucket_dict