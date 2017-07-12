from pyspark.sql.functions import collect_list,size,monotonically_increasing_id

def getDupRows(df):

	#Adding a dummy index column to return indices
	tdf = df.withColumn("RowNoIndex", monotonically_increasing_id())
	tdf=tdf.groupBy(df.columns).agg(collect_list("RowNoIndex").alias("dup_ids")).where(size("dup_ids") > 1)
	dup_ids_list = tdf.select("dup_ids").rdd.flatMap(lambda x:x).collect()
	dup_ids = []
	for dup in dup_ids_list:
		dup_ids = dup_ids + dup[1:]

	#Note: Indexing starts from 0
	
	return dup_ids
