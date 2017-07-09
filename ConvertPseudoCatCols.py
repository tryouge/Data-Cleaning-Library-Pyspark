import re
from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql.functions import regexp_replace, col

sc = SparkContext.getOrCreate()
sqlContext = SQLContext(sc)

def ConvertPseudoCatCols(df, start_symbols, end_symbols):

	start_r = [re.compile('^' + '\\' + s + '[0-9]+') for s in start_symbols]
	end_r = [re.compile('[0-9]+' + '\\' + s + '$') for s in end_symbols]
	if df.count() > 10000:
		sample = 1000

	if df.count() < 10:
		sample = df.count()

	else:
		sample = df.count()/10

	df_t = df.rdd.takeSample(False, sample, seed=0)
	df_t = sqlContext.createDataFrame(df_t)

	string_cols = []
	for (a,b) in df.dtypes:
			if b == 'string':
				string_cols.append(a)


	pseudo_cat_cols = dict()

	for cols in string_cols:
		col_elements = df_t.select(cols).rdd.flatMap(lambda x:x).collect()

		for i in range(len(start_symbols)):
			if all(start_r[i].match(ele) for ele in col_elements):
				pseudo_cat_cols[cols]='^'+'\\'+ start_symbols[i]

		for i in range(len(end_symbols)):
			if all(end_r[i].match(ele) for ele in col_elements):
				pseudo_cat_cols[cols] = '\\' + end_symbols[i] + '$'

	df_r = df
	for i in pseudo_cat_cols:
		df_r=df_r.withColumn(i+'_new', regexp_replace(col(i), pseudo_cat_cols[i], ""))
		df_r=df_r.withColumn(i+'_new', col(i+'_new').cast('float').alias(i+'_new'))
		df_r=df_r.drop(i)

	return df_r
