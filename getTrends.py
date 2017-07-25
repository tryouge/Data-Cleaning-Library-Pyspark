from pyspark.sql.functions import log10, col, bround
from pyspark import SparkContext
from pyspark.sql import SQLContext

sc = SparkContext.getOrCreate()
sqlContext = SQLContext(sc)

global_trends = []

def getTrendRows(df, attrbs, targColumn, targVal, recordsNo, targVals_counts, depth, parent_attrb):

	trends = []

	#print parent_attrb

	#print depth

	#Edit this for number of recursive levels
	if depth == 3:
		return 

	for attrb in attrbs:

		df_t = df.groupBy(attrb).pivot(targColumn).count()
		df_t = df_t.select(attrb, str(targVal))

		df_t = df_t.withColumn('Support_' + str(targVal), col(str(targVal)) / float(recordsNo))
		df_t = df_t.withColumn('Confidence_' + str(targVal), col(str(targVal)) / float(targVals_counts[targVal]))
		df_t = df_t.withColumn('Lift_'+ str(targVal), col('Confidence_' + str(targVal)) / (targVals_counts[targVal] / float(recordsNo)))
		df_t = df_t.withColumn('Metric_'+ str(targVal), col('Support_' + str(targVal)) * log10(col('Lift_' + str(targVal))))

		scores = df_t.collect()
		scores = [ [attrb] + list(x) for x in scores ]
		trends = trends + scores

		#Adjust [:3] for top x rules
		trends = sorted(trends, key=lambda x:x[-1], reverse=True)[:3]

	for trend in trends:

		df_e = df.filter(col(trend[0]) == trend[1])
		df_e = df_e.drop(trend[0])
		tmp_attrbs = attrbs

		#print tmp_attrbs
		#print trend

		tmp_attrbs = [ x for x in attrbs if x!=trend[0] ]

		parent_attrb.append(str(trend[0]) + ' == ' + str(trend[1]))

		#Getting row to be inserted in dataframe
		z = parent_attrb + trend[-5:]
		z[0:-5] = [' & '.join(z[0:-5])]

		global_trends.append(z)

		getTrendRows(df_e, tmp_attrbs, targColumn, targVal, recordsNo, targVals_counts, depth+1, parent_attrb)		
		parent_attrb.pop()

	return global_trends




def getTrends(df, attrbs, targColumn, targVal):

	targVals = df.groupBy(targColumn).count()
	recordsNo = df.count()
	targVals = targVals.collect()
	targVals = [tuple(x) for x in targVals]
	targVals_counts = dict(targVals)

	df_rows = getTrendRows(df, attrbs, targColumn, targVal, recordsNo, targVals_counts, 0, [])
	trends_cols = ['Trend', 'Count of Target Val', 'Support', 'Confidence', 'Lift', 'Metric']
	df_req = sqlContext.createDataFrame(df_rows, trends_cols)
	df_req = df_req.withColumn('Support', bround(col('Support'), 3))
	df_req = df_req.withColumn('Confidence', bround(col('Confidence'), 3))
	df_req = df_req.withColumn('Lift', bround(col('Lift'), 3))
	df_req = df_req.withColumn('Metric', bround(col('Metric'), 3))

	return df_req

