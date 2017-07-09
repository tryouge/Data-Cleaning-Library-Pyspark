from pyspark.sql.functions import col, count

def DropColsByNa(df, perc_cutoff):
	df_cols = df.columns
	df_nullCounts = [df.where(col(c).isNull()).count() for c in df.columns]
	num_Cols = len(df.columns)
	num_Rows = df.count()
	t_df = df
	for i in range(num_Cols):
		if (df_nullCounts[i]*100)/float(num_Rows) > perc_cutoff:
			t_df=t_df.drop(df_cols[i])

	return t_df
