from pyspark.sql.functions import monotonically_increasing_id
from pyspark.mllib.stat import Statistics

def root(a, i):

	while a[i]!=i:
		a[i]=a[a[i]]
		i=a[i]

	return i

def union(a, size, i, j):
	ri = root(a, i)
	rj = root(a, j)

	if size[ri] < size[rj]:
		a[ri]=a[rj]
		size[rj]=size[rj]+size[ri]

	else:
		a[rj]=a[ri]
		size[ri]=size[ri]+size[rj]

def DropColsByCor(df, cor_cutoff):

	tdf = df
	dsu_dict = {}

	string_cols=[]
	for (a,b) in df.dtypes:
			if b == 'string':
				string_cols.append(a)

	for cols in string_cols:
		tdf=tdf.drop(cols)

	num_cols = len(tdf.columns)
	dsu = [i for i in range(num_cols)]
	size = [1 for i in range(num_cols)]

	features = tdf.rdd.map(lambda row:row[0:])
	corr_mat = Statistics.corr(features, method="pearson")

	for i in range(num_cols):
		for j in range(i):
			if corr_mat[i][j] > cor_cutoff:
				union(dsu, size, i, j)

	drop_cols = []
	for i in range(num_cols):
		if dsu[i]!=i:
			drop_cols.append(tdf.columns[i])

		#Setting up dictionary to save up on iterations
		if dsu[i]==i:
			dsu_dict[tdf.columns[i]] = [tdf.columns[i]]

	for i in range(num_cols):
		if dsu[i]!=i:
			ri = root(dsu, i)
			dsu_dict[tdf.columns[ri]].append(tdf.columns[i])

	for cols in drop_cols:
		tdf=tdf.drop(cols)

	string_df=df.select(string_cols)

	#Adding index to help merge both string and numeric dataframes
	tdf = tdf.withColumn("RowNoIndex", monotonically_increasing_id())
	string_df = string_df.withColumn("RowNoIndex", monotonically_increasing_id())
	tdf = tdf.join(string_df,['RowNoIndex'])
	tdf = tdf.drop('RowNoIndex')

	return dsu_dict, tdf

