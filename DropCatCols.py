from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.sql.functions import monotonically_increasing_id

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

def CheckSimilarity(df, col1, col2, similarity_cutoff):

	tdf = df
	t_count = tdf.count()

	tdf = df.groupBy([col1, col2]).count()
	tdf = tdf.sort('count', ascending=False)

	col1_entries = tdf.toPandas()[col1].unique()
	col2_entries = tdf.toPandas()[col2].unique()

	if len(col1_entries)!=len(col2_entries):
		return 0

	col1_dict = dict()
	col2_dict = dict()

	for entry in col1_entries:
		col1_dict[entry]=0

	for entry in col2_entries:
		col2_dict[entry]=0

	tdf = tdf.collect()
	s_count=0

	for row in tdf:
		if col1_dict[row[0]]==0 and col2_dict[row[1]]==0:
			col1_dict[row[0]]=1
			col2_dict[row[1]]=1
			s_count=s_count+row[2]

	s_ratio = s_count/float(t_count)

	if s_ratio > similarity_cutoff:
		return 1

	else:
		return 0


def DropCatCols(df, similarity_cutoff):

	string_cols=[]
	for (a,b) in df.dtypes:
			if b == 'string':
				string_cols.append(a)

	tdf = df[string_cols]

	num_cols = len(tdf.columns)
	dsu = [i for i in range(num_cols)]
	size = [1 for i in range(num_cols)]

	drop_cols = []
	dsu_dict = dict()

	for i in range(num_cols):
		for j in range(i):
			if CheckSimilarity(tdf, string_cols[i], string_cols[j], similarity_cutoff)==1:
				union(dsu, size, i, j)

	for i in range(num_cols):
		if dsu[i]==i:
			dsu_dict[tdf.columns[i]] = [tdf.columns[i]]

	for i in range(num_cols):
		if dsu[i]!=i:
			drop_cols.append(tdf.columns[i])
			ri = root(dsu, i)
			dsu_dict[tdf.columns[ri]].append(tdf.columns[i])

	for cols in drop_cols:
		tdf=tdf.drop(cols)

	num_df = df
	for cols in string_cols:
		num_df=num_df.drop(cols)

	tdf = tdf.withColumn("RowNoIndex", monotonically_increasing_id())
	num_df = num_df.withColumn("RowNoIndex", monotonically_increasing_id())
	tdf = tdf.join(num_df,['RowNoIndex'])
	tdf = tdf.drop('RowNoIndex')
	
	return dsu_dict, tdf	