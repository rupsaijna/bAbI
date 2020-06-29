import numpy as np
import sys
import pandas as pd

fname=sys.argv[1]

opfile_df=fname.replace('.txt','_df.pkl')
df = pd.read_pickle(opfile_df)

opfile_df_transformed=fname.replace('.txt','_df_transformed.pkl')
df_transformed = pd.read_pickle(opfile_df_transformed)

print('Originally')
print('ORIG Cols :',len(df.columns))
print('Trans Cols :',len(df_transformed.columns))

#print(df_transformed)
for col in df.columns:
	if df[col].sum()==0:
		df=df.drop(columns=[col])

for col in df_transformed.columns:
	if df_transformed[col].sum()==0:
		df_transformed=df_transformed.drop(columns=[col])

print('Removing unused')
print('ORIG Cols :',len(df.columns))
print('Trans Cols :',len(df_transformed.columns))


dum_cols=[c for c in df_transformed.columns if ('q_' not in c and c!='ANSWER')]
newdf_transformed = pd.get_dummies(df_transformed, prefix=dum_cols, columns=dum_cols)

print('Category to Binary')
print('Trans Cols :',len(newdf_transformed.columns))


drop_cols=[c for c in newdf_transformed.columns if ('_0' in c )]
newdf_transformed=newdf_transformed.drop(columns=drop_cols)

print('Drop Not presents')
print('Trans Cols :',len(newdf_transformed.columns))
#print(newdf_transformed.loc[5])

###############################################
dum_cols=[c for c in df.columns if ('q_' not in c and c!='ANSWER')]
newdf = pd.get_dummies(df, prefix=dum_cols, columns=dum_cols)

print('Category to Binary')
print('ORIG Cols :',len(newdf.columns))


drop_cols=[c for c in newdf.columns if ('_0' in c )]
newdf=newdf.drop(columns=drop_cols)

print('Drop Not presents')
print('ORIG Cols :',len(newdf.columns))
#print(newdf.loc[5])


##WRITE
newdf.to_pickle(fname.replace('.txt','')+'_relationalfeatureset.pkl')
newdf_transformed.to_pickle(fname.replace('.txt','')+'_transformed_relationalfeatureset.pkl')


f=open(fname.replace('_sentenceleveltransform','').replace('.txt','_meta.txt'),'a+')
f.write('\n'+fname+'\t'+fname.replace('.txt','')+'_relationalfeatureset.pkl'+'\t'+','.join(newdf.columns)+'\n')
f.write('\n'+fname+'\t'+fname.replace('.txt','')+'_transformed_relationalfeatureset.pkl'+'\t'+','.join(newdf_transformed.columns)+'\n')
f.close()

print('Features ready at '+ fname.replace('.txt','')+'_relationalfeatureset.pkl'+ ' and '+ fname.replace('.txt','')+'_transformed_relationalfeatureset.pkl')
