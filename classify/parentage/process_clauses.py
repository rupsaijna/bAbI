import pandas as pd
from sympy import *
import sys

#fname='generated/generated1.txt'
fname=sys.argv[1]

metaname=fname.replace('_sentenceleveltransform','').replace('.txt','_meta.txt')
clausename=fname.replace('.txt','_clauses.txt')
outname=clausename.replace('.txt','_processed.txt')

mf=open(metaname,'r')
metadetails=mf.readlines()
mf.close()
fo=open(outname,'w')
fo.write('Names:'+str(metadetails[0]))
fo.write('\nLabels:'+str(metadetails[1]))
fo.write('\nLocations:'+str(metadetails[2])+'\n\n')
fo.close()
labels=metadetails[1]
labels=labels.replace('\n','').split(',')

for md in metadetails:
	if fname in md:
		md=md.split('\t')
		features=md[2].replace('\n','').split(',')

for sb in features:
	  globals()[sb]=symbols(sb)
	  
clauses=pd.read_csv(clausename,sep='\t')
clauses=clauses.fillna(' ')

for label in labels:
	print('Processing:',label)
	fo=open(outname,'a')
	fo.write('Label:'+str(label))
	fo.close()
	
	sub_p = clauses.loc[(clauses['class']==label) & (clauses['p/n']=='positive')]

	sub_n = clauses.loc[(clauses['class']==label) & (clauses['p/n']=='negative')]


	long_string_pos=''
	for ind,row in sub_p.iterrows():
		cl=row['Clause']
		cl=cl.replace('#','~').replace('isParent(','f_').replace(')','')
		cl=cl[:-1].replace(';',' & ')
		cl=cl.replace('\n','')
		if len(cl)>0:
			if long_string_pos=='':
				long_string_pos='('+cl+')'
			else:
				long_string_pos+=' | '+'('+cl+')'
			
	long_string_neg=''
	for ind,row in sub_n.iterrows():
		cl=row['Clause']
		cl=cl.replace('#','~')
		cl=cl[:-1].replace(';',' & ')
		cl=cl.replace('\n','')
		if len(cl)>0:
			if long_string_neg=='':
				long_string_neg='('+cl+')'
			else:
				long_string_neg+=' | '+'~('+cl+')'
			
	long_string_pos_exp=parse_expr(long_string_pos)

	long_string_pos_exp=simplify(long_string_pos_exp)


	long_string_neg_exp=parse_expr(long_string_neg)

	long_string_neg_exp=simplify(long_string_neg_exp)

	long_string_combo='('+long_string_pos+')'+' & ('+long_string_neg+')'

	long_string_combo_exp=parse_expr(long_string_combo)

	long_string_combo_exp=simplify(long_string_combo_exp)

	fo=open(outname,'a')
	fo.write('\nPositive Clauses:'+str(long_string_pos_exp))
	fo.write('\nNegative Clauses:'+str(long_string_neg_exp))
	fo.write('\nCombined Clauses:'+str(long_string_combo_exp)+'\n\n')
	fo.close()
	#print(to_cnf(long_string_pos))
print('Output written to:'+outname)
