import pandas as pd
import matplotlib.pyplot as plt
fname=sys.argv[1]
oname=sys.argv[2]
#fname = '../generated/generated2.txt'
df_transformed = pd.read_pickle(fname.replace('.txt','')+'_transformed_relationalfeatureset.pkl')

df_transformed_X= df_transformed.drop(columns=['ANSWER'])

df_transformed_X = df_transformed_X.astype('uint8')

correl = df_transformed_X.corr()

#plt.matshow(correl)
#plt.savefig('plot2.png')


#############
import seaborn as sns
sns.set(font_scale =0.45)

sp = sns.heatmap(correl, xticklabels=correl.columns, yticklabels=correl.columns, cmap="coolwarm", linewidths=0.5, linecolor='black')
bottom, top = sp.get_ylim()
sp.set_ylim(bottom + 0.5, top - 0.5)



fig = sp.get_figure()
fig.savefig(oname)
