import pandas as pd
import matplotlib.pyplot as plt
fname=sys.argv[1]

df_transformed = pd.read_pickle(fname.replace('.txt','')+'_transformed_relationalfeatureset.pkl')

df_transformed_X= df_transformed.drop(columns=['ANSWER'])

df_transformed_X = df_transformed_X.astype('uint8')

correl = df_transformed_X.corr()

plt.matshow(correl)
plt.savefig('plot.png')
