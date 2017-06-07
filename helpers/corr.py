import glob
import pandas as pd
from itertools import combinations
predPath = "predictions/"
models = []
for file_name in glob.glob(predPath + '*.csv'):
    df = pd.read_csv(file_name, encoding = "utf-8")

    df = df.ix[:, :1]
    df = df.rename(index=str,
                   columns = {'is_duplicate': file_name[19:]})    
    models.append(df)

order = combinations(models, 2)
for model1, model2 in order:
    print("----------------------------------------")
    print(model1.expanding(min_periods=1).corr(pairwise=True,other=model2).iloc[-1, :, :])
    
