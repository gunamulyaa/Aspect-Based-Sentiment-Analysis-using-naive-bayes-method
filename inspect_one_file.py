import pandas as pd
from pathlib import Path

base = Path(r'C:\Users\gunam\Downloads\Project\Aspect-Based-Sentiment-Analysis-using-naive-bayes-method\Dataset')
for name in ['Raw_Data.xlsx', 'Comments_1787980995.69729.xlsx']:
    p = base / name
    print(f'=== {name} ===')
    df = pd.read_excel(p, header=None)
    print('shape:', df.shape)
    for i in range(min(15, len(df))):
        row = df.iloc[i].tolist()
        print(i, row[:12])
    print()
