import pandas as pd
from pathlib import Path

base = Path(r'C:\Users\gunam\Downloads\Project\Aspect-Based-Sentiment-Analysis-using-naive-bayes-method\Dataset')
files = [base/'Raw_Data.xlsx', base/'Comments_1787980995.69729.xlsx']
for f in files:
    print(f'=== {f.name} ===')
    xls = pd.ExcelFile(f)
    print('sheets=', xls.sheet_names)
    for s in xls.sheet_names:
        df = pd.read_excel(f, sheet_name=s, header=None)
        print('sheet', s, 'shape', df.shape)
        print(df.head(10).to_string(index=False, header=False))
        print('---')
