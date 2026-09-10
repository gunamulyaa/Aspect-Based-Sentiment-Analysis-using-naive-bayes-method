import pandas as pd
from pathlib import Path

base = Path(r'C:\Users\gunam\Downloads\Project\Aspect-Based-Sentiment-Analysis-using-naive-bayes-method\Dataset')

for p in sorted(base.glob('*.xlsx')):
    print(f'FILE: {p.name}')
    try:
        df = pd.read_excel(p, header=None)
        print('shape=', df.shape)
        # show first 15 non-empty rows without giant output
        for i in range(min(12, len(df))):
            row = df.iloc[i].dropna().tolist()
            if row:
                print(f'row{i}: {row[:18]}')
        # print rows that look like headers
        print('possible header rows:')
        for i in range(min(25, len(df))):
            row = df.iloc[i].dropna().astype(str).str.lower().tolist()
            text = ' '.join(row)
            if any(k in text for k in ['comment', 'text', 'username', 'date', 'time', 'content', 'post', 'created', 'user']):
                print(' header candidate row', i, row[:20])
        print('---')
    except Exception as e:
        print('ERROR', e)
        print('---')
