import pandas as pd

filename = 'resources/taggers/example-mc-grammar-no-downsample-all-five-dev-explicit-gpu/ptest.tsv'

df = pd.read_csv(filename, header=None, delim_whitespace=True)

df.values

df2 = df.drop([1], axis=1)

# back to tsv
df2.to_csv('out.tsv', sep='\t', index=False)
