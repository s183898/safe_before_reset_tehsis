from analyze import load_df

df = load_df(overwrite=True)

print(len(df['image'].unique()))
    