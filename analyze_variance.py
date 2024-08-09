from analyze import load_df
import pandas as pd
import glob
from scipy.stats import chi2_contingency, mannwhitneyu
import statsmodels.api as sm
import statsmodels.formula.api as smf
import seaborn as sns
import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import matplotlib.colors as mcolors
import numpy as np
from statsmodels.miscmodels.ordinal_model import OrderedModel

from scipy.stats import chi2_contingency
from scipy.stats import fisher_exact


def inter_group_var(df):

    variance_df = df.groupby(['participant_id', 'Method', 'ethnicity'])['quality_ordinal'].var().reset_index()
    
    fig, ax = plt.subplots(figsize=(10, 6))  # Adjusted size for better visibility
    sns.boxplot(data=variance_df, x='ethnicity', y='quality_ordinal', hue='Method', ax=ax)

    # Adding customizations
    ax.set_title('Distribution of Quality Ordinal Variance by Ethnicity and Method')
    ax.set_xlabel('Ethnicity')
    ax.set_ylabel('Variance of Quality Ordinal')
    plt.legend(title='Method')
    plt.savefig('results/inter_group_variance3.png')

    return

    mean_method = variance_df.groupby(['ethnicity', 'Method'])['quality_ordinal'].mean().reset_index()

    fig, ax = plt.subplots(figsize=(6, 4))

    # Customizations
    sns.barplot(data=mean_method, x='ethnicity', y='quality_ordinal', hue='Method', ax=ax, palette='viridis')
    ax.set_title('Average participant variance for \'Quality\'')
    ax.set_xlabel('Ethnicity')
    ax.set_ylabel('Variance (Quality)')
    plt.legend()  
    plt.tight_layout()
    plt.savefig('results/inter_group_variance2.png')
    
    mean_method.to_latex('mean_participant_variance.tex')

    return

def inter_image_var(df):

    variance_df = df.groupby(['identifier', 'Method', 'ethnicity'])['quality_ordinal'].var().reset_index()

    mean_method = variance_df.groupby(['ethnicity', 'Method'])['quality_ordinal'].mean().reset_index()

    fig, ax = plt.subplots(figsize=(6, 4))

    # Customizations
    sns.barplot(data=mean_method, x='ethnicity', y='quality_ordinal', hue='Method', ax=ax, palette='viridis')
    ax.set_title('Average image variance for \'Quality\'')
    ax.set_xlabel('Ethnicity')
    ax.set_ylabel('Variance (Quality)')
    plt.legend()
    plt.tight_layout()  
    plt.savefig('inter_image_variance2.png')
    
    mean_method.to_latex('mean_image_variance.tex')

    return

def statistics(df):
    model = smf.mnlogit('quality_ordinal ~ C(ethnicity, Treatment(reference="caucasian")) + C(Method,Treatment(reference="normal")) + C(gender_collapsed, Treatment(reference="female"))', data=df)
    result = model.fit()
    print(result.summary())

    with open('model_output.tex', 'w') as f:
        f.write(result.summary().as_latex())


def statistics(df):
    model = smf.mnlogit('method ~ C(quality_ordinal) + C(ethnicity, Treatment(reference="caucasian")) + C(gender_collapsed, Treatment(reference="female"))', data=df)
    result = model.fit()
    print(result.summary())

    with open('model_output_method.tex', 'w') as f:
        f.write(result.summary().as_latex())

# Define a function to perform the chi-square test
def test_association(df, col1, col2):
    contingency_table = pd.crosstab(df[col1], df[col2])
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    print(f'Chi-square test between {col1} and {col2}')
    print(f'Chi-square Statistic: {chi2}, p-value: {p}\n')

def beautify_label(label):
    # Beautify and simplify labels for the LaTeX output
    label = label.replace('\\textbf{C(ethnicity, Treatment(reference="caucasian"))[T.', 'Ethnicity: ')
    label = label.replace('\\textbf{C(Method, Treatment(reference="normal"))[T.', 'Method: ')
    label = label.replace('\\textbf{C(gender_collapsed, Treatment(reference="female"))[T.', 'Gender: ')
    label = label.replace(']}', '')
    label = label.replace('}', '')  # Remove remaining closing braces
    return label

def process_latex_lines(lines):
    processed_lines = []
    for line in lines:
        if 'P>|z|' in line:  # Ensure the line contains a p-value field
            parts = line.split('&')
            if len(parts) > 5 and '}' in parts[4]:  # Additional check for numeric p-value field
                try:
                    p_value = float(parts[4].strip().replace('\\textbf{', '').replace('}', ''))
                    if p_value < 0.05:
                        parts[4] = '\\textbf{' + parts[4].strip() + '}'  # Make p-value bold if significant
                except ValueError:
                    print("Non-numeric p-value encountered, skipping formatting for this line.")
                parts[0] = beautify_label(parts[0])  # Beautify the variable labels
                line = '&'.join(parts)
        processed_lines.append(line)
    return processed_lines


def statistics_v2(df):
    model = smf.mnlogit('quality_ordinal ~ C(ethnicity, Treatment(reference="caucasian")) + C(Method, Treatment(reference="normal")) + C(gender_collapsed, Treatment(reference="female"))', data=df)
    result = model.fit()
    
    # Extract and process LaTeX summary
    summary_latex = result.summary().as_latex()
    lines = summary_latex.split('\\n')
    processed_lines = process_latex_lines(lines)
    
    new_summary_str = '\\n'.join(processed_lines)
    
    # Write to LaTeX file
    with open('model_output_v2.tex', 'w') as f:
        f.write(new_summary_str)

def ordered_model(df):
    # Fitting the ordinal logistic regression
    df['quality_ordinal'] = df['quality_ordinal'].astype(int)
    df_test_columns = ["ethnicity",""]
    print(np.asarray(df['quality_ordinal']))
    mod = OrderedModel(df['quality_ordinal'], df.drop('quality_ordinal', axis=1), distr='logit')
    res = mod.fit(method='bfgs')
    print(res.summary())

def all_hist(df):
    occupations = df["occupation"].unique()
    n_row = 2 # Adjust as necessary for your data
    n_col = 5  # Adjust as necessary for your data
    fig, axs = plt.subplots(n_row, n_col, figsize=(10, 5), dpi=250, sharey=True)
    
    # reduce white space

    sns.set_theme(style="whitegrid")

    # Iterate over each subplot and fill with bar plots
    for i, occupation in enumerate(occupations):
        ax = axs[i // n_col, i % n_col]  # Determine the subplot to use
        # Filter the data for the specific occupation
        data = df[df['occupation'] == occupation]
        
        # Create the bar plot
        sns.countplot(data=data, x='quality', hue='Method', ax=ax,
                order=["poor", "bad", "fair", "good", "excellent"], hue_order=["normal","debias"])
        if i != 0:
            # Remove the legend from all other subplots
            ax.legend_.remove()
        ax.set_title("")
        ax.text(0.90, 0.95, occupation, transform=ax.transAxes, 
                horizontalalignment='right', verticalalignment='top', 
                fontsize=12)
        ax.set_xlabel('')  # Clear the x-axis label
        ax.set_ylabel('Count')  # Set the y-axis label

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.05, hspace=0.3)

    plt.savefig(f"results/quality_ordinal_seperate_new.png") 

def test_occ(df, effect, columns, occupation,method="chi"):
    # Filter DataFrame for the specific occupation
    df_filtered = df[df['occupation'] == occupation]
    
    results = []

    results_fish = []

    # Perform Chi-square test for each variable
    for variable in columns:
        contingency_table = pd.crosstab(df_filtered[variable], df_filtered[effect]).fillna(0)
        print(contingency_table)
        # Check if the contingency table is large enough
        if contingency_table.shape[0] < 2 or contingency_table.shape[1] < 2:
            results.append({
                'Occupation': occupation,
                'Variable': variable,
                'Chi-squared Statistic': None,
                'Degrees of Freedom': None,
                'P-value': 'Not enough data',
                'Expected Frequencies': None
            })
            continue
        chi2, p_chi, dof, expected = None, None, None, None
        odds_ratio, p_fish = None, None

        # Performing the Chi-square test
        if method == "chi":
            chi2, p_chi, dof, expected = chi2_contingency(contingency_table)
        elif method == "fish":
            odds_ratio, p_fish = fisher_exact(contingency_table)
            dof = 1
            expected = None

        results_fish.append({
            'Occupation': occupation,
            'Variable': variable,
            'Odds Ratio': odds_ratio,
            'P-value': p_fish
        })
        
        # Append results
        results.append({
            'Occupation': occupation,
            'Variable': variable,
            'Chi-squared Statistic': chi2,
            'Degrees of Freedom': dof,
            'P-value': p_chi,
            'Expected Frequencies': expected
        })
    
    return results, results_fish


def test_all(df, effect, columns, method="chi"):
    columns = columns.append(effect)

    # df_reduced = df[['ethnicity', 'gender', 'quality', 'Method', 'occupation', 'sample']]
    # df_reduced = df
    
    all_results = []
    unique_occupations = df['occupation'].unique()
    for occupation in unique_occupations:
        results, results_fish = test_occ(df, effect, columns, occupation,method)
        if method == "fish":
            all_results.extend(results_fish)
        elif method == "chi":
            all_results.extend(results)


    results_df = pd.DataFrame(all_results)
    
    # dont save degrees of freedom, expected frequencies
    if method == "chi":
        results_df_smaller = results_df.drop(columns=['Degrees of Freedom', 'Expected Frequencies'])

    results_df_smaller = results_df_smaller.groupby(['Occupation', 'Variable']).first()
    results_df_smaller.to_csv(f'results/{effect}_{method}_small_new.csv', index=False)
    results_df_smaller.to_latex(f'results/{effect}_{method}_small_new.tex')                                          

    results_df.to_csv(f'results/{effect}_{method}_big_new.csv', index=False)
    results_df.to_latex(f'results/{effect}_{method}_big_new.tex', index=False)

if __name__ == '__main__':
    
    df = load_df(overwrite=False)

    # effect = "sample"
    # columns = ["ethnicity", "gender", "quality"]

    # # test_all(df)

    # test_all(df, "sample", columns, method="fish")

    #drop unintelligble
    df = df.drop(df[df['gender'] == 'unintelligible'].index)
    df = df.drop(df[df['ethnicity'] == 'unintelligible'].index)

    inter_group_var(df)
    inter_image_var(df)



    # all_hist(df)
    # all_hist(df)

    # inter_group_var(df)
    # inter_image_var(df)
    # statistics_v2(df)
    # ordered_model(df)
    # bar plot of quality ordinal for normal and debias

    # plt.show()

    # Testing the effect of 'method' on 'quality', 'ethnicity', and 'gender'
    # test_association(df, 'Method', 'quality')
    # test_association(df, 'Method', 'ethnicity')
    # test_association(df, 'Method', 'gender_collapsed')
