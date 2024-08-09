import pandas as pd
import glob
from scipy.stats import chi2_contingency, mannwhitneyu
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
import matplotlib.image as mpimg
from scipy.spatial import distance

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.metrics import structural_similarity as ssim
import cv2
from scipy.spatial import distance

import seaborn as sns
import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import matplotlib.colors as mcolors
import numpy as np


def load_df(overwrite=True):
    # Load and combine data
    if not overwrite and os.path.exists("exp_data.csv") and os.path.exists("exp_data_grouped.csv"):
        df = pd.read_csv("exp_data.csv")
        return df
    
    path = 'experiment'
    all_files = glob.glob(os.path.join(path, "**", "image_labels.csv"), recursive=True)
    
    df_list = []
    
    for f in all_files:
        # Extract participant_id from the path
        participant_id = os.path.basename(os.path.dirname(f))
        
        # Read the CSV file, skipping the first 6 rows
        temp_df = pd.read_csv(f, skiprows=range(1, 7))
        
        # Add the participant_id as a new column in the dataframe
        temp_df['participant_id'] = participant_id
        
        # Append the dataframe to the list
        df_list.append(temp_df)
    
    # Concatenate all dataframes in the list
    df = pd.concat(df_list, ignore_index=True)

    print(len(df['image'].unique()))

    # Extract labeling method from the 'image' filename
    df['Method'] = df['image'].apply(lambda x: 'normal' if 'none_normal' in x.lower() else 'debias').astype('category')
    df['sample'] = df['image'].apply(lambda x: x.split("_")[-1].replace(".png", "") if x is not np.nan else np.nan)
    df['prompt'] = df['image'].str.extract(r'(.*?)(?:afro-american|None)')
    # make a nice prompt name by removing underscores and removing "a_photo_of_the_face_of_a_"
    df['occupation'] = df['prompt'].str.replace("a_photo_of_the_face_of_a_", "").str.replace("_", " ").str.title()
    
    # Combine prompt, sample, and Method into a unique identifier
    df['identifier'] = df['occupation'] + '_' + df['sample'].astype(str) + '_' + df['Method'].astype(str)
    
    quality_mapping = {'poor': 1, 'bad': 2, 'fair': 3, 'good': 4, 'excellent': 5}
    df['quality_ordinal'] = df['quality'].map(quality_mapping)
    
    gender_categories = {1: 'female', 2: 'female', 3: 'neutral', 4: 'male', 5: 'male'}
    df['gender_collapsed'] = df['gender'].map(gender_categories)

    # df = df[df["gender"]!="unintelligible"]
    df[df['gender'] == 'unintelligible'] = np.nan
    df['gender'] = pd.to_numeric(df['gender'])

    print("before",len(df['image'].unique()))
    
    # get index of nan
    nan_index = df.isna().any(axis=1)
    print(nan_index)
    df = df.dropna()
    

    df.to_csv("exp_data.csv", index=False)

    return df

def pivot_table(df, to_latex=False):
    # Create the pivot table
    pivot = df.pivot_table(index='occupation', columns='Method', values=['gender', 'quality_ordinal'])
    
    # Create the new order for the columns
    new_order = [('gender', 'normal'), ('gender', 'debias'), ('quality_ordinal', 'normal'), ('quality_ordinal', 'debias')]
    
    # Reorder the columns in the DataFrame
    pivot = pivot.reorder_levels([0, 1], axis=1)
    pivot = pivot.sort_index(axis=1, level=0)
    pivot = pivot.loc[:, new_order]

    if to_latex:
        pivot.to_latex("pivot_table.tex")

    return pivot


def plot1(df, occs, ax):

    fig, ax = plt.subplots(figsize=(10, 5), dpi=250)

    occ_before = df[df['Method'] == 'normal']
    occ_after = df[df['Method'] == 'debias']

    for i, occ in enumerate(occs):
        
        y_pos = i
        occ_before = df[(df['occupation'] == occ) & (df['Method'] == 'normal')]
        occ_after = df[(df['occupation'] == occ) & (df['Method'] == 'debias')]

        # CI = 1.96
        occ_before_mean = occ_before['quality_ordinal'].mean()
        occ_after_mean = occ_after['quality_ordinal'].mean()
        # occ_before_std = occ_before['quality_ordinal'].std()
        # occ_after_std = occ_after['quality_ordinal'].std()
        # occ_before_count = occ_before['quality_ordinal'].count()
        # occ_after_count = occ_after['quality_ordinal'].count()

        # z = (occ_after_mean - occ_before_mean) / np.sqrt((occ_before_std**2 / occ_before_count) + (occ_after_std**2 / occ_after_count))
        # p = 1 - stats.norm.cdf(z)
        # # err bars
        # ax.errorbar(occ_before_mean, y_pos+0.4, xerr=CI*occ_before_std/np.sqrt(occ_before_count), fmt='o', color='black', capsize=4)
        # ax.errorbar(occ_after_mean, y_pos, xerr=CI*occ_after_std/np.sqrt(occ_after_count), fmt='o', color='purple', capsize=4)
        
        color = 'green' if occ_after_mean > occ_before_mean else 'red'
        # arrow from mean to mean

        width = max(0.05, abs(occ_after_mean - occ_before_mean) / 10)
        head_length = max(0.05, width)
        linewidth = 0.3
        ax.arrow(occ_before_mean, y_pos, occ_after_mean - occ_before_mean, 0, width=width, head_width=0.5, head_length=head_length, fc=color, ec=color, length_includes_head=True)
        ax.vlines(occ_before_mean, y_pos- linewidth, y_pos + linewidth, color='black', lw=2)  # at the base
    
    linewidth = 0.2

    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    
    ax.vlines(-100, -100-0.05, -100+0.05 + linewidth, color='black', lw=2, label = "before debias")  # at the base

    # ax.errorbar(-1000, -1000, 1,1, fmt='o', color='black', capsize=4, label = 'Normal 95% CI')
    # ax.errorbar(-1000, -1000, 1,1, fmt='o', color='purple', capsize=4, label = 'Debias 95% CI')

    ax.arrow(0,0,0,0, color='green', label='Quality Improvement')
    ax.arrow(0,0,0,0, color='red', label='Quality Degradation')

    # improvement_handle = Line2D([0], [0], color='green', marker='>', markersize=10, label='Quality Improvement')
    # degradation_handle = Line2D([0], [0], color='red', marker='<', markersize=10, label='Quality Degradation')

    # set x lim ylim
    ax.set_ylim(ylim[0], ylim[1])

    ax.set_yticks([i for i in range(len(occs))])
    ax.set_yticklabels(occs)
    # ax.set_xlim(2, 5)
    ax.set_xticks([2, 3, 4, 5])
    ax.set_xticklabels(["2\nBad", "3\nFair", "4\nGood", "5\nExcellent"])
    ax.set_xlim(0,xlim[1]+0.5)

    ax.set_title(f"", fontsize=16)
    ax.set_xlabel("Quality")
    ax.set_ylabel("Occupation")
    ax.set_yticklabels([occ.lower() for occ in occs], fontsize=14)

    ax.legend(ncol=1)
    labelsize_x = 12
    labelsize_y = 12

    ax.set_xlim(2, 5)
    ax.tick_params(axis='x', which='major', labelsize=labelsize_x)
    ax.tick_params(axis='y', which='major', labelsize=labelsize_y)

    # current_ylim = ax.get_ylim()  # Get the current limits
    # ax.set_ylim(current_ylim[0], current_ylim[1] * 1.5)  # Increase the upper limit by 10%

    plt.tight_layout()
    plt.savefig("results/quality_plot_1_new.png")
    plt.close()


def plot2(df, occs, ax):

    fig, ax = plt.subplots(figsize=(10, 5), dpi=250)

    occ_before = df[df['Method'] == 'normal']
    occ_after = df[df['Method'] == 'debias']

    for i,occ in enumerate(occs):
        i = i*2
        y_pos = i
        occ_before = df[(df['occupation'] == occ) & (df['Method'] == 'normal')]
        occ_after = df[(df['occupation'] == occ) & (df['Method'] == 'debias')]
        
        black_ratio_before = occ_before["ethnicity"].value_counts(normalize=True).get("black", 0)
        black_ratio_after = occ_after["ethnicity"].value_counts(normalize=True).get("black", 0)
    
        # scatter if same
        if black_ratio_before == black_ratio_after:
            ax.scatter([black_ratio_before, black_ratio_after], [y_pos, y_pos], color='blue')
        else:
            color = 'green' if abs(0.5-black_ratio_before) > abs(0.5-black_ratio_after) else 'red'
            # arrow from mean to mean

            width = max(0.15,0)
            head_width = 0.6
            head_length = min(0.02, width)
            ax.arrow(black_ratio_before, y_pos, black_ratio_after - black_ratio_before, 0, width=width, head_width=head_width, head_length=head_length, fc=color, ec=color, length_includes_head=True)
            # line = Line2D([black_ratio_before, black_ratio_after], [y_pos, y_pos], color=color, marker='>', markersize=10)
            ax.scatter([black_ratio_before], [y_pos], color="black", s=50)
            ax.scatter([black_ratio_after], [y_pos], color="purple", s=50)
            # ax.add_line(line)

    # line = Line2D([black_ratio_before, black_ratio_after], [y_pos, y_pos], color="green", marker='>', markersize=10, label='closer to Independence')
    # ax.add_line(line)
    # line = Line2D([black_ratio_before, black_ratio_after], [y_pos, y_pos], color="red", marker='>', markersize=10, label='Further from Independence')
    # ax.add_line(line)
            
    ax.arrow(-100, -100, black_ratio_after - black_ratio_before, 0, width=width, head_width=head_width, head_length=head_length, fc="green", ec="green", length_includes_head=True, label='Closer to 50%')
    ax.arrow(-100, -100, black_ratio_after - black_ratio_before, 0, width=width, head_width=head_width, head_length=head_length, fc="red", ec="red", length_includes_head=True, label='Closer to 50%')

    ax.scatter([black_ratio_before], [y_pos], color="black", s=50, label = 'Normal')
    ax.scatter([black_ratio_after], [y_pos], color="purple", s=50, label = 'Debias')

    ax.scatter(0,-10, color='blue', label = 'Stationary')

    # line at 50%
    ax.axvline(x=0.5, color='black', linestyle='--', linewidth=0.5)
    ax.set_xlabel("Percentage of 'Black'")
    ax.set_xlim(-0.05, 1)
    ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    ax.set_xticklabels(['0%', '10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%', '100%'])
    ax.set_ylim(-0.5, len(occs)*2)
    ax.set_ylabel("Occupation")
    ax.set_yticks([i*2 for i in range(len(occs))])
    ax.set_yticklabels([occ.lower() for occ in occs], fontsize=12)
    ax.set_title(f"", fontsize=16)
    ax.legend()
    # ax.legend(ncol=2)

    plt.tight_layout()
    plt.savefig("results/quality_plot_2.png")
    plt.close()



def plot3(df, occs, ax):

    fig, ax = plt.subplots(figsize=(12, 6), dpi=250)

    occ_before = df[df['Method'] == 'normal']
    occ_after = df[df['Method'] == 'debias']

    for i,occ in enumerate(occs):
        y_pos = i*2
        occ_before = df[(df['occupation'] == occ) & (df['Method'] == 'normal')]
        occ_after = df[(df['occupation'] == occ) & (df['Method'] == 'debias')]
        
        CI = 1.96
        occ_before_mean = occ_before['gender'].mean()
        occ_after_mean = occ_after['gender'].mean()
        occ_before_std = occ_before['gender'].std()
        occ_after_std = occ_after['gender'].std()
        occ_before_count = occ_before['gender'].count()
        occ_after_count = occ_after['gender'].count()

        z = (occ_after_mean - occ_before_mean) / np.sqrt((occ_before_std**2 / occ_before_count) + (occ_after_std**2 / occ_after_count))
        p = 1 - stats.norm.cdf(z)
        # err bars
        ax.errorbar(occ_before_mean, y_pos + 0.4, xerr=CI*occ_before_std/np.sqrt(occ_before_count), fmt='o', color='black', capsize=4)
        ax.errorbar(occ_after_mean, y_pos , xerr=CI*occ_after_std/np.sqrt(occ_after_count), fmt='o', color='purple', capsize=4)
        
        color = 'green' if abs(occ_after_mean-3) < abs(occ_before_mean-3) else 'red'
        # arrow from mean to mean

        width = max(0.05, abs(occ_after_mean - occ_before_mean) / 10)
        head_length = max(0.05, width)

        ax.arrow(occ_before_mean, y_pos-0.4, occ_after_mean - occ_before_mean, 0, width=width, head_width=0.5, head_length=head_length, fc=color, ec=color, length_includes_head=True)
   
    ax.arrow(0,0,0,0, color='green', label='closer to 50%')
    ax.arrow(0,0,0,0, color='red', label='Further from 50%')

    ax.set_yticks([i*2 for i in range(len(occs))])
    ax.set_yticklabels(occs)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xticklabels(["1", "2", "3", "4", "5"])
    ax.errorbar(-100,-100,0,0, fmt='o', color='black', capsize=4, label = 'Normal 95% CI')
    ax.errorbar(-100,-100,0,0, fmt='o', color='purple', capsize=4, label = 'Debias 95% CI')
    
    ax.set_ylim(-1, len(occs)*2)
    ax.set_xlim(0.7, 5.3)

    ax.axvline(x=3, color='black', linestyle='--', linewidth=0.5, label = '(Neutral)')

    ax.set_yticklabels([occ.lower() for occ in occs], fontsize=14)
    ax.set_title(f"", fontsize=16)
    ax.set_xlabel("Gender appearance")
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xticklabels(['1\nFeminine', '2','3','4','5\nMasculine'])
    
    ax.legend()

    plt.tight_layout()
    plt.savefig("results/quality_plot_3.png")
    plt.close()


def plot4(df, real_distr, ax):

    occs = list(real_distr.keys())
    
    fig, ax = plt.subplots(figsize=(10, 5), dpi=250)

    occ_before = df[df['Method'] == 'normal']
    occ_after = df[df['Method'] == 'debias']

    for i, occ in enumerate(occs):
        i = i*2
        y_pos = i
        occ_before = df[(df['occupation'] == occ) & (df['Method'] == 'normal')]
        occ_after = df[(df['occupation'] == occ) & (df['Method'] == 'debias')]
        
        black_ratio_before = occ_before["ethnicity"].value_counts(normalize=True).get("black", 0)
        black_ratio_after = occ_after["ethnicity"].value_counts(normalize=True).get("black", 0)
        
        white_ratio_before = occ_before["ethnicity"].value_counts(normalize=True).get("caucasian", 0)
        white_ratio_after = occ_after["ethnicity"].value_counts(normalize=True).get("caucasian", 0)

        black_ratio_before = black_ratio_before / (black_ratio_before+ white_ratio_before) if (black_ratio_before+ white_ratio_before)  != 0 else 0.5
        black_ratio_after = black_ratio_after / (black_ratio_after+ white_ratio_after) if (black_ratio_after+ white_ratio_after) != 0 else 0.5

        print(black_ratio_before, black_ratio_after)

        # black_ratio_before += occ_before["ethnicity"].value_counts(normalize=True).get("other", 0)
        # black_ratio_after += occ_after["ethnicity"].value_counts(normalize=True).get("other", 0)

        real = 0.5

        # scatter if same
        if black_ratio_before == black_ratio_after:
            ax.scatter([black_ratio_before, black_ratio_after], [y_pos, y_pos], color='blue')
        else:
            if real < 0:
                color = 'grey'
            else:   
                color = 'green' if abs(real-black_ratio_before) > abs(real-black_ratio_after) else 'red'

            width = max(0.15,0)
            head_width = 0.6
            head_length = min(0.02, width)
            ax.arrow(black_ratio_before, y_pos, black_ratio_after - black_ratio_before, 0, width=width, head_width=head_width, head_length=head_length, fc=color, ec=color, length_includes_head=True)
            # ax.scatter([black_ratio_before], [y_pos], color="black", s=50)
            # ax.scatter([black_ratio_after], [y_pos], color="purple", s=50)
        # make scatter with cross
        # ax.scatter([real], [y_pos], color="darkorange", s=75, marker="x")
        line_width = 0.02
        ax.vlines(black_ratio_before, y_pos-0.5, y_pos+0.5 + line_width, color='black', lw=2)  # at the base

    ax.vlines(-100, -100+0.25-0.5, -100+0.25+0.5 + line_width, color='black', lw=2, label = "Before debias")  # at the base

    # ax.scatter([-100], [-100], color="darkorange", s=75,  marker="x", label = 'Real distribution')

    ax.arrow(-100, -100, black_ratio_after - black_ratio_before, 0, width=width, head_width=head_width, head_length=head_length, fc="green", ec="green", length_includes_head=True, label = 'Closer to 50\%')
    ax.arrow(-100, -100, black_ratio_after - black_ratio_before, 0, width=width, head_width=head_width, head_length=head_length, fc="red", ec="red", length_includes_head=True, label='Further from 50\%')

    # vert line at 50%
    ax.vlines(0.5, -0.5, len(occs)*2, color='black', linestyle='--', linewidth=0.5, label = '50%')

    # ax.scatter([-100], [-100], color="black", s=50, label = 'Normal')
    # ax.scatter([-100], [-100], color="purple", s=50, label = 'Debias')

    # ax.scatter(-100,-10, color='blue', label = 'Stationary')

    # line at 50%
    # ax.axvline(x=0.s5, color='black', linestyle='--', linewidth=0.5)
    ax.set_xlabel("Percentage of Black and White")
    ax.set_xlabel("")
    ax.set_xlim(-0.05, 1)
    ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    ax.set_xticklabels(['0%\nAll Caucasian', '10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%', '100%\nAll Black'])
    ax.set_ylim(-0.5, len(occs)*2)
    ax.set_ylabel("Occupation")
    ax.set_yticks([i*2 for i in range(len(occs))])
    ax.set_yticklabels([occ.lower() for occ in occs], fontsize=12)
    ax.set_title(f"", fontsize=16)
    ax.legend()
    # ax.legend(ncol=2)

    plt.tight_layout()
    plt.savefig("results/quality_plot_4_ethnicity_new_black.png")
    plt.close()


def plot7(df, real_distr, ax):

    occs = list(real_distr.keys())
    
    fig, ax = plt.subplots(figsize=(10, 5), dpi=250)

    occ_before = df[df['Method'] == 'normal']
    occ_after = df[df['Method'] == 'debias']

    for i, occ in enumerate(occs):
        i = i*2
        y_pos = i
        occ_before = df[(df['occupation'] == occ) & (df['Method'] == 'normal')]
        occ_after = df[(df['occupation'] == occ) & (df['Method'] == 'debias')]
        
        female_ratio_before = occ_before["gender_collapsed"].value_counts(normalize=True).get("female", 0)
        female_ratio_after = occ_after["gender_collapsed"].value_counts(normalize=True).get("female", 0)
      
        female_ratio_before += occ_before["gender_collapsed"].value_counts(normalize=True).get("neutral", 0)
        female_ratio_after += occ_after["gender_collapsed"].value_counts(normalize=True).get("neutral", 0)

        real = real_distr[occ]
        # scatter if same
        if female_ratio_before == female_ratio_after:
            ax.scatter([female_ratio_before, female_ratio_after], [y_pos, y_pos], color='blue')
        else:
            if real < 0:
                color = 'grey'
            else:
                color = 'green' if abs(real-female_ratio_before) > abs(real-female_ratio_after) else 'red'

            width = max(0.15,0)
            head_width = 0.6
            head_length = min(0.02, width)
            ax.arrow(female_ratio_before, y_pos, female_ratio_after - female_ratio_before, 0, width=width, head_width=head_width, head_length=head_length, fc=color, ec=color, length_includes_head=True)
            # ax.scatter([black_ratio_before], [y_pos], color="black", s=50)
            # ax.scatter([black_ratio_after], [y_pos], color="purple", s=50)
        # make scatter with cross
        ax.scatter([real], [y_pos], color="darkorange", s=75, marker="x")
        line_width = 0.02
        ax.vlines(female_ratio_before, y_pos-0.5, y_pos+0.5 + line_width, color='black', lw=2)  # at the base

    ax.vlines(-100, -100+0.25-0.5, -100+0.25+0.5 + line_width, color='black', lw=2, label = "Before debias")  # at the base

    ax.scatter([-100], [-100], color="darkorange", s=75,  marker="x", label = 'Real distribution')

    ax.arrow(-100, -100, female_ratio_after - female_ratio_before, 0, width=width, head_width=head_width, head_length=head_length, fc="green", ec="green", length_includes_head=True, label = 'Closer to real distribution')
    ax.arrow(-100, -100, female_ratio_after - female_ratio_before, 0, width=width, head_width=head_width, head_length=head_length, fc="red", ec="red", length_includes_head=True, label='Further from real distribution')

    # ax.scatter([-100], [-100], color="black", s=50, label = 'Normal')
    # ax.scatter([-100], [-100], color="purple", s=50, label = 'Debias')

    ax.scatter(-100,-10, color='blue', label = 'Stationary')

    # line at 50%
    # ax.axvline(x=0.s5, color='black', linestyle='--', linewidth=0.5)
    ax.set_xlabel("Percentage of non-male")
    ax.set_xlabel("")
    ax.set_xlim(-0.05, 1.05)
    ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    ax.set_xticklabels(['0%\nNo non-male', '10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%', '100%\nAll non-male'])
    ax.set_ylim(-0.5, len(occs)*2)
    ax.set_ylabel("Occupation")
    ax.set_yticks([i*2 for i in range(len(occs))])
    ax.set_yticklabels([occ.lower() for occ in occs], fontsize=12)
    ax.set_title(f"", fontsize=16)
    ax.legend()
    # ax.legend(ncol=2)

    plt.tight_layout()
    plt.savefig("results/quality_plot_7_gender_non_male.png")
    plt.close()



def plot5(df, occs, ax):

    fig, ax = plt.subplots(figsize=(10, 5), dpi=250)

    occ_before = df[df['Method'] == 'normal']
    occ_after = df[df['Method'] == 'debias']

    for i, occ in enumerate(occs):
        i = i*2
        y_pos = i
        occ_before = df[(df['occupation'] == occ) & (df['Method'] == 'normal')]
        occ_after = df[(df['occupation'] == occ) & (df['Method'] == 'debias')]
        
        non_male_before = 1 - occ_before["gender_collapsed"].value_counts(normalize=True).get("male", 0)
        non_male_after = 1 - occ_after["gender_collapsed"].value_counts(normalize=True).get("male", 0)
      
        # scatter if same
        if non_male_before == non_male_after:
            ax.scatter([non_male_after, non_male_before], [y_pos, y_pos], color='blue')
        else:
            color = 'green' if abs(0.5 - non_male_before) > abs(0.5-non_male_after) else 'red'

            width = max(0.15,0)
            head_width = 0.6
            head_length = min(0.02, width)
            ax.arrow(non_male_before, y_pos, non_male_after - non_male_before, 0, width=width, head_width=head_width, head_length=head_length, fc=color, ec=color, length_includes_head=True)
            ax.scatter([non_male_before], [y_pos], color="black", s=50)
            ax.scatter([non_male_after], [y_pos], color="purple", s=50)
        # make scatter with cross
        # ax.scatter([real], [y_pos], color="darkorange", s=75, marker="x")

    # ax.scatter([-100], [-100], color="darkorange", s=75,  marker="x", label = 'Real-distribution')

    ax.arrow(-100, -100, non_male_after - non_male_before, 0, width=width, head_width=head_width, head_length=head_length, fc="green", ec="green", length_includes_head=True, label = 'Closer to 50%')
    ax.arrow(-100, -100, non_male_after - non_male_before, 0, width=width, head_width=head_width, head_length=head_length, fc="red", ec="red", length_includes_head=True, label='Further from 50%')

    ax.scatter([-100], [-100], color="black", s=50, label = 'Normal')
    ax.scatter([-100], [-100], color="purple", s=50, label = 'Debias')

    ax.scatter(-100,-10, color='blue', label = 'Stationary')

    # line at 50%
    # ax.axvline(x=0.s5, color='black', linestyle='--', linewidth=0.5)
    ax.set_xlabel("Percentage of women or neutral")
    ax.set_xlim(-0.05, 1.05)
    ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    ax.set_xticklabels(['0%', '10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%', '100%'])
    ax.set_ylim(-0.5, len(occs)*2)
    ax.set_ylabel("Occupation")
    ax.set_yticks([i*2 for i in range(len(occs))])
    ax.set_yticklabels([occ.lower() for occ in occs], fontsize=12)
    ax.set_title(f"", fontsize=16)
    ax.legend()
    # ax.legend(ncol=2)

    plt.tight_layout()
    plt.savefig("results/quality_plot_5.png")
    plt.close()


def plot6(df, occs, ax):

    fig, ax = plt.subplots(figsize=(10, 5), dpi=250)

    occ_before = df[df['Method'] == 'normal']
    occ_after = df[df['Method'] == 'debias']

    for i, occ in enumerate(occs):
        i = i*2
        y_pos = i
        occ_before = df[(df['occupation'] == occ) & (df['Method'] == 'normal')]
        occ_after = df[(df['occupation'] == occ) & (df['Method'] == 'debias')]
        
        non_male_before = occ_before["gender_collapsed"].value_counts(normalize=True).get("female", 0) + occ_before["gender_collapsed"].value_counts(normalize=True).get("neutral", 0)
        non_male_after = occ_after["gender_collapsed"].value_counts(normalize=True).get("female", 0) + occ_after["gender_collapsed"].value_counts(normalize=True).get("neutral", 0)

                # Ethnicity analysis
        black_ratio_before = occ_before["ethnicity"].value_counts(normalize=True).get("black", 0)
        other_ratio_before = occ_before["ethnicity"].value_counts(normalize=True).get("other", 0)

        black_ratio_after = occ_after["ethnicity"].value_counts(normalize=True).get("black", 0)
        other_ratio_after = occ_after["ethnicity"].value_counts(normalize=True).get("other", 0)

        non_white_before = black_ratio_before + other_ratio_before
        non_white_after = black_ratio_after + other_ratio_after


        # scatter if same
        if non_male_before == non_male_after:
            # pass
            # ax.scatter([non_male_after, non_male_before], [y_pos, y_pos], color='blue', marker='s')
            ax.scatter([non_male_after, non_male_before], [y_pos-0.25, y_pos-0.25], color='purple', s=50)

        else:
            color1 = 'green' if abs(0.5 - non_male_before) > abs(0.5-non_male_after) else 'red'

            width = max(0.15,0)
            head_width = 0.6
            head_length = min(0.02, width)
            ax.arrow(non_male_before, y_pos-0.25, non_male_after - non_male_before, 0, width=width, head_width=head_width, head_length=head_length, fc="purple", ec="purple", length_includes_head=True)
            # ax.scatter([non_male_before], [y_pos-0.25], color="black", s=50)
            # ax.scatter([non_male_after], [y_pos-0.25], color="purple", s=50)
        line_width = 0.02
        ax.vlines(non_male_before, y_pos-0.25-0.5, y_pos-0.25+0.5 + line_width, color='black', lw=2)  # at the base
            # ax.vlines(non_male_after, y_pos+0.25-0.5, y_pos-0.25+0.5 + line_width, color='blue', lw=2)  # at the base
        

        if non_white_before == non_white_after:
            ax.scatter([non_white_after, non_white_before], [y_pos+0.25, y_pos+0.25], color='orange', marker='x')
            # ax.scatter([non_white_after, non_white_before], [y_pos+0.25, y_pos+0.25], color='orange', marker='x')

        else:
            color2 = 'green' if abs(0.5 - non_male_before) > abs(0.5-non_male_after) else 'red'
            width = max(0.15,0)
            head_width = 0.6
            head_length = min(0.02, width)
            ax.arrow(non_white_before, y_pos+0.25, non_white_after - non_white_before, 0, width=width, head_width=head_width, head_length=head_length, fc="orange", ec="orange", length_includes_head=True)

        line_width = 0.02
        ax.vlines(non_white_before, y_pos+0.25-0.5, y_pos+0.25+0.5 + line_width, color='black', lw=2)  # at the base

    ax.arrow(-100, -100, non_male_after - non_male_before, 0, width=width, head_width=head_width, head_length=head_length, fc="purple", ec="purple", length_includes_head=True, label = 'Non-male debias')
    ax.arrow(-100, -100, non_male_after - non_male_before, 0, width=width, head_width=head_width, head_length=head_length, fc="orange", ec="orange", length_includes_head=True, label='Non-Caucasian debias')

    # ax.scatter(-100,-10, color='purple', label = 'Stationary non-male')
    # ax.scatter(-100,-10, color='orange', label = 'Stationary non-Caucasian')

    ax.vlines(-100, -100+0.25-0.5, -100+0.25+0.5 + line_width, color='black', lw=2, label = "original")  # at the base

    # line at 50%
    # ax.axvline(x=0.s5, color='black', linestyle='--', linewidth=0.5)
    ax.set_xlabel("Percentage of: (non-male / non-Caucasian)")
    ax.set_xlim(-0.05, 1.05)
    ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    ax.set_xticklabels(['0%', '10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%', '100%'])
    ax.set_ylim(-1, len(occs)*2+0.5)
    ax.set_ylabel("Occupation")
    ax.set_yticks([i*2 for i in range(len(occs))])
    ax.set_yticklabels([occ.lower() for occ in occs], fontsize=12)
    ax.set_title(f"", fontsize=16)
    # ax.legend(ncol=2)
    ax.legend()

    plt.tight_layout()
    plt.savefig("results/plot6_percentage.png")
    plt.close()


def numbers(df, occs):

    for i, occ in enumerate(occs):
        
        # Filter data for the current occupation and method
        occ_before = df[(df['occupation'] == occ) & (df['Method'] == 'normal')]
        occ_after = df[(df['occupation'] == occ) & (df['Method'] == 'debias')]
        
        # Ethnicity analysis
        black_ratio_before = occ_before["ethnicity"].value_counts(normalize=True).get("black", 0)
        other_ratio_before = occ_before["ethnicity"].value_counts(normalize=True).get("other", 0)

        black_ratio_after = occ_after["ethnicity"].value_counts(normalize=True).get("black", 0)
        other_ratio_after = occ_after["ethnicity"].value_counts(normalize=True).get("other", 0)

        non_white_before = black_ratio_before + other_ratio_before
        non_white_after = black_ratio_after + other_ratio_after


        # Gender analysis
        male_ratio_before = occ_before["gender_collapsed"].value_counts(normalize=True).get("male", 0)
        female_ratio_before = occ_before["gender_collapsed"].value_counts(normalize=True).get("female", 0)

        male_ratio_after = occ_after["gender_collapsed"].value_counts(normalize=True).get("male", 0)
        female_ratio_after = occ_after["gender_collapsed"].value_counts(normalize=True).get("female", 0)

        # Output results
        print(f"Occupation: {occ}")
        print("non-white before debias:", str(non_white_before*10)[:4].replace(".",",")+"cm")
        print("non-white after debias:", str(non_white_after*10)[:4].replace(".",",")+"cm")
        # print("black before debias:", str(black_ratio_before*10)[:4].replace(".",",")+"cm")
        # print("black after debias:", str(black_ratio_before*10)[:4].replace(".",",")+"cm")
        # print("Female  before debias:", str(female_ratio_before*10)[:4].replace(".",",")+"cm")
        # print("female after debias:", str(female_ratio_after*10)[:4].replace(".",",")+"cm")
        print("\n")
    



def plot123(df):

    # Plot the data
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(15, 6), dpi=250, sharey=True)


    plot1(df, ax1)
    plot2(df, ax2)
    plot3(df, ax3)

    labelsize_x = 12
    labelsize_y = 12

    # set bigger xticks
    ax1.tick_params(axis='x', which='major', labelsize=labelsize_x)
    ax2.tick_params(axis='x', which='major', labelsize=labelsize_x)
    ax3.tick_params(axis='x', which='major', labelsize=labelsize_x)

    # set bigger yticks
    ax1.tick_params(axis='y', which='major', labelsize=labelsize_y)
    ax2.tick_params(axis='y', which='major', labelsize=labelsize_y)
    ax3.tick_params(axis='y', which='major', labelsize=labelsize_y)

    current_ylim = ax1.get_ylim()  # Get the current limits
    ax1.set_ylim(current_ylim[0], current_ylim[1] * 1.5)  # Increase the upper limit by 10%
    ax2.set_ylim(current_ylim[0], current_ylim[1] * 1.5)
    ax3.set_ylim(current_ylim[0], current_ylim[1] * 1.5)

    plt.tight_layout()
    fig.subplots_adjust(wspace=0.05)  # Adjust the wspace parameter as needed
    
    plt.savefig("results/combined_plot.png")
    # plt.show()


def before_after(df, scatter_all=False):

    head_width=0.35
    head_length=0.07
    width=0.1

    pivot_data = pivot_table(df)
    quality_mapping = {1: 'Poor', 2: 'Bad', 3: 'Fair', 4: 'Good', 5: 'Excellent'}

    fig, (ax1, ax3, ax2) = plt.subplots(nrows=1, ncols=3, figsize=(15, len(pivot_data)/2), dpi=200, sharey=True)

    fig.subplots_adjust(hspace=0.5)

    y_ticks = [i for i in range(len(pivot_data))]
    y_labels = [idx.lower().replace("a_photo_of_the_face_of","").replace('_', ' ').title() for idx in pivot_data.index]
    y_lim = len(pivot_data)+ 1
    ax1.set_yticks(y_ticks)
    ax1.set_yticklabels(y_labels)
    ax1.set_ylim(-1, y_lim)

    # Gender plot
    for i, (idx, row) in enumerate(pivot_data.iterrows()):
        y_pos = i  # position for this idx
        debias_towards_center = abs(row[('gender', 'debias')] - 3) < abs(row[('gender', 'normal')] - 3)
        arrow_color = 'green' if debias_towards_center else 'red'
        
        if row[('gender', 'normal')] != row[('gender', 'debias')]:
            normal_value = row[('gender', 'normal')]
            debias_value = row[('gender', 'debias')]
            
            # ax1.arrow(row[('gender', 'normal')], y_pos, row[('gender', 'debias')] - row[('gender', 'normal')], 0, head_width=head_width, head_length=head_length, width=width, fc=arrow_color, ec=arrow_color, alpha=0.8, length_includes_head=True)
            
            line = plt.Line2D([normal_value, debias_value], [y_pos, y_pos], color=arrow_color, label='closer to Independence' if arrow_color == 'green' else 'Further from Independence')
            ax1.add_line(line)

            # Calculate midpoint
            midpoint_x = (normal_value + debias_value) / 2

            dx = debias_value - normal_value
            dy = 0

            direction = 1 if dx > 0 else -1

            # Draw an arrow in the middle
            head_length = 0.05

            ax1.arrow(midpoint_x-(head_length/3)*direction, y_pos, dx * 0.0001, dy, head_width=0.3, head_length=head_length, fc=arrow_color, ec=arrow_color)
            
            ax1.scatter([normal_value, debias_value], [y_pos, y_pos], color=arrow_color, s=10)
            ax1.scatter([abs(normal_value-normal_value)], [y_pos], color=arrow_color, s=10)
            
            if scatter_all:
                ax1.scatter([row[('gender', 'normal')], row[('gender', 'debias')]], [y_pos, y_pos], color='blue', alpha=0.8)
        else:
            ax1.scatter([row[('gender', 'normal')], row[('gender', 'debias')]], [y_pos, y_pos], color='blue', alpha=0.8)

    # Add legend
    handles, labels = ax1.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax1.legend(by_label.values(), by_label.keys())

    ax1.set_yticks(y_ticks)
    ax1.set_yticklabels(y_labels)
    ax1.set_title("Gender appearance")
    # get max and min
    xmin = min(pivot_data[('gender', 'normal')].min(), pivot_data[('gender', 'debias')].min())
    xmax = max(pivot_data[('gender', 'normal')].max(), pivot_data[('gender', 'debias')].max())
    ax1.set_xlim(1 - 0.2, 5 + 0.2)    
    ax1.set_ylim(-1, len(pivot_data))

    # Set custom x-ticks for gender plot
    ax1.set_xticks([1, 2, 3, 4, 5])
    ax1.set_xticklabels(['Feminine - 1', '2','3','4','5 - Masculine'])

    # Quality plot
    for i, (idx, row) in enumerate(pivot_data.iterrows()):
        y_pos = i  # position for this idx

        debias_quality_better = row[('quality_ordinal', 'debias')] > row[('quality_ordinal', 'normal')] 
        arrow_color = 'green' if debias_quality_better else 'red'
        
        if row[('quality_ordinal', 'normal')] != row[('quality_ordinal', 'debias')]:
            # ax2.arrow(row[('quality_ordinal', 'normal')], y_pos, row[('quality_ordinal', 'debias')] - row[('quality_ordinal', 'normal')], 0, head_width=0.35, head_length=0.05, width=0.1, fc=arrow_color, ec=arrow_color, length_includes_head=True)
            normal_value = row[('quality_ordinal', 'normal')]
            debias_value = row[('quality_ordinal', 'debias')]

            line = plt.Line2D([normal_value, debias_value], [y_pos, y_pos], color=arrow_color, label='Quality improvement' if arrow_color == 'green' else 'Quality degradation')
            ax2.add_line(line)

            # Calculate midpoint
            midpoint_x = (normal_value + debias_value) / 2

            dx = debias_value - normal_value
            dy = 0

            direction = 1 if dx > 0 else -1

            # Draw an arrow in the middle
            head_length = 0.05

            ax2.arrow(midpoint_x-(head_length/3)*direction, y_pos, dx * 0.0001, dy, head_width=0.3, head_length=head_length, fc=arrow_color, ec=arrow_color)
            
            ax2.scatter([normal_value, debias_value], [y_pos, y_pos], color=arrow_color, s=10)
            # ax2.scatter([abs(normal_value-normal_value)], [y_pos], color=arrow_color, s=10)
            
            if scatter_all:
                ax2.scatter([row[('quality_ordinal', 'normal')], row[('quality_ordinal', 'debias')]], [y_pos, y_pos], color='blue')
        else:
            ax2.scatter([row[('quality_ordinal', 'normal')], row[('quality_ordinal', 'debias')]], [y_pos, y_pos], color='blue')

    ax2.set_title("Quality")
    ax2.set_xlim(2, 5)    
    ax2.set_ylim(-0.5, len(pivot_data)+1)

    ax2.set_xticks([2, 3, 4, 5])
    ax2.set_xticklabels(["1 2\nPoor Bad", "3\nFair", "4\nGood", "5\nExcellent"])

    # Add legend
    handles, labels = ax2.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax2.legend(by_label.values(), by_label.keys())

    # Group by image and method, then apply the percentage calculation

    caucasian_df = df[df['ethnicity'] == 'black']

    # Step 2: Group by 'occupation' and 'Method', and count 'caucasian'
    caucasian_counts = caucasian_df.groupby(['occupation', 'Method']).size()

    # Step 3: Group the original df by 'occupation' and 'Method' to get total counts per group
    total_counts = df.groupby(['occupation', 'Method']).size()

    # Step 4: Calculate percentage of 'caucasian' for each group
    percentage_caucasian = (caucasian_counts / total_counts * 100).fillna(0)

    df_percentages = percentage_caucasian.unstack()

   # Iterate through the DataFrame to plot arrows
    for i, (prompt, row) in enumerate(df_percentages.iterrows()):
        normal_pct = row.get('normal', 0)  # Get 'normal' percentage, default to 0 if not present
        debias_pct = row.get('debias', 0)  # Get 'debias' percentage, default to 0 if not present

        y_pos = i  # position for this prompt
        
        if normal_pct != debias_pct:
            # Plotting the arrow

            color = 'green' if abs(debias_pct-50) < abs(normal_pct-50) else 'red'
            label = 'closer to Independence' if color == 'green' else 'Further from Independence'

            normal_value =  row.get('normal', 0)
            debias_value = row.get('debias', 0)

            line = plt.Line2D([normal_value, debias_value], [y_pos, y_pos], color=color, label=label)
            ax3.add_line(line)

            # Calculate midpoint
            midpoint_x = (normal_value + debias_value) / 2

            dx = debias_value - normal_value
            dy = 0

            direction = 1 if dx > 0 else -1

            # Draw an arrow in theiddle
            head_length = 2

            ax3.arrow(midpoint_x-(head_length/3)*direction, y_pos, dx * 0.0001, dy, head_width=0.3, head_length=head_length, fc=color, ec=color)
            
            ax3.scatter([normal_value, debias_value], [y_pos, y_pos], color=color, s=10)
            
            # ax3.arrow(normal_pct, i, debias_pct - normal_pct, 0, color='purple', head_width=head_width, head_length=4, width=width, length_includes_head=True)
            if scatter_all:
                ax3.scatter([normal_pct, debias_pct], [i, i], color='blue')
        else:
            ax3.scatter([normal_pct, debias_pct], [i, i], color='blue' , label='Stationary')

    # set line at 50%
    ax3.axvline(x=50, color='black', linestyle='--', linewidth=0.5, label = '50%')

    # Set x-axis limits
    ax3.set_xlim(-2,85)
    ax3.set_xticks(range(0, 81, 10))
    ax3.set_xticklabels([f'{x}%' for x in range(0, 81, 10)])

    # Add legend
    handles, labels = ax3.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    # change  order:
    by_label = {k: by_label[k] for k in ['closer to Independence', 'Further from Independence', 'Stationary']}
    ax3.legend(by_label.values(), by_label.keys())

    # Adding labels and title
    ax3.set_title('Percentage of "Black"')
    
    plt.savefig("results/quality_gender_movement.png")



def ethnicity_move(df, ax, scatter_all=False): 
    quality_mapping = {1: 'poor', 2: 'bad', 3: 'fair', 4: 'good', 5: 'excellent'}

    # Calculate mean and 95% confidence intervals for each group
    grouped_df = df.groupby(['ethnicity', 'Method']).agg(
        mean_quality=('quality_ordinal', 'mean'),
        count=('quality_ordinal', 'size'),
        std_dev=('quality_ordinal', 'std')
    ).reset_index()

    # Calculate the 95% confidence interval
    grouped_df['ci95'] = grouped_df.apply(lambda row: stats.t.ppf(1-0.025, row['count']-1) * row['std_dev'] / np.sqrt(row['count']), axis=1)

    # Add lower and upper bounds for normal and debias methods
    grouped_df['lower'] = grouped_df['mean_quality'] - grouped_df['ci95']
    grouped_df['upper'] = grouped_df['mean_quality'] + grouped_df['ci95']

    # Plot the data
    ethnicities = grouped_df['ethnicity'].unique()
    for i, ethnicity in enumerate(ethnicities):

        normal_row = grouped_df[(grouped_df['ethnicity'] == ethnicity) & (grouped_df['Method'] == 'normal')]
        debias_row = grouped_df[(grouped_df['ethnicity'] == ethnicity) & (grouped_df['Method'] == 'debias')]
        
        # normal_value = grouped_df[(grouped_df['ethnicity'] == ethnicity) & (grouped_df['Method'] == 'normal')]['quality_ordinal'].values[0]
        # debias_value = grouped_df[(grouped_df['ethnicity'] == ethnicity) & (grouped_df['Method'] == 'debias')]['quality_ordinal'].values[0]
        
        normal_value = normal_row['mean_quality'].values[0]
        debias_value = debias_row['mean_quality'].values[0]

        normal_lower = normal_row['lower'].values[0]
        normal_upper = normal_row['upper'].values[0]
        debias_lower = debias_row['lower'].values[0]
        debias_upper = debias_row['upper'].values[0]

        y_pos = i*2  # position for this ethnicity
        debias_towards_five = abs(debias_value - 5) < abs(normal_value - 5)
        arrow_color = 'green' if debias_towards_five else 'red'
        
        if normal_value != debias_value:
            # ax.arrow(normal_value, y_pos, debias_value - normal_value, 0, head_width=0.2, head_length=0.2, width=0.05, fc=arrow_color, ec=arrow_color, length_includes_head=True)
            
            # line = plt.Line2D([normal_value, debias_value], [y_pos, y_pos], color=arrow_color)
            # ax.add_line(line)

            # Calculate midpoint
            midpoint_x = (normal_value + debias_value) / 2

            dx = debias_value - normal_value
            dy = 0

            direction = 1 if dx > 0 else -1

            # Draw an arrow in the middle
            head_length = 0.05

            # ax.arrow(midpoint_x-(head_length/3)*direction, y_pos, dx * 0.0001, dy, head_width=0.3, head_length=head_length, fc=arrow_color, ec=arrow_color)
            head_length = max(0.05, abs(debias_value - normal_value) / 10)
            ax.arrow(normal_value, y_pos-1, debias_value - normal_value, 0, 
            head_width=0.2, head_length=head_length, width=0.07, fc=arrow_color, ec=arrow_color, length_includes_head=True)

            ax.errorbar(normal_value, y_pos-0.5, xerr=[[normal_value - normal_lower], [normal_upper - normal_value]], fmt='o', color="black", capsize=5, alpha = 1)
            ax.errorbar(debias_value, y_pos, xerr=[[debias_value - debias_lower], [debias_upper - debias_value]], fmt='o', color="purple", capsize=5, alpha = 1)

            if scatter_all:
                ax.scatter([normal_value, debias_value], [y_pos, y_pos], color='blue')
        else:
            ax.scatter([normal_value, debias_value], [y_pos, y_pos], color='blue')

    ax.errorbar(0,0,0,0, fmt='o', color="black", capsize=5, alpha = 1, label = 'normal 95% CI')
    ax.errorbar(0,0,0,0, fmt='o', color="purple", capsize=5, alpha = 1, label = 'debias 95% CI')

    # ax.set_yticks([i*2 for i in range(len(ethnicities))])
    ax.set_yticks([i*2-0.5 for i in range(len(ethnicities))])

    ax.set_yticklabels([ethnicity.replace('_', ' ').title() for ethnicity in ethnicities])
    ax.set_title("Quality Changes by: \n Ethnicity")
    ax.set_xlim(1, 5)
    ax.set_ylim(-2, len(ethnicities)*2-1)
    
    # set legend
    ax.legend()

    # Set custom x-ticks for quality plot
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xticklabels([quality_mapping[x] for x in range(1, 6)])


def gender_move_old(df, ax, scatter_all=False): 

    quality_mapping = {1: 'poor', 2: 'bad', 3: 'fair', 4: 'good', 5: 'excellent'}

    # Group by 'ethnicity' and 'Method' and calculate the mean of 'quality_ordinal'
    grouped_df = df.groupby(['gender', 'Method']).agg({'quality_ordinal': 'mean'}).reset_index()

    # Plot the data
    genders = grouped_df['gender'].unique()
    for i, gender in enumerate(genders):
        normal_value = grouped_df[(grouped_df['gender'] == gender) & (grouped_df['Method'] == 'normal')]['quality_ordinal'].values[0]
        debias_value = grouped_df[(grouped_df['gender'] == gender) & (grouped_df['Method'] == 'debias')]['quality_ordinal'].values[0]
        
        y_pos = i  # position for this gender
        debias_better = debias_value > normal_value
        arrow_color = 'green' if debias_better else 'red'
        
        if normal_value != debias_value:

            line = plt.Line2D([normal_value, debias_value], [y_pos, y_pos], color=arrow_color)
            ax.add_line(line)

            # Calculate midpoint
            midpoint_x = (normal_value + debias_value) / 2

            dx = debias_value - normal_value
            dy = 0

            direction = 1 if dx > 0 else -1

            # Draw an arrow in the middle
            head_length = 0.05

            ax.arrow(midpoint_x-(head_length/3)*direction, y_pos, dx * 0.0001, dy, head_width=0.3, head_length=head_length, fc=arrow_color, ec=arrow_color)
            
            ax.scatter([normal_value, debias_value], [y_pos, y_pos], color=arrow_color, s=10)
            ax.scatter([abs(normal_value-normal_value)], [y_pos], color=arrow_color, s=10)

            if scatter_all:
                pass
        else:
            ax.scatter([normal_value, debias_value], [y_pos, y_pos], color='blue')

    ax.set_yticks([i for i in range(len(genders))])
    ax.set_yticklabels(['Feminine  1', '2','3','4','Masculine  5'])
    ax.set_title("Gender")
    ax.set_xlim(1, 5)
    ax.set_ylim(-1, len(genders))

    # Set custom x-ticks for quality plot
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xticklabels([quality_mapping[x] for x in range(1, 6)])


def gender_move(df, ax, scatter_all=False): 
    quality_mapping = {1: 'poor', 2: 'bad', 3: 'fair', 4: 'good', 5: 'excellent'}

    # Calculate mean and 95% confidence intervals for each group
    grouped_df = df.groupby(['gender_collapsed', 'Method']).agg(
        mean_quality=('quality_ordinal', 'mean'),
        count=('quality_ordinal', 'size'),
        std_dev=('quality_ordinal', 'std')
    ).reset_index()

    # Calculate the 95% confidence interval
    grouped_df['ci95'] = grouped_df.apply(lambda row: stats.t.ppf(1-0.025, row['count']-1) * row['std_dev'] / np.sqrt(row['count']), axis=1)

    # Add lower and upper bounds for normal and debias methods
    grouped_df['lower'] = grouped_df['mean_quality'] - grouped_df['ci95']
    grouped_df['upper'] = grouped_df['mean_quality'] + grouped_df['ci95']

    # Plot the data
    ethnicities = grouped_df['gender_collapsed'].unique()
    for i, ethnicity in enumerate(ethnicities):

        normal_row = grouped_df[(grouped_df['gender_collapsed'] == ethnicity) & (grouped_df['Method'] == 'normal')]
        debias_row = grouped_df[(grouped_df['gender_collapsed'] == ethnicity) & (grouped_df['Method'] == 'debias')]
        
        # normal_value = grouped_df[(grouped_df['ethnicity'] == ethnicity) & (grouped_df['Method'] == 'normal')]['quality_ordinal'].values[0]
        # debias_value = grouped_df[(grouped_df['ethnicity'] == ethnicity) & (grouped_df['Method'] == 'debias')]['quality_ordinal'].values[0]
        
        normal_value = normal_row['mean_quality'].values[0]
        debias_value = debias_row['mean_quality'].values[0]

        normal_lower = normal_row['lower'].values[0]
        normal_upper = normal_row['upper'].values[0]
        debias_lower = debias_row['lower'].values[0]
        debias_upper = debias_row['upper'].values[0]

        y_pos = i*2  # position for this ethnicity
        debias_towards_five = abs(debias_value - 5) < abs(normal_value - 5)
        arrow_color = 'green' if debias_towards_five else 'red'
        
        if normal_value != debias_value:
            # ax.arrow(normal_value, y_pos, debias_value - normal_value, 0, head_width=0.2, head_length=0.2, width=0.05, fc=arrow_color, ec=arrow_color, length_includes_head=True)
            
            # line = plt.Line2D([normal_value, debias_value], [y_pos, y_pos], color=arrow_color)
            # ax.add_line(line)

            # Calculate midpoint
            midpoint_x = (normal_value + debias_value) / 2

            dx = debias_value - normal_value
            dy = 0

            direction = 1 if dx > 0 else -1

            # Draw an arrow in the middle
            head_length = 0.05

            # ax.arrow(midpoint_x-(head_length/3)*direction, y_pos, dx * 0.0001, dy, head_width=0.3, head_length=head_length, fc=arrow_color, ec=arrow_color)
            head_length = max(0.05, abs(debias_value - normal_value) / 10)
            ax.arrow(normal_value, y_pos-1, debias_value - normal_value, 0, 
            head_width=0.2, head_length=head_length, width=0.07, fc=arrow_color, ec=arrow_color, length_includes_head=True)

            ax.errorbar(normal_value, y_pos-0.5, xerr=[[normal_value - normal_lower], [normal_upper - normal_value]], fmt='o', color="black", capsize=5, alpha = 1)
            ax.errorbar(debias_value, y_pos, xerr=[[debias_value - debias_lower], [debias_upper - debias_value]], fmt='o', color="purple", capsize=5, alpha = 1)

            if scatter_all:
                ax.scatter([normal_value, debias_value], [y_pos, y_pos], color='blue')
        else:
            ax.scatter([normal_value, debias_value], [y_pos, y_pos], color='blue')

    ax.errorbar(0,0,0,0, fmt='o', color="black", capsize=5, alpha = 1, label = 'normal 95% CI')
    ax.errorbar(0,0,0,0, fmt='o', color="purple", capsize=5, alpha = 1, label = 'debias 95% CI')

    ax.set_yticks([i*2-0.5 for i in range(len(ethnicities))])
    # ax.set_yticks([i*2 for i in range(len(ethnicities))])

    # ax.set_yticklabels(["Female", "Neutral", "Male"])
    ax.set_title("Gender")
    ax.set_xlim(1, 5)
    ax.set_ylim(-2, len(ethnicities)*2-1)
    
    # set legend
    ax.legend()

    # Set custom x-ticks for quality plot
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xticklabels([quality_mapping[x] for x in range(1, 6)])

def combined_plot(df):
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(6, 6), dpi=250)
    
    ethnicity_move(df, ax1)
    gender_move(df, ax2)
    
    plt.tight_layout()
    plt.savefig("results/gender_race_quality_movement_2.png")


def ethnicity_gender_when_changed(df):
    # Combining 'occupation' and 'sample' to create a unique group identifier
    df['sample_group'] = df['occupation'] + df['sample'].astype(str)

    # Pivot the data to compare the ethnicity between methods
    pivot = df.pivot_table(
        index='sample_group',
        columns='Method',
        values=['ethnicity', 'quality_ordinal'],
        aggfunc={'ethnicity': 'first', 'quality_ordinal': 'mean'}
    )
    # make categorical

    # Identifying groups where ethnicity is the same and different across methods
    pivot['ethnicity_same'] = pivot['ethnicity']['normal'] == pivot['ethnicity']['debias']

    # Average quality where ethnicity remained the same
    same_ethnicity_normal = pivot.loc[pivot['ethnicity_same'], 'quality_ordinal']['normal'].mean()
    same_ethnicity_debias = pivot.loc[pivot['ethnicity_same'], 'quality_ordinal']['debias'].mean()

    # Average quality where ethnicity changed
    changed_ethnicity_normal = pivot.loc[~pivot['ethnicity_same'], 'quality_ordinal']['normal'].mean()
    changed_ethnicity_debias = pivot.loc[~pivot['ethnicity_same'], 'quality_ordinal']['debias'].mean()

    print("same (normal): ", round(same_ethnicity_normal,3))
    print("same (debias): ", round(same_ethnicity_debias,3))
    print("loss: " , round(same_ethnicity_normal-same_ethnicity_debias,3))
          
    print("changed (normal): ", round(changed_ethnicity_normal,3))
    print("changed (debias): ", round(changed_ethnicity_debias,3))
    print("loss: " , round(changed_ethnicity_normal-changed_ethnicity_debias,3))
 
    # Pivot the data to compare the gender between methods
    pivot = df.pivot_table(
        index='sample_group',
        columns='Method',
        values=['gender', 'quality_ordinal'],
        aggfunc={'gender': 'first', 'quality_ordinal': 'mean'}
    )
    # Identifying groups where gender is the same and different across methods
    pivot['gender_same'] = pivot['gender']['normal'] == pivot['gender']['debias']

    same_gender_normal = pivot.loc[pivot['gender_same'], 'quality_ordinal']['normal'].mean()
    same_gender_debias = pivot.loc[pivot['gender_same'], 'quality_ordinal']['debias'].mean()

    # Average quality where gender changed
    changed_gender_normal = pivot.loc[~pivot['gender_same'], 'quality_ordinal']['normal'].mean()
    changed_gender_debias = pivot.loc[~pivot['gender_same'], 'quality_ordinal']['debias'].mean()

    print("Gender same (normal): ", round(same_gender_normal, 3))
    print("Gender same (debias): ", round(same_gender_debias, 3))
    print("Gender loss (same): ", round(same_gender_normal - same_gender_debias, 3))

    print("Gender changed (normal): ", round(changed_gender_normal, 3))
    print("Gender changed (debias): ", round(changed_gender_debias, 3))
    print("Gender loss (changed): ", round(changed_gender_normal - changed_gender_debias, 3))

    changed_gender_values = pivot.loc[~pivot['gender_same'], 'quality_ordinal']['normal'].tolist()
    print(changed_gender_values)
    no_gender_values = pivot.loc[pivot['gender_same'], 'quality_ordinal']['normal'].tolist()
    print(no_gender_values)

    plt.boxplot([changed_gender_values, no_gender_values], labels= ["changed","didnt change"])
    plt.savefig("When_changed.png")


def population_change(df):
    # Filter data based on method
    ethnicity_counts = df.groupby(['Method', 'ethnicity']).size().unstack(fill_value=0)

    # Display the results
    print(ethnicity_counts)


def heatmap(df, column):

    # Step 2: Create a dataframe that lists the initial and following ethnicity for each method change
    # Pivot the table by 'occupation' and 'sample' as index, and 'Method' as columns, capturing the first ethnicity observed for each method.
    
    df = df[df[column]!="unintelligible"]

    transitions = df.pivot_table(index=['occupation', 'sample'], columns='Method', values=column, aggfunc='first')

    # Reset index and melt to create a long format DataFrame where each row is a sample-method combination
    transitions = transitions.reset_index().melt(id_vars=['occupation', 'sample'], value_vars=transitions.columns, var_name='Method', value_name=column)

    # Step 3: Calculate transitions by shifting ethnicity within each group to find the 'from' ethnicity
    transitions['group_after'] = transitions.groupby(['occupation', 'sample'])[column].shift(1)

    # Drop rows where ethnicity_after is NaN, which are rows without a previous method to compare
    transitions.dropna(subset=['group_after'], inplace=True)

    # Step 4: Create a complete set of all possible 'from' and 'to' combinations
    all_ethnicities = pd.Series(df[column].unique())
    cross_join = pd.merge(all_ethnicities.rename('group_after'), all_ethnicities.rename('ethnicity'), how='cross')

    # Step 5: Count the transitions and merge with the cross join to ensure all possible transitions are shown
    transition_counts = transitions.groupby(['group_after', column]).size().reset_index(name='count')
    transition_counts = pd.merge(cross_join, transition_counts, on=['group_after', column], how='left').fillna(0)

    # Convert the count column back to integer type if it was affected by fillna
    transition_counts['count'] = transition_counts['count'].astype(int)

    # Pivot the transition_counts dataframe to make it suitable for heatmap
    heatmap_data = transition_counts.pivot(index="group_after", columns=column, values="count")
    heatmap_data = heatmap_data.fillna(0)
    # Create the heatmap
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(heatmap_data, annot=True, fmt="d", cmap="YlGnBu", annot_kws={"fontsize": 16})
    plt.title(f'{column} transitions', fontsize=20)
    plt.xlabel(f'To {column}', fontsize=16)
    plt.ylabel(f'From {column}', fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=16)

    # Set font sizes for x and y labels and title
    plt.savefig(f"results/transition_1_{column}.png")

    plt.figure(figsize=(10, 8))
    ax =sns.heatmap(heatmap_data, annot=True, fmt="d", cmap="YlGnBu", annot_kws={"fontsize": 16})
    plt.title(f'{column} transitions', fontsize=20)
    plt.xlabel(f'From {column}', fontsize=16)
    plt.ylabel(f'To {column}', fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=16)

    # Set font sizes for x and y labels and title
    plt.savefig(f"results/transition_2_{column}.png")

def heatmap_with_gender_quality(df):

    # Pivot the table by 'occupation' and 'sample' as index, and 'Method' as columns, capturing the first gender observed for each method.
    transitions = df.pivot_table(index=['occupation', 'sample'], columns='Method', values='gender', aggfunc='first')

    # Reset index and melt to create a long format DataFrame where each row is a sample-method combination
    transitions = transitions.reset_index().melt(id_vars=['occupation', 'sample'], var_name='Method', value_name='gender')

    # Calculate transitions by shifting gender within each group to find the 'from' gender
    transitions['gender_after'] = transitions.groupby(['occupation', 'sample'])['gender'].shift(-1)

    # Join the average quality for each transition
    df['quality_ordinal'] = df['quality_ordinal'].astype(float)  # Ensure quality is float
    avg_quality = df.groupby(['occupation', 'sample', 'Method']).agg({'quality_ordinal': 'mean'}).reset_index()
    transitions = pd.merge(transitions, avg_quality, how='left', left_on=['occupation', 'sample', 'Method'], right_on=['occupation', 'sample', 'Method'])

    # Drop rows where gender_after is NaN, which are rows without a previous method to compare
    transitions.dropna(subset=['gender_after'], inplace=True)

    # Count the transitions and calculate the average quality
    transition_counts = transitions.groupby(['gender', 'gender_after']).agg(count=('gender', 'size'), avg_quality=('quality_ordinal', 'mean')).reset_index()

    # Pivot the transition_counts dataframe for heatmap
    count_data = transition_counts.pivot(index="gender_after", columns="gender", values="count")
    quality_data = transition_counts.pivot(index="gender_after", columns="gender", values="avg_quality")
    
    cmap = sns.color_palette("YlGnBu", as_cmap=True)
    cmap.set_bad("black")
    
    # Create the average quality heatmap    
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(quality_data, annot=True, cmap=cmap, annot_kws={"fontsize": 12}, fmt=".2f", vmin=1, vmax=5)   
    # Define the colorbar and set tick positions and labels

    colorbar = ax.collections[0].colorbar
    colorbar.set_label('Average Quality', fontsize=14)

    # Specific tick values and labels
    tick_values = [1, 2, 3, 4, 5]  # Corresponding to specific values
    tick_labels = ['1 - Poor', '2 - Bad', '3 - Fair', '4 - Good', '5 - Excellent']
    colorbar.set_ticks(tick_values)
    colorbar.set_ticklabels(tick_labels)
    colorbar.set_label('Average Quality', fontsize=14)

    plt.title('Average Quality of Gender Transitions', fontsize=20)
    plt.xlabel('From Gender', fontsize=16)
    plt.ylabel('To Gender', fontsize=16)
    plt.savefig("gender_transition_quality.png")

    # Create the count heatmap
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(count_data, annot=True, fmt=".2f", cmap="YlGnBu", annot_kws={"fontsize": 12})
    colorbar = ax.collections[0].colorbar
    colorbar.set_label('Count', fontsize=14)

    plt.title('Gender Transition Counts', fontsize=20)
    plt.xlabel('To Gender', fontsize=16)
    plt.ylabel('From Gender', fontsize=16)
    plt.savefig("results/gender_transition.png")

def heatmap_quality_count(df):

    # Pivot the table by 'occupation' and 'sample' as index, and 'Method' as columns, capturing the first ethnicity observed for each method.
    transitions = df.pivot_table(index=['occupation', 'sample'], columns='Method', values='ethnicity', aggfunc='first')

    # Reset index and melt to create a long format DataFrame where each row is a sample-method combination
    transitions = transitions.reset_index().melt(id_vars=['occupation', 'sample'], var_name='Method', value_name='ethnicity')

    # Calculate transitions by shifting ethnicity within each group to find the 'from' ethnicity
    transitions['ethnicity_after'] = transitions.groupby(['occupation', 'sample'])['ethnicity'].shift(-1)

    # Join the average quality for each transition
    df['quality_ordinal'] = df['quality_ordinal'].astype(float)  # Ensure quality is float
    avg_quality = df.groupby(['occupation', 'sample', 'Method']).agg({'quality_ordinal': 'mean'}).reset_index()
    transitions = pd.merge(transitions, avg_quality, how='left', left_on=['occupation', 'sample', 'Method'], right_on=['occupation', 'sample', 'Method'])

    # Drop rows where ethnicity_after is NaN, which are rows without a previous method to compare
    transitions.dropna(subset=['ethnicity_after'], inplace=True)

    # Count the transitions and calculate the average quality
    transition_counts = transitions.groupby(['ethnicity', 'ethnicity_after']).agg(count=('ethnicity', 'size'), avg_quality=('quality_ordinal', 'mean')).reset_index()

    # Pivot the transition_counts dataframe for heatmap
    count_data = transition_counts.pivot(index="ethnicity_after", columns="ethnicity", values="count")
    quality_data = transition_counts.pivot(index="ethnicity_after", columns="ethnicity", values="avg_quality")
        
    # Assuming 'count_data' and 'quality_data' are prepared as described
    cmap = mcolors.LinearSegmentedColormap.from_list("custom_red_green", ["red", "yellow", "green"], N=256)
    cmap.set_bad("azure")  # Handling NaN values or data gaps

    plt.figure(figsize=(12, 10))
    ax = sns.heatmap(quality_data, annot=False, cmap=cmap, fmt=".2f", vmin=1, vmax=5,
                    cbar_kws={'label': 'Average Quality'})

    # Configure colorbar
    colorbar = ax.collections[0].colorbar
    colorbar.set_label('Average Quality', fontsize=14)
    tick_values = [1, 2, 3, 4, 5]
    tick_labels = ['Poor', 'Bad', 'Fair', 'Good', 'Excellent']
    colorbar.set_ticks(tick_values)
    colorbar.set_ticklabels(tick_labels)

    # Add annotations manually
    for i, (row_val, row_count) in enumerate(zip(quality_data.values, count_data.values)):
        for j, (cell_val, cell_count) in enumerate(zip(row_val, row_count)):
            if np.isnan(cell_val):
                text = ''
            else:
                text = f'{int(cell_count):d}' if not np.isnan(cell_count) else ''
            color = 'black'  # Adjusting text color based on cell color for better visibility
            ax.text(j + 0.5, i + 0.5, text, ha='center', va='center', color=color, fontsize=18)

    plt.title('Average Quality and Count of Ethnicity Transitions', fontsize=20)
    plt.xlabel('To Ethnicity', fontsize=16)
    plt.ylabel('From Ethnicity', fontsize=16)
    plt.savefig("results/combined_transition_quality_count.png")

def heatmap_quality_count_gender(df):
    
    # Combine prompt, sample, and Method into a unique identifier
    df['identifier'] = df['occupation'] + '_' + df['sample'].astype(str) + '_' + df['Method'].astype(str)

    # Pivot the table by 'occupation' and 'sample' as index, and 'Method' as columns, capturing the first gender observed for each method.
    transitions = df.pivot_table(index=['occupation', 'sample'], columns='Method', values='gender_collapsed', aggfunc='first')

    # Reset index and melt to create a long format DataFrame where each row is a sample-method combination
    transitions = transitions.reset_index().melt(id_vars=['occupation', 'sample'], var_name='Method', value_name='gender_collapsed')

    # Calculate transitions by shifting gender within each group to find the 'from' gender
    transitions['gender_after'] = transitions.groupby(['occupation', 'sample'])['gender_collapsed'].shift(-1)

    # Join the average quality for each transition
    df['quality_ordinal'] = df['quality_ordinal'].astype(float)  # Ensure quality is float
    avg_quality = df.groupby(['occupation', 'sample', 'Method']).agg({'quality_ordinal': 'mean'}).reset_index()
    transitions = pd.merge(transitions, avg_quality, how='left', left_on=['occupation', 'sample', 'Method'], right_on=['occupation', 'sample', 'Method'])

    # Drop rows where gender_after is NaN, which are rows without a previous method to compare
    transitions.dropna(subset=['gender_after'], inplace=True)

    # Count the transitions and calculate the average quality
    transition_counts = transitions.groupby(['gender_collapsed', 'gender_after']).agg(count=('gender_collapsed', 'size'), avg_quality=('quality_ordinal', 'mean')).reset_index()

    # Pivot the transition_counts dataframe for heatmap
    count_data = transition_counts.pivot(index="gender_after", columns="gender_collapsed", values="count")
    quality_data = transition_counts.pivot(index="gender_after", columns="gender_collapsed", values="avg_quality")
    
    # Assuming 'count_data' and 'quality_data' are prepared as described
    cmap = mcolors.LinearSegmentedColormap.from_list("custom_red_green", ["red", "yellow", "green"], N=256)
    cmap.set_bad("black")  # Handling NaN values or data gaps

    plt.figure(figsize=(12, 10))
    ax = sns.heatmap(quality_data, annot=False, cmap=cmap, fmt=".2f", vmin=1, vmax=5,
                    cbar_kws={'label': 'Average Quality'})

    # Configure colorbar
    colorbar = ax.collections[0].colorbar
    colorbar.set_label('Average Quality', fontsize=14)
    tick_values = [1, 2, 3, 4, 5]
    tick_labels = ['Poor', 'Bad', 'Fair', 'Good', 'Excellent']
    colorbar.set_ticks(tick_values)
    colorbar.set_ticklabels(tick_labels)

    # Add annotations manually
    for i, (row_val, row_count) in enumerate(zip(quality_data.values, count_data.values)):
        for j, (cell_val, cell_count) in enumerate(zip(row_val, row_count)):
            if np.isnan(cell_val):
                text = ''
            else:
                text = f'{int(cell_count):d}' if not np.isnan(cell_count) else ''
            color = 'white' if cell_val > 3 else 'black'  # Adjusting text color based on cell value
            ax.text(j + 0.5, i + 0.5, text, ha='center', va='center', color=color, fontsize=12)

    plt.title('Average Quality and Count of Gender Transitions', fontsize=20)
    plt.xlabel('To Gender', fontsize=16)
    plt.ylabel('From Gender', fontsize=16)
    plt.savefig("heatmap_gender.png")



def correlation_for_transitions(df):

    def load_image(image_path):
        base=  "static/images/"
        image_path = base + image_path

        image = mpimg.imread(image_path)
        if image.shape[-1] == 4:  # Check for RGBA and convert to RGB
            image = image[:, :, :3]
        return cv2.cvtColor((image * 255).astype('uint8'), cv2.COLOR_RGB2GRAY)  # Convert to grayscale

    def calculate_ssim(img1, img2):
        return ssim(img1, img2)

    def calculate_correlation(img1, img2):
        return np.corrcoef(img1.ravel(), img2.ravel())[0, 1]

    def calculate_cosine_similarity(hist1, hist2):
        return 1 - distance.cosine(hist1, hist2)

    def extract_histogram(image):
        hist = [np.histogram(image[:, :, i], bins=256, range=(0, 255))[0] for i in range(3)]
        hist = np.concatenate(hist).ravel()
        hist = hist / np.sum(hist)  # Normalize
        return hist

    def extract_features(image_path):
        # Load the image using matplotlib
        base=  "static/images/"
        image_path = base + image_path
        image = mpimg.imread(image_path)
        # Check if image is in PNG format (RGBA) and convert to RGB
        if image.shape[2] == 4:
            image = image[:, :, :3]
        # Calculate histogram for each RGB channel and normalize
        hist = [np.histogram(image[:, :, i], bins=256, range=(0, 1))[0] for i in range(3)]
        hist = np.concatenate(hist).ravel()
        hist = hist / np.sum(hist)  # Normalize
        return hist

    # Assuming df is your DataFrame and 'image' contains paths to the images
    df['features'] = df['image'].apply(extract_features)

    # Function to calculate similarity
    def calculate_similarity(group):
        features_A = np.array(list(group['features_A']))
        features_B = np.array(list(group['features_B']))
        # Calculate average cosine similarity
        similarities = [1 - distance.cosine(features_A[i], features_B[i]) for i in range(len(features_A))]
        return np.nanmean(similarities)

    methods = df['Method'].unique()

    df_A = df[df['Method'] == methods[0]]
    df_B = df[df['Method'] == methods[1]]

    merged_df = pd.merge(df_A, df_B, on=['prompt', 'sample', 'occupation'], suffixes=('_A', '_B'))

    # Calculate similarities
    def calculate_similarities(group):
        ssim_value = calculate_ssim(group['image_gray_A'].iloc[0], group['image_gray_B'].iloc[0])
        corr_value = calculate_correlation(group['image_gray_A'].iloc[0], group['image_gray_B'].iloc[0])
        cos_sim_value = calculate_cosine_similarity(group['histogram_A'].iloc[0], group['histogram_B'].iloc[0])
        return pd.Series([ssim_value, corr_value, cos_sim_value], index=['SSIM', 'Correlation', 'Cosine_Similarity'])

    # Apply similarity calculations for each group
    results = merged_df.groupby(['prompt', 'sample', 'occupation']).apply(calculate_similarities)

    # Average results per occupation
    average_results_per_occupation = results.groupby('occupation').mean()

    # Print or return results
    print(average_results_per_occupation)


def heatmap2(df, column):
    df = df[df[column] != "unintelligible"]

    try:
        # Attempt to pivot the table
        transitions = df.pivot_table(index=['occupation', 'sample'], columns='Method', values=column, aggfunc='first')
        print("Pivot successful:", transitions.head())
    except KeyError as e:
        print("Failed during pivot:", e)
        return

    # Reset and melt
    try:
        transitions = transitions.reset_index().melt(id_vars=['occupation', 'sample'], value_vars=transitions.columns, var_name='Method', value_name=column)
        print("Melt successful:", transitions.head())
    except KeyError as e:
        print("Failed during melt:", e)
        return

    # Calculate transitions
    try:
        transitions['group_after'] = transitions.groupby(['occupation', 'sample'])[column].shift(1)
        transitions.dropna(subset=['group_after'], inplace=True)
        print("Group and shift successful:", transitions.head())
    except Exception as e:
        print("Failed during grouping and shifting:", e)
        return

    # Cross join
    try:
        all_ethnicities = pd.Series(df[column].unique())
        cross_join = pd.merge(all_ethnicities.rename('group_after'), all_ethnicities.rename(column), how='cross')
        transition_counts = transitions.groupby(['group_after', column]).size().reset_index(name='count')
        transition_counts = pd.merge(cross_join, transition_counts, on=['group_after', column], how='left').fillna(0)
        transition_counts['count'] = transition_counts['count'].astype(int)
        print("Cross join and counting successful:", transition_counts.head())
    except Exception as e:
        print("Failed during cross join or counting:", e)
        return

    # Create heatmap
    try:
        heatmap_data = transition_counts.pivot(index="group_after", columns=column, values="count")
        plt.figure(figsize=(10, 8))
        sns.heatmap(heatmap_data, annot=True, fmt="d", cmap="YlGnBu", annot_kws={"fontsize": 16})
        plt.savefig("gender_trans.png")
    except Exception as e:
        print("Failed during heatmap creation:", e)
        return


if __name__ == '__main__':

    df = load_df(overwrite=True)

    plt.style.use('ggplot')
    plt.rcParams['text.color'] = 'black'
    plt.rcParams['axes.labelcolor'] = 'black'
    plt.rcParams['xtick.color'] = 'black'
    plt.rcParams['ytick.color'] = 'black'
    plt.rcParams['axes.titlecolor'] = 'black'

    # heatmap_quality_count
    # before_after(df, grousped)

    # print(pivot_table(df, to_latex=True))
    # before_after(df)


    # print(df[df['ethnicity'] == 'black'].groupby(['occupation', 'Method']).size())
    # ethnicity_gender_when_changed(df)
    # combined_plot(df)
    # plot123(df)

    occs = df['occupation'].unique()

    occs = ['Taxi Driver ', 'Housekeeper ', 'Flight Attendant ', 'Chef ', 'Pilot  ',  'Nurse  ', 'Therapist ', 'Cook ', 'Firefighter ', 'Ceo ']
    
    order = "pilot,therapist,flight_attendant,chef,ceo,firefighter,nurse,housekeeper,cook,taxi_driver"
    # order = order.split(',')

    occs1 = [occs[i] for i in [4, 6, 2, 3, 9, 8, 5, 1, 7, 0]]

    real = {"Ceo ": -1000, 'Pilot  ':7.3,'Therapist ':22.2,'Flight Attendant ':29.7, 'Chef ':37.1,'Firefighter ':15.8,  'Nurse  ': 25.1,  'Housekeeper ':25.1, 'Cook ': 27, 'Taxi Driver ':42.9} 
    # divide all values by 100:
    real = {k: v/100 for k, v in real.items()}
    

    # real_gender = {'Firefighter ':"0.48",'Taxi Driver ':"1.2",'Chef ': "2.3", 'Pilot  ': "0.48",'Cook ': "3.96",  'Flight Attendant ': "6.48", 'Therapist ': "8.68",'Nurse  ': "8.68", 'Housekeeper ': "8.85"}
    
    real_gender = {"Ceo ": -1000, 'Pilot  ': 0.48, 'Therapist ': 8.68, 'Flight Attendant ': 6.48, 'Chef ': 2.3, 'Firefighter ': 0.48, 'Nurse  ': 8.68, 'Housekeeper ': 8.85, 'Cook ': 3.96, 'Taxi Driver ': 1.2}
    
    #get percentage by multiply with 10
    real_gender = {k: float(v)/10 for k, v in real_gender.items()}

    occs1 = list(real.keys())
    plot1(df, occs1, ax=None)
    plot2(df, occs1, ax=None)
    plot3(df, occs1, ax=None)

    plot4(df, real, ax=None)
    plot7(df, real_gender, ax=None)

    # heatmap(df)


    # combined_plot(df)
    # plot5(df, occs1, ax=None)
    # plot4(df, real, ax=None)

    numbers(df, occs)
    # ethnicity_gender_when_changed(df)

    # ethnicity_gender_when_changed(df)

    # print(df.column's)
    heatmap(df, "ethnicity")
    # df["gender"] = df["gender"].astype(str)

    heatmap2(df, "gender")
    # print(df["ethnicity"])
    # print(df["gender_collapsed"])