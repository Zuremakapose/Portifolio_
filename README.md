Name: Zurema Kapose Subject: Intro to Machine Learning Hult International Business School

Challenge:

Certain factors contribute to the health of a newborn baby. One such health measure is birth weight. Countless studies have identified factors, both preventative and hereditary, that lead to low birth weight. Your team has been hired as public health consultants to analyze and model an infant’s birth weight based on such characteristics.

Introduction of Analysis:

Analysis is a crucial aspect of data-driven project. Involving the examination and interpretation of data to derive meaningful insights and make informed decisions. In the coding section below key components such as understanding the data, identifying and removing unwanted values, and use of high-quality data analysis to perform analysis like DP in missing imputations and perform Exploratory Data Analysis.

Birthweight_data_dictionary:

| variable | label | description |

|----------|---------|---------------------------------|

| 1 | mage | mother's age in years |

| 2 | meduc | mother's education in years |

| 3 | monpre | month pregnancy that prenatal care began |

| 4 | npvis | total number of prenatal visits |

| 5 | fage | father's age in years |

| 6 | feduc | father's education in years |

| 7 | omaps | one minute apgar score |

| 8 | fmaps | five minute apgar score |

| 9 | cigs | average cigarettes per day consumed by the mother |

| 10 | drink | average drinks per week consumed by the mother |

| 11 | male | 1 if baby male |

| 12 | mwhte | 1 if mother white |

| 13 | mblck | 1 if mother black |

| 14 | moth | 1 if mother is not black or white |

| 15 | fwhte | 1 if father white |

| 16 | fblck | 1 if father black |

| 17 | foth | 1 if father is not black or white |

| 18 | bwght | birthweight in grams |

# installing phik (phi coefficient)
#%pip install phik
# %pip install seaborn
%pip install seaborn
Requirement already satisfied: seaborn in /Users/zuremakapose/anaconda3/lib/python3.11/site-packages (0.13.2)
Requirement already satisfied: numpy!=1.24.0,>=1.20 in /Users/zuremakapose/anaconda3/lib/python3.11/site-packages (from seaborn) (1.24.3)
Requirement already satisfied: pandas>=1.2 in /Users/zuremakapose/anaconda3/lib/python3.11/site-packages (from seaborn) (2.1.1)
Requirement already satisfied: matplotlib!=3.6.1,>=3.4 in /Users/zuremakapose/anaconda3/lib/python3.11/site-packages (from seaborn) (3.8.2)
Requirement already satisfied: contourpy>=1.0.1 in /Users/zuremakapose/anaconda3/lib/python3.11/site-packages (from matplotlib!=3.6.1,>=3.4->seaborn) (1.0.5)
Requirement already satisfied: cycler>=0.10 in /Users/zuremakapose/anaconda3/lib/python3.11/site-packages (from matplotlib!=3.6.1,>=3.4->seaborn) (0.12.1)
Requirement already satisfied: fonttools>=4.22.0 in /Users/zuremakapose/anaconda3/lib/python3.11/site-packages (from matplotlib!=3.6.1,>=3.4->seaborn) (4.48.1)
Requirement already satisfied: kiwisolver>=1.3.1 in /Users/zuremakapose/anaconda3/lib/python3.11/site-packages (from matplotlib!=3.6.1,>=3.4->seaborn) (1.4.5)
Requirement already satisfied: packaging>=20.0 in /Users/zuremakapose/anaconda3/lib/python3.11/site-packages (from matplotlib!=3.6.1,>=3.4->seaborn) (23.1)
Requirement already satisfied: pillow>=8 in /Users/zuremakapose/anaconda3/lib/python3.11/site-packages (from matplotlib!=3.6.1,>=3.4->seaborn) (10.0.1)
Requirement already satisfied: pyparsing>=2.3.1 in /Users/zuremakapose/anaconda3/lib/python3.11/site-packages (from matplotlib!=3.6.1,>=3.4->seaborn) (3.1.1)
Requirement already satisfied: python-dateutil>=2.7 in /Users/zuremakapose/anaconda3/lib/python3.11/site-packages (from matplotlib!=3.6.1,>=3.4->seaborn) (2.8.2)
Requirement already satisfied: pytz>=2020.1 in /Users/zuremakapose/anaconda3/lib/python3.11/site-packages (from pandas>=1.2->seaborn) (2023.3.post1)
Requirement already satisfied: tzdata>=2022.1 in /Users/zuremakapose/anaconda3/lib/python3.11/site-packages (from pandas>=1.2->seaborn) (2023.3)
Requirement already satisfied: six>=1.5 in /Users/zuremakapose/anaconda3/lib/python3.11/site-packages (from python-dateutil>=2.7->matplotlib!=3.6.1,>=3.4->seaborn) (1.16.0)
Note: you may need to restart the kernel to use updated packages.
# %pip install matplotlib
%pip install matplotlib
Requirement already satisfied: matplotlib in /Users/zuremakapose/anaconda3/lib/python3.11/site-packages (3.8.2)
Requirement already satisfied: contourpy>=1.0.1 in /Users/zuremakapose/anaconda3/lib/python3.11/site-packages (from matplotlib) (1.0.5)
Requirement already satisfied: cycler>=0.10 in /Users/zuremakapose/anaconda3/lib/python3.11/site-packages (from matplotlib) (0.12.1)
Requirement already satisfied: fonttools>=4.22.0 in /Users/zuremakapose/anaconda3/lib/python3.11/site-packages (from matplotlib) (4.48.1)
Requirement already satisfied: kiwisolver>=1.3.1 in /Users/zuremakapose/anaconda3/lib/python3.11/site-packages (from matplotlib) (1.4.5)
Requirement already satisfied: numpy<2,>=1.21 in /Users/zuremakapose/anaconda3/lib/python3.11/site-packages (from matplotlib) (1.24.3)
Requirement already satisfied: packaging>=20.0 in /Users/zuremakapose/anaconda3/lib/python3.11/site-packages (from matplotlib) (23.1)
Requirement already satisfied: pillow>=8 in /Users/zuremakapose/anaconda3/lib/python3.11/site-packages (from matplotlib) (10.0.1)
Requirement already satisfied: pyparsing>=2.3.1 in /Users/zuremakapose/anaconda3/lib/python3.11/site-packages (from matplotlib) (3.1.1)
Requirement already satisfied: python-dateutil>=2.7 in /Users/zuremakapose/anaconda3/lib/python3.11/site-packages (from matplotlib) (2.8.2)
Requirement already satisfied: six>=1.5 in /Users/zuremakapose/anaconda3/lib/python3.11/site-packages (from python-dateutil>=2.7->matplotlib) (1.16.0)
Note: you may need to restart the kernel to use updated packages.
# Package and Dataset Imports
#Import the following packages:
import pandas as pd # data science essentials
import numpy as np # mathematical essentials
import matplotlib.pyplot as plt # data visualization
import seaborn as sns # enhanced data viz

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# loading data
bwght = pd.read_excel('birthweight (1).xlsx')

kaggle_test = pd.read_csv('kaggle_test_data.csv')

# concatenating datasets together for missing value analysis and feature engineering
bwght['set'] = 'Not Kaggle'
kaggle_test ['set'] = 'Kaggle'

# concatenating both datasets together for missing values and feature engineering
df = pd.concat(objs = [bwght, kaggle_test],
                    axis = 0,
                    ignore_index = False)

# setting pandas print options
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 100)


# displaying the head of the dataset
df.head(n = 5)
bwt_id	mage	meduc	monpre	npvis	fage	feduc	omaps	fmaps	cigs	drink	male	mwhte	mblck	moth	fwhte	fblck	foth	bwght	set
0	bwt_1	28	12.0	2	10.0	31.0	17.0	8.0	9.0	0.0	0.0	0	1	0	0	1	0	0	3317.0	Not Kaggle
1	bwt_2	21	NaN	1	6.0	21.0	NaN	8.0	9.0	NaN	NaN	0	1	0	0	1	0	0	1160.0	Not Kaggle
2	bwt_3	27	15.0	2	11.0	32.0	16.0	9.0	9.0	0.0	0.0	1	1	0	0	1	0	0	4706.0	Not Kaggle
3	bwt_4	33	17.0	1	20.0	39.0	17.0	9.0	10.0	0.0	0.0	0	0	0	1	0	0	1	3289.0	Not Kaggle
4	bwt_5	30	15.0	2	12.0	36.0	16.0	9.0	9.0	NaN	NaN	1	1	0	0	1	0	0	3490.0	Not Kaggle
########################################
# standard_scaler
########################################
def standard_scaler(df):
    """
    Standardizes a dataset (mean = 0, variance = 1). Returns a new DataFrame.
    Requires sklearn.preprocessing.StandardScaler()
    
    PARAMETERS
    ----------
    df     | DataFrame to be used for scaling
    """

    # INSTANTIATING a StandardScaler() object
    scaler = StandardScaler(copy = True)


    # FITTING the scaler with the data
    scaler.fit(df)


    # TRANSFORMING our data after fit
    x_scaled = scaler.transform(df)

    
    # converting scaled data into a DataFrame
    new_df = pd.DataFrame(x_scaled)


    # reattaching column names
    new_df.columns = list(df.columns)
    
    return new_df


########################################
# plot_feature_importances
########################################
def plot_feature_importances(model, train, export = False):
    """
    Plots the importance of features from a CART model.
    
    PARAMETERS
    ----------
    model  : CART model
    train  : explanatory variable training data
    export : whether or not to export as a .png image, default False
    """
    
    # number of features to plot
    n_features = train.shape[1]
    
    # setting plot window
    fig, ax = plt.subplots(figsize=(12,9))
    
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), train.columns)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    
    if export == True:
        plt.savefig('./analysis_images/Feature_Importance.png')
# Convert birthweight into a binary classification problem (low birthweight as 1, otherwise 0)
df['low_bwght'] = (df['bwght'] < 2500).astype(int)
# See the last five row
df.tail()
bwt_id	mage	meduc	monpre	npvis	fage	feduc	omaps	fmaps	cigs	drink	male	mwhte	mblck	moth	fwhte	fblck	foth	bwght	set	low_bwght
59	bwt_427	32	12.0	2	12.0	30.0	12.0	9.0	9.0	0.0	0.0	1	1	0	0	1	0	0	NaN	Kaggle	0
60	bwt_438	40	16.0	2	12.0	34.0	13.0	8.0	9.0	NaN	NaN	1	1	0	0	1	0	0	NaN	Kaggle	0
61	bwt_449	34	12.0	1	15.0	36.0	16.0	9.0	9.0	0.0	0.0	1	1	0	0	1	0	0	NaN	Kaggle	0
62	bwt_463	29	15.0	1	10.0	28.0	14.0	9.0	9.0	0.0	0.0	1	1	0	0	1	0	0	NaN	Kaggle	0
63	bwt_468	31	17.0	2	NaN	33.0	15.0	8.0	9.0	0.0	0.0	1	1	0	0	1	0	0	NaN	Kaggle	0
# Dropping post-event horizon features
df1=df.drop(['omaps','fmaps'],axis   = 1)
# Data cleaning and preprocessing
# Find the types of data which contain the dataset
df1.dtypes
bwt_id        object
mage           int64
meduc        float64
monpre         int64
npvis        float64
fage         float64
feduc        float64
cigs         float64
drink        float64
male           int64
mwhte          int64
mblck          int64
moth           int64
fwhte          int64
fblck          int64
foth           int64
bwght        float64
set           object
low_bwght      int64
dtype: object
# Find the shape of dataset
df1.shape
(473, 19)
# Find the columns names of dataset
df1.columns
Index(['bwt_id', 'mage', 'meduc', 'monpre', 'npvis', 'fage', 'feduc', 'cigs', 'drink', 'male', 'mwhte', 'mblck', 'moth', 'fwhte', 'fblck', 'foth', 'bwght', 'set', 'low_bwght'], dtype='object')
# See the nonunique values in dataset
df1.nunique()
bwt_id       473
mage          28
meduc         12
monpre         9
npvis         25
fage          36
feduc         12
cigs          11
drink          3
male           2
mwhte          2
mblck          2
moth           2
fwhte          2
fblck          2
foth           2
bwght        272
set            2
low_bwght      2
dtype: int64
# This description occur berfore cleaning the dataset only for analyse
df1.describe()
mage	meduc	monpre	npvis	fage	feduc	cigs	drink	male	mwhte	mblck	moth	fwhte	fblck	foth	bwght	low_bwght
count	473.000000	467.000000	473.000000	452.000000	472.000000	463.000000	440.000000	441.000000	473.000000	473.000000	473.000000	473.000000	473.000000	473.000000	473.000000	409.000000	473.000000
mean	29.784355	13.680942	2.205074	11.535398	32.169492	13.866091	1.172727	0.020408	0.505285	0.871036	0.073996	0.054968	0.877378	0.073996	0.048626	3189.870416	0.169133
std	5.105664	2.136672	1.330149	3.924575	6.097656	2.222327	4.211621	0.297017	0.500501	0.335515	0.262041	0.228160	0.328350	0.262041	0.215312	748.450059	0.375266
min	16.000000	3.000000	0.000000	0.000000	18.000000	6.000000	0.000000	0.000000	0.000000	0.000000	0.000000	0.000000	0.000000	0.000000	0.000000	360.000000	0.000000
25%	26.000000	12.000000	1.000000	10.000000	28.000000	12.000000	0.000000	0.000000	0.000000	1.000000	0.000000	0.000000	1.000000	0.000000	0.000000	2780.000000	0.000000
50%	30.000000	13.000000	2.000000	12.000000	32.000000	14.000000	0.000000	0.000000	1.000000	1.000000	0.000000	0.000000	1.000000	0.000000	0.000000	3340.000000	0.000000
75%	33.000000	16.000000	3.000000	13.000000	36.000000	16.000000	0.000000	0.000000	1.000000	1.000000	0.000000	0.000000	1.000000	0.000000	0.000000	3686.000000	0.000000
max	44.000000	17.000000	8.000000	36.000000	62.000000	17.000000	30.000000	6.000000	1.000000	1.000000	1.000000	1.000000	1.000000	1.000000	1.000000	5089.000000	1.000000
# Checking each feature for missing values
df1.isnull().sum()
bwt_id        0
mage          0
meduc         6
monpre        0
npvis        21
fage          1
feduc        10
cigs         33
drink        32
male          0
mwhte         0
mblck         0
moth          0
fwhte         0
fblck         0
foth          0
bwght        64
set           0
low_bwght     0
dtype: int64
# Total missing value in dataset
df1.isnull().sum().sum()
167
# See the flag columns by the help of mv_flagger
df1.columns
Index(['bwt_id', 'mage', 'meduc', 'monpre', 'npvis', 'fage', 'feduc', 'cigs', 'drink', 'male', 'mwhte', 'mblck', 'moth', 'fwhte', 'fblck', 'foth', 'bwght', 'set', 'low_bwght'], dtype='object')
Exploratory Data Analysis (EDA)

# Not use df1 use df2 for further working because it not contain misiing values
# View Data Summary after preprocessing the dataset
# Display the first few rows of the DataFrame
df1.head()
bwt_id	mage	meduc	monpre	npvis	fage	feduc	cigs	drink	male	mwhte	mblck	moth	fwhte	fblck	foth	bwght	set	low_bwght
0	bwt_1	28	12.0	2	10.0	31.0	17.0	0.0	0.0	0	1	0	0	1	0	0	3317.0	Not Kaggle	0
1	bwt_2	21	NaN	1	6.0	21.0	NaN	NaN	NaN	0	1	0	0	1	0	0	1160.0	Not Kaggle	1
2	bwt_3	27	15.0	2	11.0	32.0	16.0	0.0	0.0	1	1	0	0	1	0	0	4706.0	Not Kaggle	0
3	bwt_4	33	17.0	1	20.0	39.0	17.0	0.0	0.0	0	0	0	1	0	0	1	3289.0	Not Kaggle	0
4	bwt_5	30	15.0	2	12.0	36.0	16.0	NaN	NaN	1	1	0	0	1	0	0	3490.0	Not Kaggle	0
# Get summary statistics
df1.describe()
mage	meduc	monpre	npvis	fage	feduc	cigs	drink	male	mwhte	mblck	moth	fwhte	fblck	foth	bwght	low_bwght
count	473.000000	467.000000	473.000000	452.000000	472.000000	463.000000	440.000000	441.000000	473.000000	473.000000	473.000000	473.000000	473.000000	473.000000	473.000000	409.000000	473.000000
mean	29.784355	13.680942	2.205074	11.535398	32.169492	13.866091	1.172727	0.020408	0.505285	0.871036	0.073996	0.054968	0.877378	0.073996	0.048626	3189.870416	0.169133
std	5.105664	2.136672	1.330149	3.924575	6.097656	2.222327	4.211621	0.297017	0.500501	0.335515	0.262041	0.228160	0.328350	0.262041	0.215312	748.450059	0.375266
min	16.000000	3.000000	0.000000	0.000000	18.000000	6.000000	0.000000	0.000000	0.000000	0.000000	0.000000	0.000000	0.000000	0.000000	0.000000	360.000000	0.000000
25%	26.000000	12.000000	1.000000	10.000000	28.000000	12.000000	0.000000	0.000000	0.000000	1.000000	0.000000	0.000000	1.000000	0.000000	0.000000	2780.000000	0.000000
50%	30.000000	13.000000	2.000000	12.000000	32.000000	14.000000	0.000000	0.000000	1.000000	1.000000	0.000000	0.000000	1.000000	0.000000	0.000000	3340.000000	0.000000
75%	33.000000	16.000000	3.000000	13.000000	36.000000	16.000000	0.000000	0.000000	1.000000	1.000000	0.000000	0.000000	1.000000	0.000000	0.000000	3686.000000	0.000000
max	44.000000	17.000000	8.000000	36.000000	62.000000	17.000000	30.000000	6.000000	1.000000	1.000000	1.000000	1.000000	1.000000	1.000000	1.000000	5089.000000	1.000000
# Check data types and missing values
df1.info()
<class 'pandas.core.frame.DataFrame'>
Index: 473 entries, 0 to 63
Data columns (total 19 columns):
 #   Column     Non-Null Count  Dtype  
---  ------     --------------  -----  
 0   bwt_id     473 non-null    object 
 1   mage       473 non-null    int64  
 2   meduc      467 non-null    float64
 3   monpre     473 non-null    int64  
 4   npvis      452 non-null    float64
 5   fage       472 non-null    float64
 6   feduc      463 non-null    float64
 7   cigs       440 non-null    float64
 8   drink      441 non-null    float64
 9   male       473 non-null    int64  
 10  mwhte      473 non-null    int64  
 11  mblck      473 non-null    int64  
 12  moth       473 non-null    int64  
 13  fwhte      473 non-null    int64  
 14  fblck      473 non-null    int64  
 15  foth       473 non-null    int64  
 16  bwght      409 non-null    float64
 17  set        473 non-null    object 
 18  low_bwght  473 non-null    int64  
dtypes: float64(7), int64(10), object(2)
memory usage: 73.9+ KB
Descriptive Statistics:

# Plot histograms
df1.hist(figsize=(13, 11))

# Adjust layout 
plt.tight_layout()

# Display the plot
plt.show()
No description has been provided for this image
#missing value imputation


df1['meduc'] = df1['meduc'].fillna(df1['meduc'].mean())

df1['npvis'] = df1['npvis'].fillna(df1['npvis'].mean())

df1['fage'] = df1['fage'].fillna(df1['fage'].mean())

df1['feduc'] = df1['feduc'].fillna(df1['feduc'].mean())

df1['cigs'] = df1['cigs'].fillna(df1['cigs'].mean())

df1['drink'] = df1['drink'].fillna(df1['drink'].mean())

df1['bwght'] = df1['bwght'].fillna(df1['bwght'].mean())


#Checking the results
df1.isnull().sum()
bwt_id       0
mage         0
meduc        0
monpre       0
npvis        0
fage         0
feduc        0
cigs         0
drink        0
male         0
mwhte        0
mblck        0
moth         0
fwhte        0
fblck        0
foth         0
bwght        0
set          0
low_bwght    0
dtype: int64
Correlation Analysis:

#correlation heatmap
# numeric_columns = X_train.select_dtypes(include=['number']).columns

# Select specific numerical columns
numerical_columns = df1.select_dtypes(include=['number']).columns

# Compute the Pearson correlation matrix for selected columns
corr = df1[numerical_columns].corr(method='pearson')


# Compute Pearson correlation matrix
corr_matrix = df1.corr(numeric_only= True)
# Visualize correlation matrix using heatmap
plt.figure(figsize=(10, 6))
plt.title("Pearson compute correlation matrix")
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", color ='darkblue',linewidths=0.5)

# Adjust layout 
plt.tight_layout()

# Display the plot
plt.show()
No description has been provided for this image
# Calculate Pearson correlation coefficients with birthweight (bwght)
correlation_with_bwght = df1.corr(numeric_only= True)['bwght']

# Sort correlation coefficients in descending order of their absolute values
sorted_correlation = correlation_with_bwght.abs().sort_values(ascending=False)

# Print strong positive and negative correlations with birthweight
strong_positive_correlations = sorted_correlation[sorted_correlation > 0.5]
strong_negative_correlations = sorted_correlation[sorted_correlation < -0.5]

print("Strong Positive Correlations with Birthweight (bwght):")
print(strong_positive_correlations)

print("\nStrong Negative Correlations with Birthweight (bwght):")
print(strong_negative_correlations)
Strong Positive Correlations with Birthweight (bwght):
bwght        1.000000
low_bwght    0.761201
Name: bwght, dtype: float64

Strong Negative Correlations with Birthweight (bwght):
Series([], Name: bwght, dtype: float64)
# Create subplots with 2 columns
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Plot scatter plot for birthweight vs. average drinks per week drink
axs[1].scatter(df1['drink'], df1['bwght'], color='lightgreen', alpha=0.5)
axs[1].set_xlabel('Average Drinks per Week')
axs[1].set_ylabel('Birthweight (grams)')
axs[1].set_title('Scatter Plot of Drinks per Week vs. Birthweight ')
axs[1].grid(True)

# Plot scatter plot for birthweight vs. average cigarettes per day cigs
axs[0].scatter(df1['cigs'], df1['bwght'], color='darkgreen', alpha=0.5)
axs[0].set_xlabel('Average Cigarettes per Day')
axs[0].set_ylabel('Birthweight (grams)')
axs[0].set_title('Scatter Plot of Cigarettes vs. Birthweight per Day')
axs[0].grid(True)

# Adjust layout 
plt.tight_layout()

# Display the plot
plt.show()
No description has been provided for this image
# Scatter plots to visualize relationships
plt.figure(figsize=(12, 6))
sns.scatterplot(data=df1, x='mage', y='bwght')
plt.title('Scatter plot of Birthweight vs Mother\'s Age')
plt.xlabel('Mother\'s Age')
plt.ylabel('Birthweight')
plt.grid(True)

# Adjust layout 
plt.tight_layout()

# Display the plot
plt.show()
No description has been provided for this image
# Boxplot: Mother's Education Level vs Birthweight
plt.figure(figsize=(12, 6))
sns.boxplot(x='meduc', y='bwght', data=df1)
plt.title("Mother's Education Level vs Birthweight")
plt.xlabel("Mother's Education Level")
plt.ylabel("Birthweight")
plt.grid(True)

# Adjust layout 
plt.tight_layout()

# Display the plot
plt.show()
No description has been provided for this image
# Pairplot: Visualizing pairwise relationships
sns.pairplot(df1[['mage', 'drink', 'npvis', 'meduc', 'bwght']])

# Adjust layout 
plt.tight_layout()

# Display the plot
plt.show()
No description has been provided for this imageFeature Engineering
import pandas as pd  # This line imports pandas with the alias pd

# Feature 1 
df1['mage_group'] = pd.cut(df1['mage'], bins=[0, 20, 30, 40, 100], labels=['<20', '20-29', '30-39', '>39'])

#Feature 2 
df1['age_diff'] = abs(df1['mage'] - df1['fage'])

#feature 3 
df1['mhealth'] = df1['cigs'] * df1['drink']
# correlation 
df1_corr = df1.corr(method = 'pearson',
                    numeric_only = True)
plt.figure(figsize = (10,8))
sns.heatmap(data = df1_corr[['bwght', 'low_bwght']],
            cmap = 'Greens',
            annot = True)
plt.title('Pearson Correlation Heatmap')
plt.show()
No description has been provided for this image
Logistic Regression model implementation

# Import model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
df1.columns
Index(['bwt_id', 'mage', 'meduc', 'monpre', 'npvis', 'fage', 'feduc', 'cigs', 'drink', 'male', 'mwhte', 'mblck', 'moth', 'fwhte', 'fblck', 'foth', 'bwght', 'set', 'low_bwght', 'mage_group', 'age_diff', 'mhealth'], dtype='object')
y_variable = 'low_bwght'

df1_features = ['mage', 'meduc', 'monpre', 'npvis', 'fage',
                'feduc', 'cigs', 'drink', 'male', 'mwhte',
                'mblck', 'moth', 'fwhte', 'fblck', 'foth', 'age_diff', 'mhealth']
# Scale the numeric features
scaler = StandardScaler()

df1[df1_features] = scaler.fit_transform(df1[df1_features])
df1.head()
bwt_id	mage	meduc	monpre	npvis	fage	feduc	cigs	drink	male	mwhte	mblck	moth	fwhte	fblck	foth	bwght	set	low_bwght	mage_group	age_diff	mhealth
0	bwt_1	-0.349855	-7.925973e-01	-0.154337	-0.400655	-0.192200	1.426882e+00	-2.890321e-01	-7.124055e-02	-1.010627	0.384783	-0.282681	-0.241175	0.373844	-0.282681	-0.226078	3317.0	Not Kaggle	0	20-29	-0.149686	-0.053542
1	bwt_2	-1.722333	3.043046e-16	-0.906928	-1.444437	-1.835653	1.376064e-16	-6.599197e-17	1.042197e-17	-1.010627	0.384783	-0.282681	-0.241175	0.373844	-0.282681	-0.226078	1160.0	Not Kaggle	1	20-29	-0.982092	-0.049236
2	bwt_3	-0.545924	6.219617e-01	-0.154337	-0.139710	-0.027855	9.715776e-01	-2.890321e-01	-7.124055e-02	0.989484	0.384783	-0.282681	-0.241175	0.373844	-0.282681	-0.226078	4706.0	Not Kaggle	0	20-29	0.405251	-0.053542
3	bwt_4	0.630486	1.565001e+00	-0.906928	2.208799	1.122562	1.426882e+00	-2.890321e-01	-7.124055e-02	-1.010627	-2.598865	-0.282681	4.146361	-2.674915	-0.282681	4.423259	3289.0	Not Kaggle	0	30-39	0.682720	-0.053542
4	bwt_5	0.042281	6.219617e-01	-0.154337	0.121236	0.629526	9.715776e-01	-6.599197e-17	1.042197e-17	0.989484	0.384783	-0.282681	-0.241175	0.373844	-0.282681	-0.226078	3490.0	Not Kaggle	0	20-29	0.682720	-0.049236
## parsing out testing data (needed for later) ##

# dataset for kaggle
kaggle_data = df1[ df1['set'] == 'Kaggle' ].copy()


# dataset for model building
df = df1[ df1['set'] == 'Not Kaggle' ].copy()


# dropping set identifier (kaggle)
kaggle_data.drop(labels = 'set',
                 axis = 1,
                 inplace = True)


# dropping set identifier (model building)
df.drop(labels = 'set',
        axis = 1,
        inplace = True)
y_variable = 'low_bwght'
x_features = ['mage', 'meduc', 'monpre', 'npvis', 'fage', 'feduc', 'cigs', 
              'drink', 'male', 'mwhte', 'mblck', 'moth', 'fwhte', 'fblck', 
              'foth', 'age_diff', 'mhealth'] 

# prepping data for train-test split
y_data = df[y_variable]


# removing non-numeric columns and missing values
x_data = df[x_features].copy().select_dtypes(include=[int, float]).dropna(axis = 1)


# storing remaining x_features after the step above
x_features = list(x_data.columns)
# train-test split with stratification
x_train, x_test, y_train, y_test = train_test_split(
            x_data,
            y_data,
            test_size    = 0.25,
            random_state = 219)

# results of train-test split
print(f"""
Original Dataset Dimensions
---------------------------
Observations (Rows): {df.shape[0]}
Features  (Columns): {df.shape[1]}


Training Data (X-side)
----------------------
Observations (Rows): {x_train.shape[0]}
Features  (Columns): {x_train.shape[1]}


Training Data (y-side)
----------------------
Feature Name:        {y_train.name}
Observations (Rows): {y_train.shape[0]}


Testing Data (X-side)
---------------------
Observations (Rows): {x_test.shape[0]}
Features  (Columns): {x_test.shape[1]}


Testing Data (y-side)
---------------------
Feature Name:        {y_test.name}
Observations (Rows): {y_test.shape[0]}""")
Original Dataset Dimensions
---------------------------
Observations (Rows): 409
Features  (Columns): 21


Training Data (X-side)
----------------------
Observations (Rows): 306
Features  (Columns): 17


Training Data (y-side)
----------------------
Feature Name:        low_bwght
Observations (Rows): 306


Testing Data (X-side)
---------------------
Observations (Rows): 103
Features  (Columns): 17


Testing Data (y-side)
---------------------
Feature Name:        low_bwght
Observations (Rows): 103
help(RandomForestClassifier)
Help on class RandomForestClassifier in module sklearn.ensemble._forest:

class RandomForestClassifier(ForestClassifier)
 |  RandomForestClassifier(n_estimators=100, *, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='sqrt', max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None, monotonic_cst=None)
 |  
 |  A random forest classifier.
 |  
 |  A random forest is a meta estimator that fits a number of decision tree
 |  classifiers on various sub-samples of the dataset and uses averaging to
 |  improve the predictive accuracy and control over-fitting.
 |  Trees in the forest use the best split strategy, i.e. equivalent to passing
 |  `splitter="best"` to the underlying :class:`~sklearn.tree.DecisionTreeRegressor`.
 |  The sub-sample size is controlled with the `max_samples` parameter if
 |  `bootstrap=True` (default), otherwise the whole dataset is used to build
 |  each tree.
 |  
 |  For a comparison between tree-based ensemble models see the example
 |  :ref:`sphx_glr_auto_examples_ensemble_plot_forest_hist_grad_boosting_comparison.py`.
 |  
 |  Read more in the :ref:`User Guide <forest>`.
 |  
 |  Parameters
 |  ----------
 |  n_estimators : int, default=100
 |      The number of trees in the forest.
 |  
 |      .. versionchanged:: 0.22
 |         The default value of ``n_estimators`` changed from 10 to 100
 |         in 0.22.
 |  
 |  criterion : {"gini", "entropy", "log_loss"}, default="gini"
 |      The function to measure the quality of a split. Supported criteria are
 |      "gini" for the Gini impurity and "log_loss" and "entropy" both for the
 |      Shannon information gain, see :ref:`tree_mathematical_formulation`.
 |      Note: This parameter is tree-specific.
 |  
 |  max_depth : int, default=None
 |      The maximum depth of the tree. If None, then nodes are expanded until
 |      all leaves are pure or until all leaves contain less than
 |      min_samples_split samples.
 |  
 |  min_samples_split : int or float, default=2
 |      The minimum number of samples required to split an internal node:
 |  
 |      - If int, then consider `min_samples_split` as the minimum number.
 |      - If float, then `min_samples_split` is a fraction and
 |        `ceil(min_samples_split * n_samples)` are the minimum
 |        number of samples for each split.
 |  
 |      .. versionchanged:: 0.18
 |         Added float values for fractions.
 |  
 |  min_samples_leaf : int or float, default=1
 |      The minimum number of samples required to be at a leaf node.
 |      A split point at any depth will only be considered if it leaves at
 |      least ``min_samples_leaf`` training samples in each of the left and
 |      right branches.  This may have the effect of smoothing the model,
 |      especially in regression.
 |  
 |      - If int, then consider `min_samples_leaf` as the minimum number.
 |      - If float, then `min_samples_leaf` is a fraction and
 |        `ceil(min_samples_leaf * n_samples)` are the minimum
 |        number of samples for each node.
 |  
 |      .. versionchanged:: 0.18
 |         Added float values for fractions.
 |  
 |  min_weight_fraction_leaf : float, default=0.0
 |      The minimum weighted fraction of the sum total of weights (of all
 |      the input samples) required to be at a leaf node. Samples have
 |      equal weight when sample_weight is not provided.
 |  
 |  max_features : {"sqrt", "log2", None}, int or float, default="sqrt"
 |      The number of features to consider when looking for the best split:
 |  
 |      - If int, then consider `max_features` features at each split.
 |      - If float, then `max_features` is a fraction and
 |        `max(1, int(max_features * n_features_in_))` features are considered at each
 |        split.
 |      - If "sqrt", then `max_features=sqrt(n_features)`.
 |      - If "log2", then `max_features=log2(n_features)`.
 |      - If None, then `max_features=n_features`.
 |  
 |      .. versionchanged:: 1.1
 |          The default of `max_features` changed from `"auto"` to `"sqrt"`.
 |  
 |      Note: the search for a split does not stop until at least one
 |      valid partition of the node samples is found, even if it requires to
 |      effectively inspect more than ``max_features`` features.
 |  
 |  max_leaf_nodes : int, default=None
 |      Grow trees with ``max_leaf_nodes`` in best-first fashion.
 |      Best nodes are defined as relative reduction in impurity.
 |      If None then unlimited number of leaf nodes.
 |  
 |  min_impurity_decrease : float, default=0.0
 |      A node will be split if this split induces a decrease of the impurity
 |      greater than or equal to this value.
 |  
 |      The weighted impurity decrease equation is the following::
 |  
 |          N_t / N * (impurity - N_t_R / N_t * right_impurity
 |                              - N_t_L / N_t * left_impurity)
 |  
 |      where ``N`` is the total number of samples, ``N_t`` is the number of
 |      samples at the current node, ``N_t_L`` is the number of samples in the
 |      left child, and ``N_t_R`` is the number of samples in the right child.
 |  
 |      ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,
 |      if ``sample_weight`` is passed.
 |  
 |      .. versionadded:: 0.19
 |  
 |  bootstrap : bool, default=True
 |      Whether bootstrap samples are used when building trees. If False, the
 |      whole dataset is used to build each tree.
 |  
 |  oob_score : bool or callable, default=False
 |      Whether to use out-of-bag samples to estimate the generalization score.
 |      By default, :func:`~sklearn.metrics.accuracy_score` is used.
 |      Provide a callable with signature `metric(y_true, y_pred)` to use a
 |      custom metric. Only available if `bootstrap=True`.
 |  
 |  n_jobs : int, default=None
 |      The number of jobs to run in parallel. :meth:`fit`, :meth:`predict`,
 |      :meth:`decision_path` and :meth:`apply` are all parallelized over the
 |      trees. ``None`` means 1 unless in a :obj:`joblib.parallel_backend`
 |      context. ``-1`` means using all processors. See :term:`Glossary
 |      <n_jobs>` for more details.
 |  
 |  random_state : int, RandomState instance or None, default=None
 |      Controls both the randomness of the bootstrapping of the samples used
 |      when building trees (if ``bootstrap=True``) and the sampling of the
 |      features to consider when looking for the best split at each node
 |      (if ``max_features < n_features``).
 |      See :term:`Glossary <random_state>` for details.
 |  
 |  verbose : int, default=0
 |      Controls the verbosity when fitting and predicting.
 |  
 |  warm_start : bool, default=False
 |      When set to ``True``, reuse the solution of the previous call to fit
 |      and add more estimators to the ensemble, otherwise, just fit a whole
 |      new forest. See :term:`Glossary <warm_start>` and
 |      :ref:`gradient_boosting_warm_start` for details.
 |  
 |  class_weight : {"balanced", "balanced_subsample"}, dict or list of dicts,             default=None
 |      Weights associated with classes in the form ``{class_label: weight}``.
 |      If not given, all classes are supposed to have weight one. For
 |      multi-output problems, a list of dicts can be provided in the same
 |      order as the columns of y.
 |  
 |      Note that for multioutput (including multilabel) weights should be
 |      defined for each class of every column in its own dict. For example,
 |      for four-class multilabel classification weights should be
 |      [{0: 1, 1: 1}, {0: 1, 1: 5}, {0: 1, 1: 1}, {0: 1, 1: 1}] instead of
 |      [{1:1}, {2:5}, {3:1}, {4:1}].
 |  
 |      The "balanced" mode uses the values of y to automatically adjust
 |      weights inversely proportional to class frequencies in the input data
 |      as ``n_samples / (n_classes * np.bincount(y))``
 |  
 |      The "balanced_subsample" mode is the same as "balanced" except that
 |      weights are computed based on the bootstrap sample for every tree
 |      grown.
 |  
 |      For multi-output, the weights of each column of y will be multiplied.
 |  
 |      Note that these weights will be multiplied with sample_weight (passed
 |      through the fit method) if sample_weight is specified.
 |  
 |  ccp_alpha : non-negative float, default=0.0
 |      Complexity parameter used for Minimal Cost-Complexity Pruning. The
 |      subtree with the largest cost complexity that is smaller than
 |      ``ccp_alpha`` will be chosen. By default, no pruning is performed. See
 |      :ref:`minimal_cost_complexity_pruning` for details.
 |  
 |      .. versionadded:: 0.22
 |  
 |  max_samples : int or float, default=None
 |      If bootstrap is True, the number of samples to draw from X
 |      to train each base estimator.
 |  
 |      - If None (default), then draw `X.shape[0]` samples.
 |      - If int, then draw `max_samples` samples.
 |      - If float, then draw `max(round(n_samples * max_samples), 1)` samples. Thus,
 |        `max_samples` should be in the interval `(0.0, 1.0]`.
 |  
 |      .. versionadded:: 0.22
 |  
 |  monotonic_cst : array-like of int of shape (n_features), default=None
 |      Indicates the monotonicity constraint to enforce on each feature.
 |        - 1: monotonic increase
 |        - 0: no constraint
 |        - -1: monotonic decrease
 |  
 |      If monotonic_cst is None, no constraints are applied.
 |  
 |      Monotonicity constraints are not supported for:
 |        - multiclass classifications (i.e. when `n_classes > 2`),
 |        - multioutput classifications (i.e. when `n_outputs_ > 1`),
 |        - classifications trained on data with missing values.
 |  
 |      The constraints hold over the probability of the positive class.
 |  
 |      Read more in the :ref:`User Guide <monotonic_cst_gbdt>`.
 |  
 |      .. versionadded:: 1.4
 |  
 |  Attributes
 |  ----------
 |  estimator_ : :class:`~sklearn.tree.DecisionTreeClassifier`
 |      The child estimator template used to create the collection of fitted
 |      sub-estimators.
 |  
 |      .. versionadded:: 1.2
 |         `base_estimator_` was renamed to `estimator_`.
 |  
 |  estimators_ : list of DecisionTreeClassifier
 |      The collection of fitted sub-estimators.
 |  
 |  classes_ : ndarray of shape (n_classes,) or a list of such arrays
 |      The classes labels (single output problem), or a list of arrays of
 |      class labels (multi-output problem).
 |  
 |  n_classes_ : int or list
 |      The number of classes (single output problem), or a list containing the
 |      number of classes for each output (multi-output problem).
 |  
 |  n_features_in_ : int
 |      Number of features seen during :term:`fit`.
 |  
 |      .. versionadded:: 0.24
 |  
 |  feature_names_in_ : ndarray of shape (`n_features_in_`,)
 |      Names of features seen during :term:`fit`. Defined only when `X`
 |      has feature names that are all strings.
 |  
 |      .. versionadded:: 1.0
 |  
 |  n_outputs_ : int
 |      The number of outputs when ``fit`` is performed.
 |  
 |  feature_importances_ : ndarray of shape (n_features,)
 |      The impurity-based feature importances.
 |      The higher, the more important the feature.
 |      The importance of a feature is computed as the (normalized)
 |      total reduction of the criterion brought by that feature.  It is also
 |      known as the Gini importance.
 |  
 |      Warning: impurity-based feature importances can be misleading for
 |      high cardinality features (many unique values). See
 |      :func:`sklearn.inspection.permutation_importance` as an alternative.
 |  
 |  oob_score_ : float
 |      Score of the training dataset obtained using an out-of-bag estimate.
 |      This attribute exists only when ``oob_score`` is True.
 |  
 |  oob_decision_function_ : ndarray of shape (n_samples, n_classes) or             (n_samples, n_classes, n_outputs)
 |      Decision function computed with out-of-bag estimate on the training
 |      set. If n_estimators is small it might be possible that a data point
 |      was never left out during the bootstrap. In this case,
 |      `oob_decision_function_` might contain NaN. This attribute exists
 |      only when ``oob_score`` is True.
 |  
 |  estimators_samples_ : list of arrays
 |      The subset of drawn samples (i.e., the in-bag samples) for each base
 |      estimator. Each subset is defined by an array of the indices selected.
 |  
 |      .. versionadded:: 1.4
 |  
 |  See Also
 |  --------
 |  sklearn.tree.DecisionTreeClassifier : A decision tree classifier.
 |  sklearn.ensemble.ExtraTreesClassifier : Ensemble of extremely randomized
 |      tree classifiers.
 |  sklearn.ensemble.HistGradientBoostingClassifier : A Histogram-based Gradient
 |      Boosting Classification Tree, very fast for big datasets (n_samples >=
 |      10_000).
 |  
 |  Notes
 |  -----
 |  The default values for the parameters controlling the size of the trees
 |  (e.g. ``max_depth``, ``min_samples_leaf``, etc.) lead to fully grown and
 |  unpruned trees which can potentially be very large on some data sets. To
 |  reduce memory consumption, the complexity and size of the trees should be
 |  controlled by setting those parameter values.
 |  
 |  The features are always randomly permuted at each split. Therefore,
 |  the best found split may vary, even with the same training data,
 |  ``max_features=n_features`` and ``bootstrap=False``, if the improvement
 |  of the criterion is identical for several splits enumerated during the
 |  search of the best split. To obtain a deterministic behaviour during
 |  fitting, ``random_state`` has to be fixed.
 |  
 |  References
 |  ----------
 |  .. [1] L. Breiman, "Random Forests", Machine Learning, 45(1), 5-32, 2001.
 |  
 |  Examples
 |  --------
 |  >>> from sklearn.ensemble import RandomForestClassifier
 |  >>> from sklearn.datasets import make_classification
 |  >>> X, y = make_classification(n_samples=1000, n_features=4,
 |  ...                            n_informative=2, n_redundant=0,
 |  ...                            random_state=0, shuffle=False)
 |  >>> clf = RandomForestClassifier(max_depth=2, random_state=0)
 |  >>> clf.fit(X, y)
 |  RandomForestClassifier(...)
 |  >>> print(clf.predict([[0, 0, 0, 0]]))
 |  [1]
 |  
 |  Method resolution order:
 |      RandomForestClassifier
 |      ForestClassifier
 |      sklearn.base.ClassifierMixin
 |      BaseForest
 |      sklearn.base.MultiOutputMixin
 |      sklearn.ensemble._base.BaseEnsemble
 |      sklearn.base.MetaEstimatorMixin
 |      sklearn.base.BaseEstimator
 |      sklearn.utils._estimator_html_repr._HTMLDocumentationLinkMixin
 |      sklearn.utils._metadata_requests._MetadataRequester
 |      builtins.object
 |  
 |  Methods defined here:
 |  
 |  __init__(self, n_estimators=100, *, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='sqrt', max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None, monotonic_cst=None)
 |      Initialize self.  See help(type(self)) for accurate signature.
 |  
 |  set_fit_request(self: sklearn.ensemble._forest.RandomForestClassifier, *, sample_weight: Union[bool, NoneType, str] = '$UNCHANGED$') -> sklearn.ensemble._forest.RandomForestClassifier
 |      Request metadata passed to the ``fit`` method.
 |      
 |      Note that this method is only relevant if
 |      ``enable_metadata_routing=True`` (see :func:`sklearn.set_config`).
 |      Please see :ref:`User Guide <metadata_routing>` on how the routing
 |      mechanism works.
 |      
 |      The options for each parameter are:
 |      
 |      - ``True``: metadata is requested, and passed to ``fit`` if provided. The request is ignored if metadata is not provided.
 |      
 |      - ``False``: metadata is not requested and the meta-estimator will not pass it to ``fit``.
 |      
 |      - ``None``: metadata is not requested, and the meta-estimator will raise an error if the user provides it.
 |      
 |      - ``str``: metadata should be passed to the meta-estimator with this given alias instead of the original name.
 |      
 |      The default (``sklearn.utils.metadata_routing.UNCHANGED``) retains the
 |      existing request. This allows you to change the request for some
 |      parameters and not others.
 |      
 |      .. versionadded:: 1.3
 |      
 |      .. note::
 |          This method is only relevant if this estimator is used as a
 |          sub-estimator of a meta-estimator, e.g. used inside a
 |          :class:`~sklearn.pipeline.Pipeline`. Otherwise it has no effect.
 |      
 |      Parameters
 |      ----------
 |      sample_weight : str, True, False, or None,                     default=sklearn.utils.metadata_routing.UNCHANGED
 |          Metadata routing for ``sample_weight`` parameter in ``fit``.
 |      
 |      Returns
 |      -------
 |      self : object
 |          The updated object.
 |  
 |  set_score_request(self: sklearn.ensemble._forest.RandomForestClassifier, *, sample_weight: Union[bool, NoneType, str] = '$UNCHANGED$') -> sklearn.ensemble._forest.RandomForestClassifier
 |      Request metadata passed to the ``score`` method.
 |      
 |      Note that this method is only relevant if
 |      ``enable_metadata_routing=True`` (see :func:`sklearn.set_config`).
 |      Please see :ref:`User Guide <metadata_routing>` on how the routing
 |      mechanism works.
 |      
 |      The options for each parameter are:
 |      
 |      - ``True``: metadata is requested, and passed to ``score`` if provided. The request is ignored if metadata is not provided.
 |      
 |      - ``False``: metadata is not requested and the meta-estimator will not pass it to ``score``.
 |      
 |      - ``None``: metadata is not requested, and the meta-estimator will raise an error if the user provides it.
 |      
 |      - ``str``: metadata should be passed to the meta-estimator with this given alias instead of the original name.
 |      
 |      The default (``sklearn.utils.metadata_routing.UNCHANGED``) retains the
 |      existing request. This allows you to change the request for some
 |      parameters and not others.
 |      
 |      .. versionadded:: 1.3
 |      
 |      .. note::
 |          This method is only relevant if this estimator is used as a
 |          sub-estimator of a meta-estimator, e.g. used inside a
 |          :class:`~sklearn.pipeline.Pipeline`. Otherwise it has no effect.
 |      
 |      Parameters
 |      ----------
 |      sample_weight : str, True, False, or None,                     default=sklearn.utils.metadata_routing.UNCHANGED
 |          Metadata routing for ``sample_weight`` parameter in ``score``.
 |      
 |      Returns
 |      -------
 |      self : object
 |          The updated object.
 |  
 |  ----------------------------------------------------------------------
 |  Data and other attributes defined here:
 |  
 |  __abstractmethods__ = frozenset()
 |  
 |  __annotations__ = {'_parameter_constraints': <class 'dict'>}
 |  
 |  ----------------------------------------------------------------------
 |  Methods inherited from ForestClassifier:
 |  
 |  predict(self, X)
 |      Predict class for X.
 |      
 |      The predicted class of an input sample is a vote by the trees in
 |      the forest, weighted by their probability estimates. That is,
 |      the predicted class is the one with highest mean probability
 |      estimate across the trees.
 |      
 |      Parameters
 |      ----------
 |      X : {array-like, sparse matrix} of shape (n_samples, n_features)
 |          The input samples. Internally, its dtype will be converted to
 |          ``dtype=np.float32``. If a sparse matrix is provided, it will be
 |          converted into a sparse ``csr_matrix``.
 |      
 |      Returns
 |      -------
 |      y : ndarray of shape (n_samples,) or (n_samples, n_outputs)
 |          The predicted classes.
 |  
 |  predict_log_proba(self, X)
 |      Predict class log-probabilities for X.
 |      
 |      The predicted class log-probabilities of an input sample is computed as
 |      the log of the mean predicted class probabilities of the trees in the
 |      forest.
 |      
 |      Parameters
 |      ----------
 |      X : {array-like, sparse matrix} of shape (n_samples, n_features)
 |          The input samples. Internally, its dtype will be converted to
 |          ``dtype=np.float32``. If a sparse matrix is provided, it will be
 |          converted into a sparse ``csr_matrix``.
 |      
 |      Returns
 |      -------
 |      p : ndarray of shape (n_samples, n_classes), or a list of such arrays
 |          The class probabilities of the input samples. The order of the
 |          classes corresponds to that in the attribute :term:`classes_`.
 |  
 |  predict_proba(self, X)
 |      Predict class probabilities for X.
 |      
 |      The predicted class probabilities of an input sample are computed as
 |      the mean predicted class probabilities of the trees in the forest.
 |      The class probability of a single tree is the fraction of samples of
 |      the same class in a leaf.
 |      
 |      Parameters
 |      ----------
 |      X : {array-like, sparse matrix} of shape (n_samples, n_features)
 |          The input samples. Internally, its dtype will be converted to
 |          ``dtype=np.float32``. If a sparse matrix is provided, it will be
 |          converted into a sparse ``csr_matrix``.
 |      
 |      Returns
 |      -------
 |      p : ndarray of shape (n_samples, n_classes), or a list of such arrays
 |          The class probabilities of the input samples. The order of the
 |          classes corresponds to that in the attribute :term:`classes_`.
 |  
 |  ----------------------------------------------------------------------
 |  Methods inherited from sklearn.base.ClassifierMixin:
 |  
 |  score(self, X, y, sample_weight=None)
 |      Return the mean accuracy on the given test data and labels.
 |      
 |      In multi-label classification, this is the subset accuracy
 |      which is a harsh metric since you require for each sample that
 |      each label set be correctly predicted.
 |      
 |      Parameters
 |      ----------
 |      X : array-like of shape (n_samples, n_features)
 |          Test samples.
 |      
 |      y : array-like of shape (n_samples,) or (n_samples, n_outputs)
 |          True labels for `X`.
 |      
 |      sample_weight : array-like of shape (n_samples,), default=None
 |          Sample weights.
 |      
 |      Returns
 |      -------
 |      score : float
 |          Mean accuracy of ``self.predict(X)`` w.r.t. `y`.
 |  
 |  ----------------------------------------------------------------------
 |  Data descriptors inherited from sklearn.base.ClassifierMixin:
 |  
 |  __dict__
 |      dictionary for instance variables (if defined)
 |  
 |  __weakref__
 |      list of weak references to the object (if defined)
 |  
 |  ----------------------------------------------------------------------
 |  Methods inherited from BaseForest:
 |  
 |  apply(self, X)
 |      Apply trees in the forest to X, return leaf indices.
 |      
 |      Parameters
 |      ----------
 |      X : {array-like, sparse matrix} of shape (n_samples, n_features)
 |          The input samples. Internally, its dtype will be converted to
 |          ``dtype=np.float32``. If a sparse matrix is provided, it will be
 |          converted into a sparse ``csr_matrix``.
 |      
 |      Returns
 |      -------
 |      X_leaves : ndarray of shape (n_samples, n_estimators)
 |          For each datapoint x in X and for each tree in the forest,
 |          return the index of the leaf x ends up in.
 |  
 |  decision_path(self, X)
 |      Return the decision path in the forest.
 |      
 |      .. versionadded:: 0.18
 |      
 |      Parameters
 |      ----------
 |      X : {array-like, sparse matrix} of shape (n_samples, n_features)
 |          The input samples. Internally, its dtype will be converted to
 |          ``dtype=np.float32``. If a sparse matrix is provided, it will be
 |          converted into a sparse ``csr_matrix``.
 |      
 |      Returns
 |      -------
 |      indicator : sparse matrix of shape (n_samples, n_nodes)
 |          Return a node indicator matrix where non zero elements indicates
 |          that the samples goes through the nodes. The matrix is of CSR
 |          format.
 |      
 |      n_nodes_ptr : ndarray of shape (n_estimators + 1,)
 |          The columns from indicator[n_nodes_ptr[i]:n_nodes_ptr[i+1]]
 |          gives the indicator value for the i-th estimator.
 |  
 |  fit(self, X, y, sample_weight=None)
 |      Build a forest of trees from the training set (X, y).
 |      
 |      Parameters
 |      ----------
 |      X : {array-like, sparse matrix} of shape (n_samples, n_features)
 |          The training input samples. Internally, its dtype will be converted
 |          to ``dtype=np.float32``. If a sparse matrix is provided, it will be
 |          converted into a sparse ``csc_matrix``.
 |      
 |      y : array-like of shape (n_samples,) or (n_samples, n_outputs)
 |          The target values (class labels in classification, real numbers in
 |          regression).
 |      
 |      sample_weight : array-like of shape (n_samples,), default=None
 |          Sample weights. If None, then samples are equally weighted. Splits
 |          that would create child nodes with net zero or negative weight are
 |          ignored while searching for a split in each node. In the case of
 |          classification, splits are also ignored if they would result in any
 |          single class carrying a negative weight in either child node.
 |      
 |      Returns
 |      -------
 |      self : object
 |          Fitted estimator.
 |  
 |  ----------------------------------------------------------------------
 |  Readonly properties inherited from BaseForest:
 |  
 |  estimators_samples_
 |      The subset of drawn samples for each base estimator.
 |      
 |      Returns a dynamically generated list of indices identifying
 |      the samples used for fitting each member of the ensemble, i.e.,
 |      the in-bag samples.
 |      
 |      Note: the list is re-created at each call to the property in order
 |      to reduce the object memory footprint by not storing the sampling
 |      data. Thus fetching the property may be slower than expected.
 |  
 |  feature_importances_
 |      The impurity-based feature importances.
 |      
 |      The higher, the more important the feature.
 |      The importance of a feature is computed as the (normalized)
 |      total reduction of the criterion brought by that feature.  It is also
 |      known as the Gini importance.
 |      
 |      Warning: impurity-based feature importances can be misleading for
 |      high cardinality features (many unique values). See
 |      :func:`sklearn.inspection.permutation_importance` as an alternative.
 |      
 |      Returns
 |      -------
 |      feature_importances_ : ndarray of shape (n_features,)
 |          The values of this array sum to 1, unless all trees are single node
 |          trees consisting of only the root node, in which case it will be an
 |          array of zeros.
 |  
 |  ----------------------------------------------------------------------
 |  Methods inherited from sklearn.ensemble._base.BaseEnsemble:
 |  
 |  __getitem__(self, index)
 |      Return the index'th estimator in the ensemble.
 |  
 |  __iter__(self)
 |      Return iterator over estimators in the ensemble.
 |  
 |  __len__(self)
 |      Return the number of estimators in the ensemble.
 |  
 |  ----------------------------------------------------------------------
 |  Methods inherited from sklearn.base.BaseEstimator:
 |  
 |  __getstate__(self)
 |      Helper for pickle.
 |  
 |  __repr__(self, N_CHAR_MAX=700)
 |      Return repr(self).
 |  
 |  __setstate__(self, state)
 |  
 |  __sklearn_clone__(self)
 |  
 |  get_params(self, deep=True)
 |      Get parameters for this estimator.
 |      
 |      Parameters
 |      ----------
 |      deep : bool, default=True
 |          If True, will return the parameters for this estimator and
 |          contained subobjects that are estimators.
 |      
 |      Returns
 |      -------
 |      params : dict
 |          Parameter names mapped to their values.
 |  
 |  set_params(self, **params)
 |      Set the parameters of this estimator.
 |      
 |      The method works on simple estimators as well as on nested objects
 |      (such as :class:`~sklearn.pipeline.Pipeline`). The latter have
 |      parameters of the form ``<component>__<parameter>`` so that it's
 |      possible to update each component of a nested object.
 |      
 |      Parameters
 |      ----------
 |      **params : dict
 |          Estimator parameters.
 |      
 |      Returns
 |      -------
 |      self : estimator instance
 |          Estimator instance.
 |  
 |  ----------------------------------------------------------------------
 |  Methods inherited from sklearn.utils._metadata_requests._MetadataRequester:
 |  
 |  get_metadata_routing(self)
 |      Get metadata routing of this object.
 |      
 |      Please check :ref:`User Guide <metadata_routing>` on how the routing
 |      mechanism works.
 |      
 |      Returns
 |      -------
 |      routing : MetadataRequest
 |          A :class:`~sklearn.utils.metadata_routing.MetadataRequest` encapsulating
 |          routing information.
 |  
 |  ----------------------------------------------------------------------
 |  Class methods inherited from sklearn.utils._metadata_requests._MetadataRequester:
 |  
 |  __init_subclass__(**kwargs) from abc.ABCMeta
 |      Set the ``set_{method}_request`` methods.
 |      
 |      This uses PEP-487 [1]_ to set the ``set_{method}_request`` methods. It
 |      looks for the information available in the set default values which are
 |      set using ``__metadata_request__*`` class attributes, or inferred
 |      from method signatures.
 |      
 |      The ``__metadata_request__*`` class attributes are used when a method
 |      does not explicitly accept a metadata through its arguments or if the
 |      developer would like to specify a request value for those metadata
 |      which are different from the default ``None``.
 |      
 |      References
 |      ----------
 |      .. [1] https://www.python.org/dev/peps/pep-0487

# Initialize and train the model
rf_model = RandomForestClassifier(criterion       ='entropy',
                                      min_samples_leaf=1,
                                      n_estimators    =350,
                                      random_state    =702,
                                      bootstrap       =False,
                                      max_depth       = 10,
                                      warm_start      =True)
rf_model.fit(x_train,y_train)
y_pred_train = rf_model.predict(x_train)
# Print the predicted birthweight
print("Predicted Birthweight:", y_pred_train)#(binary: 0 for not low birth weight, 1 for low birth weight)
Predicted Birthweight: [0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 1 0
 0 0 1 0 0 0 0 0 1 0 0 0 1 0 0 1 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 1 1 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 1 0 0
 0 1 0 1 0 0 0 1 0 0 0 0 0 1 0 0 0 0 1 0 1 1 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 1 0 0 0 0 0 0 0 0 0 1 0 0 0 1 1 0 0 1 0 0 0
 1 1 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 1 0 0
 0 0 1 0 1 0 1 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 1 0 1
 0 1 0 0 0 0 0 0 0 0]
# Evaluate the classifier's performance
accuracy = accuracy_score(y_train, y_pred_train)
print(f"Accuracy: {accuracy:.2f}")
Accuracy: 0.93
#confusion matrix

# Generate confusion matrix
cm_train = confusion_matrix(y_train, y_pred_train)

# Calculate individual values
rf_tn = cm_train[0, 0]
rf_fp = cm_train[0, 1]
rf_fn = cm_train[1, 0]
rf_tp = cm_train[1, 1]

# Print confusion matrix
print(f"""
True Negatives : {rf_tn}
False Positives: {rf_fp}
False Negatives: {rf_fn}
True Positives : {rf_tp}
""")
True Negatives : 243
False Positives: 0
False Negatives: 22
True Positives : 41

Implementation of K-Nearest Neighbors Classification

# Import KNeighbours classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
# Define and train the KNN model
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(x_train, y_train)
y_pred_knn_train = knn_model.predict(x_train)

# y_pred contains the predicted birthweight values for the testing set
# Print the predicted birthweight
print("Predicted Birthweight:", y_pred_knn_train)#(binary: 0 for not low birth weight, 1 for low birth weight)
Predicted Birthweight: [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 1 0 0 0 0 0 0 1 0
 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 1 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 1 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0]
# Evaluate the classifier's performance
accuracy = accuracy_score(y_train, y_pred_knn_train)
print(f"Accuracy: {accuracy:.2f}")
Accuracy: 0.81
# Generate confusion matrix for KNN
cm_knn_train = confusion_matrix(y_train, y_pred_knn_train)

# Calculate individual values
knn_tn = cm_knn_train[0, 0]
knn_fp = cm_knn_train[0, 1]
knn_fn = cm_knn_train[1, 0]
knn_tp = cm_knn_train[1, 1]

# Print confusion matrix for KNN
print(f"""
KNN Confusion Matrix:
True Negatives : {knn_tn}
False Positives: {knn_fp}
False Negatives: {knn_fn}
True Positives : {knn_tp}
""")
KNN Confusion Matrix:
True Negatives : 238
False Positives: 5
False Negatives: 52
True Positives : 11

Decision Tree Classification implementation

# Import the model from sklearn
from sklearn.tree import DecisionTreeClassifier
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(x_train, y_train)
y_pred_dt_train = dt_model.predict(x_train)

# Print the predicted birthweight
print("Predicted Birthweight:", y_pred_dt_train)
Predicted Birthweight: [0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 1 0 0 0 0 1 0 1 0
 0 0 1 0 0 0 0 0 1 0 0 0 1 0 0 1 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1
 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 1 0 0 1 0 0 0 0 1 0 0 1 0 1 0 0 0 0 0 1 1 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 1 1 1
 0 1 1 1 0 0 0 1 0 0 0 0 0 1 0 0 1 0 1 0 1 1 0 0 0 0 0 0 1 0 0 0 0 0 0 0 1
 0 0 0 0 0 0 1 0 0 0 0 0 0 1 0 1 0 0 0 0 0 0 0 0 0 1 1 1 0 1 1 0 0 1 0 0 0
 1 1 0 0 0 0 0 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 1 0 0 0 0 1 0 1 1 0 1
 0 0 1 0 1 0 1 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 1 0 1
 0 1 0 0 0 0 0 0 0 0]
# Evaluate the classifier's performance
accuracy = accuracy_score(y_train, y_pred_dt_train)
print(f"Accuracy: {accuracy:.2f}")
Accuracy: 1.00
cm_dt_train = confusion_matrix(y_train, y_pred_dt_train)

dt_tn = cm_dt_train[0, 0]
dt_fp = cm_dt_train[0, 1]
dt_fn = cm_dt_train[1, 0]
dt_tp = cm_dt_train[1, 1]
# Print confusion matrix for Decision Tree
print(f"""
DT Confusion Matrix:
True Negatives : {dt_tn}
False Positives: {dt_fp}
False Negatives: {dt_fn}
True Positives : {dt_tp}
""")
DT Confusion Matrix:
True Negatives : 243
False Positives: 0
False Negatives: 1
True Positives : 62

# Provide a clear explanation of the confusion matrix
print("In this confusion matrix:")
print("- TP represents the number of correctly predicted positive cases.")
print("- TN represents the number of correctly predicted negative cases.")
print("- FP represents the number of incorrectly predicted positive cases.")
print("- FN represents the number of incorrectly predicted negative cases.")
In this confusion matrix:
- TP represents the number of correctly predicted positive cases.
- TN represents the number of correctly predicted negative cases.
- FP represents the number of incorrectly predicted positive cases.
- FN represents the number of incorrectly predicted negative cases.
# x-data
x_data_kaggle = kaggle_data[x_features].copy()


# y-data
y_data_kaggle = kaggle_data[y_variable]
# Make predictions on the scaled test data
kaggle_predictions = dt_model.predict(x_data_kaggle)
# Create a DataFrame for the Kaggle predictions
kaggle_predictions_df = pd.DataFrame({
    'bwt_id': kaggle_data['bwt_id'],  # This assumes 'bwt_id' is the identifier column in your kaggle_test_data
    'low_bwght': kaggle_predictions
})
kaggle_predictions_df
bwt_id	low_bwght
0	bwt_14	0
1	bwt_16	1
2	bwt_24	0
3	bwt_30	0
4	bwt_57	1
5	bwt_61	0
6	bwt_73	1
7	bwt_75	0
8	bwt_82	0
9	bwt_87	0
10	bwt_101	0
11	bwt_102	0
12	bwt_116	1
13	bwt_119	0
14	bwt_121	0
15	bwt_125	1
16	bwt_127	0
17	bwt_136	0
18	bwt_142	0
19	bwt_145	0
20	bwt_151	0
21	bwt_159	0
22	bwt_165	0
23	bwt_166	0
24	bwt_183	1
25	bwt_184	0
26	bwt_189	0
27	bwt_208	0
28	bwt_223	0
29	bwt_233	1
30	bwt_259	1
31	bwt_272	0
32	bwt_275	0
33	bwt_276	1
34	bwt_288	1
35	bwt_291	0
36	bwt_293	1
37	bwt_301	1
38	bwt_307	0
39	bwt_319	0
40	bwt_322	0
41	bwt_325	0
42	bwt_329	1
43	bwt_333	1
44	bwt_349	0
45	bwt_352	0
46	bwt_353	0
47	bwt_361	0
48	bwt_362	1
49	bwt_363	0
50	bwt_365	1
51	bwt_372	1
52	bwt_376	0
53	bwt_389	1
54	bwt_391	1
55	bwt_392	0
56	bwt_393	0
57	bwt_404	0
58	bwt_418	1
59	bwt_427	0
60	bwt_438	0
61	bwt_449	0
62	bwt_463	0
63	bwt_468	1
# Save the predictions to a CSV file for Kaggle submission
# submission_file_path = '/birthweight_kaggle_predictions.csv'
# kaggle_predictions_df.to_csv(submission_file_path, index=False)
kaggle_predictions_df.to_csv(path_or_buf = "./results/Zurema_pred_3.csv",
                     index       =  True)
Analysis Questions

Are there any strong positive or strong negative linear (Pearson) correlations with birthweight?

Yes, as finding the correlation with birthweight give a strong positive linear relations. Because its values reached the 1 mean if strong_positive_correlations = sorted_correlation[sorted_correlation > 0.5. The correlation coefficients to identify any variables that have strong correlations (positive or negative) with birthweight.

Is there an ofﬁcial threshold that signiﬁes when birthweight gets more dangerous? In other words, is there a cutoff point between a healthy birthweight and a non-healthy birthweight?

Yes, there is an official threshold that signifies when birthweight becomes more dangerous, especially in terms of identifying low birth weight (LBW) infants. Low birth weight is defined by the World Health Organization (WHO) as a birth weight of less than 2500 grams (or 5.5 pounds) regardless of gestational age (source: https://en.wikipedia.org/wiki/Birth_weight ). After looking at the data, it can seen that a good weight at birth is a really important factor of health in life, as per the source: "to show connection between birth weight and later-life conditions, including diabetes, obesity, tobacco smoking, and intelligence. Low birth weight is associated with neonatal infection and infant mortality" (source: https://en.wikipedia.org/wiki/Birth_weight ). If you set threshold 2500 it will give the low birth weight and greater then 2500 low birth rate less. Therefore, the 2500-gram threshold serves as an important indicator for identifying infants who may require additional medical attention and support.

In the code the birthweight was converted into a binary classification problem using the threshold of 2200 grams. A birthweight of less than 2200 grams was considered as "low birthweight" with a value of 1 while a birthweight greater than or equals to 2200 grams was considered as not "low birthweight" with a value of 0.

After transforming birthweight (bwght) using this threshold, did correlations and/or phi coefﬁcients improve? Why or why not?

Yes, it was improved. If the birthweight is below a certain threshold (e.g., 2500 grams) it is considered low, you can create a binary variable indicating low birthweight (1) and normal birthweight (0). If the correlations or phi coefficients do not show significant changes or weaken after the transformation, it suggests that transforming the birthweight variable may not have improved the relationships with other variables.

Which two features in your machine learning model had the largest impact on birthweight? Present one actionable insight for each of these.

Feature
Mother's age in years (mage)

Maternal age is often associated with birthweight, with studies suggesting that both younger and older maternal ages may be linked to adverse birth outcomes. Teenage mothers (those under 20 years old) may be at higher risk of delivering low birthweight infants due to factors such as inadequate prenatal care, socio-economic status, and biological immaturity. Maternal age can impact various aspects of pregnancy and maternal health, including fertility, pregnancy complications.

In summary, maternal age is an important factor to consider in prenatal care and public health interventions aimed at promoting healthy pregnancies and birth outcomes.

Feature
The age difference feature quantifies the age gap between parents, potentially highlighting social dynamics that influence prenatal health. Larger age differences might correlate with varying levels of maturity, financial stability, and support systems, which can impact birth outcomes. This feature can serve as a proxy for generational differences in parenting attitudes and access to healthcare resources. It allows the model to account for the complex interplay between parental ages and the child's birth weight.

In addiction this relationship in a single variable, we can simplify the model without losing the nuances of parental age interaction effects.

Feature
Mother Health and Lifestyle Score Combining the mother's smoking and drinking habits into a composite score provides a singular measure of prenatal health risk behaviors. This score can capture cumulative lifestyle risks that may have a synergistic negative impact on fetal development and birth weight. The scoring system simplifies the model by reducing two variables into one while preserving their combined predictive power. It allows for the easy identification of high-risk pregnancies that may require additional care or intervention. The score standardizes different lifestyle factors, providing a clear and interpretable metric for healthcare providers and prediction models.

Mother's education in years

Higher levels of maternal education are associated with better utilization of prenatal care services. Maternal education often correlates with healthier lifestyle choices during pregnancy. Women with higher education levels are more likely to engage in behaviors such as proper nutrition, regular exercise, avoidance of harmful substances. Maternal education is associated with better cognitive development and academic achievement in children.

Present your ﬁnal model's confusion matrix and explain what each error means (false positives and false negatives). Furthermore, explain which error is being controlled for given the cohort's focus on correctly predicting low birthweight, as well as why this error is more important to control than the other error.

Confusion Matrix Explanation for Decision Tree:

True Negatives (TN): The number of correctly predicted non-low birthweight instances.

False Positives (FP): The number of instances that were predicted as low birthweight but were actually non-low birthweight.

False Negatives (FN): The number of instances that were predicted as non-low birthweight but were actually low birthweight.

True Positives (TP): The number of correctly predicted low birthweight instances.

Explanation of Errors for Decision Tree:

False Positives (FP): These are cases where the model incorrectly predicts a baby to be of low birthweight when it is actually not. This means that resources might be unnecessarily allocated to these cases.

False Negatives (FN): These are cases where the model incorrectly predicts a baby to be of non-low birthweight when it is actually low birthweight. This is a more critical error as it might lead to lack of necessary medical attention or interventions to prevent complications for the baby.

Given the cohort’s focus on correctly predicting low birthweight, the more important error to control for is False Negatives (FN). This is because predicting a baby as non-low birthweight when it’s actually low birthweight can lead to serious health complications for the baby. It’s crucial to identify these cases early to ensure timely and appropriate medical interventions. Therefore, reducing the False Negatives should be a priority to enhance the model’s effectiveness and reliability in predicting low birthweight, regardless of the model used.

Data visualization of explanation

Data visualization is a powerful tool for gaining insights from data and communicating findings effectively. Each visualization should be accompanied by sufficient explanation to ensure clarity and interpretation.

Histogram:
• Importance: The histogram displays the distribution of birth weights in the dataset, with birth weight values along the x-axis and the frequency of occurrence on the y-axis. The bars represent the number of infants falling within each birth weight range. Understanding the distribution of birth weights is crucial for identifying trends, anomalies, and potential risk factors associated with low birth weight.

• Why it's useful: This visualization allows us to compare the correlation with other variables or comparison across demographic groups, and can provide valuable insights into factors influencing birth weight outcomes. The peak of the histogram indicates the most common birth weight range, which appears to be around 1000 grams. This suggests that a considerable number of infants in the dataset have birth weights close to this value.

Boxplot for Categorical Variables:
• Importance: This visualization allows to compare the distribution of birthweight across different levels of a categorical variable, in this case, the mother’s education level (meduc).

• Why it’s useful: It can reveal if there are significant differences in birthweight across various education levels. This might help in understanding if maternal education has an impact on birthweight.

Pairplot: • Importance: A pairplot provides a grid of scatterplots for each pair of features in the dataset, along with histograms on the diagonal. • Why it’s useful: It’s a quick way to visualize relationships between variables. This can be particularly useful in identifying potential patterns, correlations, or clusters. For instance, we can quickly spot if mage (mother’s age) and drink (alcohol consumption during pregnancy) have a discernible relationship with bwght (birthweight) and with each other. These visualizations help to uncover potential patterns or relationships in the data that might not be evident through descriptive statistics alone, thus aiding in more informed data analysis and modeling decisions.
