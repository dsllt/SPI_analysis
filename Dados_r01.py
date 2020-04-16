import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import matplotlib.dates as mdates
from lmoments3 import distr
from scipy.stats import gamma
import datetime

# Set working directory
os.chdir('/Users/dayanesallet/Documents/__Marea/Dona_Francisca/_Dados/')


# Import data
chuvas = pd.read_csv('./chuvas_C_02953008.csv', sep = ';',decimal = ',')
chuvas.head()
chuvas['Data'] # is not a datetime object, therefore needs to be converted
# Create new dataframe composed of only Date and Total rainfall
chuvas_red = pd.concat([chuvas['Data'], chuvas['Total']], axis = 1)

# Put date as date type object
chuvas_red['Data'] = pd.to_datetime(chuvas_red.Data, format = '%d/%m/%Y')

# Sort values by date
chuvas_red.sort_values('Data', inplace=True)
# Remove duplicated date values
chuvas_red.drop_duplicates(subset='Data', keep='first', inplace=True, ignore_index = True)
# Identify year of start and end of dataset 
start = datetime.datetime(chuvas_red.iloc[0,0].year,1,1)
end  = datetime.datetime(chuvas_red.iloc[-1,0].year,12,1)
# Create a dataframe to store the date range between start and stop
df = pd.DataFrame()
df['date'] = pd.date_range(start = start, end = end, freq='MS')
# Merge the dataframe with dates and data (in order to have all dates within period)
df1 = df.merge(chuvas_red, how = 'left', left_on='date', right_on='Data')
df1.drop('Data', axis = 1, inplace = True)

# In a new dataframe, put date as index
chuvas_red1 = df1.set_index(['date'])
# Make a columns with months of year
chuvas_red1['moy'] = chuvas_red1.index.month
# Make a column with years of dataset
chuvas_red1['Year'] = chuvas_red1.index.year 
chuvas_red1.Year.value_counts().sort_index()
'''
q = piv.columns.value_counts().sort_index()
w = chuvas_red1.Year.value_counts().sort_index()
print(q.index)
print(w.index)
falta o ano 1958
'''
# Make a pivot table with columns for each year and rows for each month
piv = pd.pivot_table(chuvas_red1, index = [chuvas_red1.index.month], 
                     columns = [chuvas_red1.index.year], values = ['Total'])
# Columns are multi-index, drop the first level('Total') of the columns 
piv.columns = piv.columns.droplevel()
piv.columns.value_counts().sort_index()

#Insert missing year of data
piv.insert(15, "1958", 'z')
nan_value = float("NaN")
piv['1958'] = nan_value


# Create the 3 month sum
piv_1 = piv.append([piv.iloc[0,:],piv.iloc[1,:]])

month_03_df = pd.DataFrame(columns=['JFM','FMA','MAM','AMJ','MJJ','JJA','JAS','ASO','SON','OND','NDJ','DJF'])
for ind in range(0,12):
    month = month_03_df.columns[ind]
    month_03_df[month] = piv_1.iloc[ind,:]+piv_1.iloc[ind+1,:]+piv_1.iloc[ind+2,:]


normal_df = pd.DataFrame()
for column in month_03_df:
    am = month_03_df.loc[:,column].sort_values(ascending=True, na_position='last').dropna()
    paras = distr.gam.lmom_fit(am)
    G_am = gamma.cdf(am, a=list(paras.values())[0], scale = list(paras.values())[2])
    N_am = distr.nor.ppf(G_am, loc=0, scale=1)
    a = month_03_df.loc[:, column].fillna(9999)
    JFM1 =  pd.DataFrame({ column: a})
    JFM2 = pd.DataFrame({ column:am,'Normal':N_am})
    JFM2 = JFM2.reindex(sorted(JFM2.index))
    MAR_CH = JFM1.merge(JFM2,how = 'left', on = [column], left_index = True,right_index = True)
    normal_df[column] = MAR_CH['Normal']

# Reorder months  
# Rename columns
normal_df.rename(columns = {'JFM': 3, 'FMA':4, 'MAM':5, 'AMJ':6, 'MJJ':7, 
                            'JJA':8, 'JAS':9, 'ASO':10, 'SON':11, 'OND':12,
                            'NDJ':1, 'DJF':2}, inplace = True)

# Reorder columns
order = np.arange(1,13)
normal_df_order = normal_df[order]
normal_df_order_transposed = normal_df_order.transpose()
years = normal_df_order_transposed.columns

# Generate a dataframe with all indexes in one column
glu = pd.DataFrame()
for year in years:
    glu = pd.concat([glu,normal_df_order_transposed.loc[:,year]], axis = 0)
    
# Put as index of the dataframe the dates
dates = pd.date_range(start='1943-01-01',end='2018-12-01', freq='MS')
glu_index = glu.set_index(dates)


# Plot
fig, ax = plt.subplots()
ax.plot(glu_index.loc['1990':'2018',0], marker = '.', linestyle = '-')
ax.set_xlabel('Year')
ax.set_ylabel('SPI')
ax.set_title('SPI over years')
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'));




####
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.graphics.tsaplots as sgt
import statsmodels.tsa.stattools as sts
from statsmodels.tsa.seasonal import seasonal_decompose
import seaborn as sns
sns.set()
# Visualization
chuvas_red1.Total.plot(figsize = (20,5))
plt.title('Precipitation over time', size=24)
plt.show();

# Seasonality
chuvas_dropped = chuvas_red1.fillna(method='ffill') #Forward fill missing values
chuvas_dropped.dropna(inplace=True) # Remove NaN in first months
s_dec_additive = seasonal_decompose(chuvas_dropped.Total, model ='additive')
s_dec_additive.plot()
plt.show();

# Autocorrelation
sgt.plot_acf(chuvas_dropped.Total, lags = 40, zero = False)
plt.title('ACF Precipitation', size=24)
plt.show()

# Partial Autocorrelation
sgt.plot_pacf(chuvas_dropped.Total, lags = 40, zero = False)
plt.title('PACF Precipitation', size=24)
plt.show()










