############################################################
## Python Libraries

import pandas as pd
import matplotlib.pyplot as plt
import warnings
import numpy as np
import joblib
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from fbprophet.plot import add_changepoints_to_plot
import plotly.offline as py

## Config Settings
pd.set_option('display.max_columns',None)
# %matplotlib inline
warnings.filterwarnings("ignore")
py.init_notebook_mode()

#### Constants
DATA_FILE               = '/Users/kaustavsaha/startup_project/Data.csv'
CHANGEPOINT_RANGE       = 0.95 ## Change Points can also be specified by manual intervention 
INTERVAL_WIDTH          = 0.95 ## Uncertainty Interval to produce a confidence interval (95%) around the forecast
YEARLY_SEASONALITY      = True
WEEKLY_SEASONALITY      = True  
GROWTH                  = 'logistic'
CHANGEPOINT_PRIOR_SCALE = .001 ## Trend Flexibility | Increase -> More Flexibility Decrease -> Increase Rigidity

#####
def data_read(data_path):
    data = pd.read_csv(data_path)
    df   = pd.DataFrame(data)
    print(' #### Initial Data Load ####')
    print(df.head(5))
    return(df)

###############################
def date_time_feature_extractor(date,n):
    date_str      = str(date)
    date_str_list = date_str.split('/')
    data_parser   = int(date_str_list[n])
    return(data_parser)

############################
def data_augmentation(df):
    df['Month'] = df.date.apply(lambda x : date_time_feature_extractor(x,0))
    df['Day']   = df.date.apply(lambda x : date_time_feature_extractor(x,1))
    df['Year']  = df.date.apply(lambda x : date_time_feature_extractor(x,2))
    print('#####   Augmented Data  ########')
    print(df.head(5))
    return(df)

############################################
def basic_data_structure_exploration(df):
    print('#########  Data Shape  #########')
    print(df.shape)
    print('#########  Data Head   ##########')
    print(df.head())
    print('#########  Data Types   ##########')
    print(df.dtypes)
    print('######### Day Type Values ########')
    print(df.daytype.unique())
    print('########  Total number of NaN  ######')
    print(df.isna().sum())
    print('######### Unique Station Id #######')
    print(df.station_id.nunique())
    print('########  Unique Station Name  ######')
    print(df.stationname.nunique())

####################################################
## Total Rides Across Years
def total_rides_across_years(df):
    df_year_rides  = pd.DataFrame(df.groupby('Year')['rides'].sum())
    df_year_rides.plot(kind = 'bar', title = 'Total Rides in millions across years')

####################################################
## Busiest Stations
def top_total_rides_across_stations(df):
    total_station_rides  = pd.DataFrame(df.groupby('station_id')['rides'].sum())
    total_station_rides    = total_station_rides.sort_values('rides', ascending = False)
    top_station_rides = total_station_rides.head(10)
    print(top_station_rides)
    # top_station_rides.plot(kind = 'bar', ylim = (50000000, 100000000) , /
    #                      title = 'Top Stations with Total Rides (millions)')

#####################################################
## Station Count Across Years
def station_count(df):
    df_station_count  = pd.DataFrame(df.groupby('Year').station_id.nunique())
    df_station_count.plot(kind = 'bar', ylim = (140,146) , title = 'Total Number of Stations across years')

######################################################
def basic_data_visualization(df):
    total_rides_across_years(df)
    top_total_rides_across_stations(df)
    station_count(df)

########################################################
## TODO 
## def extra_visualization(df):
## Extra Visualization 
## X axis is Years
## Y axis is Box plot of Total Rides by station-ids 

########################################################
def zero_ride_station_date_analysis(df):
    
    print('##########################################')
    print('###### Stations with zero ridership cannot be used for any Time-Series Modelling') 
    station  = pd.DataFrame(df.groupby(['station_id'])['rides'].agg('sum'))
    print(station.query('rides == 0'))

########################################################
def model_data_creator(df):

    Model_Data_Repository = {}
    
    total_stations = df.station_id.unique()
    
    for individual_station in total_stations :
        model_data = df[(df['station_id'] == individual_station)]
        model_data = model_data.sort_values(by=['Year','Month','Day'])
        model_data = model_data[['date','rides']]
        Model_Data_Repository[individual_station] = model_data
     
    return(Model_Data_Repository)

##########################################################
def data_pre_processing(df):
    zero_ride_station_date_analysis(df)
    model_data_creator(df)

##########################################################
### TODO
### Imputation of Zero Rides with None as Prophet will take care of missing values


##########################################################
def model_data_creator(df):

    Model_Data_Repository = {}
    
    total_stations = df.station_id.unique()
    
    for individual_station in total_stations :
        model_data = df[(df['station_id'] == individual_station)]
        model_data = model_data.sort_values(by=['Year','Month','Day'])
        model_data = model_data[['date','rides']]
        Model_Data_Repository[individual_station] = model_data
     
    return(Model_Data_Repository)

##########################################################
def save_model(model, model_filename):
    
    ## Save Model to Disk
    model_filename = model_filename + '.sav'
    joblib.dump(model,model_filename)

##########################################################
def load_model(model_filename):
    
    ## Load Model from disk
    model_filename = model_filename + '.sav'
    model = joblib.load(model_filename)
    
    return(model)

##########################################################
## Using Facebook's Prophet Library for Univariate Time Series Analysis
def model_building(model_data,saturating_maximum):

    ## Inherently Prophet uses a Linear Model. For Growth Trends (Population of a metro-politan like 
    ## Chicago will grow in the future, hence ridership is bound to increase). I will use a logistic growth
    ## trend model. I use a configurable cap with a assumed max limit of max_ridership*1.1 
    ## (1.1 is a assumed constant that can also be an increasing sequence)
    
    model = Prophet(changepoint_range       = CHANGEPOINT_RANGE,
                    #growth                  = GROWTH ,
                    interval_width          = INTERVAL_WIDTH, 
                    yearly_seasonality      = YEARLY_SEASONALITY,
                    weekly_seasonality      = WEEKLY_SEASONALITY,
                    changepoint_prior_scale = CHANGEPOINT_PRIOR_SCALE) 
    
    fitted_model = model.fit(model_data)
    
    future = fitted_model.make_future_dataframe(periods=365)
    
    #future['cap']   = saturating_maximum   ## Saturating Maximum
    
    forecast = fitted_model.predict(future)
    
    Forecast_Plot = fitted_model.plot(forecast)
    
    Plot_Components = fitted_model.plot_components(forecast)
    
    Interactive_Plot = plot_plotly(fitted_model,forecast)
    py.iplot(Interactive_Plot)
    
    return(forecast)
    ## Change Points can be added to Forecast
    #change_point_plot = add_changepoints_to_plot(fig.gca(), fitted_model, forecast)

###########################################################################
def model_analyzer(model_name, model_data):
    
    ## Plotting the Time Series
    model_data.plot(kind='line', x='date', y='rides', 
                title='Time Series of Ride', figsize=(10,10), ylim=(0,(model_data.rides.max())*1.1))
    
    ## Column Renaming to fit Prophet Model
    model_data.rename(columns = {'date' : 'ds' , 'rides' : 'y'}, inplace = True)
    
    forecast = model_building(model_data, model_data.y.max()*1.1)

#####################################################
def model_iterator(Model_Data_Repository):
    
    Result_Repository = {}
    
    for model_name, model_data in Model_Data_Repository.items():
        
        model_analyzer(model_name, model_data)
        
        Result_Repository[model_name] = model_data
        
    return(Result_Repository)

####################################################
def model_iterator(Model_Data_Repository):
    
    Result_Repository = {}
    
    for model_name, model_data in Model_Data_Repository.items():
        
        model_analyzer(model_name, model_data)
        
        Result_Repository[model_name] = model_data
        
    return(Result_Repository)

######################################################
def driver_function():
    
    ## Service for Data Reading
    df = data_read(DATA_FILE)
    
    ## Service for Data Augmentation
    # df = data_augmentation(df)
    
    ## Service for Basic Analysis of Data Structure
    # basic_data_structure_exploration(df)
    
    ## Service for Basic Data Visualization
    # basic_data_visualization(df)
    
    ## Service for Data Imputation
    # data_pre_processing(df)
    
    ## Service for Model Data Repository Creation
    # Model_Data_Repository = model_data_creator(df)
    
    ## Service for building Result Repository
    # Result_Repository = model_iterator(Model_Data_Repository)
    
    #return(Result_Repository)
    
#######################################################
## Triggering Point
driver_function()
#Result_Repository = driver_function()

########################################################




























































































































































