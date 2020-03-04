pd.set_option('display.max_columns',None)
%matplotlib inline
warnings.filterwarnings("ignore")
py.init_notebook_mode()
rcParams['figure.figsize'] = 10, 10

## All Config Settings
TESTING_YEAR                = 2017
TRAINING_YEAR_LIST          = [2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016]
NUMBER_OF_DAYS_TESTING_YEAR = 365
DATA_FILE                   = '/Users/kaustavsaha/startup_project/Project/data/Data.csv'
CHANGEPOINT_RANGE           = 0.95 ## Change Points can also be specified by manual intervention 
INTERVAL_WIDTH              = 0.95 ## Uncertainty Interval to produce a confidence interval (95%) around the forecast
YEARLY_SEASONALITY          = True
WEEKLY_SEASONALITY          = True  
GROWTH                      = 'logistic'
CHANGEPOINT_PRIOR_SCALE     = .005 ## Trend Flexibility | Increase -> More Flexibility Decrease -> Increase Rigidity
