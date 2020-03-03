#########################################################
DATA_FILE  = '/Users/kaustavsaha/startup_project/Data.csv'
MODEL_PATH = '/Users/kaustavsaha/startup_project/Project/models/'
CHANGEPOINT_RANGE       = 0.95 ## Change Points can also be specified by manual intervention 
INTERVAL_WIDTH          = 0.95 ## Uncertainty Interval to produce a confidence interval (95%) around the forecast
YEARLY_SEASONALITY      = True
WEEKLY_SEASONALITY      = True  
GROWTH                  = 'logistic'
CHANGEPOINT_PRIOR_SCALE = .001 ## Trend Flexibility | Increase -> More Flexibility | Decrease -> Increase Rigidity
########################################################
