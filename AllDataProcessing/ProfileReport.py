# profile report del archivo /data/PMData.csv
import pandas as pd
# use pandas profiling to generate a profile report
from pandas_profiling import ProfileReport

# load the data
data = pd.read_csv('data/SHHS.csv')

# generate the profile report
profile = ProfileReport(data, title='PMData Profile Report', explorative=True)
# save the report to a file
profile.to_file('data/SHHS_profile_report.html')
