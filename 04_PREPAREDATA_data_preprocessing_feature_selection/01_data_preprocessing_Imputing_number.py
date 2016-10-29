import pandas as pd
from io import StringIO
import sys
csv_data = '''A,B,C,D
1.0,2.0,3.0,4.0
5.0,6.0,,8.0
10.0,11.0,12.0,'''

# If you are using Python 2.7, you need
# to convert the string to unicode:
# csv_data = unicode(csv_data)



def main():
  #create a data table with NaN
  df = pd.read_csv(StringIO(csv_data))
  print(df)


  ############
  # basic
  ###########
  '''
  print(df.isnull().sum())
  print(df.dropna())
  print(df.dropna(axis=1))
  # only drop rows where all columns are NaN
  print(df.dropna(how='all') )
  # drop rows that have not at least 4 non-NaN values
  print(df.dropna(thresh=4))
  # only drop rows where NaN appear in specific columns (here: 'C')
  print(df.dropna(subset=['C']))
  '''


  ################
  # Imputing missing values
  ################
  from sklearn.preprocessing import Imputer

  imr = Imputer(missing_values='NaN', strategy='mean', axis=0)
  imr = imr.fit(df)
  imputed_data = imr.transform(df.values)
  print(imputed_data)

  return 0




main()
