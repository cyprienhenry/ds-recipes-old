
# Saving time in Python for common Data Science tasks

The purpose of this workbook is to act as a quick reference guide / list of recipes to perform common data science tasks in an efficient way.

# Table of contents
* [Best practices](#best_practices)
* [Compute correlation matrix](#corr_matrix)
* [Deal with dates](#dates)
* [Deal with NAs](#NAs)
* [Deal with constant / quasi-constant variables](#constant_variables)
* [Load data from multiple files and concatenate](#load_concat)
* [Force Python to reload a module](#module_reload) 
* [Save a dataframe to .csv](#save_csv)

___

# <a name="best_practices"></a> Best practices
There are often several ways of performing a given task in Python, such as iterating on a list for instance. Below I tried to list some common tasks with the corresponding best practice.

## > Iterate on a dictionary
The idiomatic way is to use `items()` to iterate accros the dictionary.


```python
d = {'First Name': 'John', 'Last Name': 'Doe'}
for key, val in d.items():
    print('Key ' + key + ' has value ' + val)
```

    Key First Name has value John
    Key Last Name has value Doe


## > Iterate on a list and access the index
Here the trick is to use `enumerate` rather than creating an index value that we would manually increment. 
`enumerate` makes things smooth:


```python
items = ['a', 'b', 'c']
for index, item in enumerate(items, start=0):   # default is zero
    print(index, item)
```

    0 a
    1 b
    2 c


# Deal with constant and quasi-variables

___

# <a name="dates"></a> Deal with dates
The common operations involve:

* converting a `string` to a `datetime` object or the reverse operation
* changing the format of the displayed date, like removing day, month, year information and keep only time information
* computing date / time differences

## > Parse a `string` into a `datetime` object


```python
import datetime
my_date = datetime.datetime.strptime('2012-07-22 16:19:00.539570', '%Y-%m-%d %H:%M:%S.%f')
print('The created object has the following type: %s' % type(my_date))
```

    The created object has the following type: <class 'datetime.datetime'>


## > Parse a `string` into `datetime` object when loading a .csv file

When loading data from a .csv file, it may be handy to take care of the date column at the same time, rather than modifying thedate column in another command. To do so, one may define a date parser function that will be applied on the date column when reading the csv.

To demonstrate this, we generate a fake .csv file with a date column. The date will be written as '15', '16', '17' for 2015, 2016 and 2017 respectively.


```python
import pandas as pd
from pandas import datetime

df = pd.DataFrame(['15', '16', '17'], columns = ['date'])
print(df.head())

# save dataframe
df.to_csv('./data/tmp.csv', index=False)
```

      date
    0   15
    1   16
    2   17


Let's now demonstrate how to define a date parser and load the csv properly.


```python
def date_parser(x):
    return datetime.strptime('20' + x, '%Y')

pd.read_csv('./data/tmp.csv', parse_dates = [0], date_parser=date_parser)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2015-01-01</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2016-01-01</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2017-01-01</td>
    </tr>
  </tbody>
</table>
</div>



## > Format a `datetime` object to a `string`

To format a `datetime` object to a string, the function to use is `strftime()`. The list of date formatters can be found [here](https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior). 


```python
import datetime
print('Output current time as a formatted string:')
datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
```

    Output current time as a formatted string:





    '2018-01-05 11:27:51.252837'



___

# <a name='NAs'></a>Deal with NAs

Dealing with missing values may involve:
* assessing the dataset to find the rows / columns containing missing values
* removing the records containing missing values
* filling the missing values with an alternate value

## > Drop NAs using `Pandas.DataFrame.dropna(axis, how, thresh)`

Pandas can look for NA column-wise or row-wise, drop a label if any or all values are NA and use a threshold for deletion.

## > Spot NAs using `apply()` and a lambda function

A lambda function combined with `Pandas.DataFrame.apply()` may be used to spot NAs and then take some action. Things to remember :
* if `x`is a DataFrame, then `x.isnull()` returns a boolean same-sized object indicating if the values are NA. 
* applying either `all()` or `any()` on `x.isnull()` checks if any or all of the value in the DataFrame are NA 

Let's take ta toy example:


```python
import pandas as pd
import numpy as np
df = pd.DataFrame([[0, np.nan, 1], [np.nan, np.nan, 2]], columns = ['x1', 'x2', 'x3'])
print(df)

# create a boolean mask with columns containing only NAs
cols = df.apply(lambda x: all(x.isnull()), axis=0)
print('\nThe following column(s) contain only NAs:')
print(df.columns[cols].values)
print('\n')

cols = df.apply(lambda x: any(x.isnull()), axis=0)
print('The following column(s) contain at least one NA')
print(df.columns[cols].values)
```

        x1  x2  x3
    0  0.0 NaN   1
    1  NaN NaN   2
    
    The following column(s) contain only NAs:
    ['x2']
    
    
    The following column(s) contain at least one NA
    ['x1' 'x2']


___

# <a name="corr_matrix"></a> Compute correlation matrix
It can be cumbersome to get the list of the most correlated pairs of variables in a data set. Here is an example of how to do so. 

We first create a toy dataset with 20 features and 5 correlated pairs of features to play with. Then, the correlation matrix is computed using the `pandas.DataFrame.corr()` command. To extract the relevant part of the matrix, a boolean mask is created with the `numpy.triu()` command. Finally the matrix is converted to a Pandas Series with a multi-index using the `pandas.DataFrame.stack()` command.


```python
from sklearn.datasets import make_classification
import pandas as pd
import numpy as np
```


```python
X, y = make_classification(n_features=10, n_informative=3, n_redundant=5, n_classes=2,
    n_clusters_per_class=2)
```


```python
col_names = ['feature_' + str(i) for i in range(X.shape[1])]
X = pd.DataFrame(X, columns=col_names)
```


```python
# compute correlation matrix
cor_matrix = X.corr()
```

Then we want to extract the upper-part of the matrix (becauses the correlation matrix is symetrical), to do so we will generate a boolean mask array from an upper triangular matrix.


```python
mask = np.triu(np.ones(cor_matrix.shape), k=1).astype(np.bool)
print('The following mask has been generated: \n')
print(mask)
```

    The following mask has been generated: 
    
    [[False  True  True  True  True  True  True  True  True  True]
     [False False  True  True  True  True  True  True  True  True]
     [False False False  True  True  True  True  True  True  True]
     [False False False False  True  True  True  True  True  True]
     [False False False False False  True  True  True  True  True]
     [False False False False False False  True  True  True  True]
     [False False False False False False False  True  True  True]
     [False False False False False False False False  True  True]
     [False False False False False False False False False  True]
     [False False False False False False False False False False]]


When applied to our correlation matrix, it will only keep the upper part, excluding the diagonal. We use `.abs()`at the end because we are interested in variables positively and negatively correlated.


```python
upper_cor_matrix = cor_matrix.where(mask).abs()
upper_cor_matrix
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>feature_0</th>
      <th>feature_1</th>
      <th>feature_2</th>
      <th>feature_3</th>
      <th>feature_4</th>
      <th>feature_5</th>
      <th>feature_6</th>
      <th>feature_7</th>
      <th>feature_8</th>
      <th>feature_9</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>feature_0</th>
      <td>NaN</td>
      <td>0.517328</td>
      <td>0.653118</td>
      <td>0.244725</td>
      <td>0.098232</td>
      <td>0.139354</td>
      <td>0.110006</td>
      <td>0.682568</td>
      <td>0.267424</td>
      <td>0.061726</td>
    </tr>
    <tr>
      <th>feature_1</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.296403</td>
      <td>0.120035</td>
      <td>0.125074</td>
      <td>0.914892</td>
      <td>0.053613</td>
      <td>0.370075</td>
      <td>0.875100</td>
      <td>0.444038</td>
    </tr>
    <tr>
      <th>feature_2</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.229692</td>
      <td>0.198927</td>
      <td>0.622841</td>
      <td>0.153837</td>
      <td>0.544542</td>
      <td>0.530619</td>
      <td>0.452341</td>
    </tr>
    <tr>
      <th>feature_3</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.187990</td>
      <td>0.345492</td>
      <td>0.145993</td>
      <td>0.514996</td>
      <td>0.218010</td>
      <td>0.685669</td>
    </tr>
    <tr>
      <th>feature_4</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.200426</td>
      <td>0.028883</td>
      <td>0.019659</td>
      <td>0.101719</td>
      <td>0.005976</td>
    </tr>
    <tr>
      <th>feature_5</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.121003</td>
      <td>0.190213</td>
      <td>0.840338</td>
      <td>0.392408</td>
    </tr>
    <tr>
      <th>feature_6</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.007935</td>
      <td>0.043498</td>
      <td>0.021896</td>
    </tr>
    <tr>
      <th>feature_7</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.116589</td>
      <td>0.587003</td>
    </tr>
    <tr>
      <th>feature_8</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.809456</td>
    </tr>
    <tr>
      <th>feature_9</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



To make it easier to use, the columns are stacked into rows, resulting in a multi-index Pandas Series:


```python
cor_series = upper_cor_matrix.stack().sort_values(ascending=False)
print('Display the most-correlated pairs:')
cor_series[cor_series > 0.6]
```

    Display the most-correlated pairs:





    feature_1  feature_5    0.914892
               feature_8    0.875100
    feature_5  feature_8    0.840338
    feature_8  feature_9    0.809456
    feature_3  feature_9    0.685669
    feature_0  feature_7    0.682568
               feature_2    0.653118
    feature_2  feature_5    0.622841
    dtype: float64



___

# <a name='load_concat'></a> Load data from multiple files and concatenate

Python `glob.glob()` command is used to find all the files in a directory matching a path pattern (**be careful, the files are return in an unsorted order**).

`map()` command is applied to the file list to read the csv files via a lambda function.

The result is eventually concatenated using `Pandas.concat()` function.


```python
import glob
import pandas as pd
files = glob.glob('./data/file*.csv')
print('List of files found:')
print(files)
df1 = pd.concat(map(lambda file: pd.read_csv(file, sep=','), files))
df1
```

    List of files found:
    ['./data/file1.csv', './data/file2.csv']





<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>john doe</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>donald trump</td>
    </tr>
    <tr>
      <th>0</th>
      <td>3</td>
      <td>bill clinton</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>hillary clinton</td>
    </tr>
  </tbody>
</table>
</div>



# <a name='module_reload'></a>Force Python to reload a module

Once a module has been loaded using `import module_name`, running this same command again will not reload the module. 

Say you are making changes on a module and testing the result interactively in a python shell. If you have loaded the module once and want to see the new changes you have to use:

```{python}
import importlib
importlib.reload(module_name)
```

___

# <a name='save_csv'></a>Save a dataframe to .csv file

Full documentation on the `Pandas.DataFrame.to_csv()` command may be found here: [official documentation](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.to_csv.html)

Ok this one may seem obvious but if you want your `Pandas.DataFrame` to be easily readable afterwards, you have to take care:
* if your dataframe doest not have an index, I suggest that you pass `index=False` to `Pandas.DataFrame.to_csv()` so that row numbers are not saved in the .csv file. This will prevent any trouble from occuring when loading the file again
* if your dataframe already has an index that you want to keep, the default value `index=True` in `Pandas.DataFrame.to_csv()` will work fine
* if you don't have an index yet but want to save the dataframe and use an existing column as the index for future reading, then you can specify `index=True` AND `index_label=my_future_index_column`
