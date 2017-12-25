# Saving time in Python for common Data Science tasks

# Table of contents
* [DataFrame manipulation](#dataframes)
* [Dates](#dates)
* [Exploratory Data Analysis](#eda)

## <a name="dataframes"></a>DataFrames


## <a name="dates"></a> Working with dates
The list of **date formatters can be found [here](https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior)**.

### From `datetime` to `string`:
To format a `datetime`object, the function to use is `strftime()`:

```python
import datetime
datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
```

```
'2017-12-25 19:15:42.389311'
```



### From `string`to `datetime`
To parse a string into a `datetime`object, one can use:

```python
datetime.datetime.strptime('2012-07-22 16:19:00.539570', '%Y-%m-%d %H:%M:%S.%f')
```

```
datetime.datetime(2012, 7, 22, 16, 19, 0, 539570)
```




### <a name="dataframes"></a>Exploratory Data Analysis



```python
pweave -f pandoc python.pmd
pandoc -o python.html python.md --template=GitHub.html5
# pweave -f md2html python.pmd
```


