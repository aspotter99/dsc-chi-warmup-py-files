# Quick .py File Example Warmup 

Feel free to do what you need to do for the code challenge (ie, rest), but go through this at some point before you start the project in earnest

![](viz/rest.gif)

## Efficient Data Science Workflows Use Functions in .py Files

In order to avoid the clutter of jupyter notebooks and to aid collaboration, an efficient data science workflow puts most of its work into **functions**.  

These functions are then put inside **.py files** and called to run through whole chunks of processing at a time

We'll run through an example below

### Imports


```python
#run this cell w/o changes

#data manip
import pandas as pd
import numpy as np

#tests
from test_background import pkl_dump, test_obj_dict, run_test_dict, run_test
```


```python
#__SOLUTION__
#run this cell w/o changes

#data manip
import pandas as pd
import numpy as np

#tests
from test_background import pkl_dump, test_obj_dict, run_test_dict, run_test
```

**Load in** fight_songs.csv from the data folder as a dataframe


```python
#__SOLUTION__

fight_songs = pd.read_csv('data/fight_songs.csv')

fight_songs.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>school</th>
      <th>conference</th>
      <th>song_name</th>
      <th>writers</th>
      <th>year</th>
      <th>student_writer</th>
      <th>official_song</th>
      <th>contest</th>
      <th>bpm</th>
      <th>sec_duration</th>
      <th>...</th>
      <th>win_won</th>
      <th>victory_win_won</th>
      <th>rah</th>
      <th>nonsense</th>
      <th>colors</th>
      <th>men</th>
      <th>opponents</th>
      <th>spelling</th>
      <th>trope_count</th>
      <th>spotify_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>Notre Dame</td>
      <td>Independent</td>
      <td>Victory March</td>
      <td>Michael J. Shea and John F. Shea</td>
      <td>1908</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>152</td>
      <td>64</td>
      <td>...</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>6</td>
      <td>15a3ShKX3XWKzq0lSS48yr</td>
    </tr>
    <tr>
      <td>1</td>
      <td>Baylor</td>
      <td>Big 12</td>
      <td>Old Fight</td>
      <td>Dick Baker and Frank Boggs</td>
      <td>1947</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>76</td>
      <td>99</td>
      <td>...</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>5</td>
      <td>2ZsaI0Cu4nz8DHfBkPt0Dl</td>
    </tr>
    <tr>
      <td>2</td>
      <td>Iowa State</td>
      <td>Big 12</td>
      <td>Iowa State Fights</td>
      <td>Jack Barker, Manly Rice, Paul Gnam, Rosalind K...</td>
      <td>1930</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>155</td>
      <td>55</td>
      <td>...</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>4</td>
      <td>3yyfoOXZQCtR6pfRJqu9pl</td>
    </tr>
    <tr>
      <td>3</td>
      <td>Kansas</td>
      <td>Big 12</td>
      <td>I'm a Jayhawk</td>
      <td>George "Dumpy" Bowles</td>
      <td>1912</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>137</td>
      <td>62</td>
      <td>...</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>3</td>
      <td>0JzbjZgcjugS0dmPjF9R89</td>
    </tr>
    <tr>
      <td>4</td>
      <td>Kansas State</td>
      <td>Big 12</td>
      <td>Wildcat Victory</td>
      <td>Harry E. Erickson</td>
      <td>1927</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>80</td>
      <td>67</td>
      <td>...</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>3</td>
      <td>4xxDK4g1OHhZ44sTFy8Ktm</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 23 columns</p>
</div>



Notice that the `Year` column has **some weird values** in it, and is an object dtype (specifically, a string)


```python
print(fight_songs.year.value_counts().head())

type(fight_songs['year'][0])
```

    Unknown    5
    1912       4
    1915       4
    1919       3
    1909       3
    Name: year, dtype: int64





    str



Write a quick function to **turn the value `"Unknown"` into `np.nan`**, wherever it appears in the dataframe.  

**Include two parameters** (objects inside the parens of the function that are inputs used inside the function): 
- the dataframe 
- the value being replaced as `np.nan`

(but it's ok to hardcode `np.nan` as what's replacing the value)

*Don't forget the docstring!*

Run it with the correct arguments as inputs and assign it to `fight_songs`


```python
def turn_value_null(#your code here):
    '''
    write a docstring!
    '''
    #your code here 
    #that creates a variable 
    #named `frame`
    
    
    return frame
    
fight_songs = turn_value_null(#your code here)
```


```python
#__SOLUTION__

def turn_value_null(frame, value):
    '''
    data cleaning: turn argument value to null
    
    input: 
        frame: dataframe
        value_to_nan: specific value to turn to np.nan
        
    output: frame w/ all values of value_to_nan replaced w/ np.nan
    '''
    frame = frame.replace(value, np.nan)
    return frame


fight_songs = turn_value_null(fight_songs, 'Unknown')

print(f'fight_songs now has {fight_songs.year.isnull().sum()} nulls')
```

    fight_songs now has 5 nulls


Now, write a function that **removes all the nulls**.

Again, use the dataframe as a parameter to the function 

Run it with the correct arguments as inputs and assign it to `fight_songs`


```python
def drop_nulls(#your code here):
    '''
    write a docstring
    '''
    
    frame = #your code here
    
    return frame

fight_songs = drop_nulls(#your code here)
```


```python
#__SOLUTION__

def drop_nulls(frame):
    '''
    data cleaning: drop rows w/ np.nan anywhere in frame
    
    input: dataframe 
    output: dataframe w/ rows w/ np.nan dropped
    '''
    
    frame = frame.dropna(axis=0, how="any")
    
    return frame

fight_songs = drop_nulls(fight_songs)

fight_songs.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 60 entries, 0 to 64
    Data columns (total 23 columns):
    school             60 non-null object
    conference         60 non-null object
    song_name          60 non-null object
    writers            60 non-null object
    year               60 non-null object
    student_writer     60 non-null object
    official_song      60 non-null object
    contest            60 non-null object
    bpm                60 non-null int64
    sec_duration       60 non-null int64
    fight              60 non-null object
    number_fights      60 non-null int64
    victory            60 non-null object
    win_won            60 non-null object
    victory_win_won    60 non-null object
    rah                60 non-null object
    nonsense           60 non-null object
    colors             60 non-null object
    men                60 non-null object
    opponents          60 non-null object
    spelling           60 non-null object
    trope_count        60 non-null int64
    spotify_id         60 non-null object
    dtypes: int64(4), object(19)
    memory usage: 11.2+ KB


Finally, write a function to **turn the `type` of the `year` column into an `int`**

This time, have the column be a parameter

Call the function and assign it to `fight_songs['year']` (written out for you)


```python
def turn_column_int(#your code here):
    '''
    your docstring here
    '''
    
    column = #your code here
    return column

fight_songs['year'] = turn_column_int(fight_songs['year'])
```


```python
#__SOLUTION__

def turn_column_int(column):
    '''
    data cleaning: turn column to float type
    
    input: column from dataframe
    output: column as float type
    '''
    column = column.astype(int)
    return column

fight_songs['year'] = turn_column_int(fight_songs['year'])

#used for tests:
# pkl_dump([
#     (
#         fight_songs,
#         'fight_songs'
        
#     )
# ])
```


```python
#run this to check you work

run_test(fight_songs, 'fight_songs')
```

Now, write a function that **loads fight_songs.csv** into a dataframe and returns it. *(It doesn't need any parameters!)*


```python
def load_fight_songs():
    
    '''
    write your docstring here
    '''
    
    df = #your code here
    
    return df
```


```python
#__SOLUTION__

def load_fight_songs():
    
    '''
    loads in fight_songs.csv from the data folder using pd.read_csv
    
    outputs: dataframe of fight_songs.csv
    '''
    
    df = pd.read_csv('data/fight_songs.csv')
    
    return df
```

## Now the fun part:

**Write a function** (which doesn't take in any parameters) that:
- **calls** `load_fight_songs`, `turn_value_null`, `drop_nulls`, and `turn_column_int` **sequentially**
    - (make sure to include all the specific parameters of those functions called above which are necessary to make them run)
    
    
- **returns** a dataframe at the end

It should be ***the same columns, rows and data*** as the dataframe we ended up with above


```python
def load_clean_fight_songs():
    '''
    write your docstring here!
    '''
    
    
    #write your code here
```


```python
#__SOLUTION__

def load_clean_fight_songs():
    '''
    runs sequentially:
        load_fight_songs() 
            - loads fight_songs.csv
        
        df = turn_value_null(df, 'Unknown') 
            - turns values "Unknown" to np.nan
        
        df = drop_nulls(df)
            - drops null rows from df
            
        df['year'] = turn_column_float(df['year'])
            - turns 'year' column to float type
            
    result:
        fight_songs.csv loaded and cleaned
    '''
    
    
    df = load_fight_songs()
    df = turn_value_null(df, 'Unknown')
    df = drop_nulls(df)
    df['year'] = turn_column_int(df['year'])
    
    return df
```


```python
#run this cell to test your code!

fight_songs_function_test = load_clean_fight_songs()

run_test(fight_songs_function_test, 'fight_songs')
```

## Now the *really* fun part:


Open a new **text file**, and **save it** as `data_cleaning.py`

**Write out import statements for pandas and numpy**, using the same aliases we always do, in the same manner we always do

**Write out** (in order to get your fingers some muscle memory time) **all five functions** you made above, in the order you made them

At the top of `data_cleaning.py`, **write** (again, don't copy) in triple-quotes (like a docstring) the following:

'''
These functions are used to clean the fight_songs.csv dataset

load_clean_fight_songs can be used without parameters to load the csv into a dataframe, run cleaning functions, and return a clean frame

Individually, they are used to:

\- load_fight_songs: load the csv into a dataframe

\- turn_value_null: change values of "Unknown" into np.nan

\- drop_nulls: drop the rows with np.nan values

\- turn_column_int: change the 'year' column into an int type


\- load_clean_fight_songs calls the above functions sequentially and returns the frame
'''

### Now the ***REALLY*** fun part

Switch .py files with someone from the cohort

Save it in this repo as `testing_data_cleaning.py`

***Restart your kernal***

Run the cell below to test your fellow student's work!


```python
from testing_data_cleaning import load_clean_fight_songs
from test_background import pkl_dump, test_obj_dict, run_test_dict, run_test

test_frame = load_clean_fight_songs()

run_test(test_frame, 'fight_songs')
```


```python
#__SOLUTION__

from testing_data_cleaning import load_clean_fight_songs
from test_background import pkl_dump, test_obj_dict, run_test_dict, run_test

test_frame = load_clean_fight_songs

run_test(test_frame, 'fight_songs')
```

# Why This Matters

The workflow that will make you an efficient data scientist goes something like this:

- **Write preliminary code** in Jupyter Notebooks
- Complete a **small** section of code that you know completes a necessary task
- **Write that code into a function** in a .py file
- In another notebook, **import that function** and run it

#### There are -several- advantages to doing this

- **Jupyter Notebooks are MeSsY**
    - Easy to jump around cells and **lose track** of what you're doing
    - Easy to **change the value of a variable** and not remember it later
    - Not that easy to **combine work**
    
    
- Importing functions through **.py files** into another book **helps mitigate** those problems
    - Your important work is all in **one spot without the clutter** of producing that work
    - Everything's in a tidy package, and so it's **harder for variables to get re-named**
    - **Combining work becomes easier**. Instead of sharing code through Jupyter Notebooks, and having to figure out which cells to run in what order, we can share .py files where we've already put in the work of figuring out what to run in what order as we've been working
    
    
- **Saves time in the long run**
    - Might not seem worth the time investment at first, but as your projects become bigger and more sprawling the problems it helps mitigate will become laRG**ER**
    - Doing this forces a **marathon mentality over a sprint mentality**, and helps keep one focused on small, necessary tasks


![](viz/siren.gif)     ![](viz/siren.gif)
# Is This Required for the Project?
![](viz/siren.gif)     ![](viz/siren.gif)

No


### Should we try it?

Sure!  But if it seems like it's becoming a hinderance to getting stuff done, go ahead and skip it


```python

```
