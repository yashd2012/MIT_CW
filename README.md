## GUIDE FOR PYTHON SCRIPTS

### FOR ALL FOUR CODE FILES: KEY DATA FILES INCLUDE: 
#### 1) RAW DATA FILES - ALL DATAFRAMES; 
  FIELDS (LOCATIONS AND PROPERTIES), 
  REFINERIES (LOCATIONS, CONSUMPTION SLATES), 
  PIPELINES (LOCATIONS AND ATTRIBUTES),
  TERMINALS (LOCATIONS),
  FUTURE ENERGY SCENARIOS
#### 2) NETWORK PICKLE OBJECT (CONTAINS NODES AND EDGES OF FIELDS, PIPELINES, REFINERIES, TERMINALS) BASED ON RAW DATA


#### E.G. IN THE BLEND ESTIMATION ALGORITHM SCRIPT

```
aramfl_new_new = ## DATAFRAME OF FIELDS - LOCATIONS, NAMES, COUNTRIES, VOLUMES, API
...
ar_ref = ## DATAFRAME OF REFINERY COORDINATES

```


### KEY POINTS FOR EACH CODE FILE:
#### 1) BLEND ESTIMATION ALGORITHM

##### In addition to the aforementioned data files, the blend estimation algorithm uses two derived arrays - distance and connectivity matrices of all nodes in the network: (i,j)th entry  corresponds to the distance between node-i and node-j and boolean depending on whether they are connected in the network

```
## DISTANCE AND CONNECTIVITY MATRICES ALONG DISTANCE MATRIX BASED ON NODES OF THE NETWORK
with open('global_dmats_new.pickle', 'rb') as fp:
    glob_distance_mat_dict = pickle.load(fp)

with open('global_cmats_new.pickle', 'rb') as fp:
    glob_connectivity_mat_dict = pickle.load(fp)

```
##### Initializing the class - key parameters determine thresholds used by other methods such as the priority-selection, name similarity etc.

```
class country_theta:

    def __init__(self, name):
        self.name = name

    def user_inputs(self):
        self.prionum = 100
        self.pricut = 95
        self.fc_name_thresh = 0.7
        self.fc_name_low = 0.1
        self.fc_name_high = 10
.
.
.

```

#### 2) BLEND ESTIMATION ALGORITHM
#### 3) BLEND ESTIMATION ALGORITHM
#### 4) BLEND ESTIMATION ALGORITHM

