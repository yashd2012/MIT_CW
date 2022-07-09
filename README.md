## GUIDE FOR PYTHON SCRIPTS

### FOR ALL FOUR CODE FILES: KEY DATA FILES INCLUDE: 
#### RAW DATA FILES CONTAINING SUPPLY CHAIN DATA - ALL DATAFRAMES; 
  FIELDS (LOCATIONS AND PROPERTIES), 
  REFINERIES (LOCATIONS, CONSUMPTION SLATES), 
  PIPELINES (LOCATIONS AND ATTRIBUTES),
  TERMINALS (LOCATIONS),
#### NETWORK PICKLE OBJECT (CONTAINS NODES AND EDGES OF FIELDS, PIPELINES, REFINERIES, TERMINALS) BASED ON RAW DATA



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
##### Outputs are stored in a dictionary, specifically the C.I. dataframe i.e. the estimated C.I. values for each blend are stored in country-specific dataframes mapped to country key values in the dictionary

```
## SAVE THE BLEND ESTIMATION OUTPUTS IN A DICTIONARY
    with open('PATH_TO_OUTPUT_' + str(bi) + '.p', 'wb') as fp:
        pickle.dump(blend_distri, fp, protocol=pickle.HIGHEST_PROTOCOL)

```


#### 2) TRACKING FROM FIELDS TO REFINERIES

##### The outputs from the blend estimation algorithm flow into the tracking logic since the blend estimation associates fields to blends and the refinery dataset associates blends to refineries thereby creating the links from fields to refineries

##### A shortest path assessment is used along the network to determine paths from fields to refineries

```
    for bl in blends:
        blend = bl
        blend_to_refinery_path_dict_beta[country][blend] = {}
        fields = (list(theta[theta[bl]!=0][bl].index))
        refs_for_blend = ref_cs_without_lalo[ref_cs_without_lalo['Crude Stream'] == bl]
        if (refs_for_blend.shape[0]==0):
            continue
        for fname in fields:
            fn = [nx[0] for nx in nwtcopy.nodes(data=True) if ((nx[1]['typ']=='f')&(nx[1]['asset_name']==fname)&(nx[1]['asset_country']==pc))]
            if len(fn)!=1:
                continue
            for rfi in refs_for_blend.index:
                refinery  =  refs_for_blend.loc[rfi, 'Refinery']
                refcountry = refs_for_blend.loc[rfi, 'Country']

                refinery_node = [n for n,v in nwtcopy.nodes(data=True) if ((v['typ']=='r')&(v['asset_name']==refinery)&(v['asset_country']==refcountry))]
                if len(refinery_node)!=1:
                    continue
                try:
                    blend_to_refinery_path_dict_beta[country][blend][fn[0], refinery_node[0]] = (nx.shortest_path(nwtcopy,source=fn[0],target=refinery_node[0], weight='actual_distance'))
                except:
                    blend_to_refinery_path_dict_beta[country][blend][fn[0], refinery_node[0]] = "DNE"



```


#### 3) C.I. COMPUTATION ALONG TRACKED PATHS

##### In addition to the common datasets, the C.I computation script requires the pipeline and shipping C.I. dataframes that contain C.I. values for every pipeline and shipping edge in the network respectively

```
coptem_filename = ## DATAFRAME OF SEGMENT SPECIFIC PIPELINE EMISSIONS
coptem_f2r = pd.read_csv(coptem_filename, index_col=0)
opgee_shipping_f2r = ## DATAFRAME OF PATH SPECIFIC SHIPPING EMISSIONS

```

##### Loop over field to refinery paths from the previous module to compute C.I. values along source to destination pathways

```
.
.
.
                    if edge_n1n2['ty']=='kk':
                        ship_dist += (1609.34)*(edge_n1n2['shipping_distance_v3'])
                        total_dist += (1609.34)*(edge_n1n2['shipping_distance_v3'])
                        opgee_shipdf_slice = opgee_shipping_f2r.loc[str((node1, node2)),:]
                        ship_ci += opgee_shipdf_slice['tantra_gcpmj_final']

                    if edge_n1n2['ty']=='pp1':
                        pipe_dist += edge_n1n2['actual_distance']
                        total_dist += edge_n1n2['actual_distance']

                        coptem_f2r_slice = coptem_f2r[((coptem_f2r['crude_name']==bc_corrected)&(coptem_f2r['edge_pairs']==str((node1, node2))))]
                        pipe_ci_v2 += coptem_f2r_slice['CI_v2_gco2permj'].values[0]
.
.
.

```


##### The outputs are saved as a dictionary with keys denoted by the source country, blend and field-to-refinery paths:

```
.
.
.
                cdict[c][bc][fr]['path']  = frp[c][bc][fr]['path']
                cdict[c][bc][fr]['vol_bbld']  = frp[c][bc][fr]['bbld']
                cdict[c][bc][fr]['pipeline_ci_v2'] = pipe_ci_v2
                cdict[c][bc][fr]['shipping_ci'] = ship_ci
                cdict[c][bc][fr]['pipeline_distance'] = pipe_dist
                cdict[c][bc][fr]['shipping_distance'] = ship_dist
                cdict[c][bc][fr]['total_distance'] = total_dist

    ## SAVE  DICTIONARY OF EMISSIONS ALONG FIELD TO REFINERY PATHS
    dfilename = "SET_THIS_PATH_" + c + ".p"

    with open(dfilename, 'wb') as fp:
        pickle.dump(cdict, fp, protocol=pickle.HIGHEST_PROTOCOL)

.
.
.

```

#### 4) SCENARIO ANALYSIS

##### Notebook requires the dataframe containing future energy scenarios (E.J/YEAR projections for each decade, SSP scenario and model choice)

```
df_scenarios = ## DATAFRAME OF SSP SCENARIOS

```

##### Additionally, the outputs from the C.I. computation module are wrangled into a dataframe i.e. the dictionary data structure is unpacked into a tabular structure with a row for every field-to-refinery pathway; group by operations are performed on this dataframe to get different levels of aggregation that are used by the scenario analysis notebook

```
df_fr_umci = ## DATAFRAME OF CI ORGANIZED FROM FIELD TO REFINERIES
df_b_umci = ## DATAFRAME OF CI ORGANIZED AT BLEND LEVEL
df_br_umci =  ## DATAFRAME OF CI ORGANIZED FROM BLEND TO REFINERIES

```


