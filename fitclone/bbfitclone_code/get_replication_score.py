import dbclients.tantalus
tantalus_api = dbclients.tantalus.TantalusApi()
library_tickets = {'A96179B':  'SC-1341'}
library_tickets = {'SA532X3XB00210':  'SC-1341'}

library_tickets = {'A96186C':  'SC-1342'}
A96186C	SA532X3XB00210	

cell_cycle_data = []
for library_id, jira_ticket in library_tickets.iteritems():
#    features = tantalus_api.list(
    features = tantalus_api.get(
        'results',
        results_type='cell_state_prediction',
        results_version='v0.0.1',
        libraries__library_id=library_id,
        analysis__jira_ticket=jira_ticket,
    )
  file_instances = tantalus_api.get_dataset_file_instances(
tantalus_api.get
        features['id'], 'resultsdataset', results_storage_name)
    for file_instance in file_instances:
        f = client.open_file(file_instance['file_resource']['filename'])
        data = pd.read_csv(f, index_col=0)
        data['library_id'] = library_id
        cell_cycle_data.append(data)
cell_cycle_data = pd.concat(cell_cycle_data, ignore_index=True)



#################
import dbclients.tantalus
import pandas as pd

tantalus_api = dbclients.tantalus.TantalusApi()

#library_tickets = {'A96179B':  'SC-1341'}
library_tickets = {  'A96172A': 'SC-1290',    'A96149A': 'SC-1651',    'A95724A': 'SC-1314',    'A95664B': 'SC-1358',    'A95724B': 'SC-1365',    'A95632D': 'SC-1371',    'A95728A': 'SC-1379',    'A96184A': 'SC-1383',    'A96217B': 'SC-1138',    'A96219B': 'SC-1139',    'A96157C': 'SC-1645',    'A90679': 'SC-1336',    'A96165A': 'SC-1340',    'A96179B': 'SC-1285',    'A96186C': 'SC-1286',    'A95732B': 'SC-1344',    'A96174B': 'SC-1346',    'A96130B': 'SC-1159',    'A96177B': 'SC-1349',    'A96145A': 'SC-1351',    'A96162B': 'SC-1700' }

cell_cycle_data = []

results_storage_name = 'singlecelldata'

for library_id, jira_ticket in library_tickets.iteritems():
    print(library_id)
#    features = tantalus_api.list(
    features = tantalus_api.get(
        'results',
        results_type='cell_state_prediction',
        results_version='v0.0.1',
        libraries__library_id=library_id,
        analysis__jira_ticket=jira_ticket,
    )
    
    index = 1
    for feature in features:
        print(index)
        index = index + 1
        file_instances = tantalus_api.get_dataset_file_instances(feature['id'], 'resultsdataset', results_storage_name)
        f = client.open_file(file_instance['file_resource']['filename'])
        data = pd.read_csv(f, index_col=0)
        data['library_id'] = library_id
        cell_cycle_data.append(data)
        print(data)



cell_cycle_data = pd.concat(cell_cycle_data, ignore_index=True)
      
      
      
## 
library_tickets = {  'A96172A': 'SC-1290',    'A96149A': 'SC-1651',    'A95724A': 'SC-1314',    'A95664B': 'SC-1358',    'A95724B': 'SC-1365',    'A95632D': 'SC-1371',    'A95728A': 'SC-1379',    'A96184A': 'SC-1383',    'A96217B': 'SC-1138',    'A96219B': 'SC-1139',    'A96157C': 'SC-1645',    'A90679': 'SC-1336',    'A96165A': 'SC-1340',    'A96179B': 'SC-1285',    'A96186C': 'SC-1286',    'A95732B': 'SC-1344',    'A96174B': 'SC-1346',    'A96130B': 'SC-1159',    'A96177B': 'SC-1349',    'A96145A': 'SC-1351',    'A96162B': 'SC-1700' }
for library_id, jira_ticket in library_tickets.iteritems():
    print(library_id)    
    try:
        features = tantalus_api.get(
            'results',
            results_type='cell_state_prediction',
            results_version='v0.0.1',
            libraries__library_id=library_id,
            analysis__jira_ticket=jira_ticket,
        )
      
    except:
        print("An exception occurred")
      
    
    
    
features = tantalus_api.get(
    'results',
    results_type='cell_state_prediction',
    results_version='v0.0.1',
    libraries__library_id=library_id,
    analysis__jira_ticket=jira_ticket,
)    




####################################################################
####################################################################
import dbclients.tantalus
import pandas as pd
results_storage_name = 'singlecellblob_results'
tantalus_api = dbclients.tantalus.TantalusApi()
client = tantalus_api.get_storage_client(results_storage_name)





library_tickets_SA609_SA532 = {
    'A96172A': 'SC-1290',
#     'A96149A': 'SC-1651',
    'A95724A': 'SC-1314',
    'A95664B': 'SC-1358',
    'A95724B': 'SC-1365',   
    'A95632D': 'SC-1371',   
    'A95728A': 'SC-1379',   
    'A96184A': 'SC-1383',   
    'A96217B': 'SC-1138',   
    'A96219B': 'SC-1139',   
    'A96157C': 'SC-1645',   
    'A90679': 'SC-1336',   
    'A96165A': 'SC-1340',   
    'A96179B': 'SC-1285',   
    'A96186C': 'SC-1286',   
    'A95732B': 'SC-1344',   
    'A96174B': 'SC-1346',   
    'A96130B': 'SC-1159',   
    'A96177B': 'SC-1349',   
    'A96145A': 'SC-1351',   
    'A96162B': 'SC-1700',
}

# SA906a, SA906b, and SA666
library_tickets = { 
    'A96180B': 'SC-1515',  
    'A96210C': 'SC-1522',    
    'A96216A': 'SC-1637',    
    'A96175C': 'SC-1513',    
    'A96183C': 'SC-1304',    
    'A96186A': 'SC-1518',   
    'A96146A': 'SC-1624',    
    #        'A96215A': 'SC-1524',    
    'A96211C': 'SC-1258',    
    'A96172B': 'SC-1291',   
    'A96155B': 'SC-1510',    # classifier failed
    'A96220B': 'SC-1626',    # classifier failed
    'A96228B': 'SC-1649',    # classifier failed
    'A96149B': 'SC-1647',    
    'A96225B': 'SC-1538',   
    'A96225C': 'SC-1540',   
    'A96183B': 'SC-1536',    
    'A96181C': 'SC-1534',   
    'A95635B': 'SC-1373',   
    'A95662A': 'SC-1363',   
    'A95635D': 'SC-1361'
}


# SA501 
library_tickets = { 
    'A95621B':'SC-1234',
    'MF1509041':'SC-1399',
    'MF1509042':'SC-1401',
    'MF1508181':'SC-1403',
    'MF1508182':'SC-1405',
    'A95670A':'SC-1415'
}


########################
## The missing ones
########################

#library_tickets = {
    #"A96215A": "SC-1524",
#    "A96155B": "SC-1510",
#    "A96220B": "SC-1626",
#    "A96228B": "SC-1649"
# }
 
#library_tickets = { "A96215A": "SC-1524"}

#library_tickets = { "A96215A": "SC-1765"}





###'
// Comments
{
    A96215A
Traceback (most recent call last):
  File "<stdin>", line 26, in <module>
  File "/Library/Python/2.7/site-packages/sisyphus-0.0.0-py2.7.egg/dbclients/basicclient.py", line 67, in get
    raise NotFoundError("no object for {}, {}".format(table_name, fields))
dbclients.basicclient.NotFoundError: no object for resultsdataset, {'analysis': 58}
}


########################
## The missing ones - SA532X10 and all SA777
########################
#library_tickets = { "A96215A": "SC-1765"}
library_tickets = {
    'A95629A': 'SC-1651', 
    'A95618A':'',
    'A95654B':'',
    'A95628B':'',
    'A95654A':'',
    'A96192B':'',
    'A95673A':''
}


# Level 2 missing
library_tickets = {
    'A96205A':'',
    'A96244A':''
}

# SA609X3X8b 
library_tickets = {
    'A96192B':'',
    'A95654B':''
}


# The ovary ones
library_tickets = {
    'A96192B':'',
    'A95654B':''
}

# LIB_SA922_A90554B', 'LIB_SA921_A90554A', 'LIB_SA1090_A96213A')
library_tickets = {
    'A90554B':'',
    'A90554A':'',
    'A96213A':''
}


# For the WT
library_tickets = {
    'A73044B':'',
    'A73047D':'',
    'A96199B':'',
    'A96226B':''
}



# For SA501
library_tickets = { 
    'A95621B':'SC-1234',
    'MF1509041':'SC-1399',
    'MF1509042':'SC-1401',
    'MF1508181':'SC-1403',
    'MF1508182':'SC-1405',
    'A95670A':'SC-1415'
}

# For SA532 backfill
library_tickets = { 
    'A95633A':'SC-2468',
    'A95633B':'SC-2469',
    'A95703B':'SC-2219',
    'A95703A':'SC-2216',
    'A95629A':'SC-1845',
    'A96130B':'SC-1347'
}


hmmcopy_results = {}
for library_id in library_tickets.keys():
    print(library_id)
    analyses = list(tantalus_api.list('analysis',analysis_type__name='hmmcopy', input_datasets__library__library_id=library_id))
    aln_analyses = []
    for analysis in analyses:
        aligners = set()
        for dataset_id in analysis['input_datasets']:
            dataset = tantalus_api.get('sequencedataset', id=dataset_id)
            aligners.add(dataset['aligner'])
        if len(aligners) != 1:
            raise Exception('found {} aligners for analysis {}'.format(len(aligners), analysis['id']))
        aligner = aligners.pop()
        if aligner == 'BWA_ALN_0_5_7':
            aln_analyses.append(analysis)
    if len(aln_analyses) != 1:
        #raise Exception('found {} hmmcopy analyses for {}: {}'.format(
        #    len(aln_analyses), library_id, [a['id'] for a in aln_analyses]))
        analysis = aln_analyses[1]
    else:
        analysis = aln_analyses[0]
    results = tantalus_api.get('resultsdataset', analysis=analysis['id'])
    hmmcopy_results[library_id] = results

cell_cycle_data = []
for library_id in library_tickets.keys():
    print(library_id)
    try:
        classifier = tantalus_api.get('analysis',  analysis_type='cell_state_classifier', version='v0.0.1', input_results__id=hmmcopy_results[library_id]['id'])
        features = tantalus_api.get('resultsdataset', analysis=classifier['id'])
        try:
            file_instances = tantalus_api.get_dataset_file_instances(features['id'], 'resultsdataset', results_storage_name)
            for file_instance in file_instances:
                try:
                    f = client.open_file(file_instance['file_resource']['filename'])
                    data = pd.read_csv(f)
                    data['library_id'] = library_id
                    cell_cycle_data.append(data)
                except:
                    print('Some error occurred')
        except:
            print('something went south!')
    except:
        print('Classifier failed.')
    
cell_cycle_data = pd.concat(cell_cycle_data, ignore_index=True)

# cell_cycle_data.to_csv('/Users/sohrabsalehi/Desktop/SC-1311/cell_cycle.csv')
#cell_cycle_data.to_csv('/Users/sohrabsalehi/Desktop/SC-1311/cell_cycle_rest.csv')

cell_cycle_data.to_csv('/Users/sohrabsalehi/Desktop/SC-1311/cell_cycle_rest_NEW_SA609X2.csv')

cell_cycle_data.to_csv('/Users/sohrabsalehi/Desktop/SC-1311/cell_cycle_rest_NEW_SA906b.csv')

cell_cycle_data.to_csv('/Users/sohrabsalehi/Desktop/SC-1311/cell_cycle_rest_NEW_SA906b_last.csv')

cell_cycle_data.to_csv('/Users/sohrabsalehi/Desktop/SC-1311/cell_cycle_rest_NEW_SA532_SA777.csv')

cell_cycle_data.to_csv('/Users/sohrabsalehi/Desktop/SC-1311/cell_cycle_rest_NEW_missing_SA777.csv')


cell_cycle_data.to_csv('/Users/sohrabsalehi/Desktop/SC-1311/cell_cycle_rest_NEW_SA609X3X8b.csv')

cell_cycle_data.to_csv('/Users/sohrabsalehi/Desktop/SC-1311/SA922/sphase/cell_cycle.csv')

cell_cycle_data.to_csv('/Users/sohrabsalehi/Desktop/SC-1311/SA039/cell_cycle.csv')
cell_cycle_data.to_csv('/Users/sohrabsalehi/Desktop/SC-1311/SA501/cell_cycle.csv')

# Client id:
# 39037697-d441-4bce-aacc-f4eb999abfca
# OR: ebdd57b4-71cb-4e43-a21a-94010e

# Subscription and tenant id are the same as before
# 436b89a7-3b73-4644-a97b-949c4d0f19f5
# 31126879-74b8-42b9-8ae6-68b3a277ebdc

# Secret key
#  az keyvault secret download -n singlecellpipeline --vault-name production-credentials -f singlecellpipelinesecretkey.txt
# Access denied
