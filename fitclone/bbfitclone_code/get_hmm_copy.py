

# OV samples
# hmmcopy_tickets = [
#     'SC-1935',
#     'SC-1936',
#     'SC-1937',
# ]
#
# sample_ids = [
#     'SA1090',
#     'SA921',
#     'SA922',
# ]

# SC-1493  A73044B
# SC-1495  A73047D
# SC-1497  A96199B
# SC-1642  A96226B


# The wt
# hmmcopy_tickets = [
#     'SC-1493',
#     'SC-1495',
#     'SC-1497',
#     'SC-1642'
# ]
#
# sample_ids = [
#     'SA039',
#     'SA039',
#     'SA039',
#     'SA039'
# ]

# The SA501 
# hmmcopy_tickets = [
#    'SC-1415',
#    'SC-1399',
#    'SC-1401',
#    'SC-1403',
#    'SC-1405',
#    'SC-1415'
# ]
#
# sample_ids = [
#     'SA501X2XB00096',
#     'SA501X3F',
#     'SA501X3F',
#     'SA501X4F',
#     'SA501X4F',
#     'SA501X5XB00877'
# ]

# SA501 redo the 
hmmcopy_tickets = [
   'SC-1234'
]

sample_ids = [
    'SA501X2XB00096',
]


# cd ~/projects/scgenome
import dbclients
import scgenome.hmmcopy
import scgenome.utils
import csv


tantalus_api = dbclients.tantalus.TantalusApi()


local_cache_directory = '/Users/sohrabsalehi/Downloads/pythoncash'

cn_data = []
segs_data = []
metrics_data = []
align_metrics_data = []
reads_data = []


#    hmmcopy_data = hmmcopy.load_cn_data(sample_ids=sample_ids,  additional_reads_cols = ['map'])

for jira_ticket in hmmcopy_tickets:
    print('Attempting jira_ticket = {}'.format(jira_ticket))
    analysis = tantalus_api.get( 'analysis', analysis_type__name='hmmcopy', jira_ticket=jira_ticket)
    hmmcopy = scgenome.hmmcopy.HMMCopyData(jira_ticket, local_cache_directory)
    hmmcopy_data = hmmcopy.load_cn_data(sample_ids=sample_ids)
    cn_data.append(hmmcopy_data['hmmcopy_reads'])
    segs_data.append(hmmcopy_data['hmmcopy_segs'])
    metrics_data.append(hmmcopy_data['hmmcopy_metrics'])
    align_metrics_data.append(hmmcopy_data['align_metrics'])

cn_data = scgenome.utils.concat_with_categories(cn_data)
segs_data = scgenome.utils.concat_with_categories(segs_data)
metrics_data = scgenome.utils.concat_with_categories(metrics_data)
align_metrics_data = scgenome.utils.concat_with_categories(align_metrics_data)

# read the reads
for jira_ticket in hmmcopy_tickets:
    analysis = tantalus_api.get( 'analysis', analysis_type__name='hmmcopy', jira_ticket=jira_ticket)
    hmmcopy = scgenome.hmmcopy.HMMCopyData(jira_ticket, local_cache_directory)
    hmmcopy_data = hmmcopy.load_cn_data(sample_ids=sample_ids)
    reads_data.append(hmmcopy_data['reads'])

reads_data = scgenome.utils.concat_with_categories(reads_data)

i = 0
for d in ['SA1090', 'SA921', 'SA922']:
    #reads_data[i].to_csv('/Users/sohrabsalehi/Desktop/SC-1311/SA922/raw/{}_{}.csv'.format(d, 'reads_data'))
    cn_data[i].to_csv('/Users/sohrabsalehi/Desktop/SC-1311/SA922/raw/{}_{}.csv'.format(d, 'cn_data'))
    i = i + 1


cn_data.to_csv('/Users/sohrabsalehi/Desktop/SC-1311/SA922/raw/{}_{}.csv'.format('all', 'cn_data'))


dic = {'cn_data': cn_data,  'segs_data': segs_data, 'metrics_data':metrics_data, 'align_metrics_data': align_metrics_data}

for dname, dlist in dic.items():
    dlist.to_csv('/Users/sohrabsalehi/Desktop/SC-1311/SA501/raw/backfill2/{}.csv'.format(dname))



for dname, dlist in dic.items():
    i = 0
    for d in ['SA1090', 'SA921', 'SA922']:
        tmp_dat = dlist[i]
        tmp_dat.to_csv('/Users/sohrabsalehi/Desktop/SC-1311/SA922/raw/{}_{}.csv'.format(d, dname))
        i = i + 1



