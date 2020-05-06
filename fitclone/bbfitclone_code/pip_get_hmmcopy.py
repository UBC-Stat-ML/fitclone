# cd ~/projects/scgenome
import sys 
sys.path.append('/Users/sohrabsalehi/projects/scgenome')
#sys.path.append('/Users/sohrabsalehi/projects/new_scgenome/scgenome')

import os
import getopt
import pandas as pd
import dbclients
import scgenome.hmmcopy
import scgenome.utils
import csv

tantalus_api = dbclients.tantalus.TantalusApi()


def get_cn_data(hmmcopy_tickets, sample_ids, outdir):
    local_cache_directory = os.path.join(outdir, 'pythoncash')
    if not os.path.exists(local_cache_directory):
        os.mkdir(local_cache_directory)
    
    cn_data = []
    segs_data = []
    metrics_data = []
    align_metrics_data = []

    print(hmmcopy_tickets)
    print(type(hmmcopy_tickets))

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

    dic = {'cn_data': cn_data,  'segs_data': segs_data, 'metrics_data':metrics_data, 'align_metrics_data': align_metrics_data}

    for dname, dlist in dic.items():
        dlist.to_csv(os.path.join(outdir, '{}.csv'.format(dname)))

def main(argv):
    inputfile = ''
    outputfile = ''
    try:
        opts, args = getopt.getopt(argv,"hi:o:",["metafile="])
    except getopt.GetoptError:
        print ('pip_get_hmmcopy.py -i <metafile>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print ('pip_get_hmmcopy.py -i <metafile>')
            sys.exit()
        elif opt in ("-i", "--metafile"):
            metafile = arg
    
    print(metafile)     
    data = pd.read_csv(metafile)
    print(data)
    hmmcopy_tickets = data['jira_ticket'].values.tolist()
    sample_ids = data['sample_id'].values.tolist()
    outdir = os.path.dirname(os.path.realpath(metafile))
    get_cn_data(hmmcopy_tickets,  sample_ids, outdir)

if __name__ == "__main__":
   main(sys.argv[1:])















