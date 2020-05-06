#$ -S /bin/sh

# $1 the remote server e.g., sscore1
# $2 the datatag
# $3 zip file to move
function setupfiles() 
{
  ssh $1 "mkdir -p /home/ssalehi/Desktop/SC-1311/${2}/processed_data/bincount/" 
  rsync -azP ~/Desktop/SC-1311/${2}/processed_data/bincount/$3 $1:/home/ssalehi/Desktop/SC-1311/$2/processed_data/bincount/
  ssh $1 "unzip /home/ssalehi/Desktop/SC-1311/$2/processed_data/bincount/${3} -d /home/ssalehi/Desktop/SC-1311/$2/processed_data/bincount/"
}

function download_tree() 
{
  rsync -zarv --include="*/" --include="*.pdf" --exclude="*" sscore3:/home/ssalehi/projects/corrupt2_/corrupt-nextflow/deliverables/bip_sa532 .
  rsync -azP  sscore3:/home/ssalehi/projects/corrupt2_/corrupt-nextflow/deliverables/bip_sa532/tree.newick .
  
  rsync -zarv --include="*/" --include="*.pdf" --exclude="*" sscore4:/home/ssalehi/projects/corrupt2_/corrupt-nextflow/deliverables/bip_sa906a .
  rsync -azP  sscore3:/home/ssalehi/projects/corrupt2_/corrupt-nextflow/deliverables/bip_sa906a/tree.newick .

}

function setupforshahlab()
{
    rsync -azP ~/Desktop/SC-1311/${2}/processed_data/bincount/$3 $1:/shahlab/ssalehi/scratch/corrupt/data/
    ssh $1 "unzip /shahlab/ssalehi/scratch/corrupt/data/${3} -d /shahlab/ssalehi/scratch/corrupt/data/"
    echo /shahlab/ssalehi/scratch/corrupt/data/${3}
}

# setupforshahlab shahlab14 SA906b sa906b_cnvs_corrupt.csv.zip

function check_eta()
{
  ssh $1 < remote.sh
}

if [[ $1 = 'setupfiles' ]]
then
  setupfiles $2 $3 $4
elif [[ $1 = 'download_tree' ]]
then
  download_tree $2
elif [[ $1 = 'eta' ]]
then 
  check_eta $2
elif [[ $1 = 'setupforshahlab' ]]
then
  setupforshahlab $2 $3 $4
else
  'Not implemented'
fi
  
  

