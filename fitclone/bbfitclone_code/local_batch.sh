#$ -S /bin/sh

function get_rnd_str()
{
  if [ $HOST = "noah" ]
  then
    cat /dev/urandom | env LC_CTYPE=C tr -dc 'a-zA-Z0-9' | fold -w 15 | head -n 1
  else
    echo cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 15 | head -n 1
  fi
}

function get_date_str()
{
  echo `date '+%Y_%m_%d__%H_%M_%S'`
}


function git_dir_name()
{
  new_name="${1}_${2}_$(get_date_str)_$(get_rnd_str)"
  echo $new_name
}


# SA532 
function run_batch()
{
  #./nextflow run local_test.nf -resume --lowerFraction 0.01 --tipInclusionProbabilities ~/Desktop/SC-1311/SA532/processed_data/bincount/sa532_cnvs_corrupt_downsampled_50_quality_threshold_75.csv
  #mv ~/projects/corrupt-nextflow/deliverables/local_test ~/projects/corrupt-nextflow/deliverables/local_test_sa532
  # SA609
  #./nextflow run local_test.nf -resume --lowerFraction 0.02 --tipInclusionProbabilities ~/Desktop/SC-1311/SA609/processed_data/bincount/sa609_cnvs_corrupt_downsampled_40_quality_threshold_75.csv
  #mv ~/projects/corrupt-nextflow/deliverables/local_test ~/projects/corrupt-nextflow/deliverables/local_test_sa609
  # SA906b
  #./nextflow run local_test.nf -resume --lowerFraction 0.01 --tipInclusionProbabilities ~/Desktop/SC-1311/SA906b/processed_data/bincount/sa906b_cnvs_corrupt_downsampled_33_quality_threshold_75.csv
  #mv ~/projects/corrupt-nextflow/deliverables/local_test ~/projects/corrupt-nextflow/deliverables/local_test_sa906b
  echo this is a function in the making...
}

# SA906b
#./nextflow run local_test.nf -resume --lowerFraction 0.01 --tipInclusionProbabilities ~/Desktop/SC-1311/SA906b/processed_data/bincount/sa906b_cnvs_corrupt_downsampled_33_quality_threshold_75_no_padding.csv

#./nextflow run local_test.nf -resume --lowerFraction 0.01 --tipInclusionProbabilities ~/Desktop/SC-1311/SA906b/processed_data/bincount/sa906b_cnvs_corrupt_downsampled_33_quality_threshold_75.csv
#mv ~/projects/corrupt-nextflow/deliverables/local_test ~/projects/corrupt-nextflow/deliverables/local_test_sa906b_new_ploidy_padding

# Plot
# For each datatag
# 0. datatg, 1. fraction,  2. delta_mat file path, 3. new_directory
# a. run corrupt
# b. generate random string for name
# c. mv the file
# do the plotting

datatags=(SA532 SA609 SA906a SA666)
fractions=(0.01 0.03 0.01 0.04)
delta_files=(sa532_cnvs_corrupt_downsampled_50_quality_threshold_75.csv sa609_cnvs_corrupt_downsampled_40_quality_threshold_75.csv sa906a_cnvs_corrupt_downsampled_33_quality_threshold_75.csv sa666609_cnvs_corrupt_downsampled_33_quality_threshold_75.csv)
dir_names=('1' '3' '4' '5')
ND=${#datatags[@]}
# Set the output dir name
for (( i=0; i<$ND; i++ ))
do
    dir_names[$i]=$(git_dir_name local_test ${datatags[$i]})
    echo ${dir_names[$i]}
done    

for (( i=0; i<$ND; i++ ))
do  
  echo ~/Desktop/SC-1311/${datatags[$i]}/processed_data/bincount/${delta_files[$i]}
  ./nextflow run local_test.nf -resume --lowerFraction ${fractions[$i]} --tipInclusionProbabilities ~/Desktop/SC-1311/${datatags[$i]}/processed_data/bincount/${delta_files[$i]}
  echo ~/projects/corrupt-nextflow/deliverables/${dir_names[$i]}
  mv ~/projects/corrupt-nextflow/deliverables/local_test ~/projects/corrupt-nextflow/deliverables/${dir_names[$i]}
done    


# Plotting
for (( i=0; i<$ND; i++ ))
do
  Rscript --vanilla ~/projects/fitness/fitclone_driver.R --datatag ${datatags[$i]} --original_tree_path ~/projects/corrupt-nextflow/deliverables/${dir_names[$i]}/tree.newick --nchains 1
done    




#echo ${ARRAY[2]}

#dir_names[]

#for i in ${farm_hosts[@]}; do
#        su $login -c "scp $httpd_conf_new ${i}:${httpd_conf_path}"
#        su $login -c "ssh $i sudo /usr/local/apache/bin/apachectl graceful"

#done















