#$ -S /bin/sh
export CORWDIR="/Users/sohrabsalehi/projects/corrupt-nextflow/work"
ID1=$(ls -td -1 $CORWDIR/** | head -n1);
ID2=$(ls -td -1 $ID1/** |  head -n1); 
ID3=$(tail -n1 $ID2/samples/fnr.csv);  
ID4=$(echo $ID3 | awk -F "," '{print $1/10000}')
echo $ID4

# Creation time
# find ./arguments.tsv -maxdepth 1 -printf '%p %Cs\n'
C1=$(stat -f "%B" $ID2/arguments.tsv)
C2=$(stat -f "%m" $ID2/samples/fnr.csv)
C3=$(expr $C2 - $C1)

# Remaining time?
X=$(echo "$C3/$ID4" | bc -l)
ETA=$(echo "(1 - $ID4) * $X" | bc -l)
# Print nicely
echo $ETA | awk '{printf "%0.2f\n", $0}'
date -u -d @${ETA} +"%T"
# Capture how many days
DAYS=$(echo "$ETA/(24*3600)" | bc -l)
DAYS=$(echo $DAYS | awk '{printf "%d\n", $0}')

echo 'ETA is ' $DAYS 'days and '  $(date -u -d @${ETA} +"%T")  ' hours'

 
stat -f "%B" $ID2/arguments.tsv