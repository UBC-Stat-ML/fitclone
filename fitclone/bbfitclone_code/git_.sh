#$ -S /bin/sh
cd /Users/sohrabsalehi/projects/fitClone-master/figures/presubmission
git config --get remote.origin.url
git pull
#git add SA666
#git add SA609
#git add SA532
#git add SA906a
#git add SA906b
git add $1
git commit -m 'Added the newest batch of figures'
git push