#$ -S /bin/sh
cd /Users/sohrabsalehi/projects/fitness_files
git config --get remote.origin.url
git pull
git add $1
git commit -m 'Added the newest batch of figures'
git push
