#$ -S /bin/sh
cd /Users/sohrabsalehi/projects/fitness

echo "NOT UPDATING PYTHON CODE FILES..."
#jupyter nbconvert --to script  paper_experiments_dlp.ipynb
#jupyter nbconvert --to script SC_DLP_sa501_exp.ipynb

# rsync.shahssd
rsync -azP --include='*.'{pyx,py,sh,R} --exclude="*" ./ shahlab15:/ssd/sdb1/ssalehi/projects/fitness
rsync -azP --include="*.yaml" --exclude="*" ./batchconfigs/ shahlab15:/ssd/sdb1/ssalehi/projects/fitness/batchconfigs

# rsync.shah15
rsync -azP --include='*.'{pyx,py,sh,R} --exclude="*" ./ shahlab15:/home/ssalehi/projects/fitness
rsync -azP --include="*.yaml" --exclude="*" ./batchconfigs/ shahlab15:/home/ssalehi/projects/fitness/batchconfigs

rsync -azP ~/projects/fitclone/figures/raw/supp/SA501/corrupt/ shahlab15:/home/ssalehi/projects/fitclone/figures/raw/supp/SA501/corrupt
rsync -azP ~/projects/fitclone/figures/raw/supp/SA609/corrupt/ shahlab15:/home/ssalehi/projects/fitclone/figures/raw/supp/SA609/corrupt
rsync -azP ~/projects/fitclone/figures/raw/supp/SA532/corrupt/ shahlab15:/home/ssalehi/projects/fitclone/figures/raw/supp/SA532/corrupt
rsync -azP ~/projects/fitclone/figures/raw/supp/SA906a/corrupt/ shahlab15:/home/ssalehi/projects/fitclone/figures/raw/supp/SA906a/corrupt
rsync -azP ~/projects/fitclone/figures/raw/supp/SA906b/corrupt/ shahlab15:/home/ssalehi/projects/fitclone/figures/raw/supp/SA906b/corrupt
rsync -azP ~/projects/fitclone/figures/raw/supp/SA039/corrupt/ shahlab15:/home/ssalehi/projects/fitclone/figures/raw/supp/SA039/corrupt
rsync -azP ~/projects/fitclone/figures/raw/supp/SA666/corrupt/ shahlab15:/home/ssalehi/projects/fitclone/figures/raw/supp/SA666/corrupt
rsync -azP ~/projects/fitclone/figures/raw/supp/SA609X3X8a/corrupt/ shahlab15:/home/ssalehi/projects/fitclone/figures/raw/supp/SA609X3X8a/corrupt
rsync -azP ~/projects/fitclone/figures/raw/supp/SA609bRx8p/corrupt/ shahlab15:/home/ssalehi/projects/fitclone/figures/raw/supp/SA609bRx8p/corrupt
rsync -azP ~/projects/fitclone/figures/raw/supp/SA609aRx8p/corrupt/ shahlab15:/home/ssalehi/projects/fitclone/figures/raw/supp/SA609aRx8p/corrupt
rsync -azP ~/projects/fitclone/figures/raw/supp/SA609aRx4p/corrupt/ shahlab15:/home/ssalehi/projects/fitclone/figures/raw/supp/SA609aRx4p/corrupt
rsync -azP ~/projects/fitclone/figures/raw/supp/SA000/corrupt/ shahlab15:/home/ssalehi/projects/fitclone/figures/raw/supp/SA000/corrupt
echo "NOT UPDATING PYTHON CODE FILES..."