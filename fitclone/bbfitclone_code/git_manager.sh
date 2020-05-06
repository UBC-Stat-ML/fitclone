#$ -S /bin/sh

function git_fit_add() 
{
  cd /Users/sohrabsalehi/projects/fitClone-master/figures/presubmission
  git config --get remote.origin.url
  git pull
  git add $1
  git commit -m 'Added the newest batch of figures'
  git push
}

function git_fit_rm()
{
  # list files
  cd /Users/sohrabsalehi/projects/fitClone-master/figures/presubmission
#  git config --get remote.origin.url
#  git pull
  for file in ./$1/*.png
  do
      if [[ -f $file ]]
      then
        echo $file
        git rm $file
      fi
  done
  git commit -m 'Added the newest batch of figures'
  git push
}



if [ $1 = "git_fit_add" ]
then
  git_fit_add $2
elif [ $1 = 'git_fit_rm' ]
then
  git_fit_rm $2
else
  echo "Not implemented"
fi
