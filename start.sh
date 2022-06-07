#!/bin/bash

#$-l rt_F=1
#$-l h_rt=5:00:00
#$-j y
#$-cwd

echo start
source /home/acc12973zh/environment/mimic_env/bin/activate
source /etc/profile.d/modules.sh
module load gcc/11.2.0
module load python/3.8/3.8.13


python regression.py

echo end
