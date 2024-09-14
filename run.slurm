#!/bin/bash
#SBATCH --partition=contrib-gpuq                    # the DGX only belongs in the 'gpu'  partition
#SBATCH --qos=cs_dept                          # need to select 'gpu' QoS
#SBATCH --job-name=gitpush_tgtgame_fovg_ptupcdr-job
#SBATCH --output=/scratch/dmeher/slurm_outputs/gitpush_tgtgame_fovg_ptupcdr.%j.out
#SBATCH --error=/scratch/dmeher/slurm_outputs/gitpush_tgtgame_fovg_ptupcdr.%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1                 # up to 128; 
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --mem-per-cpu=80GB                 # memory per CORE; total memory is 1 TB (1,000,000 MB)
#SBATCH --export=ALL
#SBATCH --time=5-00:00:00                   # set to 1hr; please choose carefully
#SBATCH --mail-type=BEGIN,END,FAIL     # NONE,BEGIN,END,FAIL,REQUEUE,ALL,...
#SBATCH --mail-user=dmeher@gmu.edu   # Put your GMU email address here

set echo
umask 0027

# to see ID and state of GPUs assigned
nvidia-smi

#source /scratch/dmeher/custom_env/recguru_env/bin/activate
module load gnu10
module load python
module load git
source /scratch/dmeher/custom_env/ptu_env/bin/activate

python gitpush.py \
	--repo-path ../Research-PTUPCDR \
	--backup-path ../../../gitbackup/cs798_ptubackup/ \
	--data-folder ./data \
	--results-cs798-folder ./results_cs798 \
	--results-llmasrec-folder ./results_llmasrec \
	--branch sep14024 \
	--commit "Moved data and results folders, updated readme."\
	--log-file ./summmary_gitpush.txt	

#python cs798_statistical_analysis.py

#python ./entry.py --base_model MF --epoch 10 --task 2 --filename 'mae_rmse_tgtvideogame.csv'

#python entry.py \
#	--process_data_mid 0 \
#	--process_data_ready 1\
#	--test_data 3videogame_test 

#STEP1: CHANGE CONFIG FILE TO RESPECTIVE DOMAIN PAIRS YOU WANT. CHANGE UID AND IID COUNT. UID REPRESENTS UNIQUE UID AND IID REPRESENTS TOTAL. [I GOT ERROR WHEN I SET IID TO UNIQUE IID SO I KEEP IT MORE]
#STEP2: CHANGE ENTRY.PY TO ADD DOMAIN IN MID PART CODE. CHANGE THE DOMAIN PAIR COUNT, AND SPLIT OF DATASET
#STEP3: RUN ENTRY.PY WITH MID 1 READY 1 CODE. ENSURE TO CHANGE THE JOB OUPUT ERORR FILE NAMES TO BRING DISTINCTION AMONG OUTPUT FILE. YOU WILL GET MAE RMSE VALUES OVER THERE. MAKE SURE TO CHANGE THE TEST FILE AS PER TARGET DOMAIN. DATA PREPARATION CODE INSTRUCTIONS ARE IN DATASET/BASELINE_DATA FOLDER.
#STEP4: RUN ENTRY.PY WITH BASEMODE EPOCH TASK FILENAME. ENSURE TO CHANGE THE JOBNAME OUTPUT ERROR FILENAME.ENSURE TO CHANGE THE TASK AND FILENAME AS WELL.
