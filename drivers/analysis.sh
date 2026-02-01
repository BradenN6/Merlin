#!/bin/bash --login                                                                                                                                                                                  
#SBATCH --job-name=nebular-lines                                                                                                                                                             
#SBATCH --output=logs/job_%A_%a.out                                                                                                                                                              
#SBATCH --error=logs/job_%A_%a.err                                                                                                                                                               
#SBATCH --array=0-160                                                                                                                                                                                                                                                                                                                                                               
#SBATCH --ntasks=1                                                                                                                                                                              
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=3G                                                                                                                                                                                                                                                                                                                                            
#SBATCH --time=00-02:00:00                                                                                                                                                                       
#SBATCH --partition=blackhole  

# Specify the directory containing the files
directory="/scratch/zt1/project/ricotti-prj/user/ricotti/GC-Fred/CC-Fiducial" 

# Specify the files
file_format="output_*/info_0*.txt" 

# Specify the Python script to run
script="main.py" 

# List of files to run as a job array
input_files=$(ls -d $directory/output_*/info_0*.txt)

# Specify path to python install
# include yt, numpy, matplotlib, astropy, scipy
python_path=""

input_file=${input_files[$SLURM_ARRAY_TASK_ID]}

# Echo for diagnostics
slice_num=$(echo "$input_file" | grep -oP 'info_\K[0-9]+')
padded_slice_num=$(printf "%05d" "$slice_num")
output_file="stdout-$padded_scan_num.txt"
err_file="stderr-$padded_scan_num.txt"

echo "SLURM Job ID: $SLURM_ARRAY_JOB_ID"
echo "SLURM Task ID: $SLURM_ARRAY_TASK_ID"
echo "Using input file: $input_file"

echo "which python: $(which python)"
echo "Python Path: $python_path"
echo "Using input file: $input_file"
echo "Scan: $padded_scan_num"

# Run the script                                                                                                                                                                        
$python_path $script $input_file
