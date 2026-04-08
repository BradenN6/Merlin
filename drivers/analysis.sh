#!/bin/bash --login 
#SBATCH --account=ricotti-prj-astr                                                                                                                                                                                 
#SBATCH --job-name=nebular-lines                                                                                                                                                             
#SBATCH --output=logs/job_%A_%a.out                                                                                                                                                              
#SBATCH --error=logs/job_%A_%a.err                                                                                                                                                               
#SBATCH --array=0-162                                                                                                                                                                                                                                                                                                                                                               
#SBATCH --ntasks=1                                                                                                                                                                              
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=3G                                                                                                                                                                                                                                                                                                                                            
#SBATCH --time=00-02:00:00 

# Specify the directory containing the files
directory="/scratch/zt1/project/ricotti-prj/user/ricotti/GC-Fred/CC-Fiducial" 

# Specify the files
file_format="output_*/info_0*.txt" 

# Specify the Python script to run
script="init-driver.py" 

# List of files to run as a job array
input_files=($(ls -d $directory/output_*/info_0*.txt))

# Specify path to python install
# include yt, numpy, matplotlib, astropy, scipy
python_path="/cvmfs/hpcsw.umd.edu/spack-software/2023.11.20/views/2023/linux-rhel8-zen2/gcc@11.3.0/python-3.10.10/mpi-nocuda/linux-rhel8-zen2/gcc/11.3.0/bin/python3"

input_file=${input_files[$SLURM_ARRAY_TASK_ID]}

# Echo for diagnostics
slice_num=$(echo "$input_file" | grep -oP 'info_\K[0-9]+')
padded_slice_num=$(printf "%05d" "$slice_num")
output_file="stdout-$padded_slice_num.txt"
err_file="stderr-$padded_slice_num.txt"

echo "SLURM Job ID: $SLURM_ARRAY_JOB_ID"
echo "SLURM Task ID: $SLURM_ARRAY_TASK_ID"
echo "Using input file: $input_file"

echo "which python: $(which python)"
echo "Python Path: $python_path"
echo "Using input file: $input_file"
echo "Scan: $padded_scan_num"

# Run the script                                                                                                                                                                        
$python_path $script $input_file
