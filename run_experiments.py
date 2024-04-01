import os
import shutil

# Define the directories to search for Python files
directories = ["ExpCleanLabel", "ExpInBounds", "ExpTriggerSize"]

# Path to the job_executer.sh template
job_executer_template_path = "job_executer.sh"

# Iterate over each directory
for directory in directories:
    # Iterate over each file in the directory
    for filename in os.listdir(directory):
        # if filename.endswith(".py"):
        if "CovType" in filename and filename.endswith(".py"):
            # Construct the full path to the Python file
            python_file_path = os.path.join(directory, filename)
            
            # Create a new job_executer file for this Python file
            new_job_executer_path = f"{directory}_{filename}_job_executer.sh"
            shutil.copy(job_executer_template_path, new_job_executer_path)
            
            # Add the command to run the Python file at the end of the new job_executer file
            with open(new_job_executer_path, "a") as new_job_file:
                new_job_file.write(f"\n# Run the Python script\npython {python_file_path}\n")
            
            # Submit the job using sbatch
            os.system(f"sbatch {new_job_executer_path}")