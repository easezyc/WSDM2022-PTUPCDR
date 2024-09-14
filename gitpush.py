import os
import subprocess
import shutil
import argparse
import sys
from datetime import datetime

def log_message(log_file_path, message):
    with open(log_file_path, "a") as log_file:
        log_file.write(message + "\n")
    print(message)

def log_date_time(log_file_path):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file_path, "a") as log_file:
        log_file.write(f"Log Date and Time: {current_time}\n\n")
    print(f"Log Date and Time: {current_time}")

def check_folder_exists(folder_path, log_file_path):
    if not os.path.exists(folder_path):
        log_message(log_file_path, f"Error: Folder {folder_path} does not exist.")
        sys.exit(1)
    else:
        log_message(log_file_path, f"Folder {folder_path} exists.")

def move_folder(folder_path, target_path, log_file_path):
    folder_name = os.path.basename(folder_path)
    # Move the folder
    if os.path.exists(folder_path):
        shutil.move(folder_path, os.path.join(target_path, folder_name))
        log_message(log_file_path, f"Moved {folder_name} to {target_path}")
    else:
        log_message(log_file_path, f"Error: Folder {folder_name} not found at {folder_path}.")
        sys.exit(1)

def create_readme(repo_path, log_file_path):
    readme_content = """
    DATA FOLDER HAS BEEN MOVED TO ANOTHER LOCATION TO MAKE GITHUB PUSH POSSIBLE.
    TO RUN THIS CODE:
    STEP 1: CREATE a 'data' FOLDER. CREATE 3 MORE FOLDERS INSIDE THAT: 'raw', 'ready', 'mid'. PUT YOUR DATASET FILES IN JSON.GZ IN THE 'raw' FOLDER.
    STEP 2: CHANGE 'config.json' AS PER YOUR REQUIREMENTS INCLUDING NAME OF THE DATASET AND UID/IID COUNT. MAKE NECESSARY CHANGES IN 'entry.py' BASED ON THESE CHANGES.
    STEP 3: RUN 'run.slurm' FILE TO RUN ON HOPPER USING 'entry.py' WITH MID AND READY SET TO 1 TO PREPARE THE DATASET. ONCE FINISHED, RUN 'entry.py' WITH OTHER PARAMETERS. PARAMETERS ALREADY EXIST IN THE SLURM FILE. JUST UNCOMMENT THEM.
    STEP 4: THE RESULTS WILL BE STORED IN THE OUTPUT FILE SPECIFIED IN THE SLURM FILE AND USER-BASED MAE/RMSE CALCULATIONS WILL BE STORED IN ./results_cs798/.
    """

    readme_file_path = os.path.join(repo_path, "readme.txt")
    with open(readme_file_path, "w") as file:
        file.write(readme_content.strip())
    log_message(log_file_path, "Created readme.txt with instructions.")

def push_to_github(repo_path, branch_name, commit_message, log_file_path):
    # Change to the repository directory
    os.chdir(repo_path)

    # Run git commands
    try:
        # Checkout to the branch
        subprocess.run(["git", "checkout", "-b", branch_name], check=True)

        # Add files to the staging area
        subprocess.run(["git", "add", "."], check=True)

        # Commit the changes
        subprocess.run(["git", "commit", "-m", commit_message], check=True)

        # Push to the remote repository
        subprocess.run(["git", "push", "origin", branch_name], check=True)

        log_message(log_file_path, "Code pushed to GitHub successfully!")
    except subprocess.CalledProcessError as e:
        log_message(log_file_path, f"Error occurred during git push: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Move folders, create readme, and push to GitHub.")

    # Add arguments for repo path and backup path
    parser.add_argument('--repo-path', type=str, required=True, help="Path to the local repository.")
    parser.add_argument('--backup-path', type=str, required=True, help="Path to backup large folders.")

    # Add individual arguments for each folder path relative to repo-path
    parser.add_argument('--data-folder', type=str, required=True, help="Relative or full path to data folder from repo-path.")
    parser.add_argument('--results-cs798-folder', type=str, required=True, help="Relative or full path to results_cs798 folder from repo-path.")
    parser.add_argument('--results-llmasrec-folder', type=str, required=True, help="Relative or full path to results_llmasrec folder from repo-path.")

    # Add arguments for git push
    parser.add_argument('--branch', type=str, required=True, help="Branch name to push to GitHub.")
    parser.add_argument('--commit', type=str, required=True, help="Commit message for GitHub.")

    # Add argument for log file name
    parser.add_argument('--log-file', type=str, required=True, help="Name of the log file to store the process steps.")

    # Parse the arguments
    args = parser.parse_args()

    # Create log file path based on input
    log_file_path = os.path.join(args.repo_path, args.log_file)

    # Log the date and time
    log_date_time(log_file_path)

    # Step 1: Calculate the full paths for each folder relative to repo-path
    data_folder_path = os.path.abspath(os.path.join(args.repo_path, args.data_folder))
    results_cs798_folder_path = os.path.abspath(os.path.join(args.repo_path, args.results_cs798_folder))
    results_llmasrec_folder_path = os.path.abspath(os.path.join(args.repo_path, args.results_llmasrec_folder))

    # Backup folder paths
    backup_data_folder = os.path.join(args.backup_path, os.path.basename(data_folder_path))
    backup_results_cs798_folder = os.path.join(args.backup_path, os.path.basename(results_cs798_folder_path))
    backup_results_llmasrec_folder = os.path.join(args.backup_path, os.path.basename(results_llmasrec_folder_path))

    # Step 2: Check if each folder exists
    check_folder_exists(data_folder_path, log_file_path)
    check_folder_exists(results_cs798_folder_path, log_file_path)
    check_folder_exists(results_llmasrec_folder_path, log_file_path)

    # Step 3: Move each folder to the backup location
    move_folder(data_folder_path, args.backup_path, log_file_path)
    move_folder(results_cs798_folder_path, args.backup_path, log_file_path)
    move_folder(results_llmasrec_folder_path, args.backup_path, log_file_path)

    # Step 4: Create readme file explaining folder structure
    create_readme(args.repo_path, log_file_path)

    # Step 5: Push code to GitHub with provided branch name and commit message
    push_to_github(args.repo_path, args.branch, args.commit, log_file_path)

    # Step 6: Move the folders back to their original places after successful git push
    move_folder(backup_data_folder, os.path.dirname(data_folder_path), log_file_path)
    move_folder(backup_results_cs798_folder, os.path.dirname(results_cs798_folder_path), log_file_path)
    move_folder(backup_results_llmasrec_folder, os.path.dirname(results_llmasrec_folder_path), log_file_path)
