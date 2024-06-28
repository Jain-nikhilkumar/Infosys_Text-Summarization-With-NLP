from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import os
import json

def authenticate_drive():
    # Save credentials to a file
    credentials = os.getenv('GDRIVE_CREDENTIALS')
    with open('credentials.json', 'w') as f:
        f.write(credentials)
    
    # Authenticate and create the PyDrive client
    gauth = GoogleAuth()
    gauth.LoadCredentialsFile('credentials.json')
    drive = GoogleDrive(gauth)
    return drive

def list_files_in_folder(drive, folder_id):
    file_list = drive.ListFile({'q': f"'{folder_id}' in parents and trashed=false"}).GetList()
    return file_list

def download_file_from_folder(drive, folder_id, file_name, destination):
    file_list = list_files_in_folder(drive, folder_id)
    for file in file_list:
        if file['title'] == file_name:
            file.GetContentFile(destination)
            print(f"Downloaded {file_name} to {destination}")
            return
    print(f"File {file_name} not found in the folder.")

if __name__ == "__main__":
    folder_id = '1Ba65pa0dbWpKPufLFr0bHI-DUJwy568s'  # Replace with your folder ID
    file_name = 'env'  # The name of the file you want to download
    destination = 'env'  # The destination where you want to save the file
    
    drive = authenticate_drive()
    download_file_from_folder(drive, folder_id, file_name, destination)
