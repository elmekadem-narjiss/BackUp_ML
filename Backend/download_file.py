from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.auth.transport.requests import Request
import io
import os
import json

def authenticate_drive():
    # Load client secrets
    with open('client_secrets.json', 'r') as f:
        client_secrets = json.load(f)
    
    # Load existing token
    creds = None
    if os.path.exists('token.json'):
        with open('token.json', 'r') as f:
            token_data = json.load(f)
        creds = Credentials(
            token=token_data['token'],
            refresh_token=token_data['refresh_token'],
            token_uri=token_data['token_uri'],
            client_id=token_data['client_id'],
            client_secret=token_data['client_secret'],
            scopes=token_data['scopes']
        )
    
    # If credentials are invalid or expired, refresh or re-authenticate
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'client_secrets.json',
                scopes=['https://www.googleapis.com/auth/drive']
            )
            raise Exception(
                "Token is invalid and cannot run interactive auth in CI"
            )
        
        # Save refreshed credentials
        with open('token.json', 'w') as f:
            json.dump({
                'token': creds.token,
                'refresh_token': creds.refresh_token,
                'token_uri': creds.token_uri,
                'client_id': creds.client_id,
                'client_secret': creds.client_secret,
                'scopes': creds.scopes
            }, f)
    
    return build('drive', 'v3', credentials=creds)

def download_file(service, file_name, destination):
    # Search for the file by name
    results = service.files().list(
        q=f"name='{file_name}'",
        spaces='drive',
        fields='files(id, name)'
    ).execute()
    files = results.get('files', [])
    
    if not files:
        raise Exception(f"No file named '{file_name}' found in Google Drive")
    
    file_id = files[0]['id']
    request = service.files().get_media(fileId=file_id)
    fh = io.FileIO(destination, 'wb')
    downloader = MediaIoBaseDownload(fh, request)
    
    done = False
    while not done:
        status, done = downloader.next_chunk()
        print(f"Download {int(status.progress() * 100)}%")
    
    print(f"File '{file_name}' downloaded to '{destination}'")

if __name__ == '__main__':
    try:
        service = authenticate_drive()
        download_file(
            service,
            'lstm_predictions_charger.csv',
            'lstm_predictions_charger.csv'
        )
    except Exception as e:
        print(f"Error: {str(e)}")
        exit(1)
