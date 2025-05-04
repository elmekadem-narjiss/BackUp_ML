from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
import io
import os
from googleapiclient.errors import HttpError

SCOPES = ['https://www.googleapis.com/auth/drive']
FILE_ID = '1nT6AH5scHrteA7LkumeSaylsHpnX-X1d'  # Nouvel ID du fichier
OUTPUT_PATH = 'lstm_predictions_charger.csv'

def download_file():
    try:
        creds = None
        if os.path.exists('token.json'):
            creds = Credentials.from_authorized_user_file('token.json', SCOPES)
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file('client_secrets.json', SCOPES)
                creds = flow.run_local_server(port=0)
            with open('token.json', 'w') as token:
                token.write(creds.to_json())
        service = build('drive', 'v3', credentials=creds)
        request = service.files().get_media(fileId=FILE_ID)
        fh = io.FileIO(OUTPUT_PATH, 'wb')
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while done is False:
            status, done = downloader.next_chunk()
            print(f"Download {int(status.progress() * 100)}%.")
        print(f"Fichier téléchargé avec succès : {OUTPUT_PATH}")
    except HttpError as error:
        print(f"Erreur lors du téléchargement du fichier : {error}")
        if error.resp.status == 404:
            print(f"Fichier avec l'ID {FILE_ID} introuvable. Vérifiez l'ID ou les permissions.")
        elif error.resp.status == 403:
            print(f"Accès refusé pour le fichier avec l'ID {FILE_ID}. Vérifiez les permissions sur Google Drive.")
        raise
    except Exception as e:
        print(f"Erreur inattendue : {e}")
        raise

if __name__ == '__main__':
    download_file()
