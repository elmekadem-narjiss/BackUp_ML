import os
import sys
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import io

print("Début de l'étape Download LSTM predictions from Google Drive")

try:
    SCOPES = ['https://www.googleapis.com/auth/drive']
    TOKEN_FILE = 'token.json'

    if not os.path.exists(TOKEN_FILE):
        print("Erreur : token.json n'existe pas.")
        sys.exit(1)

    creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
    service = build('drive', 'v3', credentials=creds)

    def download_file(file_name, destination):
        results = service.files().list(q=f"name='{file_name}' and 'root' in parents", fields="files(id, name)").execute()
        files = results.get('files', [])
        if not files:
            print(f"Fichier {file_name} non trouvé.")
            return False
        file_id = files[0]['id']
        request = service.files().get_media(fileId=file_id)
        fh = io.FileIO(destination, 'wb')
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
            print(f"Téléchargement: {int(status.progress() * 100)}%")
        print(f"Téléchargement terminé : {destination}")
        return True

    if not download_file('lstm_predictions_charger.csv', 'lstm_predictions_charger.csv'):
        sys.exit(1)

except Exception as e:
    print(f"Erreur : {e}")
    sys.exit(1)
