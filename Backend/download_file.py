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

    print("Vérification de l'existence de token.json...")
    if not os.path.exists(TOKEN_FILE):
        print("Erreur : token.json n'existe pas.")
        sys.exit(1)
    print("token.json trouvé.")

    print("Chargement des credentials depuis token.json...")
    creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
    print("Credentials chargées avec succès.")

    print("Construction du service Google Drive...")
    service = build('drive', 'v3', credentials=creds)
    print("Service Google Drive construit avec succès.")

    def download_file(file_name, destination):
        print(f"Recherche du fichier {file_name} dans Google Drive...")
        results = service.files().list(q=f"name='{file_name}' and 'root' in parents", fields="files(id, name)").execute()
        files = results.get('files', [])
        if not files:
            print(f"Fichier {file_name} non trouvé dans Google Drive.")
            return False
        file_id = files[0]['id']
        print(f"Fichier trouvé, ID: {file_id}")

        print(f"Téléchargement de {file_name} vers {destination}...")
        request = service.files().get_media(fileId=file_id)
        fh = io.FileIO(destination, 'wb')
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while done is False:
            status, done = downloader.next_chunk()
            print(f"Téléchargement {file_name}: {int(status.progress() * 100)}%")
        print(f"Téléchargement terminé : {destination}")
        return True

    if download_file('lstm_predictions_charger.csv', 'lstm_predictions_charger.csv'):
        print("Fichier téléchargé avec succès.")
    else:
        print("Échec du téléchargement du fichier.")
        sys.exit(1)

except Exception as e:
    print(f"Erreur lors du téléchargement : {str(e)}")
    sys.exit(1)
