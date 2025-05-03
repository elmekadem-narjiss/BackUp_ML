import os
import time
import json
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.chrome.options import Options
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import io

# Configuration
COLAB_URL = "https://colab.research.google.com/github/elmekadem-narjiss/BackUp_ML/blob/main/Backend/ppo_pipeline.ipynb"
DOWNLOAD_FOLDER = os.getcwd()
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
CREDENTIALS_FILE = 'client_secrets.json'  # Fichier d'authentification Google API

# Fonction pour exécuter le notebook dans Colab
def run_colab_notebook():
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument(f"--download.default_directory={DOWNLOAD_FOLDER}")

    driver = webdriver.Chrome(options=chrome_options)
    driver.get(COLAB_URL)
    print("Page Colab chargée")

    # Attendre que le bouton "Run all" soit disponible
    wait = WebDriverWait(driver, 60)
    run_all_button = wait.until(EC.presence_of_element_located((By.XPATH, '//div[@aria-label="Run all cells"]')))
    print("Bouton 'Run all' trouvé")

    # Exécuter toutes les cellules
    action_chains = ActionChains(driver)
    action_chains.move_to_element(run_all_button).click().perform()
    print("Exécution de toutes les cellules...")

    # Attendre la fin de l'exécution (jusqu'à 10 minutes)
    time.sleep(600)  # Ajuste selon la durée d'exécution de ton notebook
    print("Exécution terminée (ou timeout atteint)")

    driver.quit()

# Fonction pour télécharger un fichier depuis Google Drive
def download_from_drive(file_name, destination):
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    if not creds or not creds.valid:
        flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
        creds = flow.run_local_server(port=0)
        with open('token.json', 'w') as token:
            token.write(creds.to_json())

    service = build('drive', 'v3', credentials=creds)

    # Rechercher le fichier dans Google Drive
    results = service.files().list(q=f"name='{file_name}' and 'root' in parents", fields="files(id, name)").execute()
    files = results.get('files', [])

    if not files:
        print(f"Fichier {file_name} non trouvé dans Google Drive.")
        return False

    file_id = files[0]['id']
    request = service.files().get_media(fileId=file_id)
    fh = io.FileIO(destination, 'wb')
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while done is False:
        status, done = downloader.next_chunk()
        print(f"Téléchargement {file_name}: {int(status.progress() * 100)}%")
    return True

# Fonction principale
def main():
    # Étape 1 : Exécuter le notebook dans Colab
    run_colab_notebook()

    # Étape 2 : Télécharger les fichiers de métriques depuis Google Drive
    metrics_files = ['ppo_bess_model_metrics.json', 'evaluation_metrics.json']
    for file_name in metrics_files:
        success = download_from_drive(file_name, file_name)
        if not success:
            print(f"Échec du téléchargement de {file_name}")
            return

    print("Automatisation terminée avec succès.")

if __name__ == "__main__":
    main()