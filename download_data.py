from sys import stdout
import requests
import config
import os
import zipfile
import warnings

URL = "https://docs.google.com/uc?export=download"
CHUNK_SIZE = 32768


def check_data_available(dataset_path):
    if not os.path.exists(dataset_path):
        return False
    else:
        return True


def download_file_from_google_drive(dataset_name, id_download, unzip=True):
    dataset_target = os.path.join(config.DATA_PATH, dataset_name)
    destination = os.path.join(config.DATA_PATH, dataset_name + '.zip')

    if not check_data_available(dataset_target):
        print(f"Downloading {dataset_name} to {dataset_target}")

        session = requests.Session()

        response = session.get(URL, params={'id': id_download}, stream=True)
        assert response.status_code == 200, f'Token is invalid'
        token = get_confirm_token(response)

        if token:
            params = {'id': id_download, 'confirm': token}
            response = session.get(URL, params=params, stream=True)

        save_response_content(response, destination)

        if unzip:
            print(destination)
            unzip_download(destination, config.DATA_PATH)
            assert check_data_available(dataset_target), f'Data was not successful downloaded'

        delete_zip(destination)
    else:
        print(f"{dataset_name} data set already available")


def unzip_download(destination, dataset_path):
    try:
        print('Unzipping...', end='')
        stdout.flush()
        with zipfile.ZipFile(destination, 'r') as z:
            z.extractall(dataset_path)
        print('Done.')
    except zipfile.BadZipfile:
        warnings.warn('Ignoring `unzip` since "{}" does not look like a valid zip file'.format(destination))


def delete_zip(zip_path):
    try:
        os.remove(zip_path)
    except OSError as e:  # if failed, report it back to the user
        warnings.warn("Error: {} - {}.".format(e.filename, e.strerror))


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None


def save_response_content(response, destination):
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


if __name__ == "__main__":
    download_file_from_google_drive(dataset_name=config.DATA_NAME_TEST, id_download=config.DOWNLOAD_TEST)
