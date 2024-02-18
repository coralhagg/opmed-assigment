import json
import requests

# Base URL of your FHIR server
fhir_server_url = 'http://localhost:8080/fhir'

# List of JSON files to process and upload
json_files = [
    'appointments.json',
    'locations.json',
    'patients.json',
    'practitioners.json',
    'slots.json'
]


def upload_fhir_bundle(bundle):
    """
    Uploads a single FHIR Bundle to the FHIR server.
    """
    headers = {
        'Content-Type': 'application/fhir+json',
        'Accept': 'application/fhir+json'
    }
    response = requests.post(fhir_server_url, json=bundle, headers=headers)

    if response.status_code in [200, 201]:
        print('Successfully uploaded bundle.')
    else:
        print(f'Failed to upload bundle. Status code: {response.status_code}, Response: {response.text}')


def process_and_upload(file_path):
    """
    Processes a JSON file and uploads each FHIR Bundle found within.
    """
    with open(file_path, 'r') as file:
        data = json.load(file)

    # Iterate through each key in the JSON file
    for key, value in data.items():
        if isinstance(value, dict) and value.get('resourceType') == 'Bundle':
            upload_fhir_bundle(value)


for json_file in json_files:
    print(f'Processing and uploading {json_file}...')
    process_and_upload(json_file)
