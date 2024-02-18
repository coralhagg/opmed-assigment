# HAPI FHIR Server Setup and Running `main.py`

This guide walks you through setting up a HAPI FHIR server using Docker and running a Python script to interact with it.

## Prerequisites

- Ensure you have Docker installed on your machine. If not, you can download it from [Docker's official website](https://www.docker.com/products/docker-desktop).

## Step 1: Start the HAPI-FHIR Server

1. Open your terminal or command prompt.
2. Pull the latest HAPI FHIR server Docker image by running the following command:

   ```sh
   docker pull hapiproject/hapi:latest
3. Start the HAPI FHIR server by executing:
    ```sh
    docker run -p 8080:8080 hapiproject/hapi:latest

## Step 2: Run the main.py Python Script

