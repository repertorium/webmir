# WebMIR – Automatic Chant Indexation System

WebMIR is a web-based system for the automatic detection and indexation of medieval chants in digitised manuscripts. It combines OMR, OCR, and MIR techniques to identify chants and link textual and melodic information, providing a visual interface for browsing processed manuscripts and inspecting detected chants.

The system includes a Python backend implementing the MIR pipeline and a React-based web frontend for interaction and visualisation.

---

## Components

- **Backend**: Python + FastAPI, MIR-based chant detection, task queue for long-running jobs
- **Frontend**: React + Vite, manuscript browsing and results visualisation
- **Deployment**: Docker-based, no local dependencies required

---

## Deployment with Docker

### Prerequisites
- Docker
- Docker Compose (`docker compose`)

### Build and run

```bash
docker compose up -d --build

```

### Reverse Proxy (Apache)
To expose the application at `/webmir/` using Apache:

- Enable required modules:

    ```bash
    sudo a2enmod proxy proxy_http

    ```

- Create a virtual host configuration (for instance, `/etc/apache2/sites-available/webmir.conf`):

    ```bash
    <VirtualHost *:8082>
        ProxyPreserveHost On

        ProxyPass        /webmir/api/  http://127.0.0.1:9001/
        ProxyPassReverse /webmir/api/  http://127.0.0.1:9001/

        ProxyPass        /webmir/      http://127.0.0.1:9000/
        ProxyPassReverse /webmir/      http://127.0.0.1:9000/

    </VirtualHost>

    ```

- Ensure Apache listens on the port (`/etc/apache2/ports.conf`):

    ```bash
    Listen 8082
    ```

- Enable the site and restart Apache:

    ```bash
    sudo a2ensite webmir.conf
    sudo systemctl restart apache2
    ```

- Access the site in:

    ```bash
    http://<server>:8082/webmir/
    ```
