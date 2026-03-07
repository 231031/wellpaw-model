# WellPaw Model Service

This service runs the WellPaw model with Docker Compose.

## Prerequisites

- Docker and Docker Compose installed
- Model files available in the `model/` directory

## Quick Start

1. Create the shared Docker network (one-time setup):
   ```docker network create wellpaw-shared-nw```

2. Put model files in the `model/` folder.

3. Start the service:
   ```docker-compose up -d```
