# WellPaw Model Service

Flask/Gunicorn service for WellPaw model inference. The Dockerfile builds a
production container that listens on port `50002`.

## Prerequisites

- Docker
- Download the model files from
  [Google Drive](https://drive.google.com/drive/folders/1pJAOh5FqdQy7q7wJ7sJneLv8ZEi28_Tn?usp=sharing)
  and place them in `model/`:
  - `model/best_dog.onnx`
  - `model/best_cat.onnx`
- Mapping files in `mapping/`:
  - `mapping/dog.json`
  - `mapping/cat.json`

The Docker build excludes `model/`, so the model files must be mounted into the
container at runtime.

## Build Production Image

Build the image from the Dockerfile:

```sh
docker build -t wellpaw-model:prod .
```

## Run Production Container

Start the service:

```sh
docker run -d \
  --name wellpaw_model \
  --restart unless-stopped \
  --memory=7680m \
  -p 50002:50002 \
  -e PORT=50002 \
  -e PYTHONUNBUFFERED=1 \
  -v "$(pwd)/model:/app/model:ro" \
  -v "$(pwd)/mapping:/app/mapping:ro" \
  wellpaw-model:prod
```

Verify the service is running:

```sh
curl http://localhost:50002/healthz
```

Expected response:

```json
{"status":200}
```

View logs:

```sh
docker logs -f wellpaw_model
```

Stop and remove the container:

```sh
docker rm -f wellpaw_model
```

## Prediction Endpoints

- `POST /predict/dog`
- `POST /predict/cat`

Send JSON with a base64-encoded image in the `image` field:

```sh
curl -X POST http://localhost:50002/predict/dog \
  -H "Content-Type: application/json" \
  -d '{"image":"<base64-image>"}'
```

## Troubleshooting

- `Conflict. The container name "/wellpaw_model" is already in use`: run
  `docker rm -f wellpaw_model` before starting a new container.
- Model load errors: confirm `model/best_dog.onnx` and `model/best_cat.onnx`
  exist on the host before starting the container.
- Port conflicts: stop the process using port `50002`, or change the host port
  in the run command from `-p 50002:50002` to another host port, for example
  `-p 50003:50002`.
