#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${ENV_FILE:-${SCRIPT_DIR}/.env}"

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "Error: env file not found at ${ENV_FILE}"
  exit 1
fi

echo "Loading environment from ${ENV_FILE}"
set -a
# shellcheck disable=SC1090
source "${ENV_FILE}"
set +a

MODEL_DIR="${MODEL_DIR:-${SCRIPT_DIR}/model}"
GCS_BUCKET="${GCS_BUCKET:-}"
GCS_MODEL_OBJECTS="${GCS_MODEL_OBJECTS:-best_dog.pt,best_cat.pt}"
GCP_SA_KEY_FILE="${GCP_SA_KEY_FILE:-${GOOGLE_APPLICATION_CREDENTIALS:-}}"

if [[ -z "${GCS_BUCKET}" ]]; then
  echo "Error: GCS_BUCKET is required."
  exit 1
fi

if [[ -z "${GCP_SA_KEY_FILE}" ]]; then
  echo "Error: GCP_SA_KEY_FILE (or GOOGLE_APPLICATION_CREDENTIALS) is required."
  exit 1
fi

if [[ ! -f "${GCP_SA_KEY_FILE}" ]]; then
  echo "Error: service account key file not found: ${GCP_SA_KEY_FILE}"
  exit 1
fi

if ! command -v gcloud >/dev/null 2>&1; then
  echo "Error: gcloud CLI is not installed on this VM."
  exit 1
fi

DOWNLOAD_TOOL=""
if gcloud storage cp --help >/dev/null 2>&1; then
  DOWNLOAD_TOOL="gcloud-storage"
elif command -v gsutil >/dev/null 2>&1; then
  DOWNLOAD_TOOL="gsutil"
else
  echo "Error: neither 'gcloud storage cp' nor 'gsutil' is available on this VM."
  exit 1
fi

mkdir -p "${MODEL_DIR}"

echo "Authenticating with service account key: ${GCP_SA_KEY_FILE}"
gcloud auth activate-service-account --key-file="${GCP_SA_KEY_FILE}" --quiet

IFS=',' read -r -a MODEL_OBJECTS <<< "${GCS_MODEL_OBJECTS}"

for object_name_raw in "${MODEL_OBJECTS[@]}"; do
  object_name="$(echo "${object_name_raw}" | xargs)"
  if [[ -z "${object_name}" ]]; then
    continue
  fi
  object_name="${object_name#/}"
  GCS_URI="gs://${GCS_BUCKET}/${object_name}"

  echo "Downloading ${GCS_URI} -> ${MODEL_DIR}/${object_name##*/}"
  if [[ "${DOWNLOAD_TOOL}" == "gcloud-storage" ]]; then
    gcloud storage cp "${GCS_URI}" "${MODEL_DIR}/${object_name##*/}"
  else
    gsutil cp "${GCS_URI}" "${MODEL_DIR}/${object_name##*/}"
  fi
done

echo "Model download completed successfully."
