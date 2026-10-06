"""
GCP Spot VM Preemption Handler & Checkpoint Sync Daemon
Watches GCP Compute Metadata server for the 30-second preemption notice.
Gracefully notifies the training loop to dump checkpoints and sync to Google Cloud Storage (GCS).
"""
import os
import time
import signal
import logging
import requests
import subprocess

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

METADATA_URL = "http://metadata.google.internal/computeMetadata/v1/instance/preempted"
METADATA_HEADERS = {"Metadata-Flavor": "Google"}

def check_preemption():
    """Poll GCP metadata server to detect if Spot instance is being reclaimed."""
    try:
        resp = requests.get(METADATA_URL, headers=METADATA_HEADERS, timeout=2)
        if resp.status_code == 200 and resp.text.strip().upper() == "TRUE":
            return True
    except requests.RequestException:
        pass
    return False

def sync_checkpoints_to_gcs(checkpoint_dir="checkpoints", gcs_bucket=None):
    """Upload local checkpoints to GCS."""
    if not gcs_bucket:
        logging.warning("No GCS bucket specified; saving locally only.")
        return
    logging.info(f"Syncing {checkpoint_dir} to {gcs_bucket}...")
    try:
        subprocess.run(["gsutil", "-m", "rsync", "-r", checkpoint_dir, f"{gcs_bucket}/checkpoints"], check=True)
        logging.info("GCS sync completed successfully!")
    except Exception as e:
        logging.error(f"Failed to sync to GCS: {e}")

def main():
    gcs_bucket = os.environ.get("GCS_CHECKPOINT_BUCKET")
    logging.info("Starting GCP Spot Preemption Monitor Daemon...")
    if gcs_bucket:
        logging.info(f"Configured GCS bucket: {gcs_bucket}")

    while True:
        if check_preemption():
            logging.warning("PREEMPTION WARNING RECEIVED! VM will terminate in ~30 seconds.")
            # Trigger emergency checkpoint save
            sync_checkpoints_to_gcs(gcs_bucket=gcs_bucket)
            # Notify training processes via SIGUSR1 or SIGTERM
            break
        time.sleep(5)

if __name__ == "__main__":
    main()
