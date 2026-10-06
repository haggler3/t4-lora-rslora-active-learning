#!/usr/bin/env bash
# Run the LoRA/rsLoRA/active-learning experiment on a GCP T4 spot VM and pull results back.
# Usage (Git Bash):  ./run_on_gcp.sh [create|upload|run|status|fetch|stop|delete]
# Re-running `run` after a spot preemption resumes: finished runs are skipped.
set -euo pipefail
cd "$(dirname "$0")"

VM=${VM:-lora-al-t4}
ZONE=${ZONE:-us-central1-a}
# gcloud on this machine needs a Python >= 3.10; use the one bundled with the SDK.
export CLOUDSDK_PYTHON=${CLOUDSDK_PYTHON:-"$LOCALAPPDATA/Google/Cloud SDK/google-cloud-sdk/platform/bundledpython/python.exe"}
ssh_vm() { gcloud compute ssh "$VM" --zone="$ZONE" --quiet --command="$1"; }

case "${1:-run}" in
  create)
    gcloud compute instances create "$VM" --zone="$ZONE" --machine-type=n1-standard-8 \
      --accelerator=type=nvidia-tesla-t4,count=1 --provisioning-model=SPOT \
      --instance-termination-action=STOP --maintenance-policy=TERMINATE \
      --image-family=pytorch-2-9-cu129-ubuntu-2204-nvidia-580 \
      --image-project=deeplearning-platform-release \
      --boot-disk-size=100GB --boot-disk-type=pd-balanced --scopes=cloud-platform ;;
  upload)
    # plink mangles binary stdin, so ship the code as base64 text.
    b64=$(tar czf - lora_rslora_al.py make_report.py | base64 -w0)
    ssh_vm "mkdir -p ~/exp && echo $b64 | base64 -d | tar xzf - -C ~/exp
            python3 -m pip install -q --user -U pip
            python3 -m pip install -q --user 'transformers>=4.45' 'peft>=0.13' datasets matplotlib
            # The DL image ships a torchaudio that fails to load and breaks transformers imports.
            sudo python3 -m pip uninstall -y -q torchaudio || true" ;;
  run)
    gcloud compute instances start "$VM" --zone="$ZONE" 2>/dev/null || true
    ssh_vm "cd ~/exp && TRANSFORMERS_VERBOSITY=error nohup python3 lora_rslora_al.py >> run.log 2>&1 < /dev/null & disown; echo started" ;;
  status)
    ssh_vm "grep -E 'FINAL|ALL DONE|Traceback' ~/exp/run.log | tail -15" ;;
  fetch)
    mkdir -p results
    ssh_vm "cd ~/exp && python3 make_report.py results > /dev/null && tar czf - --exclude=.hf_cache results | base64 -w0" \
      | base64 -d | tar xzf - ;;
  stop)   gcloud compute instances stop "$VM" --zone="$ZONE" ;;
  delete) gcloud compute instances delete "$VM" --zone="$ZONE" ;;
esac
