# Terraform Infrastructure: GCP Spot T4 Perception VM

Automated, reproducible Infrastructure-as-Code (IaC) configuration for deploying an NVIDIA Tesla T4 GPU Spot instance on Google Cloud Platform with integrated Google Cloud Storage (GCS) checkpoint persistence.

## Resources Provisioned
- **Google Compute Engine Spot Instance**: `n1-standard-4` (4 vCPUs, 15 GB RAM) attached to **1x NVIDIA Tesla T4 GPU** (16 GB GDDR6 VRAM) at spot discount pricing (~$0.11/hr).
- **Prebaked Deep Learning Image**: Uses `deeplearning-platform-release/pytorch-2-9-cu129-ubuntu-2204-nvidia-580`, equipped with NVIDIA Driver 580, CUDA 12.9, and PyTorch 2.9.
- **Preemption Checkpoint GCS Bucket**: Dedicated regional storage bucket configured with object versioning and 30-day lifecycle management for fault-tolerant Spot preemption snapshots.
- **Dedicated Service Account**: Least-privilege IAM configuration granting object management on the checkpoint bucket to the VM.

## Prerequisites
1. [Terraform CLI](https://developer.hashicorp.com/terraform/downloads) (>= 1.5.0 installed).
2. Authenticated Google Cloud account:
   ```bash
   gcloud auth application-default login
   ```

## Quickstart

### 1. Initialize Configuration
```bash
cd infra/terraform
cp terraform.tfvars.example terraform.tfvars
```
Update `terraform.tfvars` with your GCP `project_id`.

### 2. Plan and Deploy
```bash
terraform init
terraform plan
terraform apply
```

### 3. Access the Spot Instance
Upon completion, Terraform outputs the public IP and direct SSH command:
```bash
gcloud compute ssh qlora-icl-spot --zone=us-central1-a
```

### 4. Tear Down
To destroy the VM and stop all cloud spend:
```bash
terraform destroy
```
