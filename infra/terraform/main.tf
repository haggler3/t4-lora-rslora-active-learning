terraform {
  required_version = ">= 1.5.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = ">= 5.0.0"
    }
    random = {
      source  = "hashicorp/random"
      version = ">= 3.5.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
  zone    = var.zone
}

resource "random_id" "suffix" {
  byte_length = 4
}

# Fetch the latest prebaked PyTorch + CUDA 12.9 + NVIDIA 580 Deep Learning image
data "google_compute_image" "deeplearning_image" {
  family  = var.image_family
  project = var.image_project
}

# Google Cloud Storage bucket for Spot VM preemption checkpoint snapshots
resource "google_storage_bucket" "checkpoint_bucket" {
  count                       = var.enable_checkpoint_bucket ? 1 : 0
  name                        = "qlora-icl-checkpoints-${var.project_id}-${random_id.suffix.hex}"
  location                    = var.region
  force_destroy               = false
  uniform_bucket_level_access = true

  versioning {
    enabled = true
  }

  lifecycle_rule {
    condition {
      age = 30 # Retain checkpoint versions for 30 days
    }
    action {
      type = "Delete"
    }
  }
}

# Dedicated Service Account for Perception VM
resource "google_service_account" "vm_sa" {
  account_id   = "qlora-icl-runner-${random_id.suffix.hex}"
  display_name = "LoRA experiment Spot VM Service Account"
}

# Grant Storage Object Admin on the checkpoint bucket
resource "google_storage_bucket_iam_member" "bucket_admin" {
  count  = var.enable_checkpoint_bucket ? 1 : 0
  bucket = google_storage_bucket.checkpoint_bucket[0].name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.vm_sa.email}"
}

# Spot VM with NVIDIA Tesla T4 GPU
resource "google_compute_instance" "spot_vm" {
  name         = var.instance_name
  machine_type = var.machine_type
  zone         = var.zone

  # Spot VM Configuration
  scheduling {
    preemptible                 = true
    provisioning_model          = "SPOT"
    automatic_restart           = false
    on_host_maintenance         = "TERMINATE"
    instance_termination_action = "STOP"
  }

  # 1x NVIDIA Tesla T4 Accelerator
  guest_accelerator {
    type  = var.gpu_type
    count = var.gpu_count
  }

  boot_disk {
    initialize_params {
      image = data.google_compute_image.deeplearning_image.self_link
      size  = var.boot_disk_size_gb
      type  = var.boot_disk_type
    }
  }

  network_interface {
    network = "default"
    access_config {
      # Ephemeral public IP for SSH access
    }
  }

  service_account {
    email  = google_service_account.vm_sa.email
    scopes = ["https://www.googleapis.com/auth/cloud-platform"]
  }

  metadata = {
    GCS_CHECKPOINT_BUCKET = var.enable_checkpoint_bucket ? "gs://${google_storage_bucket.checkpoint_bucket[0].name}" : ""
  }

  lifecycle {
    ignore_changes = [
      metadata["ssh-keys"],
    ]
  }
}
