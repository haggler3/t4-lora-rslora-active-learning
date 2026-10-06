variable "project_id" {
  description = "Google Cloud Project ID"
  type        = string
}

variable "region" {
  description = "GCP Region"
  type        = string
  default     = "us-central1"
}

variable "zone" {
  description = "GCP Zone supporting NVIDIA T4 GPUs"
  type        = string
  default     = "us-central1-a"
}

variable "instance_name" {
  description = "Name of the Spot VM instance"
  type        = string
  default     = "qlora-icl-spot"
}

variable "machine_type" {
  description = "Compute Engine machine type"
  type        = string
  default     = "n1-standard-4"
}

variable "gpu_type" {
  description = "NVIDIA accelerator type"
  type        = string
  default     = "nvidia-tesla-t4"
}

variable "gpu_count" {
  description = "Number of GPUs attached to the instance"
  type        = number
  default     = 1
}

variable "boot_disk_size_gb" {
  description = "Boot disk size in gigabytes"
  type        = number
  default     = 100
}

variable "boot_disk_type" {
  description = "Boot disk storage type"
  type        = string
  default     = "pd-balanced"
}

variable "image_project" {
  description = "GCP image project for prebaked Deep Learning images"
  type        = string
  default     = "deeplearning-platform-release"
}

variable "image_family" {
  description = "Deep Learning VM image family with prebaked PyTorch and CUDA"
  type        = string
  default     = "pytorch-2-9-cu129-ubuntu-2204-nvidia-580"
}

variable "enable_checkpoint_bucket" {
  description = "Whether to provision a GCS bucket for spot preemption checkpointing"
  type        = bool
  default     = true
}
