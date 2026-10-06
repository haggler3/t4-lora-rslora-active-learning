output "instance_name" {
  description = "Name of the provisioned Spot VM"
  value       = google_compute_instance.spot_vm.name
}

output "instance_zone" {
  description = "Zone where the Spot VM is running"
  value       = google_compute_instance.spot_vm.zone
}

output "instance_external_ip" {
  description = "Public IP address of the Spot VM"
  value       = google_compute_instance.spot_vm.network_interface[0].access_config[0].nat_ip
}

output "ssh_command" {
  description = "Google Cloud SDK command to SSH directly into the Spot VM"
  value       = "gcloud compute ssh ${google_compute_instance.spot_vm.name} --zone=${google_compute_instance.spot_vm.zone}"
}

output "gcs_checkpoint_bucket" {
  description = "GCS bucket URI configured for preemption checkpoint synchronization"
  value       = var.enable_checkpoint_bucket ? "gs://${google_storage_bucket.checkpoint_bucket[0].name}" : "None"
}
