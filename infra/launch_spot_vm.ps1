# PowerShell script to launch a T4 GPU Spot VM on GCP
param (
    [string]$InstanceName = "qlora-icl-spot",
    [string]$Zone = "us-central1-a",
    [string]$MachineType = "n1-standard-4",
    [int]$BootDiskSizeGb = 100
)

Write-Host "=== 1. Checking GCP Credentials ===" -ForegroundColor Cyan
$activeAccount = gcloud auth list --filter=status:ACTIVE --format="value(account)"
if (-not $activeAccount) {
    Write-Host "No active GCP account found! Please run: gcloud auth login" -ForegroundColor Red
    exit 1
}
Write-Host "Active Account: $activeAccount" -ForegroundColor Green

$project = gcloud config get-value project
if (-not $project -or $project -eq "(unset)") {
    Write-Host "No active project set! Please run: gcloud config set project <YOUR_PROJECT_ID>" -ForegroundColor Red
    exit 1
}
Write-Host "Active Project: $project" -ForegroundColor Green

Write-Host "`n=== 2. Launching Spot VM with NVIDIA T4 GPU ($InstanceName) ===" -ForegroundColor Cyan
Write-Host "Zone: $Zone | Type: $MachineType | GPU: 1x NVIDIA T4 (Spot Instance)"

gcloud compute instances create $InstanceName `
    --project=$project `
    --zone=$Zone `
    --machine-type=$MachineType `
    --accelerator="type=nvidia-tesla-t4,count=1" `
    --provisioning-model=SPOT `
    --instance-termination-action=STOP `
    --image-family="pytorch-2-9-cu129-ubuntu-2204-nvidia-580" `
    --image-project="deeplearning-platform-release" `
    --maintenance-policy=TERMINATE `
    --boot-disk-size="${BootDiskSizeGb}GB" `
    --boot-disk-type="pd-balanced" `
    --scopes="cloud-platform"

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n=== 3. Instance Created Successfully! ===" -ForegroundColor Green
    $externalIp = gcloud compute instances describe $InstanceName --zone=$Zone --format="value(networkInterfaces[0].accessConfigs[0].natIP)"
    Write-Host "External IP: $externalIp" -ForegroundColor Yellow
    Write-Host "To SSH into the instance, run:" -ForegroundColor Cyan
    Write-Host "  gcloud compute ssh $InstanceName --zone=$Zone" -ForegroundColor White
} else {
    Write-Host "`nFailed to launch instance. If this is a GPU quota error, check Quotas > GPUs (all regions) or try another zone like us-east1-c or us-central1-f." -ForegroundColor Red
}
