# spin up TPU VM
PROJECT_ID=camp-blue-431854084
TPU_NAME=mayank-1
ZONE=asia-northeast1-b
ACCELERATOR_TYPE=v6e-1
RUNTIME_VERSION=tpu-ubuntu2204-base
DISK_NAME=mayank

# gcloud compute tpus tpu-vm create $TPU_NAME \
#     --project=$PROJECT_ID \
#     --zone=$ZONE \
#     --accelerator-type=$ACCELERATOR_TYPE \
#     --version=$RUNTIME_VERSION \
#     --data-disk source=projects/$PROJECT_ID/zones/$ZONE/disks/$DISK_NAME,mode=read-write

gcloud compute tpus tpu-vm ssh $TPU_NAME --project $PROJECT_ID --zone $ZONE

# gcloud compute tpus tpu-vm ssh $TPU_NAME --project $PROJECT_ID --zone $ZONE --worker=all --command="sudo mkdir -p /mnt/disks/$MOUNT_DIR"
