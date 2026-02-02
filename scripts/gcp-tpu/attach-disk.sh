# spin up TPU VM
PROJECT_ID=camp-blue-431854084
TPU_NAME=node-1
ZONE=asia-northeast1-b
DISK=projects/$PROJECT_ID/zones/$ZONE/disks/mayank-fs

gcloud alpha compute tpus tpu-vm attach-disk $TPU_NAME \
    --project=$PROJECT_ID \
    --zone=$ZONE \
    --disk=$DISK \
    --mode=read-only
