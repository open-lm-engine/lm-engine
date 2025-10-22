# spin up TPU VM
PROJECT_ID=camp-blue-431854084
TPU_NAME=node-1
ZONE=asia-northeast1-b
ACCELERATOR_TYPE=v6e-1
RUNTIME_VERSION=v2-alpha-tpuv6e
DISK=projects/$PROJECT_ID/zones/$ZONE/disks/mayank-fs

gcloud compute tpus tpu-vm ssh $TPU_NAME --project $PROJECT_ID --zone $ZONE
