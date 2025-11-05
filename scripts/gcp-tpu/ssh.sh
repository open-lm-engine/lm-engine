# spin up TPU VM
PROJECT_ID=camp-blue-431854084
TPU_NAME=node-1
ZONE=asia-northeast1-b

gcloud compute tpus tpu-vm ssh $TPU_NAME --project $PROJECT_ID --zone $ZONE --worker 1
