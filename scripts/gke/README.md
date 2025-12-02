<!-- **************************************************
Copyright (c) 2025, Mayank Mishra
************************************************** -->

# Job management on a Kubernetes cluster

```shell
# create job
kubectl apply -f <your-job-yaml>
# delete job
kubectl delete job <your-job-name>
```

# Manage a ray cluster

```shell
# launch ray cluster
kubectl apply -f ray-cluster.yml
# delete ray cluster
kubectl delete RayCluster <your-cluster-name>
```
