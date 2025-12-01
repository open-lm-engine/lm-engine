# Job management on a Kubernetes cluster

```shell
# create job
kubectl apply -f <your-job-yaml>
# delete job
kubectl delete job <your-job-name>
```

# Manage a ray cluster

kubectl apply -f ray-cluster.yml
kubectl delete RayCluster <your-cluster-name>
