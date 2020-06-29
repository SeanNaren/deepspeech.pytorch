# Training deepspeech.pytorch on Kubernetes using TorchElastic

Below are instructions to train a model using a GKE cluster and take advantage of pre-emptible VMs using [TorchElastic](https://pytorch.org/elastic/master/index.html). 

```
gcloud container clusters create torchelastic \
    --machine-type=n1-standard-2 \
    --disk-size=15Gi \
    --zone=us-west1-b \
    --cluster-version=1.15 \
    --num-nodes=3 --min-nodes=0 --max-nodes=3 \
    --enable-autoscaling \
    --scopes=storage-full

# Add GPU pool
gcloud container node-pools create gpu-pool --cluster torchelastic \
    --accelerator type=nvidia-tesla-v100,count=1\
    --machine-type=n1-standard-4 \
    --disk-size=25Gi \
    --zone=us-west1-b \
    --preemptible \
    --num-nodes=1 --min-nodes=0 --max-nodes=1 \
    --enable-autoscaling \
    --scopes=storage-full
```

We use pre-emptive nodes to reduce costs. The code handles interruptions by saving state to GCS periodically.

## Set up ElasticJob

```
git clone https://github.com/pytorch/elastic.git
cd elastic/kubernetes

kubectl apply -k config/default
```

### Setup Volume

First we create a drive to store our data. The drive is fairly small and can be managed under the volumes tab on GCP. Modify the config as needs be.

```
cd deepspeech.pytorch/kubernetes/
gcloud compute disks create --size 10Gi audio-data --zone us-west1-b
kubectl apply -f data/storage.yaml
kubectl apply -f data/persistent_volume.yaml
```

### Download Data

We run a job to download and extract the data onto our drive. Modify the config to match whatever data you'd like to download.

In our example we download and extract the AN4 data using the deepspeech.pytorch docker image.

```
kubectl apply -f data/transfer_data.yaml
kubectl logs transfer-data --namespace=elastic-job # Monitor logs to determine when the process is complete
kubectl delete -f data/transfer_data.yaml # Delete to free up resources
```

### Install CRD/CUDA for Training

```
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded.yaml
kubectl apply -f https://raw.githubusercontent.com/pytorch/elastic/master/kubernetes/config/samples/etcd.yaml
kubectl get svc -n elastic-job
```

### Training

#### GCS Model Store

To store the checkpoint models, we use [Google Cloud Storage](https://cloud.google.com/storage). Create a bucket and make sure to modify `checkpointing.gcs_bucket=deepspeech-1234` to `train.yaml` to point to the bucket that the cluster has access to.

```
kubectl apply -f train.yaml
```
