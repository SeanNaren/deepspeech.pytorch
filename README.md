# DeepSpeech KubeFlow 

This branch contains the code to run DeepSpeech on KubeFlow and Kubernetes as shown in our blog post [here](). We assume you have access to a GCP project with
the appropriate GPU resources to use this. We use LibriSpeech 100 to measure our benchmarks. Modifications have been made internally, however the overall speedups should be available in this repo.

## Installation

Follow the instructions [here](https://www.kubeflow.org/docs/started/getting-started-gke/) 
**till you set up the GCP auth data** then follow the below commands to setup kubeflow.

We assume you're using an OSX based machine, however most of these commands can be ran on Linux as well.

### Install additional dependencies
```bash
brew install ksonnet/tap/ks
brew install jq
pip install ys
```

### Set up config for gcloud

Make sure to set these zones 
```bash
export ZONE=us-central1-a # Change this to the zone you prefer, ensure that there is GPU availability in that zone on GCP
gcloud config set compute/zone ${ZONE}
```

### Set up kubeflow
```bash
# Set up env variables for cmds
export CLIENT_ID=<CLIENT_ID from OAuth page>
export CLIENT_SECRET=<CLIENT_SECRET from OAuth page>
export KUBEFLOW_SRC=/Users/sean.narenthiran/kubeflow/ # Ensure absolute path
export KUBEFLOW_TAG=master
export PROJECT=deepspeech # Ensure these are set to the values you've used when setting up kubeflow
export KFAPP=deepspeech # Ensure these are set to the values you've used when setting up kubeflow
export DEPLOYMENT_NAME=deepspeech

mkdir ${KUBEFLOW_SRC}
cd ${KUBEFLOW_SRC}
curl https://raw.githubusercontent.com/kubeflow/kubeflow/${KUBEFLOW_TAG}/scripts/download.sh | bash
${KUBEFLOW_SRC}/scripts/kfctl.sh init ${KFAPP} --platform gcp --project ${PROJECT}
cd ${KFAPP}
${KUBEFLOW_SRC}/scripts/kfctl.sh generate platform
${KUBEFLOW_SRC}/scripts/kfctl.sh apply platform
${KUBEFLOW_SRC}/scripts/kfctl.sh generate k8s
${KUBEFLOW_SRC}/scripts/kfctl.sh apply k8s

kubectl -n kubeflow get  all # Ensure you see nodes, will take a few minutes to spin everything up
```

### Adding GPUs to be provisioned

To auto-scale GPUs we need to update the created nodes.

Using the navigation menu select kubernetes engine. Then select clusters and click on the deepspeech cluster.

Once you're in the cluster, press edit at the top to configure the node. Change the auto-provision settings to look like the below to use GPUs, and click save at the bottom.

![Cluster Preview](./img/cluster_image.png)

### Creating an NFS server

To add our datasets to a shared drive for the k8 cluster, we need to create a drive and a node to manage file transfer to it. We will be creating a [filestore](https://www.kubeflow.org/docs/gke/cloud-filestore/) to serve. Note that this is also [priced](https://cloud.google.com/filestore/pricing).

```bash
cd ${KUBEFLOW_SRC}
cp ${KUBEFLOW_SRC}/deployment/gke/deployment_manager_configs/gcfs.yaml ${KFAPP}/gcp_config/
nano ${KFAPP}/gcp_config/gcfs.yaml
```

You will need to change the zone and project. Look below for the fixed template:

```
# Modify this instance to create a GCFS file store.
# 1. Change the zone to the desired zone
# 2. Change the instanceId to the desired id
# 3. Change network if needed
# 4. Change the capacity if desired.
resources:
- name: filestore
  type: gcp-types/file-v1beta1:projects.locations.instances
  properties:
    parent: projects/deepspeech/locations/us-central1-a # Ensure to set this to the right zone
    # Any name of the instance would do
    instanceId: deepspeech
    tier: STANDARD
    description: Filestore for Kubeflow
    networks:
    - network: default
    fileShares:
    - name: kubeflow
      capacityGb: 1024
```

```bash
cd ${KFAPP}
. env.sh
cd ${KFAPP}
yq -r ".resources[0].properties.instanceId=\"${DEPLOYMENT_NAME}\"" ${KFAPP}/gcp_config/gcfs.yaml > ${KFAPP}/gcp_config/gcfs.yaml.new
mv ${KFAPP}/gcp_config/gcfs.yaml.new ${KFAPP}/gcp_config/gcfs.yaml

# apply changes
cd ${KFAPP}
${KUBEFLOW_SRC}/scripts/kfctl.sh apply platform

```

This will make the drive, however we now need to mount it to kubeflow. We will need to define the storage capacity and IP address that our node is set on.
To figure out the IP run the below command:

```bash
 gcloud --project=${PROJECT} beta filestore instances list
 export GCFS_INSTANCE_IP_ADDRESS=<IP ADDRESS FROM ABOVE COMMAND>
 export GCFS_STORAGE=1024 # Set to use the entire drive
```

There should be one drive, and the IP can be found there.


```bash
cd ${KUBEFLOW_SRC}/${KFAPP}/ks_app
ks generate google-cloud-filestore-pv google-cloud-filestore-pv --name="kubeflow-gcfs" --storageCapacity="${GCFS_STORAGE}" --serverIP="${GCFS_INSTANCE_IP_ADDRESS}"
ks apply default -c google-cloud-filestore-pv
# apply changes
cd ${KUBEFLOW_SRC}/${KFAPP}
${KUBEFLOW_SRC}/scripts/kfctl.sh apply platform
```

We now need to create a node to manage file transfer to and from the drive.

In the GCP UI, go to compute engine and select VM instances. Select Create Instance. Change the name to nfs-server, change instance location to the same as defined above (us-central1-a for us) and change the Boot Instance type to Ubuntu 16 LTS, and click Create.

It will take a few minutes to create the instance, but once it is up you should be able to SSH to it via the  (click the button next to the instance name). Once you are on the node follow the below commands:

```bash
# From the GCP SSH terminal to the NFS server
sudo apt-get -y update
sudo apt-get install nfs-common
sudo mkdir /gcfs
sudo mount GCFS_INSTANCE_IP_ADDRESS/kubeflow /gcfs # swap GCFS_INSTANCE_IP_ADDRESS for the address of the filestore we found above
df -h
```

You should now see the drive mounted.

### Downloading Data

In this example we'll use LibriSpeech generated via the script found in the data/ folder. Internally we have our datasets stored in the cloud for easier access.

```bash
cd deepspeech.pytorch/ # clone the repo in this example to download librispeech
pip install -r requirements.txt
cd data/ && python librispeech.py --target-dir /gcfs/librispeech/ # Run this on the nfs-server to download the appropriate data to our NFS drive
```

### Running the pipeline

Once everything is set-up, the pipeline can be run and training can occur. To do so, run the below command:

```bash
cd deepspeech.pytorch/
kubectl create -f deepspeech.yaml
```

We can see the status of pipelines using the UI. Check under workloads to see your jobs. They will remain unschedulable until the GPUs have been provisioned and ready to run, so this may take a few minutes to do.
To check the progress of the GPUS, under clusters select deepspeech, and then select nodes. You should see two new GPU nodes, select them and check the running pods to see if a nvidia-gpu-driver pod is starting up.

After a while the pipeline will run, and results can be seen via the logs by selecting the workload job.
 
You can also see the status of the node via the terminal:

```
kubectl get pods
```

### Nuking Everything

To get rid of everything we suggest follow the below steps:

First go to the Deployment Manager page in the GCP UI, and delete the deployment. This will remove any running nodes etc.
Go to the Compute page, and make sure all nodes have been terminated. Finally go to the Disks page, and delete any stray disks.
Finally run the below command:

```
cd ${KFAPP}
${KUBEFLOW_SRC}/scripts/kfctl.sh delete all
```