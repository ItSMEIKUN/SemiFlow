## SemiFlow

Research Artifact for our **ACSAC 2025** paper: "SemiFlow: Website Fingerprinting Using Semi-Supervised Learning"

We provide the source code for the tool. To facilitate the use of the tool and review by reviewers, we provide a [`Docker image`](https://zenodo.org/records/15618129/files/SemiFlow.tar) that contains the components needed to execute the tool.

## Prerequisites

### Hardware dependencies

We used a server with an Intel Xeon Silver 4214R CPU (12 cores, 2.40GHz), 90GB of RAM, a 30GB system disk, and an NVIDIA RTX 3080 Ti GPU (12GB of video memory) as the experimental platform on which we ran all experiments. We placed the datasets in our provided mirrors so that the experiments could be run directly for evaluation.

### Software dependencies
```
- python 3.8.10
	-torch 2.0.0+cu118
	-torchvision 0.15.1+cu118
	-networkx 3.1
	-numpy 1.24.4
	-scikit-learn 1.3.2
```
The listed software dependencies form the basis of the technical realization of this study and can be appropriately adjusted according to the specific hardware configuration. Given the high demand for computational resources during the evaluation process, it is recommended to prioritize the use of GPU-based acceleration solutions to improve the efficiency of the experiments.
To minimize the workload of the reviewers, we packaged all the required environment and software dependencies into a [`Docker image`](https://zenodo.org/records/15618129/files/SemiFlow.tar).

### Dataset
All the datasets have packed into the docker image. We have also provided a [download link](https://zenodo.org/records/15621086) for the datasets via OneDrive.
We use publicly available datasets for model experimentation and evaluation. We use the following datasets:

1. AWF dataset [1]: this dataset consists of monitored websites from 1200 Alexa top sites and unmonitored websites from 400,000 Alexa top sites. This dataset was collected in 2016.
2. DF Dataset [2]. This dataset consists of monitored and unmonitored websites crawled from Alexa top sites. Like the AWF dataset, this dataset was collected in 2016.

When we use datasets from these papers, they should be properly cited in our artifact work.

```
[1] Vera Rimmer, Davy Preuveneers, Marc Juarez, Tom Van Goethem, and WouterJoosen. 2018.Automated Website Fingerprinting through Deep Learning. In Proceedings of the 25nd Network and Distributed System Security Symposium (NDSS 2018). Internet Society.
[2] Payap Sirinam, Mohsen Imani, Marc Juarez, and Matthew Wright. 2018.DeepFingerprinting: Undermining Website Fingerprinting Defenses with Deep Learning.The 25th ACM SIGSAC Conference on Computer and Communications Security(CCS'18)(2018).
```
1. For the AWF datasets, we use the CW (Closed World) and OW (Open World) datasets. You can download and use these processed datasets:

 * [CW](https://zenodo.org/records/15621086/files/AWF_CW.npz): Contains 100 monitored websites, with 400 traffic samples per website.
 * [OW](https://zenodo.org/records/15621086/files/AWF_OW.npz): Contains 200,000 unmonitored websites, with 1 traffic sample per website.

2. For the DF dataset we used the CW (Closed World) dataset:

 * [CW](https://zenodo.org/records/15621086/files/DF_CW.npz): Contains 95 monitored websites, with 400 traffic samples per website.
 * [OW](https://zenodo.org/records/15621086/files/DF_OW.npz): Contains 200,000 unmonitored websites, with 1 traffic sample per website.

You also download and use the original dataset via the link：[AWF](https://github.com/DistriNet/DLWF) and [DF](https://github.com/msrocean/Tik_Tok).

## Artifact Evaluation

### Installation: Import Docker Image

Download the packed [`Docker image`](https://zenodo.org/records/15618129/files/SemiFlow.tar), then run the commands below to build a docker container.

1. Import the packed docker image.

   ```
   docker load -i semiflow.tar
   ```

2. Build a docker container and start a bash shell.

   ```
   docker run --gpus all --shm-size=50g -it semiflow /bin/bash
   ```
   You can specify the size of the container's shared memory shm-size as appropriate.

### (1) Closed World Experiment

In this experiment, it is assumed that the victim only visits monitored websites.
The default command is:

```
python train.py
```
You can specify the following parameters:
- `--dataset` : choices = ['AWF', 'DF'] Default: AWF
- `--device`  : choices = ['cuda', 'cuda:0', 'cpu'] Default: cuda
It's best to use a GPU to speed up calculations.
- `--num_workers`  : choices = 0~Number of CPU cores. Default: 5
Preferably a number of 5 or more.
- `--train_epochs`  : choices = 300 rounds or more. Default: 400
- `--batch_size`  : choices = ['50', '100', '200']. Default: 50
- `--n_label`  : choices = ['5', '10', '20', '50']. Default: 5

Thus to carry out the AWF dataset at n=10 and using 8 number of cores the command is:
```
python train.py --num_workers 8 --n_label 10
```

### (2) Open World Experiment
In this experiment, it is a binary categorization problem considering that victims visit unmonitored websites. When this experiment is required, set the following parameters for the experiment:
- `--setting`  : choices = OW
- `-n_ow`  : choices = ['1', '2', '4', '5', '10']. Default: 1
The size of the open world in the test set represents the number associated with the number of closed worlds in the test set, e.g. 2 in the AWF dataset represents 200 (number of tags per site in the closed world) * 100 (number of site categories) * 2 = 40k number of open sites of size.

Thus, the size of the open world in the AWF dataset is 20k sized commands:
```
python train.py --setting OW --n_ow 1    or   python train.py --setting OW --n_ow 1
```

Of course, you can also modify parameters such as the number of training rounds, the number of working threads, and so on.

## Code Structure

```
├── datasets/						  // Location of data set storage
├── models/							// Model as well as the loss function.
├── tools/							 // Other methods used
├── train.py						   // Core code for training the model
```
