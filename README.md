# automl_joint_optimizer

This repo contains code for a joint optimizer that uses K-Nearest Neigbhor method to find the best next suggestion points for evaluation.

Clone: git clone https://github.com/lhvu2/automl_joint_optimizer.git

# Setup:

1. conda create --name knnopt python==3.6.10

This will create a new conda environment named 'knnopt'

2. conda activate knnopt

3. git clone https://github.com/rdturnermtl/bbo_challenge_starter_kit.git 

This will clone the starter kit with bayesmark framework to evaluate the joint optimizer.
Alternatively, unzip file: bbo_challenge_starter_kit-master.zip from this repo. Note that this starter kit zip file is owned by Ryan Turner. 

4. cd bbo_challenge_starter_kit
5. pip install -r environment.txt
6. cp -r baseopt-knn example_submissions/

This will copy the knn optimizer into the folder 'example_submissions'.

8. Run the knn optimizer: ./run_local.sh ./example_submissions/baseopt-knn 3

