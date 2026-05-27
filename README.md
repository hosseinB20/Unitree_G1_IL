# 1. Environment Setup and Installation 
To conduct this project [unitree_sim_isaaclab](https://github.com/unitreerobotics/unitree_sim_isaaclab/tree/main) was used to create unitree robot environment in Isaac Lab and [Robomimic](https://github.com/ARISE-Initiative/robomimic) was used to to train the robot by imitation learning algorithm (BC). You can refer to each repository for installation and more information. 
Only the modified and customized codes are provided in this repository. For duplicating the result you should place the codes in related folders of unitree_sim_isaaclab. 
# 2. Dataset
The dataset provided in [Hugging Face](https://huggingface.co/datasets/unitreerobotics/G1_Dex1_PickPlaceCylinder_Dataset_Sim) was downloaded and used for training the unitree G1 robot for the task Isaac-PickPlace-Cylinder-G129-Dex1-Joint. 
# 3. Unitree_G1 BC Algorithm
1- Run below commands to convert the dataset to the format which is acceptable for Robomimic (hdf5) and split it for train and validation. 
```
python ~/patch_g1_robomimic_hdf5.py
```
```
cd ~/robomimic
python robomimic/scripts/split_train_val.py \
  --dataset ~/g1_pickplace_robomimic.hdf5 \
  --ratio 0.1
```
2- Run this command to train the model via Robomimic.
```
cd ~/robomimic
python robomimic/scripts/train.py \
  --config ~/g1_bc.json \
  --dataset ~/g1_pickplace_robomimic.hdf5
```
