import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from constants import (
    base_dir_detections_cd,
    base_dir_detections_fd,
    base_dir_groundtruth,
    base_dir_metric_cd,
    base_dir_metric_fd,
    img_size_cd,
    img_size_fd,
    num_windows,
)
from dataset.dataloader import CustomDatasetFromImages
from utils import utils_detector


def save_args(file_path: str, args: Any) -> None:
    """
    Save training arguments to checkpoint directory.
    
    Args:
        file_path: Path to the current script file
        args: Argument namespace from argparse
    """
    # Copy the script file
    script_name = os.path.basename(file_path)
    shutil.copy(file_path, os.path.join(args.cv_dir, script_name))
    
    # Save arguments as JSON
    args_dict = vars(args)
    args_dict['device'] = str(args.device)  # Convert device to string for JSON serialization
    args_path = os.path.join(args.cv_dir, 'args.json')
    with open(args_path, 'w') as f:
        json.dump(args_dict, f, indent=4)
    
    # Also save as text for easy reading
    with open(os.path.join(args.cv_dir, 'args.txt'), 'w') as f:
        f.write(str(args))
    
    print(f"Saved arguments to {args.cv_dir}")

def read_json(filename):
    with open(filename) as dt:
        data = json.load(dt)
    return data

def xywh2xyxy(x):
    y = np.zeros(x.shape)
    y[:,0] = x[:, 0] - x[:, 2] / 2.
    y[:,1] = x[:, 1] - x[:, 3] / 2.
    y[:,2] = x[:, 0] + x[:, 2] / 2.
    y[:,3] = x[:, 1] + x[:, 3] / 2.
    return y

def get_detected_boxes(policy, file_dirs, metrics, set_labels):
    for index, file_dir_st in enumerate(file_dirs):
        counter = 0
        for xind in range(num_windows):
            for yind in range(num_windows):
                # ---------------- Read Ground Truth ----------------------------------
                outputs_all = []
                # Read Ground Truth
                gt_path = Path(base_dir_groundtruth) / f'{file_dir_st}_{xind}_{yind}.txt'
                
                if not gt_path.exists():
                    counter += 1
                    continue
                
                gt = np.loadtxt(gt_path).reshape([-1, 5])
                targets = np.hstack((np.zeros((gt.shape[0], 1)), gt))
                targets[:, 2:] = xywh2xyxy(targets[:, 2:])
                # ----------------- Read Detections -------------------------------
                if policy[index, counter] == 1:
                    # Use fine detector (HR)
                    preds_dir = Path(base_dir_detections_fd) / f'{file_dir_st}_{xind}_{yind}.npy'
                    targets[:, 2:] *= img_size_fd
                else:
                    # Use coarse detector (LR)
                    preds_dir = Path(base_dir_detections_cd) / f'{file_dir_st}_{xind}_{yind}.npy'
                    targets[:, 2:] *= img_size_cd
                    
                # Load predictions if they exist
                if preds_dir.exists():
                    preds = np.load(preds_dir).reshape([-1, 7])
                    outputs_all.append(torch.from_numpy(preds))
                    
                set_labels += targets[:, 1].tolist()
                metrics += utils_detector.get_batch_statistics(outputs_all, torch.from_numpy(targets), 0.5)
                
                counter += 1
                
    return metrics, set_labels

def read_offsets(image_ids, num_actions, device: torch.device = torch.device('cuda')):
    offset_fd = torch.zeros((len(image_ids), num_actions), device=device)
    offset_cd = torch.zeros((len(image_ids), num_actions), device=device)
    
    for index, img_id in enumerate(image_ids):
        img_id += '.npy'
        fd_path = Path(base_dir_metric_fd) / img_id
        cd_path = Path(base_dir_metric_cd) / img_id
        
        if fd_path.exists():
            offset_fd[index, :] = torch.from_numpy(np.load(fd_path).flatten())
        else:
            print(f"Warning: Fine detector metrics not found for {img_id}")
        
        if cd_path.exists():
            offset_cd[index, :] = torch.from_numpy(np.load(cd_path).flatten())
        else:
            print(f"Warning: Coarse detector metrics not found for {img_id}")

    return offset_fd, offset_cd

def performance_stats(policies, rewards):
    # Print the performace metrics including the average reward, average number
    # and variance of sampled num_patches, and number of unique policies
    policies = torch.cat(policies, 0)
    rewards = torch.cat(rewards, 0)

    reward = rewards.mean()
    num_unique_policy = policies.sum(1).mean()
    variance = policies.sum(1).std()

    policy_set = [p.cpu().numpy().astype(int).astype(str) for p in policies]
    policy_set = set([''.join(p) for p in policy_set])

    return reward, num_unique_policy, variance, policy_set

def compute_reward(offset_fd, offset_cd, policy, beta, sigma):
    """
    Args:
        offset_fd: np.array, shape [batch_size, num_actions]
        offset_cd: np.array, shape [batch_size, num_actions]
        policy: np.array, shape [batch_size, num_actions], binary-valued (0 or 1)
        beta: scalar
        sigma: scalar
    """
    # Reward function favors policies that drops patches only if the classifier
    # successfully categorizes the image
    offset_cd += beta
    reward_patch_diff = (offset_fd - offset_cd)*policy + -1*((offset_fd - offset_cd)*(1-policy))
    reward_patch_acqcost = (policy.size(1) - policy.sum(dim=1)) / policy.size(1)
    reward_img = reward_patch_diff.sum(dim=1) + sigma * reward_patch_acqcost
    reward = reward_img.unsqueeze(1)
    return reward.float()

def get_transforms(img_size):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform_train = transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    transform_test = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    return transform_train, transform_test

def get_dataset(img_size, root='data/'):
    transform_train, transform_test = get_transforms(img_size)
    trainset = CustomDatasetFromImages(root+'train.csv', transform_train)
    testset = CustomDatasetFromImages(root+'valid.csv', transform_test)

    return trainset, testset

def set_parameter_requires_grad(model, feature_extracting):
    # When loading the models, make sure to call this function to update the weights
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def get_model(num_output, pretrained=True):
    if pretrained:
        weights = models.ResNet34_Weights.IMAGENET1K_V1
        agent = models.resnet34(weights=weights)
    else:
        agent = models.resnet34(weights=None)

    set_parameter_requires_grad(agent, feature_extracting=False)

    # Replace final FC layer
    num_ftrs = agent.fc.in_features
    agent.fc = torch.nn.Linear(num_ftrs, num_output)

    for param in agent.fc.parameters():
        param.requires_grad = True

    return agent
