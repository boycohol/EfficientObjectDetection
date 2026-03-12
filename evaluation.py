"""
Unified Evaluation for Policy Gradient and DQN Models (Single GPU)

Evaluates:
1. Trained policy (Policy Gradient or DQN)
2. Always-HR baseline (100% high-resolution)
3. Always-LR baseline (0% high-resolution)

Usage:
    # Policy Gradient model
    python evaluation.py --load checkpoints/pg_best.pth --method pg
    
    # DQN model
    python evaluation.py --load checkpoints/dqn_best.pth --method dqn
    
    # Auto-detect method from checkpoint
    python evaluation.py --load checkpoints/best.pth
"""

import os
import torch
import torch.utils.data as data
import torch.nn.functional as F
import numpy as np
import tqdm
import argparse
from pathlib import Path
import json
import time
from collections import OrderedDict

from utils import utils, utils_detector
from constants import num_actions


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Unified Evaluation for Policy Gradient and DQN'
    )
    
    # Model parameters
    parser.add_argument('--load', required=True, 
                       help='checkpoint to load')
    parser.add_argument('--method', type=str, default='auto',
                       choices=['pg', 'dqn'],
                       help='method type: pg (policy gradient) or dqn')
    parser.add_argument('--cv_dir', default='cv/evaluation/', 
                       help='results directory')
    
    # Data parameters
    parser.add_argument('--data_dir', default='data/', 
                       help='data directory')
    parser.add_argument('--img_size', type=int, default=None,
                       help='image size (auto-detect from checkpoint if None)')
    parser.add_argument('--batch_size', type=int, default=256, 
                       help='batch size')
    parser.add_argument('--num_workers', type=int, default=8, 
                       help='dataloader workers')
    
    # RL parameters (override checkpoint values if specified)
    parser.add_argument('--beta', type=float, default=None,
                       help='coarse detector bias (use checkpoint value if None)')
    parser.add_argument('--sigma', type=float, default=None,
                       help='HR cost weight (use checkpoint value if None)')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda', 
                       help='device (cuda or cpu)')
    
    # Evaluation options
    parser.add_argument('--save_predictions', action='store_true',
                       help='save per-image predictions')
    parser.add_argument('--verbose', action='store_true',
                       help='print detailed per-batch statistics')
    
    args = parser.parse_args()
    return args


def get_policy_pg(agent, inputs, device, deterministic=True):
    """
    Get policy from Policy Gradient model.
    
    Args:
        agent: Policy network
        inputs: Input images
        device: Device
        deterministic: If True, use greedy policy (prob > 0.5)
        
    Returns:
        policy: Binary action vector [batch_size, num_actions]
        probs: Action probabilities [batch_size, num_actions]
    """
    probs = torch.sigmoid(agent(inputs))
    
    if deterministic:
        policy = (probs > 0.5).float()
    else:
        # Sample from Bernoulli distribution
        from torch.distributions import Bernoulli
        distr = Bernoulli(probs)
        policy = distr.sample()
    
    return policy, probs


def get_policy_dqn(agent, inputs, device):
    """
    Get policy from DQN model.
    
    Args:
        agent: DQN network
        inputs: Input images
        device: Device
        
    Returns:
        policy: Binary action vector [batch_size, num_actions]
        q_values: Q-values [batch_size, num_actions]
    """
    q_values = agent(inputs)
    
    # Greedy policy: use HR where Q-value > 0
    policy = (q_values > 0).float()
    
    return policy, q_values


def evaluate_policy(agent, testloader, args, method, policy_mode='learned'):
    """
    Evaluate a policy on the test set.
    
    Args:
        agent: Policy network
        testloader: Test data loader
        args: Arguments
        method: 'pg' or 'dqn'
        policy_mode: 'learned', 'always_hr', or 'always_lr'
        
    Returns:
        Dictionary containing evaluation results
    """
    agent.eval()
    
    # Storage
    rewards_list = []
    policies_list = []
    probs_or_qvals_list = []
    local_metrics = []
    local_set_labels = []
    
    # Per-image predictions (if requested)
    predictions = [] if args.save_predictions else None
    
    # Description for progress bar
    if policy_mode == 'learned':
        desc = f'Evaluating Trained Policy ({method.upper()})'
    elif policy_mode == 'always_hr':
        desc = 'Evaluating Always-HR Baseline'
    elif policy_mode == 'always_lr':
        desc = 'Evaluating Always-LR Baseline'
    else:
        desc = 'Evaluating'
    
    with torch.no_grad():
        pbar = tqdm.tqdm(testloader, desc=desc, total=len(testloader))
        
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs = inputs.to(args.device, non_blocking=True)
            
            # Get policy based on mode
            if policy_mode == 'always_hr':
                # Baseline: always use high-resolution
                policy = torch.ones(inputs.size(0), num_actions, 
                                   device=args.device)
                probs_or_qvals = policy.clone()
                
            elif policy_mode == 'always_lr':
                # Baseline: always use low-resolution
                policy = torch.zeros(inputs.size(0), num_actions,
                                    device=args.device)
                probs_or_qvals = policy.clone()
                
            else:  # learned policy
                if method == 'pg':
                    policy, probs_or_qvals = get_policy_pg(agent, inputs, args.device)
                elif method == 'dqn':
                    policy, probs_or_qvals = get_policy_dqn(agent, inputs, args.device)
                else:
                    raise ValueError(f"Unknown method: {method}")
            
            # Get pre-computed detector metrics
            offset_fd, offset_cd = utils.read_offsets(targets, num_actions, args.device)
            
            # Compute reward
            reward = utils.compute_reward(offset_fd, offset_cd, policy,
                                         args.beta, args.sigma)
            
            # Evaluate detections
            batch_metrics, batch_labels = utils.get_detected_boxes(
                policy.cpu(), targets, [], []
            )
            local_metrics.extend(batch_metrics)
            local_set_labels.extend(batch_labels)
            
            # Store results
            rewards_list.append(reward.cpu())
            policies_list.append(policy.cpu())
            probs_or_qvals_list.append(probs_or_qvals.cpu())
            
            # Per-image predictions
            if args.save_predictions and policy_mode == 'learned':
                for i, target in enumerate(targets):
                    predictions.append({
                        'image_id': target,
                        'policy': policy[i].cpu().numpy().tolist(),
                        'probs_or_qvals': probs_or_qvals[i].cpu().numpy().tolist(),
                        'reward': reward[i].item(),
                        'sparsity': policy[i].sum().item()
                    })
            
            # Update progress bar
            if args.verbose or batch_idx % 10 == 0:
                pbar.set_postfix({
                    'reward': f'{reward.mean().item():.4f}',
                    'sparsity': f'{policy.sum(1).mean().item():.2f}/{num_actions}',
                    'hr_usage': f'{policy.mean().item()*100:.1f}%'
                })
    
    # Compute detection metrics
    results = {}
    
    if len(local_metrics) > 0:
        true_positives, pred_scores, pred_labels = [
            np.concatenate(x, 0) for x in list(zip(*local_metrics))
        ]
        
        precision, recall, AP, f1, ap_class = utils_detector.ap_per_class(
            true_positives, pred_scores, pred_labels, local_set_labels
        )
        
        results['detection'] = {
            'AP': float(AP[0]) if len(AP) > 0 else 0.0,
            'AR': float(recall.mean()) if len(recall) > 0 else 0.0,
            'precision': float(precision.mean()) if len(precision) > 0 else 0.0,
            'recall': float(recall.mean()) if len(recall) > 0 else 0.0,
            'f1': float(f1.mean()) if len(f1) > 0 else 0.0,
            'num_detections': int(len(true_positives)),
            'num_gt_objects': int(len(local_set_labels))
        }
    else:
        results['detection'] = {
            'AP': 0.0,
            'AR': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'num_detections': 0,
            'num_gt_objects': 0
        }
    
    # Compute policy statistics
    rewards = [torch.cat(rewards_list, 0)]
    policies = [torch.cat(policies_list, 0)]
    probs_or_qvals = torch.cat(probs_or_qvals_list, 0)
    
    reward, sparsity, variance, policy_set = utils.performance_stats(policies, rewards)
    
    results['policy'] = {
        'mean_reward': float(reward),
        'mean_sparsity': float(sparsity),
        'variance': float(variance),
        'unique_policies': int(len(policy_set)),
        'hr_usage_percent': float(sparsity / num_actions * 100),
        'lr_usage_percent': float((num_actions - sparsity) / num_actions * 100),
    }
    
    # Method-specific statistics
    if method == 'pg' and policy_mode == 'learned':
        results['policy']['mean_prob'] = float(probs_or_qvals.mean())
        results['policy']['prob_std'] = float(probs_or_qvals.std())
    elif method == 'dqn' and policy_mode == 'learned':
        results['policy']['mean_q_value'] = float(probs_or_qvals.mean())
        results['policy']['q_value_std'] = float(probs_or_qvals.std())
        results['policy']['q_positive_ratio'] = float((probs_or_qvals > 0).float().mean() * 100)
    
    # Add predictions if saved
    if predictions:
        results['predictions'] = predictions
    
    return results


def print_results(results, title="TEST RESULTS"):
    """Pretty print evaluation results"""
    print('\n' + '='*80)
    print(title)
    
    
    # Detection metrics
    print('\nDetection Performance:')
    print(f"  Average Precision (AP): {results['detection']['AP']:.4f}")
    print(f"  Average Recall (AR):    {results['detection']['AR']:.4f}")
    print(f"  Precision:              {results['detection']['precision']:.4f}")
    print(f"  Recall:                 {results['detection']['recall']:.4f}")
    print(f"  F1 Score:               {results['detection']['f1']:.4f}")
    
    print(f"\nDataset Coverage:")
    print(f"  Total Detections:       {results['detection']['num_detections']}")
    print(f"  Ground Truth Objects:   {results['detection']['num_gt_objects']}")
    
    # Policy metrics
    print('\nPolicy Performance:')
    print(f"  Mean Reward:            {results['policy']['mean_reward']:.4f}")
    print(f"  Mean Sparsity:          {results['policy']['mean_sparsity']:.2f} / {num_actions}")
    print(f"  HR Image Usage:         {results['policy']['hr_usage_percent']:.1f}%")
    print(f"  LR Image Usage:         {results['policy']['lr_usage_percent']:.1f}%")
    print(f"  Policy Variance:        {results['policy']['variance']:.4f}")
    print(f"  Unique Policies:        {results['policy']['unique_policies']}")
    
    # Method-specific metrics
    if 'mean_prob' in results['policy']:
        print(f"\nPolicy Gradient Statistics:")
        print(f"  Mean Probability:       {results['policy']['mean_prob']:.4f}")
        print(f"  Prob Std Dev:           {results['policy']['prob_std']:.4f}")
    
    if 'mean_q_value' in results['policy']:
        print(f"\nDQN Statistics:")
        print(f"  Mean Q-value:           {results['policy']['mean_q_value']:.4f}")
        print(f"  Q-value Std Dev:        {results['policy']['q_value_std']:.4f}")
        print(f"  Q>0 (use HR):           {results['policy']['q_positive_ratio']:.1f}%")
    
    # Runtime if available
    if 'runtime' in results:
        print(f"\nRuntime:")
        print(f"  Evaluation Time:        {results['runtime']:.2f}s")
    
    


def compare_with_baselines(results_policy, results_hr, results_lr):
    """Compare trained policy with baselines"""
    print('COMPARISON WITH BASELINES')
    
    
    # Baseline 1: Always-HR
    print('\nBaseline 1: Always High-Resolution (100% HR)')
    print(f"  AP:        {results_hr['detection']['AP']:.4f}")
    print(f"  AR:        {results_hr['detection']['AR']:.4f}")
    print(f"  Reward:    {results_hr['policy']['mean_reward']:.4f}")
    print(f"  HR Usage:  100.0%")
    if 'runtime' in results_hr:
        print(f"  Runtime:   {results_hr['runtime']:.2f}s")
    
    # Baseline 2: Always-LR
    print('\nBaseline 2: Always Low-Resolution (0% HR)')
    print(f"  AP:        {results_lr['detection']['AP']:.4f}")
    print(f"  AR:        {results_lr['detection']['AR']:.4f}")
    print(f"  Reward:    {results_lr['policy']['mean_reward']:.4f}")
    print(f"  HR Usage:  0.0%")
    if 'runtime' in results_lr:
        print(f"  Runtime:   {results_lr['runtime']:.2f}s")
    
    # Trained Policy
    print('\nTrained Policy (Adaptive)')
    print(f"  AP:        {results_policy['detection']['AP']:.4f}")
    print(f"  AR:        {results_policy['detection']['AR']:.4f}")
    print(f"  Reward:    {results_policy['policy']['mean_reward']:.4f}")
    print(f"  HR Usage:  {results_policy['policy']['hr_usage_percent']:.1f}%")
    if 'runtime' in results_policy:
        print(f"  Runtime:   {results_policy['runtime']:.2f}s")
    
    # Comparative Analysis
    print('Comparative Analysis:')
    
    # AP comparison
    ap_policy = results_policy['detection']['AP']
    ap_hr = results_hr['detection']['AP']
    ap_lr = results_lr['detection']['AP']
    
    ap_retention = (ap_policy / ap_hr * 100) if ap_hr > 0 else 0
    ap_improvement = (ap_policy / ap_lr - 1) * 100 if ap_lr > 0 else 0
    
    print(f"\nAverage Precision (AP):")
    print(f"  vs Always-HR:  {ap_retention:.1f}% retained")
    print(f"  vs Always-LR:  {ap_improvement:+.1f}% improvement")
    
    # AR comparison
    ar_policy = results_policy['detection']['AR']
    ar_hr = results_hr['detection']['AR']
    ar_lr = results_lr['detection']['AR']
    
    ar_retention = (ar_policy / ar_hr * 100) if ar_hr > 0 else 0
    ar_improvement = (ar_policy / ar_lr - 1) * 100 if ar_lr > 0 else 0
    
    print(f"\nAverage Recall (AR):")
    print(f"  vs Always-HR:  {ar_retention:.1f}% retained")
    print(f"  vs Always-LR:  {ar_improvement:+.1f}% improvement")
    
    # Efficiency
    hr_usage = results_policy['policy']['hr_usage_percent']
    hr_savings = 100 - hr_usage
    
    print(f"\nEfficiency:")
    print(f"  HR Usage:      {hr_usage:.1f}%")
    print(f"  HR Savings:    {hr_savings:.1f}%")
    
    


def save_results(all_results, args, method):
    """Save all results to disk"""
    os.makedirs(args.cv_dir, exist_ok=True)
    
    # Save comprehensive results
    results_file = Path(args.cv_dir) / f'evaluation_results_{method}.json'
    
    # Prepare for JSON (remove predictions if large)
    results_to_save = {
        'method': method,
        'checkpoint': str(args.load),
        'hyperparameters': {
            'beta': args.beta,
            'sigma': args.sigma,
            'img_size': args.img_size,
        },
        'trained_policy': {
            'detection': all_results['policy']['detection'],
            'policy': {k: v for k, v in all_results['policy']['policy'].items() 
                      if k != 'predictions'}
        },
        'baseline_always_hr': {
            'detection': all_results['always_hr']['detection'],
            'policy': all_results['always_hr']['policy']
        },
        'baseline_always_lr': {
            'detection': all_results['always_lr']['detection'],
            'policy': all_results['always_lr']['policy']
        }
    }
    
    with open(results_file, 'w') as f:
        json.dump(results_to_save, f, indent=4)
    
    print(f'\n Results saved to {results_file}')
    
    # Save predictions separately if available
    if args.save_predictions and 'predictions' in all_results['policy']:
        predictions_file = Path(args.cv_dir) / f'predictions_{method}.json'
        with open(predictions_file, 'w') as f:
            json.dump(all_results['policy']['predictions'], f, indent=4)
        print(f' Predictions saved to {predictions_file}')
    
    # Save summary CSV
    summary_file = Path(args.cv_dir) / f'summary_{method}.csv'
    with open(summary_file, 'w') as f:
        f.write('Method,AP,AR,Precision,F1,Reward,HR_Usage,Unique_Policies\n')
        
        for name, results in [('Trained_Policy', all_results['policy']),
                             ('Always_HR', all_results['always_hr']),
                             ('Always_LR', all_results['always_lr'])]:
            f.write(f"{name},"
                   f"{results['detection']['AP']:.4f},"
                   f"{results['detection']['AR']:.4f},"
                   f"{results['detection']['precision']:.4f},"
                   f"{results['detection']['f1']:.4f},"
                   f"{results['policy']['mean_reward']:.4f},"
                   f"{results['policy']['hr_usage_percent']:.2f},"
                   f"{results['policy']['unique_policies']}\n")
    
    print(f' Summary saved to {summary_file}')


def main():
    """Main evaluation function"""
    # Parse arguments
    args = parse_args()
    
    # Set device
    args.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # Load checkpoint
    print(f'\nLoading checkpoint from {args.load}')
    checkpoint = torch.load(args.load, map_location=args.device)
    
    # Load hyperparameters from checkpoint
    if 'args' in checkpoint:
        ckpt_args = checkpoint['args']
        
        if args.beta is None:
            args.beta = ckpt_args.beta
        if args.sigma is None:
            args.sigma = ckpt_args.sigma
        if args.img_size is None:
            args.img_size = ckpt_args.img_size
    else:
        # Use defaults if not in checkpoint
        if args.beta is None:
            args.beta = 0.05
        if args.sigma is None:
            args.sigma = 0.25
        if args.img_size is None:
            args.img_size = 172
    
    print(f'\nHyperparameters:')
    print(f'  Method:     {args.method.upper()}')
    print(f'  Beta:       {args.beta}')
    print(f'  Sigma:      {args.sigma}')
    print(f'  Image size: {args.img_size}')
    
    # Load dataset
    print('\nLoading test dataset...')
    _, testset = utils.get_dataset(args.img_size, args.data_dir)
    
    testloader = data.DataLoader(
        testset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if args.device.type == 'cuda' else False
    )
    
    print(f'Test set: {len(testset)} images')

    # Initialize model
    print('\nInitializing model...')
    agent = utils.get_model(num_actions)
    
    # Load model weights
    if 'agent' in checkpoint:
        agent.load_state_dict(checkpoint['agent'])
    elif 'policy_net' in checkpoint:
        agent.load_state_dict(checkpoint['policy_net'])
    else:
        raise ValueError("Could not find model weights in checkpoint")
    
    agent = agent.to(args.device)
    
    # Print configuration
    print('EVALUATION CONFIGURATION')
    
    print(f'  Method:      {args.method.upper()}')
    print(f'  Checkpoint:  {args.load}')
    if 'epoch' in checkpoint:
        print(f'  Epoch:       {checkpoint["epoch"]}')
    if 'reward' in checkpoint:
        print(f'  Train reward: {checkpoint["reward"]:.4f}')
    print(f'  Device:      {args.device}')
    print(f'  Batch size:  {args.batch_size}')
    print(f'  Test samples: {len(testset)}')
    print(f'  Num actions: {num_actions}')
    print(f'  Beta:        {args.beta}')
    print(f'  Sigma:       {args.sigma}')
    
    
    
    # Run evaluations
    all_results = {}
    
    # Evaluate trained policy
    print(f'Evaluating Trained {args.method.upper()} Policy')
    t0 = time.time()
    results_policy = evaluate_policy(agent, testloader, args, args.method, 'learned')
    results_policy['runtime'] = time.time() - t0
    all_results['policy'] = results_policy
    print_results(results_policy, f"TRAINED {args.method.upper()} POLICY RESULTS")
    
    # Evaluate Always-HR baseline
    
    print('Evaluating Always-HR Baseline')
    t0 = time.time()
    results_hr = evaluate_policy(agent, testloader, args, args.method, 'always_hr')
    results_hr['runtime'] = time.time() - t0
    all_results['always_hr'] = results_hr
    print_results(results_hr, "ALWAYS-HR BASELINE RESULTS")
    
    # Evaluate Always-LR baseline
    
    print('Evaluating Always-LR Baseline')
    t0 = time.time()
    results_lr = evaluate_policy(agent, testloader, args, args.method, 'always_lr')
    results_lr['runtime'] = time.time() - t0
    all_results['always_lr'] = results_lr
    print_results(results_lr, "ALWAYS-LR BASELINE RESULTS")
    
    
    # Compare results
    compare_with_baselines(results_policy, results_hr, results_lr)
    
    
    # Save results
    
    save_results(all_results, args, args.method)
    
    print('\n Evaluation complete!')
    print(f'  Results directory: {args.cv_dir}')


if __name__ == '__main__':
    main()