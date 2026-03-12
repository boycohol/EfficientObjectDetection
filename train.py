"""
Train Policy Network with Distributed Data Parallel (DDP)

Single GPU:
    python train.py --batch_size 512 --lr 1e-4

Multi-GPU (e.g., 4 GPUs):
    torchrun --nproc_per_node=4 train.py --batch_size 512 --lr 1e-4
    
    # OR using older pytorch.distributed.launch:
    python -m torch.distributed.launch --nproc_per_node=4 train.py --batch_size 512 --lr 1e-4
"""
import os
import sys
import torch
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import tqdm
import torch.optim as optim
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
import argparse
from pathlib import Path
from tensorboard_logger import configure, log_value
from torch.distributions import Bernoulli

from utils import utils, utils_detector
from constants import base_dir_metric_cd, base_dir_metric_fd, num_actions


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Policy Network Training with DDP')
    
    # Training parameters
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size per GPU')
    parser.add_argument('--max_epochs', type=int, default=10000, help='total epochs to run')
    parser.add_argument('--epoch_step', type=int, default=1000, help='epochs for lr decay')
    parser.add_argument('--test_epoch', type=int, default=10, help='test every N epochs')
    
    # Data parameters
    parser.add_argument('--data_dir', default='data/', help='data directory')
    parser.add_argument('--img_size', type=int, default=172, help='policy network image size')
    parser.add_argument('--num_workers', type=int, default=8, help='dataloader workers')
    
    # Model parameters
    parser.add_argument('--load', default=None, help='checkpoint to load')
    parser.add_argument('--cv_dir', default='cv/tmp/', help='checkpoint directory')
    
    # RL parameters
    parser.add_argument('--alpha', type=float, default=0.8, help='exploration factor')
    parser.add_argument('--beta', type=float, default=0.1, help='coarse detector bias')
    parser.add_argument('--sigma', type=float, default=0.5, help='HR cost weight')
    
    # Distributed parameters
    parser.add_argument('--local_rank', type=int, default=-1, help='local rank for distributed training')
    parser.add_argument('--dist_backend', type=str, default='nccl', help='distributed backend')
    parser.add_argument('--dist_url', type=str, default='env://', help='url for distributed setup')
    
    # Other parameters
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--save_freq', type=int, default=50, help='checkpoint save frequency')
    
    args = parser.parse_args()
    return args


def setup_distributed():
    """Initialize distributed training environment"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    elif 'LOCAL_RANK' in os.environ:
        # torchrun
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        # Single GPU or CPU
        rank = 0
        world_size = 1
        local_rank = 0
    
    return rank, world_size, local_rank


def init_distributed_mode(args):
    """Initialize distributed training"""
    rank, world_size, local_rank = setup_distributed()
    
    args.rank = rank
    args.world_size = world_size
    args.local_rank = local_rank
    args.distributed = world_size > 1
    
    if args.distributed:
        torch.cuda.set_device(local_rank)
        args.device = torch.device('cuda', local_rank)
        
        # Initialize process group
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=world_size,
            rank=rank
        )
        
        # Synchronize
        dist.barrier()
        
        if is_main_process():
            print(f'Distributed training initialized:')
            print(f'  World size: {world_size}')
            print(f'  Rank: {rank}')
            print(f'  Local rank: {local_rank}')
            print(f'  Device: {args.device}')
    else:
        args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if is_main_process():
            print(f'Single GPU training on device: {args.device}')
    
    return args


def is_main_process():
    """Check if current process is main process"""
    return not dist.is_initialized() or dist.get_rank() == 0


def cleanup_distributed():
    """Cleanup distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()


def set_seed(seed, rank=0):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed + rank)
        torch.cuda.manual_seed_all(seed + rank)


def reduce_tensor(tensor, world_size):
    """Reduce tensor across all processes"""
    if not dist.is_initialized():
        return tensor
    
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt


def gather_tensors(tensor):
    """Gather tensors from all processes"""
    if not dist.is_initialized():
        return [tensor]
    
    world_size = dist.get_world_size()
    tensor_list = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(tensor_list, tensor)
    return tensor_list


class MetricLogger:
    """Logger that only logs from main process"""
    def __init__(self, log_dir):
        self.is_main = is_main_process()
        if self.is_main:
            configure(str(log_dir), flush_secs=5)
    
    def log(self, tag, value, step):
        if self.is_main:
            log_value(tag, value, step)


def train(epoch, agent, trainloader, optimizer, args, logger):
    """
    Train the policy network for one epoch.
    
    Args:
        epoch: Current epoch number
        agent: Policy network (DDP wrapped)
        trainloader: Training data loader
        optimizer: Optimizer
        args: Arguments
        logger: Metric logger
    """
    agent.train()
    
    # Set epoch for DistributedSampler
    if args.distributed:
        trainloader.sampler.set_epoch(epoch)
    
    rewards, rewards_baseline, policies = [], [], []
    total_loss = 0.0
    num_batches = 0
    
    # Progress bar only on main process
    if is_main_process():
        pbar = tqdm.tqdm(trainloader, desc=f'Train Epoch {epoch}')
    else:
        pbar = trainloader
    
    for batch_idx, (inputs, targets) in enumerate(pbar):
        # Move inputs to device
        inputs = inputs.to(args.device, non_blocking=True)
        
        # Forward pass through agent
        probs = torch.sigmoid(agent(inputs))
        
        # Temperature scaling for exploration (Equation 16)
        alpha_hp = np.clip(args.alpha + epoch * 0.001, 0.6, 0.95)
        probs = probs * alpha_hp + (1 - alpha_hp) * (1 - probs)
        
        # Sample actions from Bernoulli distribution
        distr = Bernoulli(probs)
        policy_sample = distr.sample()
        
        # Baseline policy (deterministic, no gradients)
        with torch.no_grad():
            policy_map = (probs > 0.5).float()
        
        # Get pre-computed detector performance metrics
        offset_fd, offset_cd = utils.read_offsets(targets, num_actions, args.device)
        
        # Compute rewards
        with torch.no_grad():
            reward_map = utils.compute_reward(
                offset_fd, offset_cd, policy_map, args.beta, args.sigma
            )
        
        reward_sample = utils.compute_reward(
            offset_fd, offset_cd, policy_sample, args.beta, args.sigma
        )
        
        # Advantage function
        advantage = reward_sample.float() - reward_map.float()
        
        # REINFORCE loss
        loss = -distr.log_prob(policy_sample)
        loss = (loss * advantage.detach()).mean()
        
        # Optimization step
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Store metrics
        rewards.append(reward_sample.detach())
        rewards_baseline.append(reward_map)
        policies.append(policy_sample.detach())
        total_loss += loss.item()
        num_batches += 1
        
        # Update progress bar (main process only)
        if is_main_process() and batch_idx % 10 == 0:
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'reward': f'{reward_sample.mean().item():.4f}',
                'sparsity': f'{policy_sample.sum(1).mean().item():.2f}'
            })
    
    # Gather metrics from all processes
    if args.distributed:
        # Convert lists to tensors for gathering
        rewards_tensor = torch.cat(rewards, 0)
        rewards_baseline_tensor = torch.cat(rewards_baseline, 0)
        policies_tensor = torch.cat(policies, 0)
        
        # Gather from all processes
        all_rewards = gather_tensors(rewards_tensor)
        all_rewards_baseline = gather_tensors(rewards_baseline_tensor)
        all_policies = gather_tensors(policies_tensor)
        
        # Concatenate on main process
        if is_main_process():
            rewards = [torch.cat(all_rewards, 0).cpu()]
            rewards_baseline = [torch.cat(all_rewards_baseline, 0).cpu()]
            policies = [torch.cat(all_policies, 0).cpu()]
    
    # Compute performance statistics (main process only)
    if is_main_process():
        reward, sparsity, variance, policy_set = utils.performance_stats(policies, rewards)
        baseline_reward = torch.cat(rewards_baseline, 0).mean().item()
        avg_loss = total_loss / num_batches
        
        print(f'\nTrain Epoch {epoch}:')
        print(f'  Loss: {avg_loss:.4f}')
        print(f'  Reward: {reward:.4f} | Baseline: {baseline_reward:.4f}')
        print(f'  Sparsity: {sparsity:.3f} | Variance: {variance:.3f}')
        print(f'  Unique Policies: {len(policy_set)}')
        
        # Log to tensorboard
        logger.log('train_loss', avg_loss, epoch)
        logger.log('train_reward', reward, epoch)
        logger.log('train_sparsity', sparsity, epoch)
        logger.log('train_variance', variance, epoch)
        logger.log('train_baseline_reward', baseline_reward, epoch)
        logger.log('train_unique_policies', len(policy_set), epoch)
        logger.log('train_advantage', reward - baseline_reward, epoch)
        logger.log('learning_rate', optimizer.param_groups[0]['lr'], epoch)


def test(epoch, agent, testloader, args, logger):
    """
    Test the policy network.
    
    Args:
        epoch: Current epoch number
        agent: Policy network (DDP wrapped)
        testloader: Test data loader
        args: Arguments
        logger: Metric logger
    """
    # Get the underlying module
    if args.distributed:
        model = agent.module
    else:
        model = agent
    
    model.eval()
    
    local_metrics, local_set_labels = [], []
    rewards_gpu, policies_gpu = [], []
    
    with torch.no_grad():
        if is_main_process():
            pbar = tqdm.tqdm(testloader, desc=f'Test Epoch {epoch}')
        else:
            pbar = testloader
        
        for inputs, targets in pbar:
            # Move inputs to device
            inputs = inputs.to(args.device, non_blocking=True)
            
            # Forward pass
            probs = torch.sigmoid(model(inputs))
            
            # Deterministic policy (greedy)
            policy = (probs > 0.5).float()
            
            # Get pre-computed metrics
            offset_fd, offset_cd = utils.read_offsets(targets, num_actions, args.device)
            
            # Compute reward
            reward = utils.compute_reward(offset_fd, offset_cd, policy, 
                                         args.beta, args.sigma)
            
            batch_metrics, batch_labels = utils.get_detected_boxes(
                policy.cpu(), targets, [], []
            )
            local_metrics.extend(batch_metrics)
            local_set_labels.extend(batch_labels)
            
            # Store for gathering
            rewards_gpu.append(reward)
            policies_gpu.append(policy)
    
    # Compute local detection metrics
    if len(local_metrics) > 0:
        true_positives, pred_scores, pred_labels = [
            np.concatenate(x, 0) for x in list(zip(*local_metrics))
        ]
        
        precision, recall, AP, f1, ap_class = utils_detector.ap_per_class(
            true_positives, pred_scores, pred_labels, local_set_labels
        )
        
        # Convert to tensors for gathering
        local_AP = torch.tensor(AP[0] if len(AP) > 0 else 0.0, device=args.device)
        local_AR = torch.tensor(recall.mean() if len(recall) > 0 else 0.0, device=args.device)
        local_precision = torch.tensor(precision.mean() if len(precision) > 0 else 0.0, device=args.device)
        local_F1 = torch.tensor(f1.mean() if len(f1) > 0 else 0.0, device=args.device)
        local_num_detections = torch.tensor(len(true_positives), device=args.device)
        local_num_gt = torch.tensor(len(local_set_labels), device=args.device)
    else:
        # No detections
        local_AP = torch.tensor(0.0, device=args.device)
        local_AR = torch.tensor(0.0, device=args.device)
        local_precision = torch.tensor(0.0, device=args.device)
        local_F1 = torch.tensor(0.0, device=args.device)
        local_num_detections = torch.tensor(0, device=args.device)
        local_num_gt = torch.tensor(0, device=args.device)
    
    # Gather metrics from all ranks
    if args.distributed:
        world_size = dist.get_world_size()
        
        # Gather detection metrics
        all_AP = [torch.zeros_like(local_AP) for _ in range(world_size)]
        all_AR = [torch.zeros_like(local_AR) for _ in range(world_size)]
        all_precision = [torch.zeros_like(local_precision) for _ in range(world_size)]
        all_F1 = [torch.zeros_like(local_F1) for _ in range(world_size)]
        all_num_detections = [torch.zeros_like(local_num_detections) for _ in range(world_size)]
        all_num_gt = [torch.zeros_like(local_num_gt) for _ in range(world_size)]
        
        dist.all_gather(all_AP, local_AP)
        dist.all_gather(all_AR, local_AR)
        dist.all_gather(all_precision, local_precision)
        dist.all_gather(all_F1, local_F1)
        dist.all_gather(all_num_detections, local_num_detections)
        dist.all_gather(all_num_gt, local_num_gt)
        
        # Gather rewards and policies
        rewards_tensor = torch.cat(rewards_gpu, 0)
        policies_tensor = torch.cat(policies_gpu, 0)
        
        all_rewards = gather_tensors(rewards_tensor)
        all_policies = gather_tensors(policies_tensor)
        
        if is_main_process():
            # Average detection metrics across ranks
            AP = torch.stack(all_AP).mean().item()
            AR = torch.stack(all_AR).mean().item()
            precision = torch.stack(all_precision).mean().item()
            F1 = torch.stack(all_F1).mean().item()
            total_detections = torch.stack(all_num_detections).sum().item()
            total_gt = torch.stack(all_num_gt).sum().item()
            
            # Concatenate policies and rewards
            rewards = [torch.cat(all_rewards, 0).cpu()]
            policies = [torch.cat(all_policies, 0).cpu()]
    else:
        # Single GPU
        AP = local_AP.item()
        AR = local_AR.item()
        precision = local_precision.item()
        F1 = local_F1.item()
        total_detections = local_num_detections.item()
        total_gt = local_num_gt.item()
        
        rewards = [torch.cat(rewards_gpu, 0).cpu()]
        policies = [torch.cat(policies_gpu, 0).cpu()]
    
    # Compute and print metrics (main process only)
    if is_main_process():
        print(f'\nTest Epoch {epoch}:')
        print(f'  AP: {AP:.4f} | AR: {AR:.4f}')
        print(f'  Precision: {precision:.4f} | F1: {F1:.4f}')
        print(f'  Detections: {int(total_detections)} | GT Objects: {int(total_gt)}')
        
        # Log detection metrics
        logger.log('test_AP', AP, epoch)
        logger.log('test_AR', AR, epoch)
        logger.log('test_precision', precision, epoch)
        logger.log('test_F1', F1, epoch)
        
        # Compute policy statistics
        reward, sparsity, variance, policy_set = utils.performance_stats(policies, rewards)
        
        print(f'  Reward: {reward:.4f}')
        print(f'  Sparsity: {sparsity:.3f} | Variance: {variance:.3f}')
        print(f'  Unique Policies: {len(policy_set)}')
        print(f'  HR Usage: {sparsity/num_actions*100:.1f}%')
        
        # Log policy metrics
        logger.log('test_reward', reward, epoch)
        logger.log('test_sparsity', sparsity, epoch)
        logger.log('test_variance', variance, epoch)
        logger.log('test_unique_policies', len(policy_set), epoch)
        logger.log('test_hr_usage', sparsity/num_actions*100, epoch)
        
        return reward
    
    return 0.0

def save_checkpoint(agent, optimizer, epoch, reward, args, is_best=False):
    """Save checkpoint (main process only)"""
    if not is_main_process():
        return
    
    # Get underlying model state dict
    if args.distributed:
        model_state_dict = agent.module.state_dict()
    else:
        model_state_dict = agent.state_dict()
    
    state = {
        'agent': model_state_dict,
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'reward': reward,
        'args': args,
    }
    
    # Save regular checkpoint
    checkpoint_path = Path(args.cv_dir) / f'ckpt_E_{epoch}_R_{reward:.4f}.pth'
    torch.save(state, checkpoint_path)
    
    # Save latest checkpoint
    latest_path = Path(args.cv_dir) / 'latest.pth'
    torch.save(state, latest_path)
    
    # Save best checkpoint
    if is_best:
        best_path = Path(args.cv_dir) / 'best_model.pth'
        torch.save(state, best_path)
        print(f'Saved best model with reward {reward:.4f}')


def main():
    """Main training function"""
    # Parse arguments
    args = parse_args()
    
    # Initialize distributed training
    args = init_distributed_mode(args)
    
    # Set random seed
    set_seed(args.seed, args.rank)
    
    # Create checkpoint directory (main process only)
    if is_main_process():
        os.makedirs(args.cv_dir, exist_ok=True)
        utils.save_args(__file__, args)
    
    # Wait for main process to create directory
    if args.distributed:
        dist.barrier()
    
    # Initialize logger
    logger = MetricLogger(Path(args.cv_dir) / 'log')
    
    # Load datasets
    if is_main_process():
        print('\nLoading datasets...')
    
    trainset, testset = utils.get_dataset(args.img_size, args.data_dir)
    
    # Create distributed samplers
    if args.distributed:
        train_sampler = DistributedSampler(
            trainset,
            num_replicas=args.world_size,
            rank=args.rank,
            shuffle=True,
            seed=args.seed
        )
        test_sampler = DistributedSampler(
            testset,
            num_replicas=args.world_size,
            rank=args.rank,
            shuffle=False
        )
    else:
        train_sampler = None
        test_sampler = None
    
    # Create data loaders
    trainloader = data.DataLoader(
        trainset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    testloader = data.DataLoader(
        testset,
        batch_size=args.batch_size,
        sampler=test_sampler,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Initialize model
    if is_main_process():
        print('Initializing model...')
    
    agent = utils.get_model(num_actions)
    agent = agent.to(args.device)
    
    # Load checkpoint if specified
    start_epoch = 0
    best_reward = float('-inf')
    
    # Wrap model with DDP
    if args.distributed:
        agent = DDP(
            agent,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=False
        )
    
    # Initialize optimizer
    optimizer = optim.Adam(agent.parameters(), lr=args.lr)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=args.epoch_step,
        gamma=0.1
    )
    
    # Print configuration (main process only)
    if is_main_process():
        print('Training Configuration:')
        print(f'  Device: {args.device}')
        print(f'  Distributed: {args.distributed}')
        if args.distributed:
            print(f'  World size: {args.world_size}')
            print(f'  Rank: {args.rank}')
        print(f'  Batch size per GPU: {args.batch_size}')
        print(f'  Total batch size: {args.batch_size * args.world_size}')
        print(f'  Learning rate: {args.lr}')
        print(f'  Alpha: {args.alpha}')
        print(f'  Beta: {args.beta}')
        print(f'  Sigma: {args.sigma}')
        print(f'  Num actions: {num_actions}')
        print(f'  Train samples: {len(trainset)}')
        print(f'  Test samples: {len(testset)}')
    
    # Training loop
    try:
        for epoch in range(start_epoch, start_epoch + args.max_epochs + 1):
            if args.distributed:
                train_sampler.set_epoch(epoch)

            # Train
            train(epoch, agent, trainloader, optimizer, args, logger)
            print(f'Finished training epoch {epoch}')
            # Test
            if epoch % args.test_epoch == 0:
                reward = test(epoch, agent, testloader, args, logger)
                
                # Save checkpoint
                is_best = reward > best_reward
                if is_best:
                    best_reward = reward
                
                if epoch % args.save_freq == 0 or is_best:
                    save_checkpoint(agent, optimizer, epoch, reward, args, is_best)
            
            # Step scheduler
            scheduler.step()
            
            # Synchronize processes
            if args.distributed:
                dist.barrier()
    
    except KeyboardInterrupt:
        if is_main_process():
            print('Training interrupted')
        
        # Save interrupted checkpoint
        if is_main_process():
            if args.distributed:
                model_state_dict = agent.module.state_dict()
            else:
                model_state_dict = agent.state_dict()
            
            state = {
                'agent': model_state_dict,
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'args': args,
            }
            
            interrupt_path = Path(args.cv_dir) / f'ckpt_interrupted_E_{epoch}.pth'
            torch.save(state, interrupt_path)
            print(f'Saved interrupted checkpoint to {interrupt_path}')
    
    finally:
        # Cleanup
        cleanup_distributed()


if __name__ == '__main__':
    main()