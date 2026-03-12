import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import tqdm
import argparse
from pathlib import Path

from utils.DQN_pytorch import ReplayMemory, Transition, EpsilonGreedy  #  Import class
from utils import utils, utils_detector
from constants import num_actions

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='DQN Policy Network Training (DDP)')
    
    # Training parameters
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=128, 
                       help='batch size per GPU')
    parser.add_argument('--max_epochs', type=int, default=1000)
    parser.add_argument('--epoch_step', type=int, default=1000, help='epochs for lr decay')
    parser.add_argument('--num_workers', type=int, default=8)
    
    # DQN parameters
    parser.add_argument('--memory_size', type=int, default=10000,
                       help='replay memory size PER GPU')
    parser.add_argument('--eps_start', type=float, default=0.9)
    parser.add_argument('--eps_end', type=float, default=0.01)
    parser.add_argument('--eps_decay', type=int, default=2500)
    
    # Reward parameters
    parser.add_argument('--beta', type=float, default=0.01)
    parser.add_argument('--sigma', type=float, default=0.02)
    
    # Data parameters
    parser.add_argument('--data_dir', default='data/')
    parser.add_argument('--img_size', type=int, default=172)
    parser.add_argument('--cv_dir', default='cv/dqn_ddp/')
    
    # Distributed parameters
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--dist_backend', type=str, default='nccl')
    parser.add_argument('--dist_url', type=str, default='env://')
    
    # Checkpointing
    parser.add_argument('--save_freq', type=int, default=50,
                       help='save checkpoint every N epochs')
    parser.add_argument('--test_freq', type=int, default=10,
                       help='test every N epochs')
    
    args = parser.parse_args()
    return args


def setup_distributed():
    """Initialize distributed training environment"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    elif 'LOCAL_RANK' in os.environ:
        rank = int(os.environ.get('RANK', 0))
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
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
        
        dist.barrier()
        
        if is_main_process():
            print(f'Distributed DQN training initialized:')
            print(f'  World size: {world_size}')
            print(f'  Rank: {rank}')
            print(f'  Local rank: {local_rank}')
            print(f'  Backend: {args.dist_backend}')
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


def gather_tensors(tensor):
    """Gather tensors from all processes"""
    if not dist.is_initialized():
        return [tensor]
    
    world_size = dist.get_world_size()
    tensor_list = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(tensor_list, tensor)
    
    return tensor_list


def optimize_model(memory, policy_net, optimizer, args):
    """
    Perform one step of DQN optimization.
    
    Key difference from standard DQN:
    - No next_state (one-shot decision)
    - Q_target = reward (no future value)
    """
    if len(memory) < args.batch_size:
        return None
    
    # Sample batch from replay memory
    transitions = memory.sample(args.batch_size)
    batch = Transition(*zip(*transitions))
    
    # Prepare batch tensors
    state_batch = torch.cat(batch.state)    # [batch_size, 3, 172, 172]
    action_batch = torch.cat(batch.action)  # [batch_size, num_patches]
    reward_batch = torch.cat(batch.reward)  # [batch_size, 1]
    
    # Compute Q(s, a) for actions taken
    q_values = policy_net(state_batch)  # [batch_size, num_patches]
    
    # Weighted sum of Q-values based on actions
    state_action_values = (q_values * action_batch).sum(dim=1, keepdim=True)
    
    # Compute target Q-values (no next state)
    target_q_values = reward_batch  # [batch_size, 1]
    
    # Compute loss
    criterion = nn.SmoothL1Loss()  # Huber loss
    loss = criterion(state_action_values, target_q_values)
    
    #  Optimization step
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()
    
    return loss.item()


def train_epoch(epoch, policy_net, trainloader, optimizer, memory, action_selector, args):
    """Train for one epoch using DQN"""
    
    if args.distributed:
        agent = policy_net.module
    else:
        agent = policy_net

    agent.train()
    epoch_losses = []
    epoch_rewards = []
    epoch_hr_usage = []
    
    # Progress bar only on main process
    if is_main_process():
        pbar = tqdm.tqdm(trainloader, desc=f'Train Epoch {epoch}')
    else:
        pbar = trainloader

    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs = inputs.to(args.device, non_blocking=True)
        
        # Forward pass: Get Q-values from policy network
        q_values = agent(inputs)  # [batch_size, num_patches]
        
        #  Select actions using epsilon-greedy
        actions = action_selector.select_action(q_values, args.device)
        
        # Compute reward based on actions
        offset_fd, offset_cd = utils.read_offsets(targets, num_actions, args.device)
        reward = utils.compute_reward(offset_fd, offset_cd, actions, 
                                      args.beta, args.sigma)
        
        # Store transition in replay memory
        for i in range(inputs.size(0)):
            memory.push(
                inputs[i:i+1].detach(),      # state
                actions[i:i+1].detach(),     # action
                reward[i:i+1].detach()       # reward
            )
        
        # Optimize the policy network
        loss = optimize_model(memory, agent, optimizer, args)
        
        # Logging
        if loss is not None:
            epoch_losses.append(loss)
        epoch_rewards.append(reward.mean().item())
        epoch_hr_usage.append(actions.mean().item() * 100)
        
        if is_main_process() and batch_idx % 10 == 0:
            pbar.set_postfix({
                'loss': f'{np.mean(epoch_losses[-100:]):.4f}' if epoch_losses else 'N/A',
                'reward': f'{np.mean(epoch_rewards[-100:]):.3f}',
                'hr_usage': f'{np.mean(epoch_hr_usage[-100:]):.1f}%',
                'eps': f'{action_selector.get_epsilon():.3f}'
            })

    if args.distributed:
        # Convert to tensors
        avg_loss = torch.tensor(np.mean(epoch_losses) if epoch_losses else 0.0,
                               device=args.device)
        avg_reward = torch.tensor(np.mean(epoch_rewards), device=args.device)
        avg_hr = torch.tensor(np.mean(epoch_hr_usage), device=args.device)
        
        # All-reduce to get global average
        dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(avg_reward, op=dist.ReduceOp.SUM)
        dist.all_reduce(avg_hr, op=dist.ReduceOp.SUM)
        
        avg_loss = avg_loss.item() / args.world_size
        avg_reward = avg_reward.item() / args.world_size
        avg_hr = avg_hr.item() / args.world_size
    else:
        avg_loss = np.mean(epoch_losses) if epoch_losses else 0.0
        avg_reward = np.mean(epoch_rewards)
        avg_hr = np.mean(epoch_hr_usage)
    
    # Print epoch summary (main process only)
    if is_main_process():
        print(f'\nEpoch {epoch} Summary:')
        print(f'  Avg Loss:     {avg_loss:.4f}')
        print(f'  Avg Reward:   {avg_reward:.3f}')
        print(f'  Avg HR Usage: {avg_hr:.1f}%')
        print(f'  Epsilon:      {action_selector.get_epsilon():.3f}')
        print(f'  Memory Size:  {len(memory)} (per GPU)')
    
    return {
        'loss': avg_loss,
        'reward': avg_reward,
        'hr_usage': avg_hr,
    }


def test(epoch, policy_net, testloader, args):
    """
    Test the policy network.
    
    Args:
        epoch: Current epoch number
        policy_net: Policy network (DDP wrapped)
        testloader: Test data loader
        args: Arguments
    """
    # Get the underlying module (unwrap DDP)
    if args.distributed:
        model = policy_net.module
    else:
        model = policy_net
    
    model.eval()
    
    local_metrics, local_set_labels = [], []
    rewards_gpu, policies_gpu = [], []
    
    with torch.no_grad():
        if is_main_process():
            pbar = tqdm.tqdm(testloader, desc=f'Test Epoch {epoch}')
        else:
            pbar = testloader
        
        for inputs, targets in pbar:
            inputs = inputs.to(args.device, non_blocking=True)
            
            # Forward pass
            q_values = model(inputs)
            
            # GREEDY action selection (no epsilon during testing)
            policy = (q_values > 0).float()
            
            # Get pre-computed metrics
            offset_fd, offset_cd = utils.read_offsets(targets, num_actions, args.device)
            
            # Compute reward
            reward = utils.compute_reward(offset_fd, offset_cd, policy, args.beta, args.sigma)
            
            #  Each rank evaluates its own detections
            batch_metrics, batch_labels = utils.get_detected_boxes(
                policy.cpu(), targets, [], []
            )
            local_metrics.extend(batch_metrics)
            local_set_labels.extend(batch_labels)
            
            rewards_gpu.append(reward)
            policies_gpu.append(policy)
    
    #  Compute local detection metrics
    if len(local_metrics) > 0:
        true_positives, pred_scores, pred_labels = [
            np.concatenate(x, 0) for x in list(zip(*local_metrics))
        ]
        precision, recall, AP, f1, ap_class = utils_detector.ap_per_class(
            true_positives, pred_scores, pred_labels, local_set_labels
        )
        
        local_AP = torch.tensor(AP[0] if len(AP) > 0 else 0.0, device=args.device)
        local_AR = torch.tensor(recall.mean() if len(recall) > 0 else 0.0, device=args.device)
        local_precision = torch.tensor(precision.mean() if len(precision) > 0 else 0.0, device=args.device)
        local_F1 = torch.tensor(f1.mean() if len(f1) > 0 else 0.0, device=args.device)
    else:
        local_AP = torch.tensor(0.0, device=args.device)
        local_AR = torch.tensor(0.0, device=args.device)
        local_precision = torch.tensor(0.0, device=args.device)
        local_F1 = torch.tensor(0.0, device=args.device)
    
    #  Gather metrics from all ranks
    if args.distributed:
        world_size = dist.get_world_size()
        
        all_AP = [torch.zeros_like(local_AP) for _ in range(world_size)]
        all_AR = [torch.zeros_like(local_AR) for _ in range(world_size)]
        all_precision = [torch.zeros_like(local_precision) for _ in range(world_size)]
        all_F1 = [torch.zeros_like(local_F1) for _ in range(world_size)]
        
        dist.all_gather(all_AP, local_AP)
        dist.all_gather(all_AR, local_AR)
        dist.all_gather(all_precision, local_precision)
        dist.all_gather(all_F1, local_F1)
        
        # Gather rewards and policies
        rewards_tensor = torch.cat(rewards_gpu, 0)
        policies_tensor = torch.cat(policies_gpu, 0)
        
        all_rewards = gather_tensors(rewards_tensor)
        all_policies = gather_tensors(policies_tensor)
        
        if is_main_process():
            AP = torch.stack(all_AP).mean().item()
            AR = torch.stack(all_AR).mean().item()
            precision = torch.stack(all_precision).mean().item()
            F1 = torch.stack(all_F1).mean().item()
            
            rewards = [torch.cat(all_rewards, 0).cpu()]
            policies = [torch.cat(all_policies, 0).cpu()]
    else:
        AP = local_AP.item()
        AR = local_AR.item()
        precision = local_precision.item()
        F1 = local_F1.item()
        
        rewards = [torch.cat(rewards_gpu, 0).cpu()]
        policies = [torch.cat(policies_gpu, 0).cpu()]
    
    # Compute and print metrics (main process only)
    if is_main_process():
        print(f'\nTest Epoch {epoch}:')
        print(f'  AP: {AP:.4f} | AR: {AR:.4f}')
        print(f'  Precision: {precision:.4f} | F1: {F1:.4f}')
        
        # Compute policy statistics
        reward, sparsity, variance, policy_set = utils.performance_stats(policies, rewards)
        
        print(f'  Reward: {reward:.4f}')
        print(f'  Sparsity: {sparsity:.3f} | Variance: {variance:.3f}')
        print(f'  Unique Policies: {len(policy_set)}')
        
        return reward
    
    return 0.0


def save_checkpoint(agent, optimizer, action_selector, epoch, reward, args, is_best=False):
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
        'epsilon': action_selector.get_epsilon(),
        'steps_done': action_selector.steps_done,
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
    # Parse arguments
    args = parse_args()
    
    # Initialize distributed mode
    args = init_distributed_mode(args)
    
    # Create checkpoint directory
    if is_main_process():
        Path(args.cv_dir).mkdir(parents=True, exist_ok=True)
    
    # Wait for directory creation
    if args.distributed:
        dist.barrier()

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

    # Initialize replay memory and action selector
    memory = ReplayMemory(capacity=args.memory_size)
    action_selector = EpsilonGreedy(
        eps_start=args.eps_start,
        eps_end=args.eps_end,
        eps_decay=args.eps_decay
    )
    
    # Tracking
    start_epoch = 1
    best_reward = float('-inf')

    if is_main_process():
        print('DQN TRAINING CONFIGURATION (DDP)')
        print(f'Device: {args.device}')
        print(f'Distributed: {args.distributed}')
        if args.distributed:
            print(f'  World size: {args.world_size}')
            print(f'  Rank: {args.rank}')
            print(f'  Backend: {args.dist_backend}')
        print(f'Batch size per GPU: {args.batch_size}')
        print(f'Total batch size: {args.batch_size * args.world_size}')
        print(f'Learning rate: {args.lr}')
        print(f'Training samples: {len(trainset)}')
        print(f'Test samples: {len(testset)}')
        print(f'Memory capacity per GPU: {args.memory_size}')
        print(f'Epsilon: {args.eps_start} → {args.eps_end} (decay={args.eps_decay})')
        print(f'Beta: {args.beta}')
        print(f'Sigma: {args.sigma}')
    
    # Training loop
    for epoch in range(start_epoch, args.max_epochs + 1):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        
        #  Train
        train_stats = train_epoch(epoch, agent, trainloader, optimizer, 
                                  memory, action_selector, args)
        
        #  Update learning rate
        scheduler.step()
        
        # Test periodically
        if epoch % args.test_freq == 0:
            if is_main_process():
                print(f'\nTesting at epoch {epoch}...')
            
            #  Test
            reward = test(epoch, agent, testloader, args)
            
            # Save checkpoint
            is_best = reward > best_reward
            if is_best:
                best_reward = reward
            
            if epoch % args.save_freq == 0 or is_best:
                save_checkpoint(agent, optimizer, action_selector, 
                              epoch, reward, args, is_best)
        
        # Synchronize
        if args.distributed:
            dist.barrier()
    
    # Cleanup
    cleanup_distributed()
    
    if is_main_process():
        print('Training complete!')


if __name__ == '__main__':
    main()