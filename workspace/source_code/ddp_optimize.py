# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from torch.cuda import nvtx
import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

import argparse
import os
import random
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

torch.set_warn_always(False)
import signal
import time

class GracefulKiller:
    kill_now = False
    def __init__(self):
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, signum, frame):
        self.kill_now = True

# Instantiate globally so main() can check the exit status
killer = GracefulKiller()

def set_random_seeds(random_seed=0):
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

def evaluate(model, device, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    return accuracy

def main():
    num_epochs_default = 3 
    batch_size_default = 256 
    learning_rate_default = 0.1
    random_seed_default = 0
    model_dir_default = "./saved_models"
    model_filename_default = "resnet_distributed.pth"

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--local_rank", type=int, help="Local rank. Necessary for using the torch.distributed.launch utility.")
    parser.add_argument("--num_epochs", type=int, help="Number of training epochs.", default=num_epochs_default)
    parser.add_argument("--batch_size", type=int, help="Training batch size for one process.", default=batch_size_default)
    parser.add_argument("--learning_rate", type=float, help="Learning rate.", default=learning_rate_default)
    parser.add_argument("--random_seed", type=int, help="Random seed.", default=random_seed_default)
    parser.add_argument("--model_dir", type=str, help="Directory for saving models.", default=model_dir_default)
    parser.add_argument("--model_filename", type=str, help="Model filename.", default=model_filename_default)
    parser.add_argument("--resume", action="store_true", help="Resume training from saved checkpoint.")
    argv = parser.parse_args()

    local_rank = argv.local_rank
    num_epochs = argv.num_epochs
    batch_size = argv.batch_size
    learning_rate = argv.learning_rate
    random_seed = argv.random_seed
    model_dir = argv.model_dir
    model_filename = argv.model_filename
    resume = argv.resume
    
    if local_rank is None:
        local_rank = int(os.environ["LOCAL_RANK"])
        print('Local rank ', local_rank)

    model_filepath = os.path.join(model_dir, model_filename)

    set_random_seeds(random_seed=random_seed)
    
    # Let initialization errors surface naturally instead of hiding them
    torch.distributed.init_process_group(backend="nccl")

    model = torchvision.models.resnet18(pretrained=False)
    device = torch.device("cuda:{}".format(local_rank))
    model = model.to(device)
    ddp_model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

    if resume == True:
        map_location = {"cuda:0": "cuda:{}".format(local_rank)}
        ddp_model.load_state_dict(torch.load(model_filepath, map_location=map_location))

    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_set = torchvision.datasets.CIFAR10(root="../data", train=True, download=False, transform=transform)
    test_set = torchvision.datasets.CIFAR10(root="../data", train=False, download=False, transform=transform)

    train_sampler = DistributedSampler(dataset=train_set)
    num_workers = min(os.cpu_count(), 8)
    
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True,
        drop_last=True,
    )
    
    test_loader = DataLoader(
        dataset=test_set,
        batch_size=128,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-5)
    fp16_scaler = torch.amp.GradScaler("cuda")
    
    try:
        for epoch in range(num_epochs):
            # Integrate the GracefulKiller check directly into the training loop
            if killer.kill_now:
                print(f"Local Rank {local_rank} exiting gracefully due to shutdown signal.")
                break

            print("Local Rank: {}, Epoch: {}, Training ...".format(local_rank, epoch))
            if epoch == 2:
                torch.cuda.cudart().cudaProfilerStart()
                
            if epoch % 10 == 0:
                if local_rank == 0:
                    accuracy = evaluate(model=ddp_model, device=device, test_loader=test_loader)
                    os.makedirs(model_dir, exist_ok=True)
                    torch.save(ddp_model.state_dict(), model_filepath)
                    print("-" * 75)
                    print("Epoch: {}, Accuracy: {}".format(epoch, accuracy))
                    print("-" * 75)
            
            nvtx.range_push("Train")
            ddp_model.train()
            nvtx.range_pop() 
    
            nvtx.range_push("Data loading")
            for data in train_loader:
                nvtx.range_pop() 
                nvtx.range_push("Copy to device")
                inputs, labels = data[0].to(device, non_blocking=True), data[1].to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                nvtx.range_pop() 
                
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=True):                     
                    nvtx.range_push("Forward pass")    
                    outputs = ddp_model(inputs)
                    loss = criterion(outputs, labels)
                    nvtx.range_pop() 
                
                nvtx.range_push("Backward pass")
                fp16_scaler.scale(loss).backward()    
                fp16_scaler.step(optimizer)
                fp16_scaler.update()
                nvtx.range_pop() 
                
                nvtx.range_push("Data loading") 
            
            # Pop the final dangling "Data loading" range at the end of the loader loop
            nvtx.range_pop() 
            
            if epoch == 2:
                torch.cuda.cudart().cudaProfilerStop() 

        # FIXED: Moved outside the epoch loop. Cleans up only when training is fully done.
        if dist.is_initialized():
            dist.destroy_process_group()
        
    except Exception as e:
        print(f"Exception caught at inner: {e}")
        if dist.is_initialized():
            dist.destroy_process_group()

if __name__ == "__main__":
    # Remove the sleep loop wrapper; let torchrun handle execution lifetime
    main()