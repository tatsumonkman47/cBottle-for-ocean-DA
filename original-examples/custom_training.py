# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""
Complete example training script using the training loop infrastructure in the
repository. By using the TrainingLoopBase class, this code supports the
following features

- distributed training
- gradient accumulation
- network checkpointing
- restartable training
- metric logging to tensorboard
- custom loss functions and datasets

"""

import sys
import torch
import torch.utils.data
from cbottle.training import loop
from cbottle.models import ModelConfigV1
from cbottle.datasets.samplers import InfiniteSequentialSampler
from cbottle import loss
import cbottle.distributed
import cbottle.checkpointing
from cbottle.training_stats import report
from cbottle.datasets.base import BatchInfo, TimeUnit

batch_size = 10
label_dim = 5
# Batch shape
condition_channels = 2
c = 10  # 10 channels
t = 1  # 1 frame for image generation, >1 for video
x = 12 * 64**2  # number of pixels in HPX64


class MockDataset(torch.utils.data.Dataset):
    def __init__(self, split="train"):
        self.split = split
        self.batch_info = BatchInfo(
            channels=[f"channel_{i}" for i in range(c)],
            time_step=1,
            time_unit=TimeUnit.HOUR,
        )

    def __len__(self) -> int:
        return {"train": 1000, "test": 100}[self.split]

    def __getitem__(self, i: int):
        return {
            "target": torch.randn(c, t, x),
            "condition": torch.randn(condition_channels, t, x),
            # the network takes arbitrary sample-dependent labels
            # this can encode meaningful information like the source dataset (ERA5 vs ICON), lead time etc.
            "class_labels": torch.zeros(label_dim),
            # frame-dependent temporal encodings
            "second_of_day": torch.tensor([0] * t),
            "day_of_year": torch.tensor([0] * t),
        }


class MyLoop(loop.TrainingLoopBase):
    # The model config is used by cbottle.models.get_model to return the model
    # this metadata is saved to the checkpoint so that we can load it more easily later on
    @property
    def model_config(self) -> ModelConfigV1:
        return ModelConfigV1(
            architecture="unet_hpx64",
            out_channels=c,
            condition_channels=condition_channels,
            time_length=t,
            level=6,
        )

    def get_loss_fn(self):
        return loss.EDMLoss(distribution="log_uniform")

    def get_data_loaders(self, batch_gpu: int):
        dataset = MockDataset("train")
        # needs to be an infinite iterator
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_gpu, sampler=InfiniteSequentialSampler(dataset)
        )
        test_dataset = MockDataset("test")
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_gpu)
        return dataset, loader, test_loader

    def get_optimizer(self, parameters) -> torch.optim.Optimizer:
        return torch.optim.Adam(parameters, lr=0.0001)

    def train_step(self, target, **kwargs):
        # batch is the same as returned by dataloader

        def eval_net(x, t):
            output = self.ddp(target, t, **kwargs)
            return output

        # the loss function returns a dataclass with these fields
        #   total: torch.Tensor
        #   denoising: torch.Tensor
        #   sigma: torch.Tensor
        #   classification: torch.Tensor | None = None
        loss_output: loss.Output = self.loss_fn(eval_net, images=target)

        # The training loop and optimizer only need to know the total loss
        return loss_output.total

    def validate(self, net):
        # net is not used here, but exists just for legacy reasons
        for batch in self.valid_loader:
            # stage the batch to the device
            batch = self._stage_dict_batch(batch)
            with torch.no_grad():
                loss = self.train_step(**batch)
            # report any scalar metrics to be logged
            report("test_loss", loss)


if __name__ == "__main__":
    output_dir = "output/"
    if len(sys.argv) > 1:
        output_dir = sys.argv[1]

    cbottle.distributed.init()
    loop = MyLoop(
        run_dir=output_dir,
        # for testing use these settings for short training
        steps_per_tick=1,
        total_ticks=1,
        state_dump_ticks=1,
        # gradient accumulation is built-in. This batch_size is defined as the
        # number of samples per gradient descent step across all gpus.
        batch_size=batch_size * cbottle.distributed.get_world_size(),
        # this is the number of samples per gpu---reduce this to change memory
        # usage
        batch_gpu=batch_size,
    )
    loop.setup()

    # restart from last checkpoint if rerunning
    try:
        loop.resume_from_rundir("output/")
    except FileNotFoundError:
        print("Starting from scratch")

    loop.train()

    # open checkpoint for later use
    with cbottle.checkpointing.Checkpoint(
        f"{output_dir}/training-state-{loop.batch_size:09d}.checkpoint"
    ) as checkpoint:
        model = checkpoint.read_model()
        print(model)
