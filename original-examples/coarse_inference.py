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
import cbottle.inference
import torch
from cbottle.visualizations import visualize
from cbottle.datasets.dataset_3d import get_dataset
import matplotlib.pyplot as plt

ds = get_dataset(dataset="amip")
loader = torch.utils.data.DataLoader(ds, batch_size=1)
batch = next(iter(loader))

# hurricane
# 27.6648° N, 81.5158° W
model = cbottle.inference.load("cbottle-3d-tc")
out, coords = model.sample(
    batch, guidance_pixels=model.get_guidance_pixels([-81.5], [27.6])
)
c = coords.batch_info.channels.index("uas")
visualize(out[0, c, 0], nest=True)
plt.savefig("hurricane.png")


model = cbottle.inference.load("cbottle-3d-moe")
out, coords = model.sample(batch)


c = coords.batch_info.channels.index("rsut")
plt.clf()
visualize(out[0, c, 0], nest=True)
plt.savefig("out.png")
