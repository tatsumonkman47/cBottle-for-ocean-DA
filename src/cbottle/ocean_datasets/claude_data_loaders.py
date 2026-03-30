import torch
import os
from torch.utils.data import Dataset
import numpy as np
import xarray as xr
import threading
import fsspec
from functools import lru_cache
import time
import glob

import sys
if os.path.exists('/home/tm3076/projects/NYU_SWOT_project/'):
    sys.path.append('/home/tm3076/projects/NYU_SWOT_project/Inpainting_Pytorch_gen/SWOT-inpainting-DL/src')
    sys.path.append('/home/tm3076/projects/NYU_SWOT_project/SWOT-data-analysis/src')
elif os.path.exists('/home.ufs/tm3076/swot_SUM03/SWOT_project/'):
    sys.path.append('/home.ufs/tm3076/swot_SUM03/SWOT_project/SWOT-inpainting-DL/src')
    sys.path.append('/home.ufs/tm3076/swot_SUM03/SWOT_project/SWOT-data-analysis/src')
elif os.path.exists('/scratch/tm3076/project/'):
    sys.path.append('/scratch/tm3076/project/SWOT-inpainting-DL/src')
    sys.path.append('/scratch/tm3076/project/SWOT-data-analysis/src')
import interp_utils

_thread_local = threading.local()

class llc4320_dataset(Dataset):
    def __init__(self, data_dir, mid_timestep, N_t, patch_coords,
                 infields, outfields,
                 in_mask_list, out_mask_list,
                 in_transform_list, out_transform_list,
                 standards=None, N=128, L_x=512e3, L_y=512e3,
                 squeeze=False, return_metadata=False,
                 return_masks=False, time_loading=False,
                 regrid_SWOT=False, cloud_rho=0.7,
                 squeeze_single_channel=True,
                 # New optimization parameters
                 preload_cloud_masks=True, cloud_cache_size=1000,
                 dtime = 1, # Time discretization, if you are using fullt
                 zarr_cache_size=256*1024**2):

        self.data_dir = data_dir
        self.mid_timestep = mid_timestep
        self.N_t = N_t
        self.patch_coords = patch_coords
        self.infields = infields
        self.outfields = outfields
        self.in_mask_list = in_mask_list
        self.out_mask_list = out_mask_list
        self.in_transform_list = in_transform_list
        self.out_transform_list = out_transform_list
        self.N = N; self.L_x = L_x; self.L_y = L_y
        self.squeeze = squeeze
        self.return_meta_data = return_metadata
        self.return_masks = return_masks
        self.time_loading = time_loading
        self.regrid_SWOT = regrid_SWOT
        self.cloud_rho = cloud_rho
        self.dtime = dtime
        
        # Optimization parameters
        self.preload_cloud_masks = preload_cloud_masks
        self.cloud_cache_size = cloud_cache_size
        self.zarr_cache_size = zarr_cache_size

        climpath = ""
        if os.path.exists('/home/tm3076/shaferlab/tatsu/cBottle-for-ocean-DA/'):
            climpath = '/home/tm3076/shaferlab/tatsu/cBottle-for-ocean-DA/src/cbottle/ocean_datasets/data'
        elif os.path.exists("/home.ufs/tm3076/swot_SUM03/SWOT_project/"): 
            climpath = '/home.ufs/tm3076/swot_SUM03/SWOT_project/SWOT-inpainting-DL/data'
        elif os.path.exists("/scratch/tm3076/project/"):
            climpath = '/scratch/tm3076/project/SWOT-inpainting-DL/data'
        # Load SST climatology for seasonal normalization
        self.SST_mean_climatology = xr.open_dataset(os.path.join(climpath,"SST_NP_daily_climatology.nc"))

        if standards is None:
            standards = {"mean_ssh":0., "std_ssh":1., "mean_sst":0., "std_sst":1., "extra_mean_tuning":0.,}
        
        # Pre-compile transforms to avoid repeated function creation
        self.transforms = self._create_transforms(standards)

        # Pre-compute random selections for cloud masks to avoid repeated computation
        if self.preload_cloud_masks:
            self._precompute_cloud_selections()

    def _create_transforms(self, standards):
        """Pre-create transform functions to avoid repeated lambda creation"""
        def make_standardize(mean=None, std=1.0):
            if mean is not None:
                def fn(x):
                    arr = x.values if isinstance(x, xr.DataArray) else x
                    return torch.from_numpy(((arr - mean) / std).astype(np.float32))
            else:
                def fn(x):
                    arr = x.values if isinstance(x, xr.DataArray) else x
                    return torch.from_numpy((arr / std).astype(np.float32))
            return fn
        def make_std_samplewise(std=1.0):
            def fn(x):
                arr = x.values if isinstance(x, xr.DataArray) else x
                mean_val = arr.mean()
                return torch.from_numpy(((arr - mean_val) / std).astype(np.float32))
            return fn
        def make_seasonal_standardize(std=5.0,extra_mean_tuning=0):
            """Create seasonal standardization function that uses daily climatological mean"""
            def fn(x):
                arr = x.values if isinstance(x, xr.DataArray) else x
                #print(f"arr.shape {arr.shape}")
                # Get the time slice indices for this sample
                time_start = self.mid_timestep - self.dtime*(self.N_t//2)
                time_end = self.mid_timestep + self.dtime*(self.N_t//2 + self.N_t%2)
                time_indices = np.arange(time_start, time_end, self.dtime)
                if self.dtime > 1: # convert from hourly to daily timesteps if dtime is greater than 1
                    time_indices = time_indices//24
                # Extract climatological means for the corresponding daily indices
                # Handle potential out-of-bounds indices by wrapping or clamping
                climatology_length = len(self.SST_mean_climatology.SST)  
                clipped_indices = np.clip(time_indices, 0, climatology_length - 1)
                # Get climatological means for this time window
                clim_means = self.SST_mean_climatology.SST.isel(time=clipped_indices).values  
                # Subtract climatological mean from each time step
                # arr shape is typically (time, y, x), clim_means shape is (time,)
                if len(arr.shape) == 3:  # (time, y, x)
                    # Broadcast climatological means to match spatial dimensions
                    clim_means_broadcast = clim_means[:, np.newaxis, np.newaxis]
                    normalized_arr = (arr - clim_means_broadcast - extra_mean_tuning) / std
                elif len(arr.shape) == 2:  # (y, x) - single time step
                    # Use single climatological value
                    clim_mean = clim_means[self.N_t//2] if len(clim_means) > self.N_t//2 else clim_means[0]
                    normalized_arr = (arr - clim_mean - extra_mean_tuning) / std
                else:
                    # Fallback: use mean of climatological values
                    clim_mean = clim_means.mean()
                    normalized_arr = (arr - clim_mean - extra_mean_tuning) / std
                return torch.from_numpy(normalized_arr.astype(np.float32))
            return fn
        return {
            "std_ssh_norm": make_standardize(std=standards["std_ssh"]),
            "std_sst_norm": make_standardize(std=standards["std_sst"]),
            "std_mean_ssh_norm": make_std_samplewise(std=standards["std_ssh"]),
            "std_mean_sst_norm": make_std_samplewise(std=standards["std_sst"]),
            "std_global_mean_ssh_norm": make_standardize(mean=standards["mean_ssh"], std=standards["std_ssh"]),
            "std_global_mean_sst_norm": make_standardize(mean=standards["mean_sst"], std=standards["std_sst"]),
            "std_seasonal_mean_sst_norm": make_seasonal_standardize(std=standards["std_sst"],extra_mean_tuning=standards["extra_mean_tuning"]),
            "no_transform": lambda x: torch.from_numpy(x.values.astype(np.float32)) if isinstance(x, xr.DataArray) else torch.from_numpy(x.astype(np.float32))
        }

    def _precompute_cloud_selections(self):
        """Pre-compute cloud mask selections to avoid repeated catalog queries"""
        # This will be populated in _init_worker_local
        self.cloud_selections_cache = None

    def __len__(self):
        return self.patch_coords.shape[0]
    
    def __getitem__(self, idx):
        pid = str(int(self.patch_coords[idx, 2])).zfill(3)
        coords = self.patch_coords[idx]
        meta = {"patch_ID": pid, "patch_coords": coords, "mid_timestep": self.mid_timestep}
        if not getattr(_thread_local, 'initialized', False):
            self._init_worker_local()
        invar, inmask = self._load_fields(pid, self.infields, self.in_transform_list, self.in_mask_list)
        # Reuse invar/inmask when outfields are identical to avoid double I/O
        if (self.outfields == self.infields
                and self.out_transform_list == self.in_transform_list):
            outvar = invar.clone()
            # Apply output masks (may differ from input masks)
            if self.out_mask_list != self.in_mask_list:
                _, outmask = self._load_masks_only(pid, outvar, self.out_mask_list)
                outvar = outvar * outmask / (inmask + 1e-8)  # undo input mask, apply output mask
            else:
                outmask = inmask.clone()
        elif self.outfields:
            outvar, outmask = self._load_fields(pid, self.outfields, self.out_transform_list, self.out_mask_list)
        else:
            outvar = torch.zeros((self.N_t, 1, self.N, self.N), dtype=torch.float32)
            outmask = torch.zeros_like(outvar)
        if self.squeeze:
            invar, outvar = invar.squeeze(), outvar.squeeze()
            inmask, outmask = inmask.squeeze(), outmask.squeeze()
        result = [invar, outvar]
        if self.return_masks:
            result.extend([inmask, outmask])
        if self.return_meta_data:
            result.append(meta)
        if len(result) == 1 and self.squeeze_single_channel:
            return result[0]
        return tuple(result)

    def _init_worker_local(self):
        """Initialize per-worker resources with optimizations"""
        # Use larger cache for better performance
        fs = fsspec.filesystem("file", default_fill_cache=True, 
                              block_size=self.zarr_cache_size)
        _thread_local.fs = fs
        # Lazy load SWOT swaths or numpy mask catalog
        if self.regrid_SWOT:
            _thread_local.swot_ds = [
                xr.open_zarr(fs.get_mapper(f"{self.data_dir}/SWOT_swaths_488/hawaii_c488_p015.zarr")),
                xr.open_zarr(fs.get_mapper(f"{self.data_dir}/SWOT_swaths_488/hawaii_c488_p028.zarr"))
            ]
        else:
            
            _thread_local.swot_npy = np.load(
                f"{self.data_dir}/swot_npy_mask_4km.npy", mmap_mode="r") * 1
            _thread_local.swot_science_npy = [np.load(science_mask) 
                                              for science_mask in sorted(glob.glob(f"{self.data_dir}/example_science_phase/SWOT*.npy"))
                                             ]
            _thread_local.swot_science_cycle_npy = [np.load(science_mask) 
                                              for science_mask in sorted(glob.glob(f"{self.data_dir}/example_science_phase/cycle_SWOT*.npy"))
                                             ]
        # Load and process cloud catalog once
        _thread_local.cloud_catalog = xr.open_zarr(
            fs.get_mapper(f"{self.data_dir}/catalog.zarr")).compute()
        _thread_local.cloud_catalog_rho = _thread_local.cloud_catalog.where(
            _thread_local.cloud_catalog.rho >= self.cloud_rho, drop=True)
        # Pre-create cloud mask mappers for all patch IDs
        cloud_mapper_dir = f"{self.data_dir}/HRS_SST_tiles/agg_cloud_masks_zarr/"
        mapper_dict = {}
        unique_patch_ids = np.unique(_thread_local.cloud_catalog_rho.patch_id.values)
        for patch_id in unique_patch_ids:
            pid2 = str(int(patch_id)).zfill(3)
            mapper_dict[pid2] = fs.get_mapper(f"{cloud_mapper_dir}/{pid2}.zarr")
        _thread_local.cloud_mask_mapper = mapper_dict
        # Pre-compute cloud selections for faster sampling
        if self.preload_cloud_masks:
            self._precompute_cloud_mask_selections()
        # Cache for zarr datasets to avoid repeated opening
        _thread_local.zarr_cache = {}
        _thread_local.initialized = True

    def _precompute_cloud_mask_selections(self):
        """Pre-compute random selections for cloud masks"""
        cc = _thread_local.cloud_catalog_rho
        n_samples = min(self.cloud_cache_size, len(cc.i_time))
        # Pre-sample indices
        indices = np.random.choice(len(cc.i_time), size=n_samples, replace=False)
        _thread_local.cloud_selections = []
        for idx in indices:
            samp = cc.isel(i_time=idx)
            pid2 = str(int(samp.patch_id.values)).zfill(3)
            patch_timestep = int(samp.patch_timestep.values)
            _thread_local.cloud_selections.append((pid2, patch_timestep))

    @lru_cache(maxsize=128)
    def _get_cached_zarr_dataset(self, path, worker_id=None):
        """Cache zarr datasets to avoid repeated opening"""
        if worker_id is None:
            worker_id = threading.get_ident()
        cache_key = f"{path}_{worker_id}"
        if cache_key not in _thread_local.zarr_cache:
            mapper = _thread_local.fs.get_mapper(path)
            _thread_local.zarr_cache[cache_key] = xr.open_zarr(mapper, consolidated=True, chunks={})
        return _thread_local.zarr_cache[cache_key]

    def _load_fields(self, pid, fields, tkeys, mask_keys):
        """Optimized field loading with caching"""
        vars, masks = [], []
        for fld, tk, mask_key in zip(fields, tkeys, mask_keys):
            # Use cached zarr opening
            if "fullt" in fld:
                path = f"{self.data_dir}/{fld}/{pid}.zarr"
                d = self._get_cached_zarr_dataset(path, threading.get_ident())
                time_slice = slice(self.mid_timestep - self.dtime*(self.N_t//2),
                                      self.mid_timestep + self.dtime*(self.N_t//2 + self.N_t%2),
                                      self.dtime)
                d = d.isel(time=time_slice)
            else:
                path = f"{self.data_dir}/{fld}_allpatches.zarr"
                ds = self._get_cached_zarr_dataset(path, threading.get_ident())
                d = ds.loc[{"patch": int(pid)}]
                d = d.isel(time=slice(self.mid_timestep - self.N_t//2,
                                      self.mid_timestep + self.N_t//2 + self.N_t%2))
            if isinstance(d, xr.Dataset):
                d = next(iter(d.data_vars.values()))
            # Apply transform directly
            ten = self.transforms[tk](d)
            mask = self._mask_dispatch(mask_key, pid, ten.shape)
            vars.append(ten * mask)
            masks.append(mask)
        return torch.stack(vars, 1), torch.stack(masks, 1)

    def _load_masks_only(self, pid, var_tensor, mask_keys):
        """Generate masks without re-loading data.

        Parameters
        ----------
        pid : str
            Patch ID.
        var_tensor : Tensor
            Already-loaded variable tensor ``(N_t, C, H, W)`` — only used to
            infer the per-field spatial shape for ``_mask_dispatch``.
        mask_keys : list[str]
            One mask key per channel.

        Returns
        -------
        (var_tensor, masks) where masks has the same shape as var_tensor.
        """
        masks = []
        for ch_idx, mask_key in enumerate(mask_keys):
            shape = var_tensor[:, ch_idx, ...].shape  # (N_t, H, W)
            masks.append(self._mask_dispatch(mask_key, pid, shape))
        return var_tensor, torch.stack(masks, 1)

    def _mask_dispatch(self, mask_key, pid, shape):
        if (mask_key is None) or ("None" in mask_key):
            return torch.ones(shape, dtype=torch.float32)
        if "null_field" in mask_key.lower():
            return torch.zeros(shape, dtype=torch.float32)
        elif "swot" in mask_key.lower():
            sampling = "all"
            version = "random"
            if "calval" in mask_key.lower():
                version = "calval"
            if "science" in mask_key.lower():
                version = "science"
            if "cycle" in mask_key.lower():
                version = "cycle"
            if "central" in mask_key.lower():
                sampling = "central"
            if "random" in mask_key.lower():
                sampling = "random"
            if "nadir" in mask_key.lower():
                result = (self._get_swot_mask(pid, version, sampling) + 
                         self._get_nadir_mask(pid)) > 0
            else:
                result = self._get_swot_mask(pid, version, sampling)
        elif "nadir" in mask_key.lower():
            result = self._get_nadir_mask(pid)
        elif "cloud_rho" in mask_key.lower():
            result = self._get_cloud_rho_mask_optimized()
        else:
            raise ValueError(f"Unknown mask type: {mask_key}")
            
        if self.time_loading:
            print(f"[Timer] Mask '{mask_key}' generated")
        return result

    def _get_cloud_rho_mask_optimized(self):
        """Heavily optimized cloud mask generation"""
        if hasattr(_thread_local, 'cloud_selections') and _thread_local.cloud_selections:
            # Use pre-computed selections
            masks = []
            # Random choice from list of tuples - sample indices first
            selection_indices = np.random.choice(len(_thread_local.cloud_selections), size=self.N_t, replace=True)
            selections = [_thread_local.cloud_selections[i] for i in selection_indices]
            # Batch process masks
            current_pid = None
            current_ds = None
            for pid2, patch_timestep in selections:
                # Only reload dataset if patch changes
                if current_pid != pid2:
                    mapper = _thread_local.cloud_mask_mapper[pid2]
                    current_ds = xr.open_zarr(mapper, consolidated=True)
                    current_pid = pid2
                # Direct numpy operation for speed
                sst_data = current_ds.sst_filtered_q5.isel(time=patch_timestep).values
                mask = ~np.isnan(sst_data)
                masks.append(torch.from_numpy(mask.astype(np.float32)))
            return torch.stack(masks)
        else:
            # Fallback to original method but optimized
            return self._get_cloud_rho_mask_fallback()

    def _get_cloud_rho_mask_fallback(self):
        """Fallback method with some optimizations"""
        cc = _thread_local.cloud_catalog_rho
        masks = []
        # Pre-generate all random indices
        random_indices = np.random.randint(0, len(cc.i_time), size=self.N_t)
        for idx in random_indices:
            samp = cc.isel(i_time=idx)
            pid2 = str(int(samp.patch_id.values)).zfill(3)
            mapper = _thread_local.cloud_mask_mapper[pid2]
            d = xr.open_zarr(mapper, consolidated=True)
            m = ~np.isnan(d.sst_filtered_q5.isel(time=int(samp.patch_timestep)).values)
            masks.append(torch.from_numpy(m.astype(np.float32)))
        return torch.stack(masks)

    def _get_swot_mask(self, pid, version, sampling):
        """Optimized SWOT mask generation"""
        if self.regrid_SWOT:
            sw_corner, ne_corner = [-154.5, 35.3], [-147.5, 42.3]
            lat_max, lat_min, l_step, lon_i = 9000, 2000, 4, np.random.randint(5)
            lon = np.random.uniform(sw_corner[0], ne_corner[0])
            lat = np.random.uniform(sw_corner[1], ne_corner[1])
            ds = np.random.choice(_thread_local.swot_ds)
            m0 = interp_utils.grid_everything(
                _thread_local.swot_ds[0].ssha, lat=lat, lon=lon,
                n=self.N, L_x=self.L_x, L_y=self.L_y).values
            m1 = interp_utils.grid_everything(
                _thread_local.swot_ds[1].ssha, lat=lat, lon=lon,
                n=self.N, L_x=self.L_x, L_y=self.L_y).values
            m01 = np.stack([m0, m1])
        elif version == "cycle":
            # The science phase generally consists of 5-7 12-hrly passes moving East-West
            # pick one at random here for sampling / training
            m_science = _thread_local.swot_science_cycle_npy[np.random.randint(len(_thread_local.swot_science_cycle_npy))]
            mask = np.zeros([self.N_t] + list(m_science.shape)[-2:])
            center_mask, center_science, = len(mask)//2, len(m_science)//2
            if len(mask) >= len(m_science): 
                # Science fits within mask time length
                mask_0 = center_mask - (len(m_science)//2)
                mask_1 = mask_0 + len(m_science)
                mask[mask_0:mask_1] += (m_science > 0)
            else:  
                # The science phase time series is larger than the mask length,
                # Trim the science phase mask to fit the ssh mask
                m_science_0 = center_science - (len(mask) //2)
                m_science_1 = m_science_0 + len(mask)
                mask += (m_science[m_science_0:m_science_1] > 0)
            return torch.from_numpy(mask)
        elif version == "science":
            # The science phase generally consists of 5-7 12-hrly passes moving East-West
            # pick one at random here for sampling / training
            m_science = _thread_local.swot_science_npy[np.random.randint(len(_thread_local.swot_science_npy))]
            mask = np.zeros([self.N_t] + list(m_science.shape)[-2:])
            center_mask, center_science, = len(mask)//2, len(m_science)//2
            if len(mask) >= len(m_science): 
                # Science fits within mask time length
                mask_0 = center_mask - (len(m_science)//2)
                mask_1 = mask_0 + len(m_science)
                mask[mask_0:mask_1] += (m_science > 0)
            else:  
                # The science phase time series is larger than the mask length,
                # Trim the science phase mask to fit the ssh mask
                m_science_0 = center_science - (len(mask) //2)
                m_science_1 = m_science_0 + len(mask)
                mask += (m_science[m_science_0:m_science_1] > 0)
            return torch.from_numpy(mask)
        else:
            i_rand = np.random.randint(64, 225-64)
            j_rand = np.random.randint(128, 800-64)
            m01 = _thread_local.swot_npy[:, j_rand-64:j_rand+64, i_rand-64:i_rand+64]
        if np.random.randint(2) < 1:
            m01 = m01[::-1, ...]
        if sampling == "central":
            mask = np.zeros([self.N_t] + list(m01.shape)[-2:], dtype=np.float32)
            mask[self.N_t//2, :, :] = m01[0]
        elif sampling == "all":
            if self.N_t > 1:
                mask_broadcast = np.broadcast_to(m01, (self.N_t//2 + self.N_t%2, 2, 128, 128))
                mask = mask_broadcast.reshape(self.N_t + self.N_t%2, 128, 128)[:self.N_t]
            else:
                mask = m01[np.random.randint(1)]
                mask = mask.astype(np.float32)
        return torch.from_numpy(mask)

    def _get_nadir_mask(self, patch_ID, version="random", sample_time="1D"):
        """Optimized nadir mask generation"""
        try:
            rand_index = np.random.randint(422)
            path = f"{self.data_dir}/copernicus_nadir_SSH_daily/{rand_index:03}.zarr"
            da = self._get_cached_zarr_dataset(path, threading.get_ident())
            random_tile = da.sla_filtered
        except Exception as e:
            fallback_path = f"{self.data_dir}/copernicus_nadir_SSH_daily/002.zarr"
            da = self._get_cached_zarr_dataset(fallback_path, threading.get_ident())
            if self.time_loading:
                print(f"BAD MAPPER {rand_index} Exception: {e}")
            random_tile = da.sla_filtered
        # Temporal downsampling + slicing
        time_len = len(random_tile.time)
        mid = np.random.randint(self.N_t // 2, time_len - self.N_t // 2)
        sliced = random_tile.isel(time=slice(mid - self.N_t//2, mid + self.N_t//2 + self.N_t%2))        
        # Optimized mask computation
        mask_values = np.where(sliced.values > 0, 1.0, 0.0).astype(np.float32)
        return torch.from_numpy(mask_values)


