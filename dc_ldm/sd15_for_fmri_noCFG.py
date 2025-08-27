import numpy as np
import wandb
import torch
import torch.nn as nn
import os
from contextlib import contextmanager
from einops import rearrange, repeat
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import torch.nn.functional as F
from sc_mbm.mae_for_fmri import fmri_encoder

# Import Stable Diffusion 1.5 components
from diffusers import StableDiffusionPipeline, DDIMScheduler
import pytorch_lightning as pl
from PIL import Image
import os

def create_model_from_config(config, num_voxels, global_pool):
    model = fmri_encoder(num_voxels=num_voxels, patch_size=config.patch_size, embed_dim=config.embed_dim,
                depth=config.depth, num_heads=config.num_heads, mlp_ratio=config.mlp_ratio, global_pool=global_pool) 
    return model

class cond_stage_model(nn.Module):
    """
    fMRI encoder and connector for SD1.5.
    """
    def __init__(self, metafile, num_voxels, cond_dim=768, global_pool=True): 
        super().__init__()
        # prepare pretrained fmri mae 
        model = create_model_from_config(metafile['config'], num_voxels, global_pool)
        model.load_checkpoint(metafile['model'])
        self.mae = model
        self.fmri_seq_len = model.num_patches
        self.fmri_latent_dim = model.embed_dim
        
        if global_pool == False:
            self.channel_mapper = nn.Sequential(
                nn.Conv1d(self.fmri_seq_len, self.fmri_seq_len // 2, 1, bias=True),
                nn.Conv1d(self.fmri_seq_len // 2, 77, 1, bias=True)  
            )
        
        # Map to cross attention dimension
        self.dim_mapper = nn.Linear(self.fmri_latent_dim, cond_dim, bias=True)
        self.global_pool = global_pool

    def forward(self, x):
        # n, c, w = x.shape
        latent_crossattn = self.mae(x)
        if self.global_pool == False:
            latent_crossattn = self.channel_mapper(latent_crossattn)
        latent_crossattn = self.dim_mapper(latent_crossattn)
        out = latent_crossattn
        return out

class fSD15(pl.LightningModule):
    """
    Stable Diffusion 1.5 adapter for fMRI-conditioned image generation.
    """
    
    def __init__(self, metafile, num_voxels, device=torch.device('cpu'),
                 model_id="runwayml/stable-diffusion-v1-5",
                 logger=None, ddim_steps=30, global_pool=True, use_time_cond=True,
                 cfg_scale: float = 3.0):
        super().__init__()
        
        # Store initialization parameters
        self.device_target = device
        self.ddim_steps = ddim_steps
        self.global_pool = global_pool
        self.use_time_cond = use_time_cond
        self.metafile = metafile
        self.model_id = model_id
        # Classifier-Free Guidance scale for generation (1.0 disables CFG)
        self.guidance_scale = cfg_scale
        
        # Initialize model components
        self._init_sd15_components()
        self._init_fmri_conditioning(metafile, num_voxels)
        self._init_training_parameters()
        self._init_ema(device)
        
        # Move to target device
        self.to(device)
        
        # Setup logging
        if logger is not None:
            logger.watch(self, log="all", log_graph=False)
            
        print(f"{self.__class__.__name__} initialized successfully")
    
    def _init_sd15_components(self):
        """Initialize Stable Diffusion 1.5 components with pipeline"""

        
        # Load base pipeline 
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch.float32,
            safety_checker=None,
            requires_safety_checker=False,
            use_safetensors=True if "safetensors" in self.model_id else False
        )
        self.pipeline.set_progress_bar_config(disable=True)
        
        # Extract individual components
        self.unet = self.pipeline.unet
        self.vae = self.pipeline.vae
        self.tokenizer = self.pipeline.tokenizer
        self.text_encoder = None  # Will be replaced with fMRI encoder
        
        # Configure scheduler for training (DDIM for consistency)
        self.scheduler = DDIMScheduler.from_config(self.pipeline.scheduler.config)
        
        # Store model configuration
        self.model_dtype = torch.float32
        self.cross_attention_dim = self.unet.config.cross_attention_dim
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        
        
        print(f"SD1.5 components loaded (cross_attention_dim: {self.cross_attention_dim}, vae_scale_factor: {self.vae_scale_factor})")
    
    def _init_fmri_conditioning(self, metafile, num_voxels):
        """Initialize fMRI conditioning components"""
        print("Initializing fMRI conditioning...")
        
        # Create fMRI condition stage model
        self.cond_stage_model = cond_stage_model(
            metafile, num_voxels, self.cross_attention_dim  , global_pool=self.global_pool
        )
        
        # Ensure consistent dtype
        self.cond_stage_model = self.cond_stage_model.to(dtype=self.model_dtype)
        self.cond_stage_trainable = True
        
    
    def _configure_pipeline_for_mode(self, mode="generation"):
        """Configure pipeline for specific mode (training/generation)"""
        if mode == "generation":
            # Apply memory optimizations for generation
            self.pipeline.enable_attention_slicing()
            self.pipeline.enable_vae_slicing()
        elif mode == "training":
            # Disable optimizations that might interfere with training
            self.pipeline.disable_attention_slicing()
            self.pipeline.disable_vae_slicing()
        
        # Ensure pipeline is on correct device
        self.pipeline = self.pipeline.to(self.device)
        
        return self.pipeline
    

    def _init_training_parameters(self):
        """Initialize training-related parameters"""
        # Training configuration
        self.train_cond_stage_only = False
        self.eval_avg = True
        self.learning_rate = 5.3e-5  # Following official SD training
        self.validation_count = 0
        # Classifier-Free Guidance training: probability to drop conditioning
        self.cond_drop_prob = 0.0
        
        # Loss weights (following LDM approach)
        self.l_simple_weight = 1.0
        self.original_elbo_weight = 0.0  # Match original LDM default
        self.learn_logvar = False
        self.logvar_init = 0.0
    
        
        # Initialize VLB weights 
        self._init_vlb_weights()
        
        print("Training parameters initialized")
    
    def _init_vlb_weights(self):
        """Initialize VLB weights for loss calculation"""
        num_timesteps = self.scheduler.config.num_train_timesteps
        betas = torch.linspace(
            self.scheduler.config.beta_start, 
            self.scheduler.config.beta_end, 
            num_timesteps
        )
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        # Calculate VLB weights (following original DDPM paper)
        lvlb_weights = betas ** 2 / (2 * (1 - alphas) * (1 - alphas_cumprod))
        lvlb_weights[0] = lvlb_weights[1]  # Fix first timestep
        self.register_buffer('lvlb_weights', lvlb_weights, persistent=False)
        
        # Learnable log variance for improved loss weighting
        if self.learn_logvar:
            self.logvar = nn.Parameter(
                torch.full((num_timesteps,), self.logvar_init), 
                requires_grad=True
            )
        else:
            self.register_buffer(
                'logvar', 
                torch.full((num_timesteps,), self.logvar_init)
            )
    
    def _init_ema(self, device):
        """Initialize EMA if enabled"""
        # EMA settings (following original LDM)
        self.use_ema = True
        self.ema_decay = 0.9999
        
        if self.use_ema:
            from dc_ldm.sd15_ema import MultiModelEMA
            models_dict = {
                'unet': self.unet,
                'cond': self.cond_stage_model
            }
            self.model_ema = MultiModelEMA(models_dict, decay=self.ema_decay, device=device)
            print(f"EMA initialized (decay={self.ema_decay})")
        else:
            print("EMA disabled")
    
    @contextmanager
    def ema_scope(self, context=None):
        """EMA context manager (following original LDM)"""
        if self.use_ema:
            models_dict = {
                'unet': self.unet,
                'cond': self.cond_stage_model
            }
            if context is not None:
                print(f"{context}: Switched to EMA weights")
            
            with self.model_ema.average_parameters(models_dict):
                yield None
        else:
            yield None
    
    def freeze_vae(self):
        """Freeze VAE parameters"""
        for param in self.vae.parameters():
            param.requires_grad = False
    
    def unfreeze_vae(self):
        """Unfreeze VAE parameters"""  
        for param in self.vae.parameters():
            param.requires_grad = True
    
    def freeze_unet(self):
        """Freeze UNet parameters"""
        for param in self.unet.parameters():
            param.requires_grad = False
    
    def unfreeze_unet(self):
        """Unfreeze UNet parameters"""
        for param in self.unet.parameters():
            param.requires_grad = True
    
    def freeze_first_stage(self):
        """Freeze VAE (equivalent to original freeze_first_stage)"""
        self.freeze_vae()
    
    def unfreeze_first_stage(self):
        """Unfreeze VAE (equivalent to original unfreeze_first_stage)"""
        self.unfreeze_vae()
    
    def freeze_whole_model(self):
        """Freeze all model parameters"""
        for param in self.parameters():
            param.requires_grad = False
    
    def unfreeze_whole_model(self):
        """Unfreeze all model parameters"""
        for param in self.parameters():
            param.requires_grad = True
    
    def get_learned_conditioning(self, fmri_input):
        """Get fMRI conditioning for cross-attention"""
        return self.cond_stage_model(fmri_input)
    
    def encode_first_stage(self, x):
        """Encode image to latent space using VAE"""
        # Ensure input has the same dtype as model
        x = x.to(dtype=self.model_dtype)
        # Ensure correct channel format: [batch, channels, height, width]
        if x.shape[-1] == 3:  # if format is [batch, height, width, channels]
            x = rearrange(x, 'b h w c -> b c h w')
        return self.vae.encode(x).latent_dist
    
    def decode_first_stage(self, z):
        """Decode latent to image using VAE"""
        z = z / self.vae.config.scaling_factor
        return self.vae.decode(z).sample
    
    def forward(self, x_start, t, fmri_condition):
        """Forward pass for training (following official Diffusers training approach)"""
        # Encode image to latent space using unified pipeline VAE
        with torch.no_grad():
            # Convert to tensor if needed and ensure correct format
            if isinstance(x_start, np.ndarray):
                x_start = torch.tensor(x_start, dtype=self.model_dtype, device=self.device)
            x_start = x_start.to(dtype=self.model_dtype, device=self.device)
            
            # Ensure correct channel format: [batch, channels, height, width]
            if len(x_start.shape) == 4 and x_start.shape[-1] == 3:
                x_start = rearrange(x_start, 'b h w c -> b c h w')
            
            # Encode to latent space using unified VAE
            posterior = self.vae.encode(x_start).latent_dist
            latents = posterior.sample() * self.vae.config.scaling_factor
        
        # Sample noise (following official approach)
        noise = torch.randn_like(latents, device=latents.device, dtype=latents.dtype)
        
        # Add noise to latents (forward diffusion process)
        noisy_latents = self.scheduler.add_noise(latents, noise, t)
        
        # Get fMRI conditioning (ensure correct dtype)
        fmri_condition = fmri_condition.to(dtype=self.model_dtype, device=self.device)
        encoder_hidden_states = self.get_learned_conditioning(fmri_condition)
        encoder_hidden_states = encoder_hidden_states.to(dtype=self.model_dtype)

        # Classifier-Free Guidance style conditioning dropout during training
        if getattr(self, 'cond_drop_prob', 0.0) > 0.0:
            batch_size = encoder_hidden_states.shape[0]
            drop_mask = torch.bernoulli(
                torch.full((batch_size, 1, 1), self.cond_drop_prob, device=encoder_hidden_states.device)
            )
            encoder_hidden_states = encoder_hidden_states * (1.0 - drop_mask)
        
        # Predict noise using UNet 
        model_pred = self.unet(
            noisy_latents,
            t,
            encoder_hidden_states=encoder_hidden_states,
            return_dict=False
        )[0]
        
        return model_pred, noise
    
    def training_step(self, batch, batch_idx):
        """Training step following official Diffusers training approach"""
        # Get batch data
        images = batch['image']
        fmri_data = batch['fmri']
        
        # Ensure correct data types and device placement
        images = images.to(dtype=self.model_dtype, device=self.device)
        fmri_data = fmri_data.to(dtype=self.model_dtype, device=self.device)
        
        # Sample random timesteps for each image (following official approach)
        batch_size = images.shape[0]
        timesteps = torch.randint(
            0, self.scheduler.config.num_train_timesteps, 
            (batch_size,), device=self.device, dtype=torch.int64
        )
        
        # Forward pass with improved error handling
        try:
            model_pred, target = self.forward(images, timesteps, fmri_data)
        except Exception as e:
            self.log('train/forward_error', 1.0)
            raise e
        
        # Calculate loss following official Diffusers approach
        loss_dict = {}
        
        # Primary loss: MSE between predicted and target noise
        if self.scheduler.config.prediction_type == "epsilon":
            # Predicting noise (standard approach)
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        elif self.scheduler.config.prediction_type == "v_prediction":
            # Predicting v-parameterization
            target = self.scheduler.get_velocity(images, target, timesteps)
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        else:
            raise ValueError(f"Unknown prediction type {self.scheduler.config.prediction_type}")
        
        # Store primary loss
        loss_dict['train/loss'] = loss
        loss_dict['train/loss_simple'] = loss  # For compatibility with LDM logging
        
        if self.original_elbo_weight > 0:
            # Calculate per-sample loss for VLB weighting
            loss_per_sample = F.mse_loss(model_pred.float(), target.float(), reduction='none')
            loss_per_sample = loss_per_sample.mean(dim=list(range(1, len(loss_per_sample.shape))))
            
            # Apply VLB weights
            loss_vlb = (self.lvlb_weights[timesteps] * loss_per_sample).mean()
            loss_dict['train/loss_vlb'] = loss_vlb
            
            # Combine losses
            loss = loss + self.original_elbo_weight * loss_vlb
            loss_dict['train/loss'] = loss
        
        # Log metrics with proper synchronization
        self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        
        # Update EMA after successful forward pass
        if self.use_ema:
            models_dict = {
                'unet': self.unet,
                'cond': self.cond_stage_model
            }
            self.model_ema.update(models_dict)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step with image generation and metrics"""

        if batch_idx != 0:
            return
        
        # Initialize validation counter if not exists
        if not hasattr(self, 'validation_count'):
            self.validation_count = 0
    
        #if self.validation_count % 2 == 0 and self.trainer.current_epoch != 0:
        if self.validation_count >= 25 and (self.validation_count - 25) % 5 == 0:
            # Full validation with EMA weights
            with self.ema_scope("Validation"):
                self.full_validation(batch)
                # Generate train dataset grid every full validation
                self.generate_train_grid()
        else:
            # Quick validation: only compute validation loss
            images = batch['image']
            fmri_data = batch['fmri']
            
            images = images.to(dtype=self.model_dtype)
            fmri_data = fmri_data.to(dtype=self.model_dtype)
            
            bsz = images.shape[0]
            timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, (bsz,), device=self.device)
            
            noise_pred, noise = self.forward(images, timesteps, fmri_data)
            val_loss = F.mse_loss(noise_pred, noise)
            
            self.log('val/loss_simple', val_loss, prog_bar=True)
        
        self.validation_count += 1
    
    
    @torch.no_grad()
    def full_validation(self, batch):
        """Full validation with image generation and evaluation metrics"""
        print("Running full validation with image generation...")
        
        # Generate images from validation batch
        grid, all_samples = self.generate_validation_images(batch, num_samples=4,limit=113)
        
        # Calculate evaluation metrics
        metric_values, metric_names = self.get_eval_metric(all_samples, avg=self.eval_avg)
        
        # Log images to wandb and save locally
       
        grid_img = Image.fromarray(grid.astype(np.uint8))
        
        # Save grid image locally use wandb file name
        save_dir = os.path.join('results', 'validation_grids')
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'{self.logger.experiment.name}_epoch_{self.trainer.current_epoch:04d}.png')
        grid_img.save(save_path)
        
        # Log to wandb if logger exists
        if hasattr(self.logger, 'experiment'):
            import wandb
            self.logger.experiment.log({
                'val/generated_images': wandb.Image(grid_img),
                'epoch': self.trainer.current_epoch
            })
        
        # Log metrics
        metric_dict = {f'val/{name}': value for name, value in zip(metric_names, metric_values)}
        self.log_dict(metric_dict, prog_bar=True)
        
        print(f"Validation metrics: {metric_dict}")
    
    @torch.no_grad()
    def generate_validation_images(self, batch, num_samples=3, limit=5):
        """Generate images for validation"""
        
        # Prepare validation data 
        val_data = []
        images = batch['image'][:limit]  # Limit number of samples
        fmri_data = batch['fmri'][:limit]
        
        for i in range(len(images)):
            # Convert image format to match expected format
            img = images[i]
            if img.shape[0] == 3:  # if [C, H, W]
                img = rearrange(img, 'c h w -> h w c')
            
            fmri = fmri_data[i]
            
            val_data.append({
                'image': img.cpu().numpy(),
                'fmri': fmri.cpu().numpy()
            })
        
        # Use the existing generate method
        grid, samples = self.generate(val_data, num_samples, self.ddim_steps, HW=None, limit=limit)
        
        return grid, samples
    
    @torch.no_grad()
    def generate_train_grid(self):
        """Generate 10x5 grid from first 10 samples of train dataset for wandb logging"""
        try:
            # Get train dataloader from trainer
            if not hasattr(self.trainer, 'train_dataloader') or self.trainer.train_dataloader is None:
                print("Train dataloader not available, skipping train dataset grid generation")
                return
            
            train_loader = self.trainer.train_dataloader
            
            # Get first 10 samples from train dataset
            train_data = []
            sample_count = 0
            
            for batch in train_loader:
                images = batch['image']
                fmri_data = batch['fmri']
                
                batch_size = images.shape[0]
                for i in range(batch_size):
                    if sample_count >= 10:  # Only need first 10 samples
                        break
                    
                    # Convert image format to match expected format
                    img = images[i]
                    if img.shape[0] == 3:  # if [C, H, W]
                        img = rearrange(img, 'c h w -> h w c')
                    
                    fmri = fmri_data[i]
                    
                    train_data.append({
                        'image': img.cpu().numpy(),
                        'fmri': fmri.cpu().numpy()
                    })
                    sample_count += 1
                
                if sample_count >= 10:
                    break
            
            if len(train_data) == 0:
                print("No train data available for grid generation")
                return
            
            print(f"Generating train dataset grid with {len(train_data)} samples...")
            
            # Generate 5 samples per fMRI input (10x5 grid: 10 rows, 5 columns)
            grid, samples = self.generate(train_data, num_samples=4, ddim_steps=self.ddim_steps, HW=None, limit=10)
            
            # Convert grid to PIL Image
            grid_img = Image.fromarray(grid.astype(np.uint8))
            
            # Save grid image locally
            save_dir = os.path.join('results', 'train_dataset_grids')
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f'train_grid_epoch_{self.trainer.current_epoch:04d}.png')
            grid_img.save(save_path)
            
            # Log to wandb if logger exists
            if hasattr(self.logger, 'experiment'):
                import wandb
                self.logger.experiment.log({
                    'train/generated_grid': wandb.Image(grid_img, caption=f"Train Dataset Grid - Epoch {self.trainer.current_epoch}"),
                    'epoch': self.trainer.current_epoch
                })
                print(f"Train dataset grid logged to wandb for epoch {self.trainer.current_epoch}")
            
        except Exception as e:
            print(f"Error generating train dataset grid: {e}")
            # Don't raise the error to avoid interrupting training
    
    def get_eval_metric(self, samples, avg=True):
        """Calculate evaluation metrics - same as in stageB_ldm_finetune.py"""
        from eval_metrics import get_similarity_metric
        
        metric_list = ['mse', 'pcc', 'ssim', 'psm']
        res_list = []
        
        gt_images = [img[0] for img in samples]
        gt_images = rearrange(np.stack(gt_images), 'n c h w -> n h w c')
        samples_to_run = np.arange(1, len(samples[0])) if avg else [1]
        
        for m in metric_list:
            res_part = []
            for s in samples_to_run:
                pred_images = [img[s] for img in samples]
                pred_images = rearrange(np.stack(pred_images), 'n c h w -> n h w c')
                res = get_similarity_metric(pred_images, gt_images, method='pair-wise', metric_name=m)
                res_part.append(np.mean(res))
            res_list.append(np.mean(res_part))
        
        # Add classification metric
        res_part = []
        for s in samples_to_run:
            pred_images = [img[s] for img in samples]
            pred_images = rearrange(np.stack(pred_images), 'n c h w -> n h w c')
            res = get_similarity_metric(pred_images, gt_images, 'class', None, 
                            n_way=50, num_trials=1000, top_k=1, device='cuda')
            res_part.append(np.mean(res))
           
          
        
        res_list.append(np.mean(res_part))
        res_list.append(np.max(res_part) if res_part else 0.0)
        metric_list.append('top-1-class')
        metric_list.append('top-1-class (max)')
        
        return res_list, metric_list
    
    def configure_optimizers(self):
        """Configure optimizer following official Diffusers training approach"""
        # Determine which parameters to optimize
        params_to_optimize = []
        
        if self.train_cond_stage_only:
            print(f"{self.__class__.__name__}: Training conditional encoder and cross-attention layers only")
            
            # Add fMRI encoder parameters
            if self.cond_stage_trainable:
                params_to_optimize.extend(self.cond_stage_model.parameters())
                print(f"Added {sum(p.numel() for p in self.cond_stage_model.parameters())} conditional encoder parameters")
            
            # Add specific UNet parameters for conditioning
            unet_params = []
            conditioning_keywords = [
                'attn2',              # Cross-attention layers
                'norm2',              # Layer normalization for cross-attention
                'time_embedding',     # Time embedding layers
            ]
            
            for name, param in self.unet.named_parameters():
                if any(keyword in name for keyword in conditioning_keywords):
                    param.requires_grad = True
                    unet_params.append(param)
                else:
                    param.requires_grad = False
            
            params_to_optimize.extend(unet_params)
            print(f"Added {len(unet_params)} UNet conditioning parameters ({sum(p.numel() for p in unet_params)} total)")
            
        else:
            # Train all parameters (full fine-tuning)
            print(f"{self.__class__.__name__}: Training all parameters")
            
            if self.cond_stage_trainable:
                params_to_optimize.extend(self.cond_stage_model.parameters())
            
            # Add all UNet parameters
            params_to_optimize.extend(self.unet.parameters())
        
        # Filter out parameters that don't require gradients
        params_to_optimize = [p for p in params_to_optimize if p.requires_grad]
        total_params = sum(p.numel() for p in params_to_optimize)
        print(f"Total parameters to optimize: {total_params:,}")
        
        # Create optimizer following official approach
        optimizer = torch.optim.AdamW(
            params_to_optimize,
            lr=self.learning_rate,
            betas=(0.9, 0.999),  # Standard betas for stable diffusion training
            weight_decay=1e-2,   # Standard weight decay
            eps=1e-08,          # Standard epsilon
        )
        
        # Optional: Learning rate scheduler (following official approach)
        if hasattr(self, 'use_scheduler') and self.use_scheduler:
            from diffusers.optimization import get_cosine_schedule_with_warmup
            
            # Estimate total training steps (if not provided)
            if not hasattr(self, 'max_train_steps'):
                # This would typically be calculated based on dataset size and epochs
                self.max_train_steps = 10000  # Default fallback
            
            lr_scheduler = get_cosine_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=500,  # Standard warmup steps
                num_training_steps=self.max_train_steps,
            )
            
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": lr_scheduler,
                    "interval": "step",  # Update every step
                    "frequency": 1,
                }
            }
        
        return optimizer
    
    def finetune(self, trainer, dataset, test_dataset, batch_size, lr, output_path, config=None):
        """Fine-tuning interface compatible with original fLDM"""
        # Store config
        self.main_config = config
        self.output_path = output_path
        self.learning_rate = lr
        self.eval_avg = config.eval_avg if config else True
        
        # Setup training
        print('\n##### Fine-tuning SD1.5 with fMRI conditioning #####')
        
        # Create data loaders
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False, num_workers=4)
        
        # Configure training strategy
        self.unfreeze_whole_model()
        self.freeze_first_stage()  # Keep VAE frozen
        self.train_cond_stage_only = True
        
        # Start training
        trainer.fit(self, train_loader, val_dataloaders=val_loader)
        
        # Save checkpoint
        self.unfreeze_whole_model()
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': config,
            'state': torch.random.get_rng_state()
        }, os.path.join(output_path, 'checkpoint.pth'))
        
        print("Fine-tuning completed!")
    
    @torch.no_grad()
    def generate(self, fmri_embedding, num_samples, ddim_steps, HW=None, limit=None, state=None, cfg_scale: float = 3.0):
        """
        Generate images from fMRI embeddings using official SD1.5 pipeline approach
        Compatible interface with original fLDM.generate() but optimized for trained models
        """
        all_samples = []
        
        # Set random state if provided (following original LDM)
        if state is not None:
            torch.cuda.set_rng_state(state)
        
        # Determine image dimensions (following official SD1.5 standards)
        if HW is None:
            height, width = 256, 256  # Standard SD1.5 resolution
        else:
            height, width = HW[0], HW[1]
        
        print(f"Generating images at {height}x{width} resolution with {ddim_steps} steps")
        
        # Use EMA weights for generation (following original LDM best practice)
        with self.ema_scope("Generation"):
            # Configure unified pipeline for generation mode
            pipeline = self._configure_pipeline_for_mode("generation")
            
            # Set model to evaluation mode
            self.eval()
            
            for count, item in enumerate(fmri_embedding):
                if limit is not None and count >= limit:
                    break
                    
                # Prepare data with improved error handling
                fmri_data, gt_image = self.prepare_generation_inputs(item)
                
                print(f"Rendering {num_samples} examples for sample {count+1}/{len(fmri_embedding)}")
                
                # Resolve guidance scale and prepare (cond, uncond) if needed
                guidance_scale = self.guidance_scale 
                use_cfg = guidance_scale > 1.0
                if use_cfg:
                    encoder_hidden_states, uncond_encoder_hidden_states = self.prepare_fmri_conditioning(
                        fmri_data, num_samples, return_uncond=True
                    )
                else:
                    encoder_hidden_states = self.prepare_fmri_conditioning(fmri_data, num_samples)
                    uncond_encoder_hidden_states = None

                # Generate images using official pipeline approach
                generated_images = self.fmri_to_images(
                    pipeline, encoder_hidden_states, num_samples, ddim_steps, height, width,
                    guidance_scale=guidance_scale,
                    uncond_encoder_hidden_states=uncond_encoder_hidden_states
                )
                
                # Process and combine with ground truth
                processed_gt = self.process_ground_truth_image(gt_image, generated_images.shape[-2:])
                
                # Combine GT and generated images
                combined_samples = torch.cat([processed_gt.cpu(), generated_images.cpu()], dim=0)
                all_samples.append(combined_samples)
        
        # Create visualization grid with official styling
        grid_np, samples_np = self.create_visualization_grid(all_samples, num_samples)
        
        print(f"Generated {len(all_samples)} sets of images successfully!")
        return grid_np, samples_np
    

    
    def prepare_generation_inputs(self, item):
        """Prepare fMRI data and ground truth image for generation"""
        fmri_data = item['fmri']
        gt_image = item['image']
        
        # Handle ground truth image format conversion
        if isinstance(gt_image, np.ndarray):
            if len(gt_image.shape) == 3 and gt_image.shape[-1] == 3:  # [H, W, C]
                gt_image = rearrange(gt_image, 'h w c -> 1 c h w')
            gt_image = torch.tensor(gt_image, dtype=self.model_dtype, device=self.device)
        
        # Handle fMRI data conversion
        if isinstance(fmri_data, np.ndarray):
            fmri_data = torch.tensor(fmri_data, dtype=self.model_dtype, device=self.device)
        else:
            fmri_data = fmri_data.to(dtype=self.model_dtype, device=self.device)
        
        return fmri_data, gt_image
    
    def prepare_fmri_conditioning(self, fmri_data, num_samples, return_uncond: bool = False):
        """Prepare fMRI conditioning for batch generation.
        If return_uncond is True, also compute unconditional states by passing zeroed fMRI input through the encoder.
        """
        # Conditional branch
        fmri_batch = repeat(fmri_data, 'h w -> c h w', c=num_samples)
        encoder_hidden_states = self.get_learned_conditioning(fmri_batch)
        encoder_hidden_states = encoder_hidden_states.to(dtype=self.model_dtype)

        if not return_uncond:
            return encoder_hidden_states

        # Unconditional branch: zero fmri input then encode
        zeroed_fmri = torch.zeros_like(fmri_data)
        zeroed_batch = repeat(zeroed_fmri, 'h w -> c h w', c=num_samples)
        uncond_encoder_hidden_states = self.get_learned_conditioning(zeroed_batch)
        uncond_encoder_hidden_states = uncond_encoder_hidden_states.to(dtype=self.model_dtype)

        return encoder_hidden_states, uncond_encoder_hidden_states
    
    def process_ground_truth_image(self, gt_image, target_size):
        """Process ground truth image for comparison with generated images"""
        # Normalize to [0, 1] range (following official normalization)
        gt_image = torch.clamp((gt_image + 1.0) / 2.0, min=0.0, max=1.0)
        
        # Resize to match generated images if needed
        if gt_image.shape[-2:] != target_size:
            gt_image = F.interpolate(
                gt_image, size=target_size, 
                mode='bilinear', align_corners=False, antialias=True
            )
        
        return gt_image
    
    def create_visualization_grid(self, all_samples, num_samples):
        """Create visualization grid following official styling"""
        # Create grid for visualization (official approach)
        grid = torch.stack(all_samples, 0)
        grid = rearrange(grid, 'n b c h w -> (n b) c h w')
        grid = make_grid(grid, nrow=num_samples+1, padding=2, normalize=True, scale_each=False)
        
        # Convert to numpy format [H, W, C] in range [0, 255]
        grid_np = (255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()).astype(np.uint8)
        samples_np = (255. * torch.stack(all_samples, 0).cpu().numpy()).astype(np.uint8)
        
        return grid_np, samples_np
    
    def fmri_to_images(self, pipeline, encoder_hidden_states, num_samples, num_inference_steps, height, width,
                       guidance_scale: float = 3.0, uncond_encoder_hidden_states=None):
        """
        Generate images using official SD1.5 pipeline approach with fMRI conditioning
        Following the exact structure of StableDiffusionPipeline.__call__()
        """
        # Set scheduler timesteps (following official approach)
        pipeline.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = pipeline.scheduler.timesteps
        
        # Prepare latent variables (following official latent preparation)
        latent_channels = pipeline.unet.config.in_channels
        latent_height = height // self.vae_scale_factor
        latent_width = width // self.vae_scale_factor
        shape = (num_samples, latent_channels, latent_height, latent_width)
        
        # Initialize latents with proper scaling (following official approach)
        latents = torch.randn(shape, device=self.device, dtype=self.model_dtype)
        latents = latents * pipeline.scheduler.init_noise_sigma
        
        # Prepare unconditional conditioning for CFG if enabled
        use_cfg = guidance_scale is not None and guidance_scale > 1.0
        if use_cfg and uncond_encoder_hidden_states is None:
            # Fallback: if not provided, use zeros-like for robustness
            uncond_encoder_hidden_states = torch.zeros_like(encoder_hidden_states)

        # Denoising loop (following official pipeline denoising structure)
        for i, t in enumerate(timesteps):
            # Scale the model input (following official scaling)
            latent_model_input = pipeline.scheduler.scale_model_input(latents, t)

            # Classifier-Free Guidance: run unconditional and conditional branches
            if use_cfg:
                latent_in = torch.cat([latent_model_input, latent_model_input], dim=0)
                context_in = torch.cat([uncond_encoder_hidden_states, encoder_hidden_states], dim=0)
                with torch.no_grad():
                    noise_pred_uncond, noise_pred_text = pipeline.unet(
                        latent_in, t, encoder_hidden_states=context_in, return_dict=False
                    )[0].chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            else:
                # Predict noise residual using UNet with fMRI conditioning
                with torch.no_grad():
                    noise_pred = pipeline.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=encoder_hidden_states,
                        return_dict=False,
                    )[0]
            
            # Compute the previous noisy sample (following official scheduler step)
            latents = pipeline.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        
        # Decode latents to images (following official VAE decoding)
        latents = 1 / pipeline.vae.config.scaling_factor * latents
        
        # Use VAE slicing for memory efficiency if enabled
        if hasattr(pipeline, '_slice_size') and pipeline._slice_size is not None:
            images = []
            for i in range(0, latents.shape[0], pipeline._slice_size):
                latent_slice = latents[i:i + pipeline._slice_size]
                image_slice = pipeline.vae.decode(latent_slice, return_dict=False)[0]
                images.append(image_slice)
            images = torch.cat(images, dim=0)
        else:
            images = pipeline.vae.decode(latents, return_dict=False)[0]
        
        # Post-processing (following official normalization)
        images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)
        
        return images
    
    def load_trained_weights(self, checkpoint_path):
        """
        Load weights from trained checkpoint
        This method allows loading your trained model for generation
        """
        print(f"Loading trained weights from: {checkpoint_path}")
        
        try:
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            
            if 'model_state_dict' in checkpoint:
                # Load the state dict
                self.load_state_dict(checkpoint['model_state_dict'], strict=False)
                print("Successfully loaded trained model weights!")
                
                # Update EMA if available
                if self.use_ema:
                    models_dict = {
                        'unet': self.unet,
                        'cond': self.cond_stage_model
                    }
                    self.model_ema.update(models_dict)
                    print("EMA weights updated!")
                
            else:
                print("No 'model_state_dict' found in checkpoint")
                
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            raise e
    
    @classmethod 
    def from_trained_checkpoint(cls, checkpoint_path, metafile, num_voxels, device=torch.device('cpu'), **kwargs):
        """
        Create fSD15 instance from trained checkpoint
        Usage: model = fSD15.from_trained_checkpoint('path/to/checkpoint.pth', metafile, num_voxels)
        """
        # Create instance
        instance = cls(metafile, num_voxels, device=device, **kwargs)
        
        # Load trained weights
        instance.load_trained_weights(checkpoint_path)
        
        return instance