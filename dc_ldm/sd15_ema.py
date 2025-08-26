import torch
import torch.nn as nn
from contextlib import contextmanager
from typing import Dict, Iterator


class SD15EMA:
    """
    SD1.5专用的EMA实现，避免LitEma的参数状态冲突问题
    """
    
    def __init__(self, model: nn.Module, decay: float = 0.9999, device=None):
        """
        初始化EMA
        Args:
            model: 需要应用EMA的模型
            decay: EMA衰减率
            device: 设备
        """
        self.decay = decay
        self.device = device or next(model.parameters()).device
        self.num_updates = 0
        
        # 存储EMA权重，使用参数的完整路径作为key
        self.shadow_params = {}
        
        # 初始化EMA权重
        self._initialize_shadow_params(model)
        
    def _initialize_shadow_params(self, model: nn.Module):
        """初始化影子参数"""
        for name, param in model.named_parameters():
            # 只为需要梯度的参数创建EMA
            if param.requires_grad:
                self.shadow_params[name] = param.data.clone().detach().to(self.device)
    
    def update(self, model: nn.Module):
        """更新EMA权重"""
        self.num_updates += 1
        
        # 动态调整衰减率（前期更快更新）
        decay = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))
        one_minus_decay = 1.0 - decay
        
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad and name in self.shadow_params:
                    # EMA更新: shadow = decay * shadow + (1-decay) * param
                    self.shadow_params[name].mul_(decay).add_(param.data, alpha=one_minus_decay)
    
    def copy_to(self, model: nn.Module):
        """将EMA权重复制到模型"""
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in self.shadow_params:
                    param.data.copy_(self.shadow_params[name])
    
    def store(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        """存储当前模型参数"""
        stored_params = {}
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in self.shadow_params:
                    stored_params[name] = param.data.clone()
        return stored_params
    
    def restore(self, model: nn.Module, stored_params: Dict[str, torch.Tensor]):
        """恢复存储的模型参数"""
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in stored_params:
                    param.data.copy_(stored_params[name])
    
    @contextmanager
    def average_parameters(self, model: nn.Module):
        """上下文管理器：临时使用EMA权重"""
        # 存储当前参数
        stored_params = self.store(model)
        
        # 切换到EMA权重
        self.copy_to(model)
        
        try:
            yield
        finally:
            # 恢复原始参数
            self.restore(model, stored_params)
    
    def state_dict(self) -> Dict:
        """获取EMA状态字典"""
        return {
            'decay': self.decay,
            'num_updates': self.num_updates,
            'shadow_params': self.shadow_params
        }
    
    def load_state_dict(self, state_dict: Dict):
        """加载EMA状态字典"""
        self.decay = state_dict['decay']
        self.num_updates = state_dict['num_updates']
        self.shadow_params = state_dict['shadow_params']


class MultiModelEMA:
    """
    多模型EMA管理器，用于同时管理UNet和条件模型的EMA
    """
    
    def __init__(self, models: Dict[str, nn.Module], decay: float = 0.9999, device=None):
        """
        初始化多模型EMA
        Args:
            models: 模型字典 {'unet': unet_model, 'cond': cond_model}
            decay: EMA衰减率
            device: 设备
        """
        self.ema_models = {}
        for name, model in models.items():
            self.ema_models[name] = SD15EMA(model, decay, device)
    
    def update(self, models: Dict[str, nn.Module]):
        """更新所有模型的EMA"""
        for name, model in models.items():
            if name in self.ema_models:
                self.ema_models[name].update(model)
    
    def copy_to(self, models: Dict[str, nn.Module]):
        """将EMA权重复制到所有模型"""
        for name, model in models.items():
            if name in self.ema_models:
                self.ema_models[name].copy_to(model)
    
    def store(self, models: Dict[str, nn.Module]) -> Dict[str, Dict[str, torch.Tensor]]:
        """存储所有模型的当前参数"""
        stored_params = {}
        for name, model in models.items():
            if name in self.ema_models:
                stored_params[name] = self.ema_models[name].store(model)
        return stored_params
    
    def restore(self, models: Dict[str, nn.Module], stored_params: Dict[str, Dict[str, torch.Tensor]]):
        """恢复所有模型的存储参数"""
        for name, model in models.items():
            if name in self.ema_models and name in stored_params:
                self.ema_models[name].restore(model, stored_params[name])
    
    @contextmanager
    def average_parameters(self, models: Dict[str, nn.Module]):
        """上下文管理器：临时使用所有模型的EMA权重"""
        # 存储当前参数
        stored_params = self.store(models)
        
        # 切换到EMA权重
        self.copy_to(models)
        
        try:
            yield
        finally:
            # 恢复原始参数
            self.restore(models, stored_params)
    
    def state_dict(self) -> Dict:
        """获取所有EMA的状态字典"""
        return {name: ema.state_dict() for name, ema in self.ema_models.items()}
    
    def load_state_dict(self, state_dict: Dict):
        """加载所有EMA的状态字典"""
        for name, ema_state in state_dict.items():
            if name in self.ema_models:
                self.ema_models[name].load_state_dict(ema_state)