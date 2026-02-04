"""
LoRA and Adapter modules for fine-tuning T3 model with emotion control.

This module provides LoRA (Low-Rank Adaptation) and Adapter layers
for efficient fine-tuning of the T3 model on emotion-labeled data.
"""

import torch
import torch.nn as nn
from typing import Optional


class LoRALayer(nn.Module):
    """
    LoRA (Low-Rank Adaptation) layer.
    
    Applies low-rank decomposition to linear layers for efficient fine-tuning.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 16,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        """
        Args:
            in_features: Input feature dimension
            out_features: Output feature dimension
            rank: LoRA rank (lower = fewer parameters)
            alpha: LoRA alpha scaling factor
            dropout: Dropout rate
        """
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # LoRA matrices
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.02)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: x @ A^T @ B^T * scaling
        
        Args:
            x: Input tensor (..., in_features)
            
        Returns:
            Output tensor (..., out_features)
        """
        # LoRA computation: x @ A^T @ B^T
        x_dropout = self.dropout(x)
        lora_output = (x_dropout @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return lora_output


class LoRALinear(nn.Module):
    """
    Linear layer with LoRA adaptation.
    
    Wraps a base linear layer and adds LoRA adaptation.
    """
    
    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int = 16,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        """
        Args:
            base_layer: Base linear layer to adapt
            rank: LoRA rank
            alpha: LoRA alpha scaling factor
            dropout: Dropout rate
        """
        super().__init__()
        self.base_layer = base_layer
        self.lora = LoRALayer(
            in_features=base_layer.in_features,
            out_features=base_layer.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: base output + LoRA output"""
        base_output = self.base_layer(x)
        lora_output = self.lora(x)
        return base_output + lora_output
    
    def merge_weights(self):
        """Merge LoRA weights into base layer (for inference)"""
        with torch.no_grad():
            merged_weight = (
                self.base_layer.weight.data +
                self.lora.lora_B @ self.lora.lora_A * self.lora.scaling
            )
            self.base_layer.weight.data = merged_weight


class AdapterLayer(nn.Module):
    """
    Adapter layer for fine-tuning.
    
    Adds a small bottleneck adapter to transformer layers.
    """
    
    def __init__(
        self,
        hidden_size: int,
        adapter_size: int = 64,
        activation: str = "gelu",
        dropout: float = 0.1,
    ):
        """
        Args:
            hidden_size: Hidden dimension of the model
            adapter_size: Adapter bottleneck dimension
            activation: Activation function
            dropout: Dropout rate
        """
        super().__init__()
        self.adapter_size = adapter_size
        
        # Down projection
        self.down_proj = nn.Linear(hidden_size, adapter_size)
        
        # Activation
        if activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        self.dropout = nn.Dropout(dropout)
        
        # Up projection
        self.up_proj = nn.Linear(adapter_size, hidden_size)
        
        # Initialize with small values
        nn.init.normal_(self.down_proj.weight, std=0.02)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.normal_(self.up_proj.weight, std=0.02)
        nn.init.zeros_(self.up_proj.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: x -> down -> activation -> dropout -> up
        
        Args:
            x: Input tensor (..., hidden_size)
            
        Returns:
            Output tensor (..., hidden_size)
        """
        # Adapter: down -> activation -> up
        down = self.down_proj(x)
        activated = self.activation(down)
        dropped = self.dropout(activated)
        up = self.up_proj(dropped)
        
        # Residual connection
        return x + up


def apply_lora_to_linear(
    module: nn.Module,
    target_modules: list = None,
    rank: int = 16,
    alpha: float = 16.0,
    dropout: float = 0.0,
) -> nn.Module:
    """
    Apply LoRA to linear layers in a module.
    
    Args:
        module: Module to apply LoRA to
        target_modules: List of module names to target (None = all Linear layers)
        rank: LoRA rank
        alpha: LoRA alpha
        dropout: Dropout rate
        
    Returns:
        Modified module with LoRA layers
    """
    if target_modules is None:
        target_modules = ["Linear"]
    
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and any(tm in name for tm in target_modules):
            # Replace with LoRALinear
            lora_linear = LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout)
            setattr(module, name, lora_linear)
        else:
            # Recursively apply to children
            apply_lora_to_linear(child, target_modules, rank, alpha, dropout)
    
    return module


def add_adapters_to_transformer(
    transformer: nn.Module,
    adapter_size: int = 64,
    adapter_layers: Optional[list] = None,
) -> nn.Module:
    """
    Add adapter layers to transformer blocks.

    Args:
        transformer: Transformer model (e.g., LlamaModel)
        adapter_size: Adapter bottleneck size
        adapter_layers: List of layer indices to add adapters to (None = all layers)

    Returns:
        Modified transformer with adapters
    """
    if not hasattr(transformer, "layers"):
        # Try common attribute names
        if hasattr(transformer, "model") and hasattr(transformer.model, "layers"):
            layers = transformer.model.layers
        else:
            raise ValueError("Could not find transformer layers")
    else:
        layers = transformer.layers

    if adapter_layers is None:
        adapter_layers = list(range(len(layers)))

    for i in adapter_layers:
        if i < len(layers):
            layer = layers[i]
            hidden_size = layer.self_attn.q_proj.out_features if hasattr(layer, "self_attn") else 1024

            # Add adapter after attention
            adapter = AdapterLayer(hidden_size, adapter_size=adapter_size)
            layer.adapter = adapter

    return transformer


def apply_lora_to_emotion_cross_attention(
    emotion_cross_attn: nn.Module,
    rank: int = 16,
    alpha: float = 16.0,
    dropout: float = 0.0,
) -> nn.Module:
    """
    Apply LoRA to emotion cross-attention module.

    Targets the following layers in EmotionCrossAttention:
    - emotion_proj: Projects emotion embedding to hidden size
    - cross_q_proj, cross_k_proj, cross_v_proj: Cross-attention Q/K/V projections
    - self_q_proj, self_k_proj, self_v_proj: Self-attention Q/K/V projections

    Args:
        emotion_cross_attn: EmotionCrossAttention module
        rank: LoRA rank
        alpha: LoRA alpha scaling factor
        dropout: Dropout rate

    Returns:
        Modified module with LoRA layers
    """
    target_layers = [
        "emotion_proj",
        "cross_q_proj", "cross_k_proj", "cross_v_proj", "cross_out_proj",
        "self_q_proj", "self_k_proj", "self_v_proj", "self_out_proj",
    ]

    for name in target_layers:
        if hasattr(emotion_cross_attn, name):
            layer = getattr(emotion_cross_attn, name)
            if isinstance(layer, nn.Linear):
                lora_linear = LoRALinear(layer, rank=rank, alpha=alpha, dropout=dropout)
                setattr(emotion_cross_attn, name, lora_linear)
                print(f"  Applied LoRA to emotion_cross_attn.{name}")

    return emotion_cross_attn


def get_emotion_trainable_params(model) -> list:
    """
    Get all trainable parameters related to emotion conditioning.

    This includes:
    - EmotionEmbeddings (64D learnable embeddings)
    - EmotionCrossAttention (cross-attention layers)
    - Any LoRA layers applied to emotion modules

    Args:
        model: ChatterboxMultilingualTTS model

    Returns:
        List of trainable parameters
    """
    trainable_params = []

    # Emotion embeddings
    if hasattr(model, "emotion_embeddings"):
        for param in model.emotion_embeddings.parameters():
            trainable_params.append(param)

    # Emotion cross-attention in T3CondEnc
    if hasattr(model, "t3") and hasattr(model.t3, "cond_enc"):
        cond_enc = model.t3.cond_enc
        if hasattr(cond_enc, "emotion_cross_attn"):
            for param in cond_enc.emotion_cross_attn.parameters():
                trainable_params.append(param)

    return trainable_params

