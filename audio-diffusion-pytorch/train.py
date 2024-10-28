from audio_diffusion_pytorch import DiffusionModel, UNetV0, VDiffusion, VSampler
import torch
model = DiffusionModel(
    net_t=UNetV0, # The model type used for diffusion (U-Net V0 in this case)
    in_channels=2, # U-Net: number of input/output (audio) channels
    channels=[8, 32, 64, 128, 256, 512, 512, 1024, 1024], # U-Net: channels at each layer
    factors=[1, 4, 4, 4, 2, 2, 2, 2, 2], # U-Net: downsampling and upsampling factors at each layer
    items=[1, 2, 2, 2, 2, 2, 2, 4, 4], # U-Net: number of repeating items at each layer
    attentions=[0, 0, 0, 0, 0, 1, 1, 1, 1], # U-Net: attention enabled/disabled at each layer
    attention_heads=8, # U-Net: number of attention heads per attention item
    attention_features=64, # U-Net: number of attention features per attention item
    diffusion_t=VDiffusion, # The diffusion method used
    sampler_t=VSampler, # The diffusion sampler used
)

# Train model with audio waveforms
audio = torch.randn(1, 2, 2**18) # [batch_size, in_channels, length]
loss = model(audio)
loss.backward()

# Turn noise into new audio sample with diffusion
noise = torch.randn(1, 2, 2**18) # [batch_size, in_channels, length]
sample = model.sample(noise, num_steps=10) # Suggested num_steps 10-100