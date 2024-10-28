from main import module_base
model = module_base.(
    checkpoint_path='logs/ckpts/2024-10-25-15-24-48/epoch=10415-valid_loss=0.003.ckpt',
    learning_rate=1e-4,
    beta1=0.9,
    beta2=0.99,
    in_channels=1,
    patch_size=16,
)
model.eval()

# Generate a sample
import torch
noise = torch.randn(1, 1, 16, 16)
with torch.no_grad():
    sample = model(noise, num_steps=100)

# Save the sample
import torchaudio
torchaudio.save('sample.wav', sample.squeeze(0), 16000)
