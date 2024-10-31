from main import module_base
from audio_diffusion_pytorch import AudioDiffusionModel, UniformDistribution

# First create the AudioDiffusionModel instance with your config parameters
audio_diffusion_model = AudioDiffusionModel(
    in_channels=2,  # from your channels config
    channels=128,
    patch_size=16,
    resnet_groups=8,
    kernel_multiplier_downsample=2,
    multipliers=[1, 2, 4, 4, 4, 4, 4],
    factors=[4, 4, 4, 2, 2, 2],
    num_blocks=[2, 2, 2, 2, 2, 2],
    attentions=[0, 0, 0, 1, 1, 1, 1],
    attention_heads=8,
    attention_features=64,
    attention_multiplier=2,
    use_nearest_upsample=False,
    use_skip_scale=True,
    diffusion_sigma_distribution=UniformDistribution()
)

# Then load the checkpoint with the audio diffusion model instance
model = module_base.Model.load_from_checkpoint(
    checkpoint_path='logs/ckpts/2024-10-25-15-24-48/epoch=10415-valid_loss=0.003.ckpt',
    lr=1e-4,
    lr_beta1=0.95,
    lr_beta2=0.999,
    lr_eps=1e-6,
    lr_weight_decay=1e-3,
    model=audio_diffusion_model,
    ema_beta=0.9999,
    ema_power=0.7
)
# Generate a sample
import torch
from audio_diffusion_pytorch import VSampler, LinearSchedule
import torchaudio
import torch
import torchaudio
from audio_diffusion_pytorch import VSampler, LinearSchedule



@torch.no_grad()
def generate_audio_with_params(
        model,
        num_samples=4,
        length=65536,  # Keep this fixed to match training
        num_steps=3,
        channels=2,
        sampling_rate=48000,
        device=None
):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)
    model.eval()
    diffusion_model = model.model_ema.ema_model.to(device)

    # Ensure length is divisible by model's patch_size (16 in your case)
    patch_size = 16
    length = (length // patch_size) * patch_size

    # Create noise input
    noise = torch.randn((num_samples, channels, length), device=device)

    # Setup sampler and schedule
    sampler = VSampler()
    schedule = LinearSchedule()

    # Generate samples
    samples = diffusion_model.sample(
        noise=noise,
        sampler=sampler,
        sigma_schedule=schedule,
        num_steps=num_steps
    )

    return samples, sampling_rate

import torch
from audio_diffusion_pytorch import VSampler, LinearSchedule

@torch.no_grad()
def generate_long_audio(
        model,
        duration_seconds=30,
        overlap_seconds=0.1,
        sampling_rate=48000,
        num_steps=50,
        channels=2,
        device=None
):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)
    model.eval()
    diffusion_model = model.model_ema.ema_model.to(device)

    # Use the model's fixed chunk size
    chunk_length = 65536  # Must match training length
    patch_size = 16
    chunk_length = (chunk_length // patch_size) * patch_size

    overlap_samples = int(overlap_seconds * sampling_rate)
    # Ensure overlap is divisible by patch_size
    overlap_samples = (overlap_samples // patch_size) * patch_size

    total_samples = int(duration_seconds * sampling_rate)

    # Initialize output array
    final_audio = torch.zeros((channels, total_samples), device=device)
    current_position = 0

    sampler = VSampler()
    schedule = LinearSchedule()

    while current_position < total_samples:
        # Generate chunk
        noise = torch.randn((1, channels, chunk_length), device=device)
        chunk = diffusion_model.sample(
            noise=noise,
            sampler=sampler,
            sigma_schedule=schedule,
            num_steps=num_steps
        )

        chunk = chunk[0]  # Remove batch dimension

        # Calculate where to place this chunk
        end_position = min(current_position + chunk_length, total_samples)
        chunk_end = end_position - current_position

        if current_position > 0:
            # Apply crossfade with previous chunk
            print(f"Crossfading at position {current_position}")
            fade_in = torch.linspace(0, 1, overlap_samples, device=device)
            fade_out = torch.linspace(1, 0, overlap_samples, device=device)

            # Apply fade in to new chunk
            chunk[:, :overlap_samples] *= fade_in.unsqueeze(0)

            # Apply fade out to previous audio
            final_audio[:, current_position:current_position + overlap_samples] *= fade_out.unsqueeze(0)

            # Add overlapped regions
            final_audio[:, current_position:current_position + overlap_samples] += \
                chunk[:, :overlap_samples]

            # Add the rest of the chunk
            if chunk_end > overlap_samples:
                final_audio[:, current_position + overlap_samples:end_position] = \
                    chunk[:, overlap_samples:chunk_end]
        else:
            # First chunk, no crossfade needed
            final_audio[:, :chunk_end] = chunk[:, :chunk_end]

        current_position += chunk_length - overlap_samples

    return final_audio.cpu(), sampling_rate

# Example usage:
try:
    # Generate a short sample first to test
    samples, sr = generate_audio_with_params(
        model,
        num_samples=1,
        num_steps=50
    )

    # Save the test sample
    audio = samples[0].cpu()
    audio = audio / torch.abs(audio).max()
    torchaudio.save(
        'test_generated_audio.wav',
        audio,
        sr,
        format='wav'
    )

    # If successful, generate longer audio
    long_audio, sr = generate_long_audio(
        model,
        duration_seconds= 15,  #was 30
        overlap_seconds=0.1,
        num_steps = 50 #was 50
    )

    # Save the long audio
    long_audio = long_audio / torch.abs(long_audio).max()
    torchaudio.save(
        'generated_long_audio.wav',
        long_audio,
        sr,
        format='wav'
    )

except Exception as e:
    print(f"Error occurred: {str(e)}")