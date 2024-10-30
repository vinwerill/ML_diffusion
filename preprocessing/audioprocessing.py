import torch
import torchaudio
import torch.nn.functional as F
import os


SAMPLE_RATE = 16000

def process_audio(audio_path: str):  #in seconds
  """
    Load one .mp3 file into array 
    Parameters:
      audio_path (str): path to audio file 
      
    Returns:
      tuple (tuple): (array, sample rate)
  """
  # Load the audio file
  # target_length = segment_seconds * SAMPLE_RATE
  waveform, original_sample_rate = torchaudio.load(audio_path)

  # Convert to mono if stereo
  if waveform.shape[0] > 1:
    waveform = torch.mean(waveform, dim=0, keepdim=True)

  # Resample to 16kHz
  new_sample_rate = SAMPLE_RATE
  resampler = torchaudio.transforms.Resample(orig_freq=original_sample_rate, new_freq=new_sample_rate)
  waveform = resampler(waveform)

  # Pad or trim the audio to target length
  # if waveform.shape[1] < target_length:
    # Pad with zeros if audio is too short
    # waveform = F.pad(waveform, (0, target_length - waveform.shape[1]))
  # else:
  #   # Trim if audio is too long
  #   waveform = waveform[:, :target_length]

  return waveform, new_sample_rate


def crop_audio(label_with_datapaths: dict, crop_data_root: str, segment_seconds=5, stride_seconds=5):
  for (label, datapaths) in label_with_datapaths.items():
    os.makedirs(f"{crop_data_root}/{label}", exist_ok=True)
    for audio_path in datapaths:
      waveform, sample_rate = process_audio(audio_path)
      segment_samples = segment_seconds * sample_rate
      stride_samples = stride_seconds * sample_rate
      
      for i, st in enumerate(range (0, waveform.shape[1]-segment_samples, stride_samples)):
        segment = waveform[:, st:st+segment_samples]
        if(segment.shape[1] < segment_samples):
          segment = F.pad(segment, (0, segment_samples - segment.shape[1]))
        filename = os.path.splitext(os.path.basename(audio_path))[0]
        torchaudio.save(f"{crop_data_root}/{label}/{filename}_seg_{i}.wav", segment, sample_rate=sample_rate)


def load_audio_with_labels(root_dir):
  labels_with_datapath = {}
  for label in os.listdir(root_dir):
    label_dir = os.path.join(root_dir, label)
    if os.path.isdir(label_dir):
      labels_with_datapath[label] = []
      for file_name in os.listdir(label_dir):
        file_path = os.path.join(label_dir, file_name)
        if file_path.endswith('.mp3'):
          labels_with_datapath[label].append(file_path)
  return labels_with_datapath
