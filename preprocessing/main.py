from sympy.core.random import sample
from transformers import AutoFeatureExtractor, ASTForAudioClassification
from datasets import load_dataset
import torch
import torchaudio
import torch.nn.functional as F
import os

SAMPLE_RATE = 16000

def process_audio(audio_path, target_length=16000*30):  # roughly 64 seconds at 16kHz
  # Load the audio file
  waveform, original_sample_rate = torchaudio.load(audio_path)

  # Convert to mono if stereo
  if waveform.shape[0] > 1:
    waveform = torch.mean(waveform, dim=0, keepdim=True)

  # Resample to 16kHz
  new_sample_rate = SAMPLE_RATE
  resampler = torchaudio.transforms.Resample(orig_freq=original_sample_rate, new_freq=new_sample_rate)
  waveform = resampler(waveform)

  # Pad or trim the audio to target length
  if waveform.shape[1] < target_length:
    # Pad with zeros if audio is too short
    waveform = F.pad(waveform, (0, target_length - waveform.shape[1]))
  else:
    # Trim if audio is too long
    waveform = waveform[:, :target_length]

  return waveform, new_sample_rate


def classify_audio(audio_path, model_name="MIT/ast-finetuned-audioset-10-10-0.4593"):
  # Load model and feature extractor
  model = ASTForAudioClassification.from_pretrained(model_name)
  feature_extractor = AutoFeatureExtractor.from_pretrained(
    model_name,
    do_normalize=True,
    return_attention_mask=True
  )

  # Process audio
  waveform, sample_rate = process_audio(audio_path)
  print(f"Processed waveform shape: {waveform.shape}")

  # Extract features
  inputs = feature_extractor(
    waveform.squeeze().numpy(),
    sampling_rate=sample_rate,
    return_tensors="pt"
  )


  # Make prediction
  model.eval()
  with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

  # Get predicted class
  predicted_class_ids = torch.argmax(logits, dim=-1).item()
  predicted_label = model.config.id2label[predicted_class_ids]

  # Get probabilities
  probs = torch.nn.functional.softmax(logits, dim=-1)
  predicted_prob = probs[0][predicted_class_ids].item()

  return {
    'label': predicted_label,
    'confidence': predicted_prob,
    'logits': logits.numpy()
  }

def crop_audio(label_with_datapaths, segment_length=5):
  for (label, datapaths) in label_with_datapaths.items():
    os.makedirs(f"dataset/cropped_data/{label}", exist_ok=True)
    for datapath in datapaths:
      waveform, sample_rate = process_audio(datapath)
      segment_samples = segment_length * sample_rate
      for i, st in enumerate(range (0, waveform.shape[1], segment_samples)):
        segment = waveform[:, st:st+segment_samples]
        if(segment.shape[1] < segment_samples):
          segment = F.pad(segment, (0, segment_samples - segment.shape[1]))
        filename = os.path.splitext(os.path.basename(datapath))[0]
        torchaudio.save(f"dataset/cropped_data/{label}/{filename}_seg_{i}.wav", segment, sample_rate=sample_rate)


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

if __name__ == "__main__":
  # Example usage
  audio_path = "bird.mp3"
  root_directory = 'dataset/鳥種清單' #make sure your dataset is in the same directory as this file
  labels_with_datapath = load_audio_with_labels(root_directory)
  crop_audio(labels_with_datapath)
  '''results = classify_audio(audio_path)
  print(f"\nPrediction Results:")
  print(f"Label: {results['label']}")
  print(f"Confidence: {results['confidence']:.2%}")'''
