from transformers import AutoFeatureExtractor, ASTForAudioClassification
from datasets import load_dataset
import torch
import torchaudio
import torch.nn.functional as F


def process_audio(audio_path, target_length=1024000):  # roughly 64 seconds at 16kHz
  # Load the audio file
  waveform, original_sample_rate = torchaudio.load(audio_path)

  # Convert to mono if stereo
  if waveform.shape[0] > 1:
    waveform = torch.mean(waveform, dim=0, keepdim=True)

  # Resample to 16kHz
  new_sample_rate = 16000
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


if __name__ == "__main__":
  # Example usage
  audio_path = "bird.mp3"
  results = classify_audio(audio_path)

  print(f"\nPrediction Results:")
  print(f"Label: {results['label']}")
  print(f"Confidence: {results['confidence']:.2%}")