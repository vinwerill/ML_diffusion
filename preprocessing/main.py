import os
import shutil
import torchaudio
from transformers import AutoFeatureExtractor, ASTForAudioClassification
import torch
import audioprocessing as ap

SEGMENT_SECONDS = 5
STRIDE_SECONDS = 1

def classify_audios(data_path, model_name="MIT/ast-finetuned-audioset-10-10-0.4593"):
  # Load model and feature extractor
  model = ASTForAudioClassification.from_pretrained(model_name)
  feature_extractor = AutoFeatureExtractor.from_pretrained(
    model_name,
    do_normalize=True,
    return_attention_mask=True
  )

  for label in os.listdir(data_path):
    label_dir = os.path.join(data_path, label)
    if os.path.isdir(label_dir):
      for file_name in os.listdir(label_dir):
        file_path = os.path.join(label_dir, file_name)
        if file_path.endswith(".wav"):
          results = classify_single_audio(file_path, model, feature_extractor)
          '''print(f"\nPrediction Results for {file_name}:")
          print(f"Label: {results['label']}")
          print(f"Confidence: {results['confidence']:.2%}")'''
          if "Bird" in results['label'] or "bird" in results['label']:
            dest_dir = f"dataset/classified_data/{label}_{SEGMENT_SECONDS}secs"
            if not os.path.exists(dest_dir):
              os.makedirs(dest_dir, exist_ok=True)
            shutil.copy(file_path, dest_dir)
          else :
            dest_dir = f"dataset/classified_data/{SEGMENT_SECONDS}+secs_others"
            if not os.path.exists(dest_dir):
              os.makedirs(dest_dir, exist_ok=True)
            shutil.copy(file_path, dest_dir)


def classify_single_audio(file_path, model, feature_extractor):
  # Process audio
  #waveform, sample_rate = ap.process_audio(file_path, 5) #resample to 16kHz
  # Extract features
  waveform, sample_rate = torchaudio.load(file_path)
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
  
  # Define root folder of birds sound
  RAW_DATA_DIR = "C:\\ML_Data\\BirdSoundGenerator\\Raw"

  # Define root folder of cropped data
  CROP_DATA_DIR = "C:\\ML_Data\\BirdSoundGenerator\\Cropped"
  
  # Create folder for cropped data if not existing
  os.makedirs(CROP_DATA_DIR, exist_ok=True)

  # data_directory = 'C:\\ML_Data\\BirdSoundGenerator\\Raw' # make sure your dataset is in the same directory as this file

  try:
    if os.path.exists(CROP_DATA_DIR):
      labels_with_datapath = ap.load_audio_with_labels(RAW_DATA_DIR)  # dict: {bird name: list of paths to .mp3 files}
      ap.crop_audio(labels_with_datapath, CROP_DATA_DIR, SEGMENT_SECONDS, STRIDE_SECONDS) # with 5 seconds segments
  except os.error:
    print("Please make sure the dataset is in the same directory as this file")

  exit(0)
  classify_audios(CROP_DATA_DIR)

  '''results = classify_audio(audio_path)
  print(f"\nPrediction Results:")
  print(f"Label: {results['label']}")
  print(f"Confidence: {results['confidence']:.2%}")'''
