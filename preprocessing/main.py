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

    # Move model to GPU if available
    print("Using", torch.cuda.get_device_name(0))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for i, label in enumerate(os.listdir(data_path)):
        print(f"{i+1}/{len(os.listdir(data_path))}", end=" ")
        print(f"Classifying {label} audios...")
        label_dir = os.path.join(data_path, label)
        if os.path.isdir(label_dir):
            for file_name in os.listdir(label_dir):
                file_path = os.path.join(label_dir, file_name)
                if file_path.endswith(".wav"):
                    results = classify_single_audio(file_path, model, feature_extractor, device)

                    dest_dir = ""
                    if "Bird" in results['label'] or "bird" in results['label']:
                        dest_dir = f"dataset/classified_data/{label}_{SEGMENT_SECONDS}secs"
                    else:
                        dest_dir = f"dataset/classified_data/{label}_{SEGMENT_SECONDS}secs/other"

                    if not os.path.exists(dest_dir):
                        os.makedirs(dest_dir, exist_ok=True)
                    dest_file_path = os.path.join(dest_dir, os.path.basename(file_path))
                    if not os.path.exists(dest_file_path):  # avoid duplicate
                        shutil.copy(file_path, dest_dir)
                    else:
                        continue

def classify_single_audio(file_path, model, feature_extractor, device):
    # Process audio
    waveform, sample_rate = torchaudio.load(file_path)
    waveform = waveform.to(device)

    # Extract features
    inputs = feature_extractor(
        waveform.squeeze().cpu().numpy(),
        sampling_rate=sample_rate,
        return_tensors="pt"
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # Make prediction
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Get prediction class
    predicted_class_ids = torch.argmax(logits, dim=-1).item()
    predicted_label = model.config.id2label[predicted_class_ids]

    # Get prediction probability
    probs = torch.nn.functional.softmax(logits, dim=-1)
    predicted_prob = probs[0][predicted_class_ids].item()

    return {
        'label': predicted_label,
        'confidence': predicted_prob,
        'logits': logits.cpu().numpy()
    }

if __name__ == "__main__":  # make sure your main is executed in the preprocessing folder
    # Define root folder of birds sound
    RAW_DATA_DIR = "dataset/鳥種清單"

    # Define root folder of cropped data
    CROP_DATA_DIR = "dataset/cropped_data"

    # Create folder for cropped data if not existing
    os.makedirs(CROP_DATA_DIR, exist_ok=True)

    print("Cropping audios...")
    try:
        if os.path.exists(CROP_DATA_DIR):
            pass
            labels_with_datapath = ap.load_audio_with_labels(RAW_DATA_DIR)  # dict: {bird name: list of paths to .mp3 files}
            ap.crop_audio(labels_with_datapath, CROP_DATA_DIR, SEGMENT_SECONDS, STRIDE_SECONDS)  # with 5 seconds segments
    except os.error:
        print("Please make sure the dataset is in the same directory as this file")

    print("Classifying audios...")
    classify_audios("dataset/cropped_data")
    print("Done!")