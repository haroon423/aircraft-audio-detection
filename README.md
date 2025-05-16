Aircraft Audio Detection

This repository contains a deep learning-based audio classification project that detects whether an audio clip contains the sound of a low-flying aircraft or not. The model is trained on a dataset of environmental audio clips and classifies audio into two classes: aircraft and no aircraft.

Features
Preprocessing audio files into Mel-spectrograms

Min-max normalization of spectrogram features

Convolutional Neural Network (CNN) for classification

Trains on segmented 5-second audio clips

Predicts aircraft presence in new audio files with confidence scores

Usage
Clone the repository

Install required Python packages (e.g., librosa, tensorflow, matplotlib)

Use the provided scripts to preprocess audio, train the model, or predict on new audio samples

model = load_model('model_path.h5')
prediction, confidence = predict_aircraft_audio('audio_file.wav', model)
print(f"Predicted class: {prediction} (1=aircraft, 0=no aircraft)")
print(f"Confidence: {confidence:.4f}")

Dataset
The model is trained on the AerosonicDB dataset.
