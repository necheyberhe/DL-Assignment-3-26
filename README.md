# Deep Learning Assignment 3  
## Melody-Conditioned Lyrics Generation using LSTM Networks

This project implements multiple recurrent neural network architectures for automatic song lyrics generation conditioned on melody information extracted from MIDI files.

The system was developed as part of a Deep Learning course assignment.

---

# Project Overview

The project investigates whether melody information can improve neural lyrics generation.

Three models were implemented and compared:

| Model | Description |
|---|---|
| Baseline Model | Lyrics-only LSTM using pretrained word embeddings |
| Global MIDI Model | LSTM conditioned on a global melody feature vector |
| Aligned MIDI Model | LSTM conditioned on time-aligned melody features |

The models generate song lyrics one word at a time using probabilistic decoding strategies.

---

# Features Implemented

## Lyrics Processing
- Text cleaning
- Tokenization
- Vocabulary construction
- Sequence generation
- Special token handling

## Word Embeddings
- Pretrained 300-dimensional GloVe embeddings
- Unknown-word random initialization

## MIDI Processing

### Global MIDI Features
Extracted per song:
- Mean pitch
- Pitch standard deviation
- Mean note duration
- Mean velocity
- Note density
- Tempo
- Drum ratio

### Time-Aligned MIDI Features
Extracted per temporal segment:
- Mean pitch
- Note count
- Mean duration
- Mean velocity

## Models
- Lyrics-only LSTM baseline
- Global melody-conditioned LSTM
- Time-aligned melody-conditioned LSTM

## Decoding Strategies
- Basic proportional sampling
- Temperature-scaled sampling
- Top-k sampling

## Evaluation
- Validation loss comparison
- Melody influence probe
- Failure mode analysis
- TensorBoard logging

---

# Project Structure

```text
project/
│
├── main.ipynb
├── README.md
│
├── generated_lyrics_results.csv
├── generated_lyrics_results.txt
├── official_test_generated_lyrics_results.csv
├── official_test_generated_lyrics_results.txt
│
├── baseline_lyrics_lstm.pth
├── global_midi_lyrics_lstm.pth
├── aligned_midi_lyrics_lstm.pth
│
├── validation_loss_comparison.png
│
├── runs/
│   └── lyrics_rnn_experiment/
│
└── Data/
    └── Archive (1)/
        ├── lyrics_train_set.csv
        ├── lyrics_test_set.csv
        └── midi_files/
# Requirements
pip install torch torchvision torchaudio
pip install pandas numpy matplotlib scikit-learn
pip install gensim pretty_midi
pip install tensorboard        

# Running the Project

Open Notebook
Open:
main.ipynb
Run all cells sequentially.
Training
The notebook trains:
Baseline model
Global MIDI model
Aligned MIDI model
Validation losses are plotted automatically.

# TensorBoard

TensorBoard logs are saved to:

runs/lyrics_rnn_experiment

Launch TensorBoard:

tensorboard --logdir=runs