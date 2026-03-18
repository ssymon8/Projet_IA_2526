#!/bin/bash

DATA_DIR="/home/ensta/ensta-baron/Projet_IA_2526/Bach10_Clean"
MODEL_TYPE="htdemucs"

echo "Démarrage du Fine-tuning Demucs sur Bach10..."
echo "Train dir: $DATA_DIR/train"
echo "Valid dir: $DATA_DIR/valid"

cd ~/Projet_IA_2526/demucs

python3 -m demucs.train \
    ++dset.wav=$DATA_DIR \
    ++dset.use_musdb=false \
    ++dset.samplerate=44100 \
    ++dset.channels=1 \
    ++dset.segment=11 \
    ++batch_size=16 \
    ++epochs=100 \
    ++tag=bach10_finetune \
    ++model=htdemucs \
    ++optim.lr=0.00005 \
    ++augment.repitch.proba=0 \
    ++test.every=999 \
    '++sources=[violin,clarinet,saxophone,bassoon]'

echo "Entraînement terminé ou arrêté."