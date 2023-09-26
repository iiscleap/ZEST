# Pitch predictor

- The relevant paths are stored in **config.py** file. Please change them accordingly
- The pitch predictor is trained using the file **pitch_attention_adv.py** file
- For training the HiFi-GAN the original F0 needs to be reconstructed. This can be done by running the code **pitch_inference.py**
- After the model is trained run **get_wav2vec_feats.py** to get the emotion features stored in a folder named **wav2vec_feats**
- Conversion for the setting **DSDT** is shown in **pitch_convert.py** file

