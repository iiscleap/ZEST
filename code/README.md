### Code Explanation

- Download the ESD dataset [https://github.com/HLTSingapore/Emotional-Speech-Data]
- Arrange the dataset in such a way that for all the speakers the train, validation and the test audio files are in separate folders
```
train
|
 ---- 0011_000051.wav
 ---- 0012_001451.wav

val
|
 ---- 0011_000001.wav
 ---- 0012_001401.wav

test
|
 ---- 0011_000021.wav
 ---- 0012_001421.wav
```
### Recontruction phase
- First go into the **EASE** folder
    - Run the commands there and extract the EASE embeddings
- Then we need to predict the F0 contour
    - In the F0_predictor folder, run **pitch_attention_adv.py**
    - Once the model is trained, execute **pitch_inference.py**
    - Also store the wav2vec emotional features using **get_wav2vec_feats.py**
- The HiFi-GAN model now needs to be trained for reconstructing the speech signal
    - In the HiFi-GAN folder follow the instruction given
    - By default the HiFi-GAN model will run for 100K steps, increase if required

### Conversion phase
- We need to convert the F0 contour
    - In the F0_predictor folder, run **pitch_convert.py**. The default setting is **DSDT**. The code needs to be changed slightly for all the other test settings
- The HiFi-GAN model now needs to be run in inference mode
    - In the HiFi-GAN folder follow the instruction given
    - By default the HiFi-GAN model will convert in the **DSDT** test setting
