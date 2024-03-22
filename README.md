# ZEST
Zero-Shot Emotion Style Transfer

Samples and code for the paper **Zero Shot Audio to Audio Emotion Transfer With Speaker Disentanglement** accepted at **ICASSP 2024**

## Ablations

We perform two ablations where we remove the two CNN-BiLSTM networks operating on the F0 contour and the HuBERT tokens. The results for these cases are provided in the diagram below. Note that there are two more entries (ZEST-no-F0-enc and ZEST-no-unit-enc) in the table than what was provided in the paper.

<img src="./images/ablations.png" width="1200px"></img>

## EASE Embeddings
We first show the utility of the adversarial learning in the speaker embedding module. <img src="./images/EASE.png" width="1200px"></img>

We show the comparison between the **x-vectors**(left of each group) and the **EASE embeddings**(right of each group) side-by-side for two different speakers in ESD dataset. The colours in the t-SNE plots are according to the 5 emotions present in the dataset. We note that x-vectors show clear emotion clusters while EASE embeddings are ignorant of the emotion information. 

## SACE Embeddings
We next show the emotion latent information learnt by the **SACE** module. We show t-SNE visualization of this latent space and colour them based on the 10 speakers and 5 emotions separately.

Emotion clusters             |  Speaker clusters
:-------------------------:|:-------------------------:
<img src="./images/emo_wav2vec.png" width="600px"></img>|  <img src="./images/spk_wav2vec.png" width="600px"></img>

## Reconstruction of F0 contour

We show 4 examples of the ability of the F0 predictor module to reconstruct the ground truth pitch contour.

Angry|  Happy | Sad | Surprise
:-------------------------:|:-------------------------:|:--------------------------:|:---------------------------:|
<img src="./images/f0_angry.png" width="600px"></img>|  <img src="./images/f0_happy.png" width="600px"></img>|  <img src="./images/f0_sad.png" width="600px"></img>|  <img src="./images/f0_surprise.png" width="600px"></img>

## Examples of F0 conversion

We show three examples of how the F0 conversion works in ZEST. We show three examples from three test settings - DSDT (source and reference speaker seen but different with unseen text), USS (Unseen source speaker with seen reference speaker/emotion and unseen text) and UTE (seen source speaker with unseen reference speaker/emotion and unseen text). The first row indicates the F0 contour of the **source** speech signal, the second row indicates pitch contour of the **reference** speech signal while the last row indicates the **converted** F0 contour.

DSDT|  USS| UTE 
:-------------------------:|:-------------------------:|:--------------------------:|
<img src="./images/F0_conversions_DSDT.png" width="600px"></img>|  <img src="./images/F0_conversions_USS.png" width="600px"></img>|  <img src="./images/F0_conversions_UTE.png" width="600px"></img>
