# HiFi-GAN

- Code is adapted from the Speech Resynthesis repository [https://github.com/facebookresearch/speech-resynthesis/tree/main]
- Run the following command
```
export CUDA_VISIBLE_DEVICES=0
cd /ZEST/code/HiFi-GAN
python train.py --checkpoint_path checkpoints/ESD/ --pitch_folder /ZEST/code/F0_predictor/f0_contours --emo_folder /ZEST/code/F0_predictor/wav2vec_feats/ --config hubert_alladv.json
```
- After the above command finishes exection run the following command for inference
```
python inference.py --checkpoint_file checkpoints/ESD/ --pitch_folder /ZEST/code/F0_predictor/f0_contours --emo_folder /ZEST/code/F0_predictor/wav2vec_feats/ --convert
```
- The codes are set for execution of the **DSDT** test setting


