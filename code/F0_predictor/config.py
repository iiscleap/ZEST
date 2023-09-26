train_datasets = {"ESD":"/home/soumyad/emoconv/ESD/train"}
val_datasets = {"ESD":"/home/soumyad/emoconv/ESD/val"}
test_datasets = {"ESD":"/home/soumyad/emoconv/ESD/test"}


train_tokens_orig = {"ESD":"/ZEST/code/train_esd.txt"}
val_tokens_orig = {"ESD":"/ZEST/code/val_esd.txt"}
test_tokens_orig = {"ESD":"/ZEST/code/test_esd.txt"}

f0_file = "ZEST/code/f0.pickle"
hparams = {
        ################################
        # Experiment Parameters        #
        ################################
        "epochs":200,
        "seed":1234,
        ################################
        # Data Parameters              #
        ################################
        "n_mel_channels":80,
        "output_classes":5,
        ################################
        # Model Parameters             #
        "encoder_embedding_dim":128,
        "speaker_embedding_dim":192,
        "emotion_embedding_dim":128,
        ################################
        # Decoder parameters
        "feed_back_last":True,
        "n_frames_per_step_decoder":1,
        "decoder_rnn_dim":512,
        "prenet_dim":[256,256],
        "max_decoder_steps":2000,
        "stop_threshold":0.5,
    
        # Attention parameters
        "attention_rnn_dim":512,
        "attention_dim":128,

        # Location Layer parameters
        "attention_location_n_filters":32,
        "attention_location_kernel_size":17,

        # PostNet parameters
        "postnet_n_convolutions":5,
        "postnet_dim":512,
        "postnet_kernel_size":5,
        "postnet_dropout":0.1,

        ################################
        # Optimization Hyperparameters #
        ################################
        "learning_rate":1e-3,
        "weight_decay":1e-4,
        "grad_clip_thresh":5.0,
        "batch_size":32,
        "warmup":7,
        "decay_rate": 0.5,
        "decay_every":7}

hifi_gan_params={"resblock_dilation_sizes":((1,3,5),(1,3,5),(1,3,5)),
                 "resblock_kernel_sizes":(3, 7, 11),
                 "upsample_kernel_sizes":(11, 8, 8, 4, 4),
                 "upsample_initial_channel":512,
                 "upsample_factors":(5, 4, 4, 2, 2),
                 "inference_padding": 5,
                 "sample_rate":16000,
                 "segment_length":8320}

transformer = {
  "encoder_layer": 4,
  "encoder_head": 2,
  "encoder_hidden": 256,
  "decoder_layer": 6,
  "decoder_head": 2,
  "decoder_hidden": 256 + 192,
  "conv_filter_size": 1024,
  "conv_kernel_size": [9, 1],
  "encoder_dropout": 0.2,
  "decoder_dropout": 0.2}

optimizerparams = {
  "betas": [0.9, 0.98],
  "eps": 0.000000001,
  "weight_decay": 0.0,
  "grad_clip_thresh": 1.0,
  "grad_acc_step": 1,
  "warm_up_step": 4000,
  "anneal_steps": [300000, 400000, 500000],
  "anneal_rate": 0.3,
  "total_step": 900000,
  "log_step": 100,
  "synth_step": 1000,
  "val_step": 1000,
  "save_step": 10000}