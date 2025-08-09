# VAD emotion scoring for audio files

## Model versioning
There are three main versions of the model architecture:

- **V1:** Uses the MelEncoder as described in [this paper](https://arxiv.org/abs/2106.03153) and the [ERes2NetV2 Speaker Recognition Model](https://modelscope.cn/models/iic/speech_eres2netv2w24s4ep4_sv_zh-cn_16k-com).
- **V2:** Utilizes the BUD-E Whisper model to extract feature embeddings and applies a simple MLP head to predict a vector of VAD scores.
- **V3:** Also uses the BUD-E Whisper model for feature embeddings, but employs a more complex two-head system to predict the means and standard deviations of VAD scores. This approach better captures the variability in emotion annotations.
- **V3s:** Similar to V3, but uses GELU activation instead of PReLU in the MLP heads. It has almost the same performance as V3, but is more efficient in terms of memory usage.

For more detailed information about the architecture, check the [architecture overview](doc/architecture_overview.md) document.

## Weights

Model weights for V3 model are available on [Hugging Face](https://huggingface.co/drakrig/vad_emotion_scorer). The model was trained on the Laion's Got Talent (Enhanced Flash Annotations and Long Captions) dataset with Method 1 annotation approach, which is described in detail in the [annotation methodology document](doc/annotation_method.md).

## Configuration files
For simplicity reason, configuratuon is done through a YAML file. Check the "vad_train_config.yaml" file for example. All parameters that must be adjusted are marked with comments. Parameters that are NOT marked with comments must stay as they are, since they are model specific and should not be changed.

## Data formatting
The annotation file is expected to be `.csv` that uses `;` as separator. The file should contain the following columns:

- `full_path` - the full path to the audio file
- `pleasure_mean` - the mean pleasure score for the audio file
- `pleasure_std` - the standard deviation of the pleasure score for the audio file
- `arousal_mean` - the mean arousal score for the audio file
- `arousal_std` - the standard deviation of the arousal score for the audio file
- `dominance_mean` - the mean dominance score for the audio file
- `dominance_std` - the standard deviation of the dominance score for the audio file
- `verified_emotion` - the verified emotion label for the audio file based on scores means. It used for balancing the emotion distribution in the dataset in case some emotions are overrepresented or underrepresented.

For more detailed information about the annotation process check the document [here](doc/annotation_method.md) or exmaine this notebook [here](doc/annotation_workflow.ipynb).

## How to abtain VAD scores for your audio files

If your dataset contains categorical emotion labels, you can use the mapping table described on page 15 of the [original paper](https://www.researchgate.net/publication/222741832_Evidence_for_a_Three-Factor_Theory_of_Emotions) to convert them to the continuous pleasure, arousal and dominance means and standard deviations.

Also, you can download extracted mapping table from Google Drive [here](https://drive.google.com/file/d/1AajCZiIwAPrQ7W2bbGpgBFyAT2b0nzz_/view?usp=sharing).

## Roadmap

- [+] Upload V3 weights to Hugging Face
- [ ] Upload updated weights to Hugging Face
- [*] Upload formatted emotion to VAD score mapping table
- [ ] Add more documentation on how to use the model

## Credits

1. [GPT SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) for initial ideas and code for certain modules.
2. [LaionAI](https://laion.ai) for the [Laion's Got Talent (Enhanced Flash Annotations and Long Captions) dataset](https://huggingface.co/datasets/laion/laions_got_talent_enhanced_flash_annotations_and_long_captions) as well as the [BUD-E Whisper](https://huggingface.co/laion/BUD-E-Whisper) model.
3. [MER2025](https://huggingface.co/datasets/MERChallenge/MER2025)