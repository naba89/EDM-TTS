# EDM-TTS

## Installation
1. Prepare the environment:
    ```bash
    conda create -n edm-tts python=3.10
    conda activate edm-tts
    ```
2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```
3. Install the `edm_tts` package:
    ```bash
    pip install -e .
    ```

## Data Preparation
**NOTE**: In case datasets are already downloaded to different paths, you have to search and replace the paths accordingly in the scripts or create symlinks.

1. Download the LibriSpeech dataset from [here](https://www.openslr.org/12) and extract to `data/librispeech/`
   ```bash
   mkdir -p data/librispeech
   wget https://www.openslr.org/resources/12/train-clean-100.tar.gz -P data/librispeech/
   tar -xvf data/librispeech/train-clean-100.tar.gz -C data/librispeech/
   ```
2. Download the LibriLight dataset from [here](https://github.com/facebookresearch/libri-light/blob/main/data_preparation/README.md#1a-downloading) and extract to `data/libri-light/unlab/`
   ```bash
   mkdir -p data/libri-light/unlab
   wget https://dl.fbaipublicfiles.com/librilight/data/small.tar -P data/libri-light/
   wget https://dl.fbaipublicfiles.com/librilight/data/medium.tar -P data/libri-light/
   wget https://dl.fbaipublicfiles.com/librilight/data/large.tar -P data/libri-light/
   tar -xvf data/libri-light/small.tar -C data/libri-light/unlab/
   tar -xvf data/libri-light/medium.tar -C data/libri-light/unlab/
   tar -xvf data/libri-light/large.tar -C data/libri-light/unlab/
   ```
3. Download the LibriHeavy manifests from [here](https://huggingface.co/datasets/pkufool/libriheavy) and extract to `data/libri-light/unlab/libriheavy/`
   ```bash
   mkdir -p data/libri-light/unlab/libriheavy
   wget https://huggingface.co/datasets/pkufool/libriheavy/resolve/main/libriheavy_cuts_small.jsonl.gz -P data/libri-light/unlab/libriheavy/
   wget https://huggingface.co/datasets/pkufool/libriheavy/resolve/main/libriheavy_cuts_medium.jsonl.gz -P data/libri-light/unlab/libriheavy/
   wget https://huggingface.co/datasets/pkufool/libriheavy/resolve/main/libriheavy_cuts_large.jsonl.gz -P data/libri-light/unlab/libriheavy/
   ```
4. Chunk the LibriHeavy manifests into manageable sizes
   ```bash
   python utility_scripts/chunk_libriheavy_manifests/chunk_libriheavy.py
   ```
5. Run the following command to create the semantic KMeans clusters for the LibriSpeech `train-clean-100` dataset:
    ```bash
    # Check the script for further customization
    python utility_scripts/hubert_kmeans/dump_features_and_kmeans.py
    ```
6. Run the following command to extract the semantic and acoustic codes for the datasets:
    ```bash
    # Train the Codec before running these commands (see Training section)
    # Check the script for further customization
    python utility_scripts/dump_tokens/dump_tokens.py --dataset_name librilight --output_dir data/librilight_codes/ --codec_model exp/edm_tts/dac/best_model
    python utility_scripts/dump_tokens/dump_tokens.py --dataset_name libriheavy-small --output_dir data/libriheavy_codes/ --codec_model exp/edm_tts/dac/best_model
    python utility_scripts/dump_tokens/dump_tokens.py --dataset_name libriheavy-medium --output_dir data/libriheavy_codes/ --codec_model exp/edm_tts/dac/best_model
    python utility_scripts/dump_tokens/dump_tokens.py --dataset_name libriheavy-large --output_dir data/libriheavy_codes/ --codec_model exp/edm_tts/dac/best_model
    ```
   

   
## Training
1. Train the Codec:
    ```bash
    # Check the script and config files for further customization
    accelerate launch --config_file configs/acc_cfg.yaml run_codec_training.py configs/dac/train_config.yaml
    ```
2. Train the Semantic-to-Acoustic model (Injection Conformer)
    ```bash
    # Check the script and config files for further customization
    accelerate launch --config_file configs/acc_cfg_deepspeed.yaml run_semantic_to_acoustic_training.py configs/injection_conformer/train_config.yaml
    ```
3. Train the Text-to-Semantic model
    ```bash
    # Check the script and config files for further customization
    accelerate launch --config_file configs/acc_cfg_deepspeed.yaml run_text_to_semantic_training.py configs/text_to_semantic_w_length/train_config.yaml
    ```
   
## Inference
```bash
# Check the script for further customization
python inference.py --text "Hello, how are you?" \
                    --speaker_prompt "path/to/speaker_prompt.wav" \
                    --output "path/to/output.wav" \
                    --codec_model "exp/edm_tts/dac/best_model" \
                    --s2a_model "exp/edm_tts/injection_conformer/best_model" \
                    --t2s_model "exp/edm_tts/text_to_semantic_w_length/best_model" 
```
