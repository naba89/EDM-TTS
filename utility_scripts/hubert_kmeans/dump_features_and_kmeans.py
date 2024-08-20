import argparse
import os
import torch
import torchaudio
from torch.utils.data import DataLoader
from transformers import AutoFeatureExtractor, AutoModel
from datasets import load_dataset
from tqdm import tqdm
import faiss
import numpy as np
from npy_append_array import NpyAppendArray


def prepare_dataset(dataset_args):
    return load_dataset(**dataset_args,
                        num_proc=32,
                        trust_remote_code=True).shuffle(seed=42)


def extract_features(batch, feature_extractor, model, layer_idx, device):
    filename = batch['file'][0]
    speech, sr = torchaudio.load(filename)
    speech = torchaudio.functional.resample(speech, sr, 16000)
    inputs = feature_extractor(speech.squeeze(0), return_tensors="pt", sampling_rate=16000)
    for key, value in inputs.items():
        inputs[key] = value.to(device)
    with torch.no_grad():
        features = model(**inputs, output_hidden_states=True).hidden_states[layer_idx]
    return features.cpu().squeeze(0)


def get_dataset_args(dataset_name):
    if dataset_name == "librispeech-train-clean-100":
        return {
            "path": "edm_tts/datasets/librispeech.py",
            "name": "clean",
            "split": "train_clean_100",
            "data_dir": "data/librispeech",  # root containing LibriSpeech/train-clean-100
        }
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--dataset_name", type=str, default="librispeech-train-clean-100")
    arg_parser.add_argument("--centroid_dir", type=str,
                            default='edm_tts/models/audio_tokenizer/'
                                    'semantic_tokenizer_hubert/semantic_cluster_centers/')
    arg_parser.add_argument("--feature_dir", type=str, default="exp/hubert_features")
    arg_parser.add_argument("--model_name", type=str, default="facebook/hubert-large-ll60k")
    arg_parser.add_argument("--layer_idx", type=int, default=18)
    arg_parser.add_argument("--num_centroids", type=int, default=1024)

    args = arg_parser.parse_args()

    dataset_args = get_dataset_args(args.dataset_name)

    # Load the dataset
    dataset = prepare_dataset(dataset_args)

    # Initialize model and feature extractor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = args.model_name
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    layer_idx = args.layer_idx

    num_samples = 1000 * args.num_centroids  # Adjust based on your dataset size and memory constraints
    total_pool = num_samples * 1
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    output_path = os.path.join(args.feature_dir, "features", model_name.split("/")[1], args.dataset_name)
    os.makedirs(output_path, exist_ok=True)
    feature_filename = os.path.join(output_path, "features.npy")

    seen_features = 0  # Tracks the number of features seen
    pbar = tqdm(total=total_pool)  # If num_samples is the feature target, this works; otherwise, consider adjusting

    with NpyAppendArray(feature_filename, delete_if_exists=True) as npaa:
        for batch in dataloader:
            features = extract_features(batch, feature_extractor, model, layer_idx, device)
            if features is not None:
                npaa.append(features.numpy())
                seen_features += features.size(0)
                pbar.update(features.size(0))

            if seen_features >= total_pool:
                break

    # Load features
    features_flat = np.load(feature_filename)

    # sample num_samples features
    np.random.seed(42)
    idx = np.random.choice(features_flat.shape[0], num_samples, replace=False)
    features_flat = features_flat[idx]

    # Perform k-means clustering
    num_centroids = args.num_centroids
    kmeans = faiss.Kmeans(features_flat.shape[1], num_centroids, niter=20, verbose=True, nredo=5, seed=42)
    kmeans.train(features_flat)
    centroids = kmeans.centroids

    # Save centroids
    centroid_path = os.path.join(args.centroid_dir,
                                 f'{model_name.split("/")[1]}_L{layer_idx}_K{num_centroids}_{args.dataset_name}.pt')
    # centroid_path = args.centroid_path
    os.makedirs(os.path.dirname(centroid_path), exist_ok=True)
    torch.save(torch.from_numpy(centroids).float(), centroid_path)


if __name__ == "__main__":
    main()
