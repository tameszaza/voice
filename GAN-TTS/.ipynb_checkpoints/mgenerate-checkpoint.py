import torch
from utils.audio import save_wav
import argparse
import os
import time
import numpy as np
from models.generator import Generator
from utils.util import mu_law_encode, mu_law_decode


def load_checkpoint(checkpoint_path, use_cuda):
    """Load the checkpoint."""
    if use_cuda:
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    return checkpoint


def attempt_to_restore(generators, checkpoint_dir, use_cuda):
    """Restore all generators from the last checkpoint."""
    checkpoint_list = os.path.join(checkpoint_dir, 'checkpoint')

    if os.path.exists(checkpoint_list):
        checkpoint_filename = open(checkpoint_list).readline().strip()
        checkpoint_path = os.path.join(checkpoint_dir, "{}".format(checkpoint_filename))
        print(f"Restoring from {checkpoint_path}")
        checkpoint = load_checkpoint(checkpoint_path, use_cuda)

        # Load each generator
        for idx, generator in enumerate(generators):
            generator_key = f"generator_{idx}"
            if generator_key in checkpoint:
                generator.load_state_dict(checkpoint[generator_key])
                print(f"Loaded weights for generator {idx}")
            else:
                print(f"Warning: No weights found for generator {idx}")
    else:
        print(f"No checkpoint found in {checkpoint_dir}")


def create_models(args, num_generators):
    """Create a list of generator models."""
    generators = [Generator(args.local_condition_dim, args.z_dim) for _ in range(num_generators)]
    return generators


def synthesis(args):
    """Synthesize audio from multiple generators."""
    # Create multiple generators
    num_generators = 2  # Adjust this number based on your MGAN setup
    generators = create_models(args, num_generators)
    if args.resume is not None:
        attempt_to_restore(generators, args.resume, args.use_cuda)

    device = torch.device("cuda" if args.use_cuda else "cpu")
    for model in generators:
        model.to(device)

    output_dir = "samples"
    os.makedirs(output_dir, exist_ok=True)

    avg_rtf = []
    for filename in os.listdir(os.path.join(args.input, 'mel')):
        start = time.time()
        conditions = np.load(os.path.join(args.input, 'mel', filename))
        conditions = torch.FloatTensor(conditions).unsqueeze(0).transpose(1, 2).to(device)

        batch_size = conditions.size(0)
        z = torch.randn(batch_size, args.z_dim).to(device).normal_(0.0, 1.0)

        for idx, model in enumerate(generators):
            # Generate audio
            audios = model(conditions, z)
            audios = audios.cpu().squeeze().detach().numpy()

            # Save the generated audio
            name = filename.split('.')[0]
            save_wav(np.asarray(audios), f'{output_dir}/{name}_generator_{idx}.wav')
            print(f"Saved generated audio from generator {idx} for {filename}")

        # Save target sample (reference audio)
        sample = np.load(os.path.join(args.input, 'audio', filename))
        sample = mu_law_decode(mu_law_encode(sample))
        save_wav(np.squeeze(sample), f'{output_dir}/{name}_target.wav')

        time_used = time.time() - start
        rtf = time_used / (len(audios) / 24000)
        avg_rtf.append(rtf)
        print(f"Time used: {time_used:.3f}, RTF: {rtf:.4f}")

    print(f"Average RTF: {sum(avg_rtf) / len(avg_rtf):.3f}")


def main():
    def _str_to_bool(s):
        """Convert string to bool (in argparse context)."""
        if s.lower() not in ['true', 'false']:
            raise ValueError(f"Argument needs to be a boolean, got {s}")
        return {'true': True, 'false': False}[s.lower()]

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='data/test', help='Directory of tests data')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of dataloader workers.')
    parser.add_argument('--resume', type=str, default="logdir", help="Path to the checkpoint directory")
    parser.add_argument('--local_condition_dim', type=int, default=80)
    parser.add_argument('--z_dim', type=int, default=128)
    parser.add_argument('--use_cuda', type=_str_to_bool, default=True)

    args = parser.parse_args()
    synthesis(args)


if __name__ == "__main__":
    main()
