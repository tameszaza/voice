"""
mmain.py

A sample extended 'main' script that trains a classification model (exactly
like the original 'main.py'), except we integrate a multi-generator (MGAN)
training flow to produce additional 'fake' data for more variety. We also
introduce an encoder to compute an orthogonal loss among the multi-generator
features, guided by code from mtrain.py (GANs).

IMPORTANT:
- You MUST adapt certain import paths and hyperparameters to match your
  own codebase.
- This code is a demonstration of how to unify classification training
  from main.py with MGAN generator logic from mtrain.py while adding
  a new generator + encoder architecture.
- The classification model (AASIST in your case) remains unchanged;
  we simply add the option to train multiple generators that produce
  "fake" data and compute an orthogonal loss among them.
- In the example below, we do a minimal generator “adversarial” loss
  (MultiResolutionSTFTLoss, or any other). In a real scenario, you
  might have a real discriminator or more intricate logic. Here, we
  keep it simple for demonstration.

Usage:
  python mmain.py --config config.json [--other-args]
"""

import argparse
import json
import os
import sys
import warnings
import time
from importlib import import_module
from pathlib import Path
from shutil import copy
from typing import Dict, List, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchcontrib.optim import SWA

# -- Suppose these come from the existing code in main.py
#    Adjust if your data, evaluate, or util modules differ in name or location
from data_utils import (Dataset_ASVspoof2019_train,
                        Dataset_ASVspoof2019_devNeval, genSpoof_list)
from evaluation import calculate_tDCF_EER
from utils import create_optimizer, seed_worker, set_seed, str_to_bool

# -- If you have a multi-resolution STFT loss from your 'mtrain.py' or elsewhere
#    If not, adapt accordingly or remove
from utils.loss import MultiResolutionSTFTLoss

# -- Add new import for MGAN models
from models.mgan import NewGenerator, NewEncoder

warnings.filterwarnings("ignore", category=FutureWarning)


###############################################################################
# Additional MGAN-like function for orth loss
###############################################################################
def calculate_orthogonal_loss(encoder, generator_outputs):
    """
    Calculate orthogonal loss between generator outputs.
    We feed each generator output into the encoder (frozen or not),
    then compute pairwise normalized dot products. The sum is the
    orth_loss we want to MINIMIZE. More orth => smaller dot => smaller loss.
    """
    with torch.no_grad():
        # each output is shape [B,1,T], pass it to encoder
        features = [encoder(g_out) for g_out in generator_outputs]
    orth_loss = 0.0
    # Compare pairwise
    for i in range(len(features)):
        for j in range(i+1, len(features)):
            f_i, f_j = features[i], features[j]
            inner = torch.sum(f_i * f_j, dim=1)
            norm = (f_i.norm(dim=1)*f_j.norm(dim=1)+1e-8)
            orth_loss += (inner / norm).mean()
    return orth_loss


###############################################################################
# Main function
###############################################################################
def main(args: argparse.Namespace) -> None:
    """
    Extended main function for classification + MGAN-based data augmentation.
    We train a standard classification model (like AASIST from main.py),
    while also training multiple generators to produce synthetic data
    that can help produce more variety (plus orthogonal loss).
    """

    # Step 1: load config
    with open(args.config, "r") as f_json:
        config = json.loads(f_json.read())
    model_config = config["model_config"]
    optim_config = config["optim_config"]
    optim_config["epochs"] = config["num_epochs"]
    track = config["track"]
    assert track in ["LA", "PA", "DF"], "Invalid track given"
    if "eval_all_best" not in config:
        config["eval_all_best"] = "True"
    if "freq_aug" not in config:
        config["freq_aug"] = "False"

    set_seed(args.seed, config)

    # define paths
    output_dir = Path(args.output_dir)
    prefix_2019 = f"ASVspoof2019.{track}"
    database_path = Path(config["database_path"])
    dev_trial_path = database_path / f"ASVspoof2019_{track}_cm_protocols/{prefix_2019}.cm.dev.trl.txt"
    eval_trial_path = database_path / f"ASVspoof2019_{track}_cm_protocols/{prefix_2019}.cm.eval.trl.txt"

    # model path
    model_tag = f"{track}_{os.path.splitext(os.path.basename(args.config))[0]}_ep{config['num_epochs']}_bs{config['batch_size']}"
    if args.comment:
        model_tag = model_tag + f"_{args.comment}"
    model_tag = output_dir / model_tag
    model_save_path = model_tag / "weights"
    eval_score_path = model_tag / config["eval_output"]

    writer = SummaryWriter(model_tag)
    os.makedirs(model_save_path, exist_ok=True)
    copy(args.config, model_tag / "config.conf")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    if device == "cpu":
        raise ValueError("GPU not detected!")

    # Step 2: define classification model (like AASIST)
    model = get_classification_model(model_config, device)
    print("Classification model #params:", sum(p.numel() for p in model.parameters()))

    # Step 3: define data loaders
    trn_loader, dev_loader, eval_loader = get_classification_loaders(database_path, args.seed, config)

    if args.eval:
        # evaluate only
        model.load_state_dict(torch.load(config["model_path"], map_location=device))
        print("Loaded model:", config["model_path"])
        print("Start evaluation (no MGAN).")
        produce_evaluation_file(eval_loader, model, device, eval_score_path, eval_trial_path)
        calculate_tDCF_EER(cm_scores_file=eval_score_path,
                           asv_score_file=database_path / config["asv_score_path"],
                           output_file=model_tag / "t-DCF_EER.txt")
        print("DONE. Exiting.")
        sys.exit(0)

    # Step 4: create multiple generators + encoder for orth loss
    num_generators = args.num_generators
    generators = []
    for i in range(num_generators):
        gen = NewGenerator(in_channels=80, z_channels=args.z_dim, hidden_size=256).to(device)
        generators.append(gen)

    encoder = NewEncoder(input_channels=1, feature_dim=128).to(device)

    # if we want to load pretrained generator weights
    if args.gan_pretrained_path is not None:
        g_ckpt = torch.load(args.gan_pretrained_path, map_location=device)
        for idx, gen in enumerate(generators):
            gen.load_state_dict(g_ckpt[f"generator_{idx}"])
        print(f"Loaded pretrained MGAN from {args.gan_pretrained_path}")

    # generator optimizers
    g_optimizers = []
    for i in range(num_generators):
        g_optim = torch.optim.Adam(generators[i].parameters(), lr=1e-4, betas=(0.9,0.999))
        g_optimizers.append(g_optim)

    # Step 5: classification model optimizer + SWA
    optim_config["steps_per_epoch"] = len(trn_loader)
    optimizer, scheduler = create_optimizer(model.parameters(), optim_config)
    optimizer_swa = SWA(optimizer)

    best_dev_eer = 1.0
    best_eval_eer = 100.0
    best_dev_tdcf = 0.05
    best_eval_tdcf = 1.0
    n_swa_update = 0
    f_log = open(model_tag / "metric_log.txt", "a")
    f_log.write("="*5 + "\n")
    metric_path = model_tag / "metrics"
    os.makedirs(metric_path, exist_ok=True)

    # stft criterion for a minimal generator usage
    stft_criterion = MultiResolutionSTFTLoss().to(device)

    # Step 6: training loop
    for epoch in range(config["num_epochs"]):
        print(f"Start epoch {epoch} / {config['num_epochs']}")
        running_loss = 0.0
        num_total = 0.0

        model.train()
        for batch_idx, (batch_x, batch_y) in enumerate(trn_loader):
            batch_size = batch_x.size(0)
            num_total += batch_size

            batch_x = batch_x.to(device)
            batch_y = batch_y.view(-1).long().to(device)

            # classification forward
            _, out_class = model(batch_x, Freq_aug=str_to_bool(config["freq_aug"]))
            # classification loss
            weight = torch.FloatTensor([0.1, 0.9]).to(device)
            criterion_cls = nn.CrossEntropyLoss(weight=weight)
            loss_cls = criterion_cls(out_class, batch_y)

            # MGAN forward
            generator_outputs = []
            for g in generators:
                z = torch.randn(batch_size, args.z_dim, device=device)
                g_out = g(batch_x, z)  # shape [B,1,Time], you might adapt
                generator_outputs.append(g_out)

            # use e.g. stft-based or any minimal generator loss
            adv_loss_sum = 0.0
            for g_out in generator_outputs:
                sc_loss, mag_loss = stft_criterion(g_out.squeeze(1), batch_x.squeeze(1))
                adv_loss_sum += (sc_loss + 2.0*mag_loss)
            adv_loss_avg = adv_loss_sum / len(generator_outputs)

            # orth
            orth_loss_val = 0.0
            if len(generators) > 1 and args.lambda_orth > 0:
                orth_loss_val = calculate_orthogonal_loss(encoder, generator_outputs)
                orth_loss_val = orth_loss_val * args.lambda_orth

            total_g_loss = adv_loss_avg + orth_loss_val

            # 1) update classifier
            optimizer.zero_grad()
            loss_cls.backward(retain_graph=True)
            optimizer.step()

            # 2) update generators
            for g_opt in g_optimizers:
                g_opt.zero_grad()
            total_g_loss.backward()
            for g_opt in g_optimizers:
                g_opt.step()

            # optional scheduler step
            if scheduler is not None:
                if optim_config["scheduler"] in ["cosine", "keras_decay"]:
                    scheduler.step()
                elif scheduler is None:
                    pass
                else:
                    raise ValueError("scheduler error:{}".format(scheduler))

            running_loss += loss_cls.item() * batch_size

        running_loss /= num_total

        # dev
        produce_evaluation_file(dev_loader, model, device, metric_path/"dev_score.txt", dev_trial_path)
        dev_eer, dev_tdcf = calculate_tDCF_EER(
            cm_scores_file=metric_path/"dev_score.txt",
            asv_score_file=database_path / config["asv_score_path"],
            output_file=metric_path / f"dev_t-DCF_EER_{epoch}.txt",
            printout=False
        )
        print(f"Epoch {epoch}, cls Loss={running_loss:.4f}, dev_eer={dev_eer:.3f}, dev_tdcf={dev_tdcf:.5f}")

        writer.add_scalar("cls_loss", running_loss, epoch)
        writer.add_scalar("dev_eer", dev_eer, epoch)
        writer.add_scalar("dev_tdcf", dev_tdcf, epoch)

        if dev_eer <= best_dev_eer:
            print("best model found at epoch", epoch)
            best_dev_eer = dev_eer
            torch.save(model.state_dict(), model_save_path / f"epoch_{epoch}_{dev_eer:.3f}.pth")

            # optional eval
            if str_to_bool(config["eval_all_best"]):
                produce_evaluation_file(eval_loader, model, device, eval_score_path, eval_trial_path)
                eval_eer, eval_tdcf = calculate_tDCF_EER(
                    cm_scores_file=eval_score_path,
                    asv_score_file=database_path / config["asv_score_path"],
                    output_file=metric_path / f"t-DCF_EER_{epoch}.txt"
                )
                log_text = f"epoch{epoch}, dev_eer={dev_eer:.4f}, eval_eer={eval_eer:.4f}, eval_tdcf={eval_tdcf:.4f}"
                print(log_text)
                f_log.write(log_text+"\n")
        writer.add_scalar("best_dev_eer", best_dev_eer, epoch)

    # final
    if args.eval_model_weights and os.path.exists(args.eval_model_weights):
        model.load_state_dict(torch.load(args.eval_model_weights, map_location=device))

    # possibly SWA
    if True:  # or if you want n_swa_update>0
        # example
        pass

    # final eval
    produce_evaluation_file(eval_loader, model, device, eval_score_path, eval_trial_path)
    final_eer, final_tdcf = calculate_tDCF_EER(
        cm_scores_file=eval_score_path,
        asv_score_file=database_path / config["asv_score_path"],
        output_file=model_tag / "t-DCF_EER.txt"
    )
    print(f"Final EER: {final_eer:.3f}, final t-DCF: {final_tdcf:.5f}")
    f_log.close()

    # Save final generator weights
    for idx, gen in enumerate(generators):
        torch.save(gen.state_dict(), model_save_path / f"final_generator_{idx}.pth")
    torch.save(encoder.state_dict(), model_save_path / "final_encoder.pth")
    print("MGAN-based classification training is complete.")


###############################################################################
# Reuse classification model from main.py
###############################################################################
def get_classification_model(model_config: Dict, device: torch.device):
    """Define classification architecture (like AASIST) from main.py's approach."""
    arch = model_config["architecture"]  # e.g. "AASIST"
    module = import_module(f"models.{arch}")  # adapt if needed
    _model = getattr(module, "Model")
    model = _model(model_config).to(device)
    return model

def get_classification_loaders(
        database_path: Path,
        seed: int,
        config: dict
) -> List[DataLoader]:
    """Make PyTorch DataLoaders for train/dev/eval just like main.py does."""
    track = config["track"]
    prefix_2019 = f"ASVspoof2019.{track}"

    trn_database_path = database_path / f"ASVspoof2019_{track}_train/"
    dev_database_path = database_path / f"ASVspoof2019_{track}_dev/"
    eval_database_path = database_path / f"ASVspoof2019_{track}_eval/"

    trn_list_path = database_path / f"ASVspoof2019_{track}_cm_protocols/{prefix_2019}.cm.train.trn.txt"
    dev_trial_path = database_path / f"ASVspoof2019_{track}_cm_protocols/{prefix_2019}.cm.dev.trl.txt"
    eval_trial_path = database_path / f"ASVspoof2019_{track}_cm_protocols/{prefix_2019}.cm.eval.trl.txt"

    d_label_trn, file_train = genSpoof_list(dir_meta=trn_list_path,
                                            is_train=True,
                                            is_eval=False)
    print("no. training files:", len(file_train))
    train_set = Dataset_ASVspoof2019_train(list_IDs=file_train,
                                           labels=d_label_trn,
                                           base_dir=trn_database_path)
    gen = torch.Generator()
    gen.manual_seed(seed)
    trn_loader = DataLoader(train_set,
                            batch_size=config["batch_size"],
                            shuffle=True,
                            drop_last=True,
                            pin_memory=True,
                            worker_init_fn=seed_worker,
                            generator=gen)

    _, file_dev = genSpoof_list(dir_meta=dev_trial_path,
                                is_train=False,
                                is_eval=False)
    print("no. validation files:", len(file_dev))
    dev_set = Dataset_ASVspoof2019_devNeval(list_IDs=file_dev, base_dir=dev_database_path)
    dev_loader = DataLoader(dev_set,
                            batch_size=config["batch_size"],
                            shuffle=False,
                            drop_last=False,
                            pin_memory=True)

    file_eval = genSpoof_list(dir_meta=eval_trial_path,
                              is_train=False,
                              is_eval=True)
    eval_set = Dataset_ASVspoof2019_devNeval(list_IDs=file_eval, base_dir=eval_database_path)
    eval_loader = DataLoader(eval_set,
                             batch_size=config["batch_size"],
                             shuffle=False,
                             drop_last=False,
                             pin_memory=True)

    return trn_loader, dev_loader, eval_loader


def produce_evaluation_file(
    data_loader: DataLoader,
    model,
    device: torch.device,
    save_path: Path,
    trial_path: Path
) -> None:
    """
    Perform classification model evaluation and save the score to a file,
    same logic as main.py. We read lines from trial_path, produce
    scores for each utterance ID in data_loader, and write them to save_path.
    """
    model.eval()
    with open(trial_path, "r") as f_trl:
        trial_lines = f_trl.readlines()
    fname_list = []
    score_list = []
    for batch_x, utt_id in data_loader:
        batch_x = batch_x.to(device)
        with torch.no_grad():
            _, batch_out = model(batch_x)
            # Suppose class 1 means "spoof" => we take out[:,1] as the score
            batch_score = batch_out[:,1].data.cpu().numpy().ravel()
        fname_list.extend(utt_id)
        score_list.extend(batch_score.tolist())

    assert len(trial_lines) == len(fname_list) == len(score_list)
    with open(save_path, "w") as fh:
        for fn, sco, trl in zip(fname_list, score_list, trial_lines):
            # e.g. 'LA_0072601 LA_0072 spoof human ??'
            parts = trl.strip().split()
            # we adapt these lines according to your actual protocol structure
            # for example:
            #   idx=0 => LA_0072601
            #   idx=1 => LA_0072
            #   idx=2 => spoof
            #   idx=3 => human
            #   ...
            # we write them in the requested format. Adjust if needed
            uttid = parts[1]
            src = parts[2]
            key = parts[3]
            assert fn == uttid, f"Mismatch {fn} vs {uttid}"
            fh.write(f"{uttid} {src} {key} {sco}\n")
    print(f"Scores saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MGAN-based classification trainer")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to config JSON file (same structure as main.py).")
    parser.add_argument("--output_dir", type=str, default="./exp_result",
                        help="Output directory for results")
    parser.add_argument("--seed", type=int, default=1234,
                        help="Random seed (default=1234)")
    parser.add_argument("--eval", action="store_true",
                        help="If set, load model and do final eval only, then exit.")
    parser.add_argument("--comment", type=str, default=None,
                        help="Extra comment appended to model_tag name.")
    parser.add_argument("--eval_model_weights", type=str, default=None,
                        help="Optional path to model weights for final evaluation.")
    # MGAN
    parser.add_argument("--gan_pretrained_path", type=str, default=None,
                        help="Path to MGAN pretrained checkpoint to init the generators.")
    parser.add_argument("--num_generators", type=int, default=2,
                        help="Number of generators for MGAN.")
    parser.add_argument("--z_dim", type=int, default=128,
                        help="Dimension of random noise for each generator.")
    parser.add_argument("--lambda_orth", type=float, default=10.0,
                        help="Weighting for orthogonal loss among generator outputs.")
    args = parser.parse_args()
    main(args)
