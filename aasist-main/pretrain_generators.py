import argparse
import torch
import torch.nn as nn
from pathlib import Path
import torchaudio
from torch.utils.data import DataLoader

from models.mgans import NewGenerator
from utils import set_seed
from data_utils import Dataset_ASVspoof2019_train, genSpoof_list

class MultiResolutionSTFTLoss(nn.Module):
    def __init__(self, fft_sizes=[1024, 2048, 512], 
                 hop_sizes=[120, 240, 50], 
                 win_lengths=[600, 1200, 240]):
        super().__init__()
        self.transforms = nn.ModuleList([
            torchaudio.transforms.Spectrogram(
                n_fft=fft, hop_length=hop, win_length=win,
                power=2.0, normalized=False)
            for fft, hop, win in zip(fft_sizes, hop_sizes, win_lengths)
        ])
        
    def forward(self, fake, real):
        loss_mag = 0
        loss_sc = 0
        
        for transform in self.transforms:
            fake_mag = transform(fake)
            real_mag = transform(real)
            
            # Magnitude loss
            loss_mag += torch.mean(torch.abs(fake_mag - real_mag))
            
            # Spectral convergence loss
            loss_sc += torch.norm(fake_mag - real_mag, p='fro') / torch.norm(real_mag, p='fro')
            
        return loss_sc, loss_mag

def pretrain_generators(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize generators
    generators = [NewGenerator().to(device) for _ in range(args.num_generators)]
    g_optimizers = [torch.optim.Adam(g.parameters(), lr=args.lr) 
                    for g in generators]
    
    # Setup data
    database_path = Path(args.database_path)
    trn_list_path = database_path / f"ASVspoof2019_{args.track}_cm_protocols/ASVspoof2019.{args.track}.cm.train.trn.txt"
    trn_database_path = database_path / f"ASVspoof2019_{args.track}_train/"
    
    d_label_trn, file_train = genSpoof_list(dir_meta=trn_list_path,
                                           is_train=True,
                                           is_eval=False)
    
    train_set = Dataset_ASVspoof2019_train(list_IDs=file_train,
                                          labels=d_label_trn,
                                          base_dir=trn_database_path)
    
    train_loader = DataLoader(train_set,
                             batch_size=args.batch_size,
                             shuffle=True,
                             drop_last=True)

    # Loss
    stft_criterion = MultiResolutionSTFTLoss().to(device)
    
    save_path = Path(args.save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    for epoch in range(args.epochs):
        total_stft_loss = 0
        
        for batch_x, _ in train_loader:
            batch_x = batch_x.to(device)
            batch_size = batch_x.size(0)
            
            z = torch.randn(batch_size, args.z_dim, device=device)
            
            # Update each generator
            for gen_idx, (generator, optimizer) in enumerate(zip(generators, g_optimizers)):
                fake = generator(batch_x, z)
                
                sc_loss, mag_loss = stft_criterion(fake.squeeze(1), batch_x.squeeze(1))
                g_loss = sc_loss + mag_loss
                
                optimizer.zero_grad()
                g_loss.backward()
                optimizer.step()
                
                total_stft_loss += g_loss.item()
                
        avg_loss = total_stft_loss / len(train_loader)
        print(f"Epoch {epoch}: Avg STFT Loss = {avg_loss:.4f}")
        
        # Save periodically
        if (epoch + 1) % args.save_interval == 0:
            state_dict = {
                f'generator_{i}': g.state_dict() 
                for i, g in enumerate(generators)
            }
            torch.save(state_dict, 
                      save_path / f'generators_ep{epoch}.pth')
    
    # Save final model
    state_dict = {
        f'generator_{i}': g.state_dict() 
        for i, g in enumerate(generators)
    }
    torch.save(state_dict, save_path / 'generators_final.pth')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--database_path', type=str, required=True)
    parser.add_argument('--track', type=str, default='LA')
    parser.add_argument('--num_generators', type=int, default=2)
    parser.add_argument('--z_dim', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--save_path', type=str, default='./pretrained_gens')
    parser.add_argument('--save_interval', type=int, default=10)
    parser.add_argument('--seed', type=int, default=1234)
    
    args = parser.parse_args()
    set_seed(args.seed, None)
    pretrain_generators(args)
