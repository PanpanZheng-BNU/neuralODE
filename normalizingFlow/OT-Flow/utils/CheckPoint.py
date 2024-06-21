import torch
import torch.nn as nn
def save_checkpoint(func:nn.Module, EPOCH, optimizer, LOSS, PATH):
  torch.save({
            'epoch': EPOCH,
            'model_state_dict': func.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': LOSS,
            }, PATH)