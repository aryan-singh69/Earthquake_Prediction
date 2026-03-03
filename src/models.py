import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiTaskCNN(nn.Module):
    """
    Multi-Task CNN for seismic signal analysis.
    
    Outputs:
    1. Detection     — Earthquake vs Noise (binary)
    2. Phase Picking — P-wave arrival time, S-wave arrival time
    3. Magnitude     — Earthquake magnitude
    4. Location      — Latitude, Longitude, Depth
    """
    def __init__(self, in_channels=3):
        super(MultiTaskCNN, self).__init__()

        #  Shared CNN Encoder (purane model se same) 
        self.conv1 = nn.Conv1d(in_channels, 32,  kernel_size=21, stride=1, padding=10)
        self.pool1 = nn.MaxPool1d(kernel_size=4, stride=4)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=15, stride=1, padding=7)
        self.pool2 = nn.MaxPool1d(kernel_size=4, stride=4)

        self.conv3 = nn.Conv1d(64, 128, kernel_size=11, stride=1, padding=5)
        self.pool3 = nn.MaxPool1d(kernel_size=4, stride=4)

        # Shared FC layer
        self.fc_shared  = nn.Linear(11904, 512)
        self.dropout     = nn.Dropout(0.5)

        # Task Heads 

        # Head 1 — Detection (Earthquake vs Noise)
        self.detection_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)  # Binary output
        )

        # Head 2 — Phase Picking (P-wave, S-wave)
        self.phase_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)  # P arrival, S arrival
        )

        # Head 3 — Magnitude
        self.magnitude_head = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)  # Single magnitude value
        )

        # Head 4 — Location (Lat, Lon, Depth)
        self.location_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 3)  # Lat, Lon, Depth
        )

    def forward(self, x):
        # Shared encoder
        x = F.relu(self.conv1(x)); x = self.pool1(x)
        x = F.relu(self.conv2(x)); x = self.pool2(x)
        x = F.relu(self.conv3(x)); x = self.pool3(x)

        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc_shared(x))
        x = self.dropout(x)

        # Task outputs
        detection  = self.detection_head(x)   # (B, 1)
        phase      = self.phase_head(x)        # (B, 2) — P, S
        magnitude  = self.magnitude_head(x)    # (B, 1)
        location   = self.location_head(x)     # (B, 3) — lat, lon, depth

        return {
            'detection':  detection,
            'phase':      phase,
            'magnitude':  magnitude,
            'location':   location
        }


# Transfer Learning Helper 
def load_pretrained_encoder(model, pretrained_path, device):
    """
    Purane SimpleCNN model se CNN weights load karo.
    Sirf conv1, conv2, conv3 reuse karo.
    """
    print("Loading pretrained encoder weights...")
    checkpoint = torch.load(pretrained_path, map_location=device, weights_only=True)

    # Sirf CNN layers copy karo
    encoder_keys = ['conv1.weight', 'conv1.bias',
                    'conv2.weight', 'conv2.bias',
                    'conv3.weight', 'conv3.bias']

    model_dict    = model.state_dict()
    pretrained    = {k: v for k, v in checkpoint.items() if k in encoder_keys}
    model_dict.update(pretrained)
    model.load_state_dict(model_dict)

    print(f"   Loaded {len(pretrained)} pretrained layers: {list(pretrained.keys())}")

    # CNN layers freeze karo — sirf heads train honge
    for name, param in model.named_parameters():
        if any(k in name for k in ['conv1', 'conv2', 'conv3']):
            param.requires_grad = False

    frozen = sum(1 for p in model.parameters() if not p.requires_grad)
    total  = sum(1 for p in model.parameters())
    print(f"   Frozen {frozen}/{total} layers — only heads will train!")

    return model


if __name__ == "__main__":
    print("Testing MultiTaskCNN...")
    model = MultiTaskCNN()
    dummy = torch.randn(4, 3, 6000)
    out   = model(dummy)

    print(f"Detection shape:  {out['detection'].shape}")   # (4, 1)
    print(f"Phase shape:      {out['phase'].shape}")       # (4, 2)
    print(f"Magnitude shape:  {out['magnitude'].shape}")   # (4, 1)
    print(f"Location shape:   {out['location'].shape}")    # (4, 3)
    print("MultiTaskCNN test passed! ")