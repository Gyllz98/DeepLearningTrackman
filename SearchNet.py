import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from models import SpectrVelCNNRegr
from check_model_complexity import print_model_complexity
import sys

class SearchNet(SpectrVelCNNRegr):
    def __init__(self, alpha=1.0, beta=1.0, gamma_conv=1.0, gamma_fc=1.0):
        # Call parent's __init__ 
        super().__init__()

        # Clear ALL existing layers from parent
        for attr in ['conv1', 'conv2', 'conv3', 'conv4', 'linear1', 'linear2', 'linear3']:
            if hasattr(self, attr):
                delattr(self, attr)

        # Convolutional layers configuration
        max_conv_layers = 4
        num_conv_layers = max(1, int(max_conv_layers * gamma_conv))
        conv_configs = [
            {'out_channels': 16, 'kernel_size': 5},
            {'out_channels': 32, 'kernel_size': 5},
            {'out_channels': 64, 'kernel_size': 5},
            {'out_channels': 128, 'kernel_size': 3},
        ][:num_conv_layers]

        # Build convolutional layers
        in_channels = 6
        self.conv_layers = nn.ModuleList()
        for i, config in enumerate(conv_configs):
            if i > 0:
                in_channels = max(1, int(conv_configs[i-1]['out_channels'] * beta))
            out_channels = max(1, int(config['out_channels'] * beta))
            
            conv_layer = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=config['kernel_size'],
                    stride=1,
                    padding=2 if config['kernel_size'] == 5 else 1
                ),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2)
            )
            self.conv_layers.append(conv_layer)

        # Compute FC input size
        fc_input_size = self._compute_fc_input_size(beta, num_conv_layers)

        # Define FC layers
        max_fc_layers = 3
        num_fc_layers = max(0, max_fc_layers * gamma_fc)

        
        # Second FC layer if gamma_fc allows
        max_fc_layers = 3
        num_fc_layers = max(1, int(max_fc_layers * gamma_fc))  # This gives us 1, 2, or 3 layers

        if num_fc_layers == 3:
            # Three FC layers
            self.linear1 = nn.Linear(fc_input_size, int(1024 * alpha))
            self.linear2 = nn.Linear(int(1024 * alpha), int(256 * alpha))
            self.linear3 = nn.Linear(int(256 * alpha), 1)
        elif num_fc_layers == 2:
            # Two FC layers
            self.linear1 = nn.Linear(fc_input_size, int(1024 * alpha))
            self.linear2 = None
            self.linear3 = nn.Linear(int(1024 * alpha), 1)
        else:  # num_fc_layers == 1
            # Single FC layer
            self.linear1 = nn.Linear(fc_input_size, 1)
            self.linear2 = None
            self.linear3 = None
    def _compute_fc_input_size(self, beta, num_conv_layers):
        """Compute the input size for the first FC layer based on conv output dimensions."""
        height, width = 74, 918
        
        # Match PyTorch's dimension calculations exactly
        for i in range(num_conv_layers):
            kernel_size = 5 if i < 3 else 3
            padding = 2 if kernel_size == 5 else 1
            
            # After conv with padding (PyTorch formula)
            height = height + 2*padding  # Add padding
            width = width + 2*padding
            
            height = (height - kernel_size) // 1 + 1  # Conv
            width = (width - kernel_size) // 1 + 1
            
            # After pooling
            height = height // 2
            width = width // 2
        
        # Get output channels of final conv layer
        out_channels = max(1, int([16, 32, 64, 128][num_conv_layers - 1] * beta))
        
        return out_channels * height * width

    def forward(self, x):
        # Pass through convolutional layers
        for conv_layer in self.conv_layers:
            x = conv_layer(x)

        # Flatten
        x = torch.flatten(x, 1)
        
        # Pass through FC layers
        if self.linear2 is not None and self.linear3 is not None:
            # Three layer case
            x = F.relu(self.linear1(x))
            x = F.relu(self.linear2(x))
            x = self.linear3(x)
        elif self.linear3 is not None:
            # Two layer case
            x = F.relu(self.linear1(x))
            x = self.linear3(x)
        else:
            # Single layer case
            x = self.linear1(x)
        return x

def calculate_params_precise(alpha, beta, gamma_conv=1.0, gamma_fc=1.0, input_height=74, input_width=918):
    """Calculate total network parameters with dynamic layer reduction"""
    # Convolutional layer configurations
    all_conv_layers = [
        {'in_channels': 6, 'out_channels': 16, 'kernel_size': 5},
        {'in_channels': 16, 'out_channels': 32, 'kernel_size': 5},
        {'in_channels': 32, 'out_channels': 64, 'kernel_size': 5},
        {'in_channels': 64, 'out_channels': 128, 'kernel_size': 3},
    ]
    
    # Determine number of conv layers
    max_conv_layers = 4
    num_conv_layers = max(1, int(max_conv_layers * gamma_conv))
    conv_layers = all_conv_layers[:num_conv_layers]
    
    # Calculate conv parameters
    conv_params = 0
    height, width = input_height, input_width
    in_channels = 6
    
    for i, layer in enumerate(conv_layers):
        if i > 0:
            in_channels = max(1, int(conv_layers[i-1]['out_channels'] * beta))
        out_channels = max(1, int(layer['out_channels'] * beta))
        kernel_size = layer['kernel_size']
        padding = 2 if kernel_size == 5 else 1
        
        # Conv params (kernel weights + bias)
        layer_params = (kernel_size**2 * in_channels * out_channels) + out_channels
        conv_params += layer_params
        
        # Use exact same dimension calculation as in _compute_fc_input_size
        height = height + 2*padding
        width = width + 2*padding
        
        height = (height - kernel_size) // 1 + 1
        width = (width - kernel_size) // 1 + 1
        
        height = height // 2
        width = width // 2

    # Final conv output channels
    final_channels = max(1, int(conv_layers[-1]['out_channels'] * beta))
    fc_input_size = final_channels * height * width
    
    # In calculate_params_precise:
    # FC layers parameters
    max_fc_layers = 3
    num_fc_layers = max(1, int(max_fc_layers * gamma_fc))
    fc_params = 0

    if num_fc_layers == 3:
        # Three FC layers
        fc1_out = int(1024 * alpha)
        fc2_out = int(256 * alpha)
        fc_params += (fc_input_size * fc1_out) + fc1_out  # First layer
        fc_params += (fc1_out * fc2_out) + fc2_out        # Second layer
        fc_params += (fc2_out * 1) + 1                    # Output layer
    elif num_fc_layers == 2:
        # Two FC layers
        fc1_out = int(1024 * alpha)
        fc_params += (fc_input_size * fc1_out) + fc1_out  # First layer
        fc_params += (fc1_out * 1) + 1                    # Output layer
    else:  # num_fc_layers == 1
        # Single FC layer
        fc_params += (fc_input_size * 1) + 1              # Direct to output
    
    return conv_params + fc_params

def determine_valid_gammas(target_params, original_params=38414929):
    """
    Determine valid gamma ranges based on parameter ratio
    """
    ratio = target_params / original_params
    
    # If we want more parameters than original, no layer removal
    if ratio >= 1.0:
        return (1.0, 1.0), (1.0, 1.0)  # (gamma_conv_range, gamma_fc_range)
        
    # Between 70-100% of parameters: no layer removal
    if ratio > 0.7:
        return (1.0, 1.0), (1.0, 1.0)
        
    # Between 40-70% of parameters: allow removing one layer of either type
    if ratio > 0.4:
        return (0.75, 1.0), (0.75, 1.0)
        
    # Between 25-40% of parameters: must remove at least one layer
    if ratio > 0.25:
        return (0.75, 1.0), (0.33, 0.75)  # Force FC layer removal
        
    # Below 25% of parameters: must remove two layers
    return (0.75, 0.75), (0.33, 0.75)  # Force both layer removals

def find_scaling_factors_grid_search(target_params, 
                                   alpha_range=(0.1, 1.5, 30),
                                   beta_range=(0.1, 1.5, 30),
                                   tolerance=0.05):
    """
    Grid search with parameter-based layer removal thresholds
    """
    # Determine valid gamma ranges based on parameter target
    (gamma_conv_min, gamma_conv_max), (gamma_fc_min, gamma_fc_max) = \
        determine_valid_gammas(target_params)
        
    # Generate all parameter combinations
    alphas = np.linspace(alpha_range[0], alpha_range[1], alpha_range[2])
    betas = np.linspace(beta_range[0], beta_range[1], beta_range[2])
    gamma_convs = np.linspace(gamma_conv_min, gamma_conv_max, 2)
    gamma_fcs = np.linspace(gamma_fc_min, gamma_fc_max, 2)
    
    best_config = None
    best_diff = float('inf')
    
    # Logging setup
    log_file = 'parameter_search.log'
    sys.stdout = open(log_file, 'w')
    
    print(f"Grid Search for Target Parameters: {target_params}")
    print(f"Parameter ratio: {target_params/38414929:.3f}")
    print(f"Valid gamma ranges - Conv: [{gamma_conv_min}, {gamma_conv_max}], FC: [{gamma_fc_min}, {gamma_fc_max}]")
    print("=" * 80)
    print(f"{'Alpha':<10}{'Beta':<10}{'Gamma Conv':<12}{'Gamma FC':<12}{'Total Params':<20}{'Rel Diff %':<15}{'Layers':<20}")
    print("-" * 80)
    
    # Search all valid combinations
    for alpha in alphas:
        for beta in betas:
            for gamma_conv in gamma_convs:
                for gamma_fc in gamma_fcs:
                    try:
                        total_params = calculate_params_precise(
                            alpha, beta, gamma_conv, gamma_fc
                        )
                        
                        abs_diff = abs(total_params - target_params)
                        rel_diff = abs_diff / target_params * 100
                        
                        layer_desc = (
                            f"{4 if gamma_conv == 1.0 else 3} conv, "
                            f"{2 if gamma_fc == 1.0 else 1} FC"
                        )
                        
                        print(f"{alpha:<10.4f}{beta:<10.4f}{gamma_conv:<12.4f}{gamma_fc:<12.4f}"
                              f"{total_params:<20,d}{rel_diff:<15.4f}{layer_desc}")
                        
                        if abs_diff < best_diff:
                            best_diff = abs_diff
                            best_config = (alpha, beta, gamma_conv, gamma_fc, 
                                         total_params, abs_diff, rel_diff)
                        
                        if rel_diff <= tolerance:
                            break
                    
                    except Exception as e:
                        print(f"Error processing config: {e}")
    
    # Restore stdout
    sys.stdout.close()
    sys.stdout = sys.__stdout__
    
    # Print results to console
    if best_config:
        alpha, beta, gamma_conv, gamma_fc, params, abs_diff, rel_diff = best_config
        # print("\n===== Best Configuration =====")
        # print(f"Alpha:          {alpha:.4f}")
        # print(f"Beta:           {beta:.4f}")
        # print(f"Gamma Conv:     {gamma_conv:.4f}")
        # print(f"Gamma FC:       {gamma_fc:.4f}")
        # print(f"Total Params:   {params:,d}")
        # print(f"Absolute Diff:  {abs_diff:,d}")
        # print(f"Relative Diff:  {rel_diff:.4f}%")
        # print(f"Architecture:   {4 if gamma_conv == 1.0 else 3} conv layers, "
        #       f"{2 if gamma_fc == 1.0 else 1} FC layers")
        # print(f"Target ratio:   {target_params/38414929:.3f}")
        # print(f"Log file:       {log_file}")
    
    return best_config

if __name__ == "__main__":
    TARGET_PARAM = (38414929// 100)
    print(f"Target parameters: {TARGET_PARAM}")
    
    result = find_scaling_factors_grid_search(
        target_params=TARGET_PARAM, 
        tolerance=0.01,
        alpha_range=(0.01, 3, 20),
        beta_range=(0.5, 2.0, 20)
    )
    
    if result:
        alpha, beta, gamma_conv, gamma_fc, total_params, abs_diff, rel_diff = result
        print("\nUsing parameters:")
        print(f"Alpha: {alpha}")
        print(f"Beta: {beta}")
        print(f"Gamma Conv: {gamma_conv}")
        print(f"Gamma FC: {gamma_fc}")
        
        # Create model and inspect its structure
        model = SearchNet(alpha=alpha, beta=beta, gamma_conv=gamma_conv, gamma_fc=gamma_fc)
        
        print("\nActual model structure:")
        conv_params = 0
        fc_params = 0
        
        print("\nConvolutional layers:")
        for i, layer in enumerate(model.conv_layers):
            conv = layer[0]  # Get Conv2d layer
            params = sum(p.numel() for p in conv.parameters())
            print(f"Conv{i+1}: in={conv.in_channels}, out={conv.out_channels}, params={params}")
            conv_params += params
            
        print("\nFully connected layers:")
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                params = sum(p.numel() for p in module.parameters())
                print(f"{name}: in={module.in_features}, out={module.out_features}, params={params}")
                fc_params += params
        
        actual_total = sum(p.numel() for p in model.parameters())
        print(f"\nTotal parameters breakdown:")
        print(f"Conv params: {conv_params:,d}")
        print(f"FC params:   {fc_params:,d}")
        print(f"Total:       {actual_total:,d}")
        print(f"Expected:    {total_params:,d}")
        print(f"Difference:  {abs(actual_total - total_params):,d}")
        
        print("\nAll model attributes:")
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                print(f"{name}: {type(module)}")