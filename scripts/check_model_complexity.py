#%%
from torch.nn import Module
from models import SpectrVelCNNRegr

def print_model_complexity(model: Module) -> None:
    """Check and print the number of parameters in the network

    Args:
        model (module): Pytorch model class
    """
    
    total_params = sum(p.numel() for p in model().parameters())
    
    print(f"Number of parameters in model {model.__name__}: {total_params} = {'{:.2e}'.format(total_params)}")

# %%
if __name__ == "__main__":
    model = SpectrVelCNNRegr
    print_model_complexity(model)

# %%
