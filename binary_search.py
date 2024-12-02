import numpy as np
import torch
from torch.nn import Module
from models import SpectrVelCNNRegr


class BinarySearch:
    def __init__(self, init_max_params, init_right_loss, model, tolerance=0.05):
        self.init_max_params = init_max_params
        self.tolerance = tolerance * self.init_max_params
        self.model = model
        self.right_loss = init_right_loss  # initial loss for max params
        self.left_loss = None
        self.left_params = 1000
        self.right_params = init_max_params

        # History to track search progress
        self.history = []  # (param count, loss)
        self.history.append((self.right_params, self.right_loss))

    def search_next_params(self, current_loss): # Correct history appending 

        # Calculate midpoint
        mid_params = (self.left_params + self.right_params) // 2

        # Check tolerance convergence
        if abs(mid_params - self.right_params) <= self.tolerance:
            return 0, self.right_params

        # Compare losses at different points
        if len(self.history) > 1 and current_loss < self.history[-2][1]:  # Middle loss better than right loss
            # Check relationship with initial loss
            if current_loss < self.history[0][1]:
                # Middle loss is best so far
                if self.history[0][1] < self.history[-2][1]:
                    # Initial loss was better than right loss
                    self.right_params = mid_params  # Reduce right boundary
                else: # Right loss was better
                    self.left_params = mid_params  # Increase left boundary
            else: # Middle loss worse than initial loss
                self.left_params = mid_params  # Increase left boundary
        else: # Right loss is better
            self.left_params = mid_params  # Increase left boundary
        
        self.history.append((mid_params, current_loss))
        return None, mid_params


    def calc_loss(self, params):
        """
        Placeholder loss function for demonstration. 
        Assumes the loss decreases as parameters increase up to a point.
        """
        # This is to recalculate if the middle params is better then both the min and max params. 
        # Also for general loss calculation. Or at least to call a the model with loss. 
        return np.exp(-params / self.init_max_params) + np.random.uniform(0, 0.01)  # Adding small noise

    def print_tolerance(self):
        print(f"Tolerance: {self.tolerance}")


if __name__ == "__main__":
    # Initialize the model
    model = SpectrVelCNNRegr()

    # Calculate total model parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model total parameters: {total_params}")

    # Binary search parameters
    init_max_params = 1000000
    current_loss = np.exp(-init_max_params / init_max_params) + np.random.uniform(0, 0.01)
    tolerance = 0.05

    # Simulate the initial loss for the max parameter count
    binary_search = BinarySearch(
        init_max_params=init_max_params,
        init_right_loss=current_loss,  # Simulated loss
        model=model,
        tolerance=tolerance
    )
    binary_search.print_tolerance()

    # Run binary search
    run = None
    next_params = binary_search.right_params
    count = 0
    while run is None:
        current_loss = binary_search.calc_loss(next_params)
        run, next_params = binary_search.search_next_params(current_loss)
        count+=1
        print(f"Iteration: {count}, Params: {next_params}, Loss: {current_loss}")
        if run == 0:
            break


    # Display history of parameter and loss
    print("Search history (params, loss):")
    for i, (params, loss) in enumerate(binary_search.history):
        print(f"Params: {params}, Loss: {loss:.4f}")
