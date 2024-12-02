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

        mid_params = (self.left_params + self.right_params) // 2 # Calculate midpoint

        # Check tolerance convergence
        if abs(mid_params - self.right_params) <= self.tolerance:
            print(f"\nMid params: {mid_params}")
            print(f"Absolute difference of mid & right params: {abs(mid_params - self.right_params)}")
            print(f"Tolerance: {self.tolerance}\n")
            return 0, self.right_params

        # Calculates the loss of min params for initial comparison
        if len(self.history) == 1:
            left_loss = self.calc_loss(self.left_params)
            self.history.append((self.left_params, left_loss))

        # Compare losses at different points
        if len(self.history) > 1 and current_loss < self.history[-1][1]:  # M < R
            print("M < R")
            if current_loss < self.history[0][1]: # M < L
                print("M < L")
                if self.history[0][1] < self.history[-1][1]: # L < R
                    print("L < R")
                    self.right_params = mid_params  # Reduce right boundary
                else: # L > R
                    print("L > R")
                    self.left_params = mid_params  # Increase left boundary
            else: # M > L
                print("M > L")
                self.left_params = mid_params  # Increase left boundary
        else: # M > R
            print("M > R")
            self.left_params = mid_params  # Increase left boundary
        
        self.history.append((mid_params, current_loss)) # Save the param count & loss
        return None, mid_params

    # Calculate loss of min. param count (1000)
    def calc_loss(self, params): 
        # This is to recalculate if the middle params is better then both the min and max params. 
        # Also for general loss calculation. Or at least to call a the model with loss. 
        return np.exp(-params / self.init_max_params) + np.random.uniform(0, 0.01)  # Random loss

    def print_tolerance(self): # Print tolerance
        print(f"Tolerance: {self.tolerance}")


if __name__ == "__main__":
    # Initialize the model
    model = SpectrVelCNNRegr() ### NOT USED

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
        if run == 0:
            break
        else:
            count+=1
            print(f"Iteration: {count}, Params: {next_params}, Loss: {current_loss}")
        


    # Display history of parameter and loss
    print("Search history (params, loss):")
    for i, (params, loss) in enumerate(binary_search.history):
        print(f"Params: {params}, Loss: {loss:.4f}")
