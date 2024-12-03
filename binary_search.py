import numpy as np
import torch

from torch.nn import Module
from models import SpectrVelCNNRegr
from SearchNet import *
from check_model_complexity import print_model_complexity


class BinarySearch:
    def __init__(self, init_max_params, init_right_loss, model, tolerance=0.05):
        self.init_max_params = init_max_params
        self.tolerance = tolerance * self.init_max_params
        self.model = model
        self.right_loss = init_right_loss  # initial loss for max params
        self.left_loss = None
        self.left_params = 1000
        self.right_params = init_max_params
        self.mid_params = (self.left_params + self.right_params) // 2 

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
            return 1, self.right_params

        # Calculates the loss of min params for initial comparison
        if len(self.history) == 1:
            left_loss = self.calc_loss_rand(self.left_params)
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

    def calc_loss(self, params):
        result = find_scaling_factors_grid_search(params) # Find the optimal alpha and beta

        if result is None: 
            raise ValueError(f"Could not find valid config for {params} parameters")
        
        alpha, beta, gamma_conv, gamma_fc, total_params, abs_diff, rel_diff = result
        model = SearchNet(alpha=alpha, beta=beta, gamma_conv=gamma_conv, gamma_fc=gamma_fc)

        # Call some function that trains the model and outputs the loss
        x = torch.randn(1, 6, 74, 918)
        target = torch.randn(1, 1)

        criterion = nn.MSELoss()
        output = model(x)
        loss = criterion(output, target)

        return loss.item() # return loss

    # Outputs random loss wrt. parameter count. - Used for testing
    def calc_loss_rand(self, params): 
        return np.exp(-params / self.init_max_params) + np.random.uniform(0, 0.01)  # Random loss

    def print_self(self): # Print tolerance
        print(f"Left parameter count: {self.left_params}")
        print(f"Right parameter count: {self.right_params}")
        print(f"Mid parameter count: {self.mid_params}")
        print(f"Right loss: {self.right_loss}")
        print(f"Tolerance: {self.tolerance}")
        print(f"Model: {self.model}")


if __name__ == "__main__":
    print(f"\n\n### Testing the binary search ###")
    # Initialize the model
    # model = SpectrVelCNNRegr
    model = SearchNet

    # Calculate total model parameters
    # total_params = sum(p.numel() for p in model.parameters())
    total_params = print_model_complexity(model)
    print(f"\n\nModel total parameters: {total_params}")

    # Binary search parameters
    init_max_params = total_params
    current_loss = np.exp(-init_max_params / init_max_params) + np.random.uniform(0, 0.01)
    tolerance = 0.05

    # Simulate the initial loss for the max parameter count
    binary_search = BinarySearch(
        init_max_params=init_max_params,
        init_right_loss=current_loss,  # Simulated loss
        model=model,
        tolerance=tolerance
    )
    binary_search.print_self()

    # Run binary search
    stop = None
    next_params = binary_search.mid_params
    count = 0
    while stop is None:
        # current_loss = binary_search.calc_loss_rand(next_params)
        current_loss = binary_search.calc_loss(next_params)
        stop, next_params = binary_search.search_next_params(current_loss)
        if stop == 1:
            break
        else:
            count+=1
            print(f"Iteration: {count}, Params: {next_params}, Loss: {current_loss}")
        


    # Display history of parameter and loss
    print("Search history (params, loss):")
    for i, (params, loss) in enumerate(binary_search.history):
        print(f"Params: {params}, Loss: {loss:.4f}")
