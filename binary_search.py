import numpy as np
from check_model_complexity import print_model_complexity
# import the model so it has the option to train with 1000 params


class BinarySearch:
    def __init__(self, init_max_params, init_right_loss, model, tolerance=0.05):
        self.init_max_params = init_max_params
        self.tolerance = tolerance * self.init_max_params
        self.model = model
        self.right_loss = init_right_loss # alt: loss = train(model)
        self.left_loss = None
        self.left_params = 1000
        self.right_params = init_max_params

        self.history = [] # (param count, loss)
        self.history.append(self.right_params, self.right_loss)

def search_next_params(self, current_loss):
    # Update history
    self.history.append(self.right_params, current_loss) 

    mid_params = (self.left_params + self.right_params) // 2

    # Tolerance ok?
    if abs(mid_params - self.right_params) <= self.tolerance:
            return 0, self.right_params
    
    # Go right or left? 
    if self.loss_history[-1] < self.loss_history[-2]: 
        # loss(m) was better than loss(r)
        calc_loss_left() 
        if self.loss_history[-1][1] < self.loss_history[0][1]:
            if self.loss_history[0][1] < self.loss_history[-2][1]:
                self.right_params = mid_params # Go right
            else:
                self.left_params = mid_params # Go left
        else:
            self.left_params = mid_params # Go left since loss(l) < loss(m)
    else:
        self.left_params = mid_params # loss(l) > loss(r)
    
    return None, mid_params

def calc_loss_left(self):
    # self.left_loss = train(self.model, self.left_params)
    # self.loss_history.append(self.left_loss)
    # self.param_history.append(self.left_params)
    pass