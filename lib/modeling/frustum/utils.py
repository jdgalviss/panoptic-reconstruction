import torch

def weight_codes():
    return torch.Tensor([
        (0.2, 0.2, 0.2),        # FLOOR
        (1.0, 1.0, 1.0),		# wall
        (1.0, 1.0, 1.0),		# floor
        (1.0, 1.0, 1.0), 		# cabinet: chair 
        (1.0, 1.0, 1.0),		# bed
        (1.0, 1.0, 1.0), 		# chair: cabinet
        (1.0, 1.0, 1.0),  		# sofa
        (1.0, 1.0, 1.0),		# table
        (1.0, 1.0, 1.0),  		# door
        (1.0, 1.0, 1.0),		# window
        (0.2, 0.2, 0.2),		# bookshelf: wall
        (0.2, 0.2, 0.2),		# picture
        (1.0, 1.0, 1.0), 		# counter
        (1.0, 1.0, 1.0),
        (1.0, 1.0, 1.0),		# desk
        (1.0, 1.0, 1.0),
        (1.0, 1.0, 1.0),		# curtain
        (1.0, 1.0, 1.0),
        (1.0, 1.0, 1.0),
        (1.0, 1.0, 1.0),
        (1.0, 1.0, 1.0),
        (1.0, 1.0, 1.0),
        (1.0, 1.0, 1.0),
        (1.0, 1.0, 1.0),
        (1.0, 1.0, 1.0),		# refrigerator
        (1.0, 1.0, 1.0),
        (1.0, 1.0, 1.0),
        (1.0, 1.0, 1.0),
        (1.0, 1.0, 1.0),		# shower curtain
        (1.0, 1.0, 1.0),
        (1.0, 1.0, 1.0),
        (1.0, 1.0, 1.0),
        (1.0, 1.0, 1.0),
        (1.0, 1.0, 1.0),  		# toilet
        (1.0, 1.0, 1.0),		# sink
        (1.0, 1.0, 1.0),
        (1.0, 1.0, 1.0),		# bathtub
        (1.0, 1.0, 1.0),
        (1.0, 1.0, 1.0),
        (1.0, 1.0, 1.0), 		# otherfurn
        (1.0, 1.0, 1.0),
        (1.0, 1.0, 1.0),
    ])