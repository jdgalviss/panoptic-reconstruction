import torch

def weight_codes():
    return torch.Tensor([
        (0.5, 0.5, 0.5),        # FLOOR
        (1.0, 1.0, 1.0),		# wall
        (1.0, 1.0, 1.0),		# floor
        (1.0, 1.0, 1.0), 		# cabinet: chair 
        (1.0, 1.0, 1.0),		# bed
        (1.0, 1.0, 1.0), 		# chair: cabinet
        (1.0, 1.0, 1.0),  		# sofa
        (1.0, 1.0, 1.0),		# table
        (1.0, 1.0, 1.0),  		# door
        (1.0, 1.0, 1.0),		# window
        (0.5, 0.5, 0.5),		# bookshelf: wall
        (0.5, 0.5, 0.5),		# picture
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

def create_color_palette():
    return torch.tensor([
        (0, 0, 0),
        (174, 199, 232),		# wall
        (152, 223, 138),		# floor
        (31, 119, 180), 		# cabinet
        (255, 187, 120),		# bed
        (188, 189, 34), 		# chair
        (140, 86, 75),  		# sofa
        (255, 152, 150),		# table
        (214, 39, 40),  		# door
        (197, 176, 213),		# window
        (148, 103, 189),		# bookshelf
        (196, 156, 148),		# picture
        (23, 190, 207), 		# counter
        (178, 76, 76),
        (247, 182, 210),		# desk
        (66, 188, 102),
        (219, 219, 141),		# curtain
        (140, 57, 197),
        (202, 185, 52),
        (51, 176, 203),
        (200, 54, 131),
        (92, 193, 61),
        (78, 71, 183),
        (172, 114, 82),
        (255, 127, 14), 		# refrigerator
        (91, 163, 138),
        (153, 98, 156),
        (140, 153, 101),
        (158, 218, 229),		# shower curtain
        (100, 125, 154),
        (178, 127, 135),
        (120, 185, 128),
        (146, 111, 194),
        (44, 160, 44),  		# toilet
        (112, 128, 144),		# sink
        (96, 207, 209),
        (227, 119, 194),		# bathtub
        (213, 92, 176),
        (94, 106, 211),
        (82, 84, 163),  		# otherfurn
        (100, 85, 144),
        (172, 172, 172),
    ])/255.0

def convert_lab_to_rgb_pt(colorgrid):
    colorgrid = colorgrid.view(-1,3)
    # convert to xyz
    y = (colorgrid[:,0] + 16.0) / 116.0
    x = (colorgrid[:,1] / 500.0) + y
    z = y - (colorgrid[:,2] / 200.0)
    #print('z', torch.sum(z<0).item())
    z[z < 0] = 0 # invalid
    xyz = torch.stack([x,y,z],1)
    mask = xyz > 0.2068966
    m1 = torch.pow(xyz[mask], 3.0)
    m0 = (xyz[~mask] - 16.0 / 116.0) / 7.787
    xyz = xyz.masked_scatter(mask, m1)
    xyz = xyz.masked_scatter(~mask, m0)
    x = xyz[:,0] * 0.55047
    y = xyz[:,1] #*1
    z = xyz[:,2] * 1.08883
    xyz = torch.stack([x,y,z],1)  
    rgb_from_xyz = torch.Tensor([[ 3.2405, -1.5372, -0.4985],
        [-0.5693,  1.8760,  0.0416],
        [ 0.0556, -0.2040,  1.0573]]).to(colorgrid.device)
    rgb = torch.matmul(rgb_from_xyz, xyz.t()).t()
    mask = rgb > 0.0031308
    m1 = 1.055 * torch.pow(rgb[mask], 1.0 / 2.4) - 0.055
    m0 = rgb[~mask] * 12.92
    rgb = rgb.masked_scatter(mask, m1)
    rgb = rgb.masked_scatter(~mask, m0)
    rgb = torch.clamp(rgb, 0, 1)
    return rgb

def convert_lab01_to_rgb_pt(colorgrid):
    # unnormalize
    colorgrid = colorgrid/2.0 + 0.5
    # print("colorgrid", colorgrid.shape)
    # print("color grid range: [{},{}]".format(colorgrid.min(), colorgrid.max()))
    sz = colorgrid.shape
    colorgrid = colorgrid.view(-1,3)
    l = colorgrid[:,:1]
    ab = colorgrid[:,1:]
    l = 100.0*l
    ab = (ab * 2.0 - 1.0) * 100.0
    colorgrid = torch.cat([l,ab],1)
    #colorgrid[:,0] = 100.0*colorgrid[:,0]
    #colorgrid[:,1:] = (colorgrid[:,1:] * 2.0 - 1.0) * 100.0
    return convert_lab_to_rgb_pt(colorgrid).view(sz)