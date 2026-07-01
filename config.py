from __future__ import annotations

from dataclasses import dataclass



'''
We've optimized memory as much as we can, 

but training with the default settings in config.py still needs about 80 GB GPU VRAM. 

Stick to the defaults when possible. If you change anything, watch these first :

TrainConfig: batch_size · pocket_atoms · node_feat_dim / cond_dim · fp_bptt (especially -1 = full BPTT)

Model backbone: HIDDEN, N_LAYERS · POCKET_HIDDEN, POCKET_LAYERS

ABCMConfig: grp_max (with group_train=True) · chain_len + chain_carry · carry_bptt · fp_starts × fp_steps (when w_fp > 0) · edge_T



'''











# framework parameters
HIDDEN = 512       
N_LAYERS = 8          
POCKET_HIDDEN = 512   
POCKET_LAYERS = 8     
PRIOR_HIDDEN = 512    
SUB_LAYERS = 2       

# 



@dataclass(frozen=True)
class TrainConfig:


    batch_size: int = 64       
    node_feat_dim: int = 16    
    cond_dim: int = 64        
    pocket_atoms: int = 2048   
    test_pts: int = 64         
    fp_bptt: int | None = None 


@dataclass(frozen=True)
class ABCMConfig:
    w_geom: float = 0.5
    w_graph: float = 0.1
    b_global: float = 0.5
    b_seed: float = 0.15
    g_init: float = 0.15
    w_edge: float = 0.2
    w_edge_op: float = 0.2
    w_seed_g: float = 0.5
    w_seed_f: float = 0.1
    w_init_g: float = 0.5
    w_init_f: float = 0.1
    z_dim: int = 32             
    edge_T: int = 20             
    delta_edit: int = 2
    alpha_prior: float = 0.5
    w_fp: float = 0.3            
    fp_steps: int = 3            # VRAM ↑ 
    fp_starts: int = 6           # VRAM ↑ 
    fp_star: float = 0.5
    w_pocket: float = 0.1
    pocket_min: float | None = None
    basin_leap: bool = True
    group_train: bool = True     
    tau: float = 0.92
    grp_min: int = 3
    grp_max: int = 6             # VRAM ↑ 
    w_basin: float = 0.3
    chain_len: int = 1           # VRAM ↑ 
    chain_ramp: bool = False
    chain_carry: bool = False    
    carry_bptt: int = 1          
