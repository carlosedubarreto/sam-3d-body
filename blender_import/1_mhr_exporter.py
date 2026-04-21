import torch
import json
import os
import numpy as np

def export_mhr_from_pt(pt_path="mhr_model.pt"):
    if not os.path.exists(pt_path):
        print(f"Error: {pt_path} not found.")
        return

    print(f"Loading MHR model from {pt_path}...")
    model = torch.jit.load(pt_path)
    
    identity = torch.zeros(1, 45)
    pose = torch.zeros(1, 204)
    extra = torch.zeros(1, 72)
    
    print("Running forward pass...")
    with torch.no_grad():
        verts, skel_state = model(identity, pose, extra)
    
    skeleton = model.character_torch.skeleton
    joint_names = list(skeleton.joint_names)
    joint_parents = [int(p) for p in skeleton.joint_parents]
    
    # Format: [tx, ty, tz, qx, qy, qz, qw, scale]
    transforms = skel_state[0].cpu().tolist()
    
    # Extract Weights
    lbs = model.character_torch.linear_blend_skinning
    skin_indices = lbs.skin_indices_flattened.cpu().tolist()
    skin_weights = lbs.skin_weights_flattened.cpu().tolist()
    vert_indices = lbs.vert_indices_flattened.cpu().tolist()
    
    # Organize weights by vertex
    # We need to know which weights belong to which vertex
    weights_data = []
    # Based on the naming 'flattened', it's likely a sparse representation
    # Let's verify the logic: if we have N vertices, each might have K influences.
    # Usually, vert_indices_flattened tells us which vertex each weight/index belongs to.
    
    # Create a list of lists for weights per vertex
    num_verts = verts.shape[1]
    weights_per_vert = [[] for _ in range(num_verts)]
    
    for i in range(len(vert_indices)):
        v_idx = vert_indices[i]
        j_idx = skin_indices[i]
        weight = skin_weights[i]
        weights_per_vert[v_idx].append((j_idx, weight))

    skeleton_data = {
        "joint_names": joint_names,
        "joint_parents": joint_parents,
        "transforms": transforms,
        "weights": weights_per_vert
    }
    
    with open("mhr_skeleton.json", "w") as f:
        json.dump(skeleton_data, f, indent=4)
        
    verts_np = verts[0].cpu().numpy()
    faces_np = model.character_torch.mesh.faces.cpu().numpy()
    
    # Rotation fix: Swap Y and Z, and negate Y to handle Blender's Z-up
    # Actually, often it's just swapping Y/Z or a -90 rotation on X.
    # Let's try simple swap first in Blender side, but export raw here.
    
    with open("mhr_mesh.obj", "w") as f:
        for v in verts_np:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for face in faces_np:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
            
    print(f"Exported mhr_skeleton.json and mhr_mesh.obj with weights.")

if __name__ == "__main__":
    export_mhr_from_pt()
