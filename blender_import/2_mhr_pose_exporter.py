import torch
import json
import pickle
import os

def export_pose(pkl_path="sam3dbody_pose.pkl", pt_path="mhr_model.pt"):
    if not os.path.exists(pkl_path) or not os.path.exists(pt_path):
        print("Error: Files not found.")
        return

    # 1. Load the pose parameters from pkl
    with open(pkl_path, 'rb') as f:
        data_list = pickle.load(f)
    
    # Take the first detected person
    pose_params = data_list[0]['mhr_model_params']
    pose_tensor = torch.from_numpy(pose_params).unsqueeze(0) # [1, 204]
    
    # 2. Load model and run forward pass to get global transforms
    print(f"Loading MHR model to compute pose transforms...")
    model = torch.jit.load(pt_path)
    
    # MHR needs Identity (45), Pose (204), Expressions (72)
    # We use the pose from the PKL
    identity = torch.zeros(1, 45)
    extra = torch.zeros(1, 72)
    
    with torch.no_grad():
        _, skel_state = model(identity, pose_tensor, extra)
    
    # skel_state is [batch, num_joints, 8]
    # Format: [tx, ty, tz, qx, qy, qz, qw, scale]
    transforms = skel_state[0].cpu().tolist()
    
    pose_data = {
        "transforms": transforms
    }
    
    with open("mhr_pose.json", "w") as f:
        json.dump(pose_data, f, indent=4)
        
    print("Exported mhr_pose.json")

if __name__ == "__main__":
    export_pose()
