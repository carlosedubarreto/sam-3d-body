import bpy
import json
import os
import mathutils

def import_and_pose_mhr(skeleton_json_path, mesh_obj_path, pose_json_path):
    # F matrix: MHR Y-up to Blender Z-up conversion
    F = mathutils.Matrix((
        (1, 0, 0, 0),
        (0, 0, -1, 0),
        (0, 1, 0, 0),
        (0, 0, 0, 1)
    ))

    def fix_coords(vec):
        return mathutils.Vector((vec[0], -vec[2], vec[1]))

    def get_mhr_matrix(t_mhr):
        # t_mhr: [tx, ty, tz, qx, qy, qz, qw, scale]
        tx, ty, tz = t_mhr[0], t_mhr[1], t_mhr[2]
        qx, qy, qz, qw = t_mhr[3], t_mhr[4], t_mhr[5], t_mhr[6]
        # mathutils.Quaternion is (w, x, y, z)
        rot = mathutils.Quaternion((qw, qx, qy, qz)).to_matrix().to_4x4()
        rot.translation = mathutils.Vector((tx, ty, tz))
        return rot

    # 1. Load Data
    if not os.path.exists(skeleton_json_path):
        print(f"Error: {skeleton_json_path} not found.")
        return

    with open(skeleton_json_path, "r") as f:
        data = json.load(f)
    joint_names = data["joint_names"]
    joint_parents = data["joint_parents"]
    rest_transforms = data["transforms"] # Global transforms at rest
    all_weights = data["weights"]

    # 2. Import Mesh
    if not os.path.exists(mesh_obj_path):
        print(f"Error: {mesh_obj_path} not found.")
        return
        
    mesh_data = bpy.data.meshes.new("MHR_Mesh")
    mesh_obj = bpy.data.objects.new("MHR_Mesh", mesh_data)
    bpy.context.collection.objects.link(mesh_obj)
    verts = []
    faces = []
    with open(mesh_obj_path, 'r') as f:
        for line in f:
            if line.startswith('v '):
                v = [float(x) for x in line.split()[1:]]
                verts.append(fix_coords(v))
            elif line.startswith('f '):
                f_indices = [int(x.split('/')[0]) - 1 for x in line.split()[1:]]
                faces.append(f_indices)
    mesh_data.from_pydata(verts, [], faces)
    mesh_data.update()

    # 3. Create Armature
    arm_data = bpy.data.armatures.new("MHR_Armature")
    arm_obj = bpy.data.objects.new("MHR_Armature", arm_data)
    bpy.context.collection.objects.link(arm_obj)
    bpy.context.view_layer.objects.active = arm_obj
    
    bpy.ops.object.mode_set(mode='EDIT')
    bones = []
    for i, name in enumerate(joint_names):
        bone = arm_data.edit_bones.new(name)
        bone.head = fix_coords(rest_transforms[i][0:3])
        bones.append(bone)
    
    for i, p in enumerate(joint_parents):
        if p != -1:
            bones[i].parent = bones[p]
    
    # Set tails for better visualization
    for i, bone in enumerate(bones):
        children = [j for j, p in enumerate(joint_parents) if p == i]
        if children:
            bone.tail = fix_coords(rest_transforms[children[0]][0:3])
            if (bone.tail - bone.head).length < 1e-4:
                 bone.tail += mathutils.Vector((0, 0, 0.01))
        else:
            if bone.parent:
                direction = bone.head - bone.parent.head
                if direction.length > 1e-4:
                    bone.tail = bone.head + direction.normalized() * 0.05
                else:
                    bone.tail = bone.head + mathutils.Vector((0, 0, 0.05))
            else:
                bone.tail = bone.head + mathutils.Vector((0, 0, 0.05))

    bpy.ops.object.mode_set(mode='OBJECT')

    # 4. Weight Painting
    if mesh_obj:
        for name in joint_names:
            mesh_obj.vertex_groups.new(name=name)
        for v_idx, v_weights in enumerate(all_weights):
            for j_idx, weight in v_weights:
                if weight > 0:
                    mesh_obj.vertex_groups[joint_names[j_idx]].add([v_idx], weight, 'REPLACE')
        mesh_obj.parent = arm_obj
        modifier = mesh_obj.modifiers.new(name="Armature", type='ARMATURE')
        modifier.object = arm_obj

    # 5. Apply Pose (Relative Transformation Approach)
    if os.path.exists(pose_json_path):
        with open(pose_json_path, "r") as f:
            pose_data = json.load(f)
        pose_transforms = pose_data["transforms"]
        
        bpy.ops.object.mode_set(mode='POSE')
        
        for i, name in enumerate(joint_names):
            bone = arm_obj.pose.bones.get(name)
            if not bone or i >= len(pose_transforms):
                continue
            
            # Get global matrices in MHR space
            m_rest = get_mhr_matrix(rest_transforms[i])
            m_pose = get_mhr_matrix(pose_transforms[i])
            
            # Calculate the transformation that takes the joint from rest to pose
            # T_mhr = M_pose * M_rest.inverted()
            delta_mhr = m_pose @ m_rest.inverted()
            
            # Convert this transformation to Blender space
            # T_blender = F * T_mhr * F_inv
            delta_b = F @ delta_mhr @ F.inverted()
            
            # Apply the transformation to the Edit Bone matrix to get target world matrix
            # TargetWorld = T_blender * EditMatrix
            target_world_matrix = delta_b @ bone.bone.matrix_local
            
            if joint_parents[i] == -1:
                # Root bone: allow translation
                bone.matrix = target_world_matrix
            else:
                # Non-root bone: keep rigid distance from parent
                bpy.context.view_layer.update()
                current_pos = bone.matrix.to_translation()
                
                new_mat = target_world_matrix.copy()
                new_mat.translation = current_pos
                bone.matrix = new_mat
            
            # Force update for hierarchy
            bpy.context.view_layer.update()
        
        bpy.ops.object.mode_set(mode='OBJECT')
        print("MHR Pose applied successfully.")

if __name__ == "__main__":
    cwd = os.getcwd()
    skeleton_path = os.path.join(cwd, "mhr_skeleton.json")
    mesh_path = os.path.join(cwd, "mhr_mesh.obj")
    pose_path = os.path.join(cwd, "mhr_pose.json")
    
    import_and_pose_mhr(skeleton_path, mesh_path, pose_path)
