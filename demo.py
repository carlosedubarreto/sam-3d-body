# Copyright (c) Meta Platforms, Inc. and affiliates.
import argparse
import os
from glob import glob
import pickle

import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml", ".sl"],
    pythonpath=True,
    dotenv=True,
)

import cv2
import numpy as np
import torch
from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator
from tools.vis_utils import visualize_sample, visualize_sample_together
from tqdm import tqdm

import pandas as pd
import pyarrow, fastparquet

#### Inicio
## Retirado do pkl2parquet que foi criado com ajuda do Cluade e Gemini, para converter PKL que adicionei como extracao do SAM 3d Body
# importante a geracao do parquet pois é o dado utilizado no SOMA-X para converter do MHR para SOMA-x
# ---------------------------------------------------------------------------
# PKL layout constants
# ---------------------------------------------------------------------------
SCALE_PARAMS_IN_PKL = 28   # stored in PKL (first N of the 68 MHR expects)
SCALE_PARAMS_TOTAL  = 68   # MHR model_params[136:204]
BODY_POSE_STORED    = 133  # body_pose_params in PKL
BODY_POSE_USED      = 130  # drop last 3 (flexible bone-length, typically 0)
MODEL_PARAMS_DIM    = 204  # total expected by mhr2soma


def frame_to_mhr_params(frame: dict) -> tuple[np.ndarray, np.ndarray]:
    """Convert one SAM 3D Body PKL frame to (shape_params, model_params).

    Returns
    -------
    shape_params : (45,) float32
    model_params : (204,) float32  layout matches mhr2soma expectations
    """
    shape_params = frame["shape_params"].astype(np.float32)           # (45,)

    # --- pack model_params (204) ---
    # [0:3] global translation: pred_cam_t is in metres, MHR expects centimetres
    translation_cm = (frame["pred_cam_t"] * 100.0).astype(np.float32) # (3,)

    # [3:6] global orientation (axis-angle, radians)
    global_rot = frame["global_rot"].astype(np.float32)               # (3,)

    # [6:136] body joint rotations: 130 values (drop last 3 flex-bone params)
    body_pose = frame["body_pose_params"][:BODY_POSE_USED].astype(np.float32)  # (130,)

    # [136:204] scale params: zero-pad from 28 → 68
    scale_full = np.zeros(SCALE_PARAMS_TOTAL, dtype=np.float32)
    scale_full[:SCALE_PARAMS_IN_PKL] = frame["scale_params"].astype(np.float32)

    model_params = np.concatenate(
        [translation_cm, global_rot, body_pose, scale_full]
    )  # (3 + 3 + 130 + 68) = 204

    assert model_params.shape == (MODEL_PARAMS_DIM,), (
        f"model_params shape {model_params.shape} != ({MODEL_PARAMS_DIM},)"
    )
    return shape_params, model_params

###### Fim

def main(args):
    if args.output_folder == "":
        output_folder = os.path.join("./output", os.path.basename(args.image_folder))
    else:
        output_folder = args.output_folder

    os.makedirs(output_folder, exist_ok=True)

    # Use command-line args or environment variables
    mhr_path = args.mhr_path or os.environ.get("SAM3D_MHR_PATH", "")
    detector_path = args.detector_path or os.environ.get("SAM3D_DETECTOR_PATH", "")
    segmentor_path = args.segmentor_path or os.environ.get("SAM3D_SEGMENTOR_PATH", "")
    fov_path = args.fov_path or os.environ.get("SAM3D_FOV_PATH", "")

    # Initialize sam-3d-body model and other optional modules
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model, model_cfg = load_sam_3d_body(
        args.checkpoint_path, device=device, mhr_path=mhr_path
    )

    human_detector, human_segmentor, fov_estimator = None, None, None
    if args.detector_name:
        from tools.build_detector import HumanDetector

        human_detector = HumanDetector(
            name=args.detector_name, device=device, path=detector_path
        )
    
    if (args.segmentor_name == "sam2" and len(segmentor_path)) or args.segmentor_name != "sam2":
        from tools.build_sam import HumanSegmentor

        human_segmentor = HumanSegmentor(
            name=args.segmentor_name, device=device, path=segmentor_path
        )
    if args.fov_name:
        from tools.build_fov_estimator import FOVEstimator

        fov_estimator = FOVEstimator(name=args.fov_name, device=device, path=fov_path)

    estimator = SAM3DBodyEstimator(
        sam_3d_body_model=model,
        model_cfg=model_cfg,
        human_detector=human_detector,
        human_segmentor=human_segmentor,
        fov_estimator=fov_estimator,
    )

    image_extensions = [
        "*.jpg",
        "*.jpeg",
        "*.png",
        "*.gif",
        "*.bmp",
        "*.tiff",
        "*.webp",
    ]
    images_list = sorted(
        [
            image
            for ext in image_extensions
            for image in glob(os.path.join(args.image_folder, ext))
        ]
    )

    for image_path in tqdm(images_list):
        outputs = estimator.process_one_image(
            image_path,
            bbox_thr=args.bbox_thresh,
            use_mask=args.use_mask,
        )

        #save parameters
        pickle_file = f"{output_folder}/{os.path.basename(image_path)[:-4]}.pkl"
        with open(pickle_file, "wb") as f:
            pickle.dump(outputs, f)

        #### Inicio
        #### Salvando o Parquet
        parquet_path = f"{output_folder}/{os.path.basename(image_path)[:-4]}.parquet"

        rows = []
        skipped = 0
        for i, frame in enumerate(outputs):
            # # Basic sanity check
            # missing = [k for k in ("shape_params", "pred_cam_t", "global_rot",
            #                     "body_pose_params", "scale_params") if k not in frame]
            # if missing:
            #     if skip_invalid:
            #         print(f"  Frame {i}: skipping — missing keys: {missing}")
            #         skipped += 1
            #         continue
            #     raise KeyError(f"Frame {i} missing required keys: {missing}")

            shape_p, model_p = frame_to_mhr_params(frame)
            rows.append({
                "shape_params": shape_p,
                "model_params": model_p,
                "mhr_valid":    True,
                # "dataset":      str(pkl_path.stem),
                "image":        str(i),
            })

        if not rows:
            raise RuntimeError("No valid frames found in the PKL file.")

        df = pd.DataFrame(rows)
        # parquet_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(parquet_path, index=False)

        print(f"Wrote {len(rows)} frames → {parquet_path}")
        if skipped:
            print(f"  ({skipped} frames skipped due to missing keys)")
        ###### Fim
        ###########################

        img = cv2.imread(image_path)
        rend_img = visualize_sample_together(img, outputs, estimator.faces)
        cv2.imwrite(
            f"{output_folder}/{os.path.basename(image_path)[:-4]}.jpg",
            rend_img.astype(np.uint8),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SAM 3D Body Demo - Single Image Human Mesh Recovery",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
                Examples:
                python demo.py --image_folder ./images --checkpoint_path ./checkpoints/model.ckpt

                Environment Variables:
                SAM3D_MHR_PATH: Path to MHR asset
                SAM3D_DETECTOR_PATH: Path to human detection model folder
                SAM3D_SEGMENTOR_PATH: Path to human segmentation model folder
                SAM3D_FOV_PATH: Path to fov estimation model folder
                """,
    )
    parser.add_argument(
        "--image_folder",
        required=True,
        type=str,
        help="Path to folder containing input images",
    )
    parser.add_argument(
        "--output_folder",
        default="",
        type=str,
        help="Path to output folder (default: ./output/<image_folder_name>)",
    )
    parser.add_argument(
        "--checkpoint_path",
        required=True,
        type=str,
        help="Path to SAM 3D Body model checkpoint",
    )
    parser.add_argument(
        "--detector_name",
        default="vitdet",
        type=str,
        help="Human detection model for demo (Default `vitdet`, add your favorite detector if needed).",
    )
    parser.add_argument(
        "--segmentor_name",
        default="sam2",
        type=str,
        help="Human segmentation model for demo (Default `sam2`, add your favorite segmentor if needed).",
    )
    parser.add_argument(
        "--fov_name",
        default="moge2",
        type=str,
        help="FOV estimation model for demo (Default `moge2`, add your favorite fov estimator if needed).",
    )
    parser.add_argument(
        "--detector_path",
        default="",
        type=str,
        help="Path to human detection model folder (or set SAM3D_DETECTOR_PATH)",
    )
    parser.add_argument(
        "--segmentor_path",
        default="",
        type=str,
        help="Path to human segmentation model folder (or set SAM3D_SEGMENTOR_PATH)",
    )
    parser.add_argument(
        "--fov_path",
        default="",
        type=str,
        help="Path to fov estimation model folder (or set SAM3D_FOV_PATH)",
    )
    parser.add_argument(
        "--mhr_path",
        default="",
        type=str,
        help="Path to MoHR/assets folder (or set SAM3D_mhr_path)",
    )
    parser.add_argument(
        "--bbox_thresh",
        default=0.8,
        type=float,
        help="Bounding box detection threshold",
    )
    parser.add_argument(
        "--use_mask",
        action="store_true",
        default=False,
        help="Use mask-conditioned prediction (segmentation mask is automatically generated from bbox)",
    )
    args = parser.parse_args()

    main(args)
