import os, sys
from typing import List
from pathlib import Path

import onnx
import torch
import onnxsim
import numpy as np
from onnxsim import simplify

from modules.xfeat import XFeat
from modules.lighterglue import LighterGlue


ONNX_OPSET_VERSION = 17


def sim(model_path):
    print(f"Simlifying {model_path}")
    onnx_model = onnx.load(model_path)
    model_simp, check = onnxsim.simplify(onnx_model)
    onnx.save(model_simp, model_path)


def export_onnx(
    xfeat_path=None,
    output_folder="onnx",
    input_shape=(1, 1, 800, 800),
    ligherglue_n_layers=3,
    dynamic=True,
    dense=False,
    top_k=512,
):
    dummy_input = torch.randn(input_shape, requires_grad=True)
    print(f'img0 shape: {dummy_input.shape}')

    # Models
    print(f'keypoints: {top_k}')
    xfeat = XFeat(weights=xfeat_path, top_k=top_k, detection_threshold=0.05).eval()

    if 1:
        # -----------------
        # Export Extractor
        # -----------------
        dynamic_axes = {
            "keypoints": {0: "num_keypoints"},
            "descriptors": {0: "num_keypoints", 1: "descriptor_dim"},
        }

        if dense:
            output_path = xfeat_path.replace(".pt", "_dense.onnx")
            xfeat.forward = xfeat.detectAndComputeDense
            dynamic_axes.update({"scales": {0: "num_keypoints"}})
            output_names = ["keypoints", "descriptors", "scales"]
        else:
            output_path = xfeat_path.replace(".pt", ".onnx")
            xfeat.forward = xfeat.detectAndCompute
            dynamic_axes.update({"scores": {0: "num_keypoints"}})
            output_names = ["keypoints", "descriptors", "scores"]

        # Add dynamic input
        if dynamic:
            dynamic_axes.update({
                "images": {1: "channel", 2: "height", 3: "width"},
            })
        else:
            print(
                f"WARNING: Exporting without --dynamic implies that the extractor's input image size will be locked to {dummy_input.shape[-2:]}"
            )
            output_path = output_path.replace(
                ".onnx",
                f"_{top_k}_{dummy_input.shape[-2]}x{dummy_input.shape[-1]}.onnx"
            )
            
        output_path = os.path.join(output_folder, os.path.basename(output_path))

        # Export model
        torch.onnx.export(
            xfeat,
            dummy_input,
            output_path,
            verbose=False,
            do_constant_folding=True,
            input_names=["images"],
            output_names=output_names,
            opset_version=ONNX_OPSET_VERSION,
            dynamic_axes=dynamic_axes if dynamic else None,
        )
        sim(output_path)

        # -----------------
        # Export Matching
        # -----------------

        # Simulate keypoints, features
        top_k = 4096 if top_k is None else top_k
        kpts = torch.rand(1, top_k, 2, dtype=torch.float32) * 2 - 1
        print(kpts.shape)
        # sys.exit(0)
        desc = torch.rand(1, top_k, 64, dtype=torch.float32)

        # Dynamic input
        dynamic_axes={
            "kpts0":    {1: "num_keypoints0"},
            "kpts1":    {1: "num_keypoints1"},
            "desc0":    {1: "num_keypoints0"},
            "desc1":    {1: "num_keypoints1"},
            "matches":  {0: "num_matches"},
            "scores":   {0: "num_matches"},
        }

        # if dense:
        #     output_matching_path = os.path.join(os.path.dirname(output_path), "matching_dense.onnx")
        #     xfeat.forward = xfeat.match_star_onnx
        #     dynamic_axes.update({"scales0": {0: "num_kpts0"},})
        #     input_names.append("scales0")
        #     input_values.append(scales)
        # else:
        #     output_matching_path = os.path.join(os.path.dirname(output_path), "matching.onnx")
        #     xfeat.forward = xfeat.match_onnx

        matcher = LighterGlue(n_layers=ligherglue_n_layers).eval()
        output_path = os.path.join(output_folder, f"lighterglue_L{ligherglue_n_layers}.onnx")
        torch.onnx.export(
            matcher,
            (kpts, kpts, desc, desc),
            output_path,
            verbose=False,
            do_constant_folding=True,
            input_names=["kpts0", "kpts1", "desc0", "desc1"],
            output_names=["matches", "scores"],
            opset_version=ONNX_OPSET_VERSION,
            dynamic_axes=None,
            # dynamic_axes=dynamic_axes if dynamic else None,
        )
        sim(output_path)



if __name__ == "__main__":
    export_onnx(
        xfeat_path="/home/tk/GNSS-Denial-UAV-Location/src/match_location/scripts/weights/xfeat.pt",
        output_folder="onnx",
        input_shape=(1, 1, 800, 800),
        ligherglue_n_layers=3,
        dynamic=True,
        dense=False,
        top_k=512,
    )