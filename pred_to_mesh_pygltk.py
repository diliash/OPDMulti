import argparse
import copy
import json
import os
import sys
from glob import glob

import numpy as np
import open3d as o3d
import pycocotools
import seaborn as sns
import trimesh
from detectron2.data import DatasetCatalog, MetadataCatalog
from tqdm import tqdm

sys.path.append("./opdformer")
from mask2former import (
    MotionVisualizer,
    add_maskformer2_config,
    add_motionnet_config,
    register_motion_instances,
)
from PIL import Image
from plyfile import PlyData, PlyElement

sys.path.append("..")
import pygltftoolkit as pygltk

S2O_COLOR_MAP_RGBA = {
    0: (0, 107, 164, 255),
    1: (255, 128, 14, 255),
    2: (44, 160, 44, 255),
    3: (171, 171, 171, 255),
}

CONFIDENCE_THRESHOLD = 0.9
IOU_THRESHOLD = 0.8


def sample_and_export_points(trimesh_mesh, triangle_instance_mask, pred_inst_triangle_map, output_path, model_id):
    for instance_id, instance_dict in pred_inst_triangle_map.items():
        if not int(instance_id) in triangle_instance_mask:
            continue
        temp_mesh = copy.deepcopy(trimesh_mesh)
        visual_copy = copy.deepcopy(temp_mesh.visual)
        temp_mesh.update_faces(triangle_instance_mask == int(instance_id))
        temp_mesh.remove_unreferenced_vertices()
        temp_mesh.visual = visual_copy.face_subset(triangle_instance_mask == int(instance_id))

        if hasattr(temp_mesh.visual, "material"):
            if isinstance(temp_mesh.visual.material, trimesh.visual.material.PBRMaterial):
                temp_mesh.visual.material = temp_mesh.visual.material.to_simple()
            elif isinstance(temp_mesh.visual, trimesh.visual.color.ColorVisuals):
                temp_mesh.visual = temp_mesh.visual.to_simple()

        if hasattr(temp_mesh.visual, "material"):
            if temp_mesh.visual.uv is None or temp_mesh.visual.material.image is None:
                result = trimesh.sample.sample_surface(temp_mesh, 100000, sample_color=False)
                points = result[0]
                colors = np.array([temp_mesh.visual.material.main_color[:3] / 255] * 100000)
            else:
                result = trimesh.sample.sample_surface(temp_mesh, 100000, sample_color=True)
                points = result[0]
                colors = result[2][:, :3] / 255
        else:
            result = trimesh.sample.sample_surface(temp_mesh, 100000, sample_color=True)
            points = result[0]
            colors = result[2][:, :3] / 255

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        downpcd = pcd.voxel_down_sample(voxel_size=0.005)
        # o3d.visualization.draw_geometries([downpcd])
        with open(f"{output_path}/{model_id}-{instance_id}.npz", "wb+") as outfile:
            np.savez(outfile, points=np.array(downpcd.points), colors=np.array(downpcd.colors), instance=np.asarray(instance_id), semantic=np.asarray(instance_dict["semantic"]))


def save_nonindexed_geometry(mesh: trimesh.Trimesh, save_path: str, export_type: str = 'obj'):
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError('Input must be a trimesh.Trimesh object')

    vertices = mesh.vertices
    faces = mesh.faces
    face_colors = mesh.visual.face_colors

    nonindexed_vertices = []
    nonindexed_faces = []

    # Create truly non-indexed vertices and faces
    for face_idx, face in enumerate(faces):
        face_vertices = vertices[face]
        start_idx = len(nonindexed_vertices)
        nonindexed_vertices.extend(face_vertices)
        nonindexed_faces.append([start_idx, start_idx + 1, start_idx + 2])

    nonindexed_mesh = trimesh.Trimesh(vertices=nonindexed_vertices, faces=nonindexed_faces, process=False)
    nonindexed_mesh.visual.face_colors = face_colors

    if export_type == 'obj':
        nonindexed_mesh.export(save_path)
    elif export_type == 'ply':
        vertex_data = np.array([(v[0], v[1], v[2]) for v in nonindexed_vertices],
                            dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])

        face_data = np.array([(f, c[0], c[1], c[2], c[3]) 
                            for f, c in zip(nonindexed_faces, face_colors)],
                            dtype=[('vertex_indices', 'i4', (3,)),
                                   ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'), ('alpha', 'u1')])

        vertex_element = PlyElement.describe(vertex_data, 'vertex')
        face_element = PlyElement.describe(face_data, 'face')

        ply_data = PlyData([vertex_element, face_element], text=True)
        ply_data.write(save_path)
    return nonindexed_mesh


def triangle_area(coordinates):
    p1, p2, p3 = coordinates
    v1 = np.array(p2) - np.array(p1)
    v2 = np.array(p3) - np.array(p1)
    area_vector = np.cross(v1, v2)
    return np.linalg.norm(area_vector) / 2.0


def compute_triangle_areas(triangle_dict):
    areas = np.array([triangle_area(triangle_dict[i]) for i in triangle_dict])
    return areas


def compute_iou(mask1, mask2, triangle_dict):
    mask1 = mask1.astype(bool)
    mask2 = mask2.astype(bool)

    triangle_areas = compute_triangle_areas(triangle_dict)

    intersection = np.logical_and(mask1, mask2).astype(np.float32) * triangle_areas
    union = np.logical_or(mask1, mask2).astype(np.float32) * triangle_areas

    iou = intersection.sum() / union.sum()
    return iou


def sort_dicts_by_field(dicts, field):
    return sorted(dicts, key=lambda x: x[field], reverse=True)


def color_to_index(color):
    return color[0] * 256 * 256 + color[1] * 256 + color[2]


def register_datasets(data_path):
    dataset_keys = ["MotionNet_train", "MotionNet_valid"]
    for dataset_key in dataset_keys:
        json = f"{data_path}/annotations/{dataset_key}.json"
        imgs = f"{data_path}/{dataset_key.split('_')[-1]}"
        register_motion_instances(dataset_key, {}, json, imgs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-p', '--pred_path', type=str, required=True, 
                        help='Path to predictions (coco_motion_results.json)')
    parser.add_argument('-d', '--dataset', type=str, required=True,
                        help='specify path to dataset')
    parser.add_argument('-r', '--renderings_dir', type=str, required=True,
                        help='specify path to renderings')
    parser.add_argument('-e', '--export_path', type=str, required=True)

    args = parser.parse_args()
    os.makedirs(f"{args.export_path}/obj")
    os.makedirs(f"{args.export_path}/parts")
    os.makedirs(f"{args.export_path}/pcd")
    os.makedirs(f"{args.export_path}/pred")
    os.makedirs(f"{args.export_path}/motion")

    with open(f"{args.pred_path}/coco_motion_results.json", "r") as f:
        predictions = json.load(f)

    model_ids = np.sort([path.split("/")[-1] for path in glob(f"{args.renderings_dir}/origin/*") if len(os.listdir(path)) > 0])

    image_id_filename_map = {}
    image_id = 0
    for model_id in model_ids:
        for i in range(3):
            filename = f"{model_id}.png-{i}.png"
            image_id_filename_map[str(image_id)] = filename
            image_id += 1

    image_id_extrinsic_map = {}

    model_id_aggregated_predictions = {}
    prediction_id_image_id_map = {}
    for idx, prediction in enumerate(predictions):
        if prediction["score"] >= CONFIDENCE_THRESHOLD:
            prediction["id"] = str(idx)
            prediction_id_image_id_map[str(idx)] = prediction["image_id"]

            model_id = image_id_filename_map[str(prediction["image_id"])].split(".")[0]
            image_n = image_id_filename_map[str(prediction["image_id"])].split(".")[1].split("-")[1]
            with open(f"{args.renderings_dir}/cameras/{model_id}/{image_n}.json", "r") as f:
                camera = json.load(f)
            image_id_extrinsic_map[str(prediction["image_id"])] = camera["camera"]["extrinsic"]["matrix"]
            if model_id in model_id_aggregated_predictions.keys():
                model_id_aggregated_predictions[model_id].append(prediction)
            else:
                model_id_aggregated_predictions[model_id] = [prediction]

    for model_id, model_predictions in tqdm(model_id_aggregated_predictions.items()):
        image_id_cache = {}
        gltf_path = f"{args.dataset}/{model_id}/{model_id}.glb"
        if not os.path.exists(gltf_path):
            continue
        gltf = pygltk.load(gltf_path)
        shape_triangles_map = {}
        for i, triangle in enumerate(gltf.vertices[gltf.faces]):
            shape_triangles_map[str(i)] = triangle
        triangle_index_instance_map = -np.ones(len(gltf.faces), dtype=np.int_)
        model_predictions = sort_dicts_by_field(model_predictions, "score")
        for prediction in model_predictions:
            image_id = str(prediction["image_id"])
            motion = {"mtype": prediction["mtype"],
                      "morigin": prediction["morigin"],
                      "maxis": prediction["maxis"],
                      "mextrinsic": prediction["mextrinsic"],
                      "gtextrinsic": image_id_extrinsic_map[image_id],
                      "image_id": image_id}
            prediction_id = prediction["id"]
            current_score = prediction["score"]
            label = prediction["category_id"]
            if image_id not in image_id_cache.keys():
                im = Image.open(f"{args.renderings_dir}/triindex/{model_id}/{image_id_filename_map[image_id][:-6]}.faceIndex{image_id_filename_map[image_id][-6:]}")
                visible_faces = -np.ones([256, 256], dtype=np.intc)
                for i in range(256):
                    for j in range(256):
                        pixel = im.getpixel((i, j))
                        if pixel[3] != 0:
                            visible_faces[j][i] = color_to_index(pixel)
                mask_instance_map_data = {}
                mask_instance_map_data[prediction_id] = {"score": current_score, "label": label, "motion": motion}
                mask_instnace_map = -np.ones([256, 256], dtype=np.int_)
                image_id_cache[image_id] = [mask_instance_map_data, visible_faces]
            else:
                mask_instance_map_data, visible_faces = image_id_cache[image_id]
                mask_instance_map_data[prediction_id] = {"score": current_score, "label": label, "motion": motion}
                image_id_cache[image_id] = [mask_instance_map_data, visible_faces]
            rle = {
                'size': prediction['segmentation']['size'],
                'counts': prediction['segmentation']['counts'].encode()
            }
            mask_decoded = np.asarray(pycocotools.mask.decode(rle), dtype=bool)
            """from matplotlib import pyplot as plt
            plt.figure(figsize=(6, 6))
            colored_mask = np.zeros((*mask_decoded.shape, 4))
            colored_mask[visible_faces > 0] = np.asarray([1, 1, 1, 1])
            colored_mask[mask_decoded > 0] = np.asarray(S2O_COLOR_MAP_RGBA[label - 1]) / 255
            print(current_score)
            plt.imshow(colored_mask)
            plt.axis('off')
            plt.show()"""

            faces_mask = np.zeros(len(gltf.faces), dtype=bool)
            faces_mask[visible_faces[mask_decoded]] = True

            currently_assigned = triangle_index_instance_map[visible_faces[mask_decoded]]
            occupied = np.where(currently_assigned >= 0)[0]

            if len(occupied):
                overlapping_instances = np.unique(currently_assigned[occupied])
                ious = []
                for overlapping_instance in overlapping_instances:
                    overlapping_instance_mask = triangle_index_instance_map == overlapping_instance
                    ious.append({"id": str(overlapping_instance), "iou": compute_iou(faces_mask, overlapping_instance_mask, shape_triangles_map)})
                ious = sort_dicts_by_field(ious, "iou")

                recompute_iou = False
                for ious_dict in ious:
                    overlapping_instance, iou = ious_dict.values()
                    if recompute_iou:
                        overlapping_instance_mask = triangle_index_instance_map == int(overlapping_instance)
                        iou = compute_iou(faces_mask, overlapping_instance_mask, shape_triangles_map)
                    overlapping_mask_instance_map_data, overlapping_visible_faces = image_id_cache[str(prediction_id_image_id_map[overlapping_instance])]
                    overlapping_instance_score = overlapping_mask_instance_map_data[str(overlapping_instance)]["score"]
                    overlapping_instance_mask = triangle_index_instance_map == int(overlapping_instance)
                    if iou >= IOU_THRESHOLD:
                        if overlapping_instance_score > current_score:
                            mask_instance_map_data, visible_faces = image_id_cache[image_id]
                            mask_instance_map_data.pop(prediction_id)
                            image_id_cache[image_id] = [mask_instance_map_data, visible_faces]

                            image_id = str(prediction_id_image_id_map[overlapping_instance])
                            prediction_id_image_id_map[prediction_id] = image_id
                            prediction_id = overlapping_instance

                            faces_mask = np.logical_or(faces_mask, overlapping_instance_mask)
                            current_score = overlapping_mask_instance_map_data[str(overlapping_instance)]["score"]
                        else:
                            faces_mask = np.logical_or(faces_mask, overlapping_instance_mask)
                            overlapping_mask_instance_map_data.pop(overlapping_instance)
                            image_id_cache[str(prediction_id_image_id_map[overlapping_instance])] = [overlapping_mask_instance_map_data, overlapping_visible_faces]
                            prediction_id_image_id_map[overlapping_instance] = image_id
                    else:
                        overlapping_mask = np.logical_and(overlapping_instance_mask, faces_mask)
                        if overlapping_instance_score > current_score:
                            faces_mask = np.logical_xor(faces_mask, overlapping_mask)
                        else:
                            faces_mask = np.logical_or(faces_mask, overlapping_mask)

            triangle_index_instance_map[faces_mask] = prediction_id
            mask_instance_map_data, visible_faces = image_id_cache[image_id]
            mask_instance_map_data[prediction_id] = {"score": current_score, "label": label, "motion": motion}
            image_id_cache[image_id] = [mask_instance_map_data, visible_faces]

        temp_mesh = trimesh.Trimesh(vertices=gltf.vertices, faces=gltf.faces, process=False)

        flattened_triangles = np.transpose(np.array([triangle.flatten() for triangle in temp_mesh.triangles]))
        kdtree = o3d.geometry.KDTreeFlann(flattened_triangles)

        face_colors = np.asarray([S2O_COLOR_MAP_RGBA[3]] * len(temp_mesh.triangles), dtype=np.uint8)
        face_instance_colors = np.asarray([S2O_COLOR_MAP_RGBA[3]] * len(temp_mesh.triangles), dtype=np.uint8)
        triangles_map = {}
        segmentation_map = {}
        segmentation_mask = -np.ones(len(temp_mesh.triangles), dtype=np.int_)
        n_instances = len(np.unique(triangle_index_instance_map))
        color_map = sns.color_palette("husl", as_cmap=True)
        pred_inst_triangle_map = {}
        triangle_instance_mask = np.zeros(len(temp_mesh.triangles), dtype=np.int_)

        with open(f"{args.renderings_dir}/triangle_maps/{model_id}.json", "r") as f:
            shape_triangles_map = json.load(f)

        # In case triangle indices of rendered mesh do not match with the ones in the gltf file
        """flattened_triangles = np.transpose(np.array([triangle.flatten() for triangle in gltf.vertices[gltf.faces]]))
        kdtree = o3d.geometry.KDTreeFlann(flattened_triangles)

        stk_to_pygltk_triangle_index_map = {}

        for stk_triangle_index, stk_triangle_vertices in shape_triangles_map.items():
            query_triangle = np.array(stk_triangle_vertices).flatten()
            _, idx, _ = kdtree.search_knn_vector_xd(query_triangle, 1)
            corresponding_index = idx[0]
            stk_to_pygltk_triangle_index_map[stk_triangle_index] = corresponding_index"""

        base_part_id = None
        for instance_id, prediction_id in enumerate(np.unique(triangle_index_instance_map)):
            stk_indexes = np.where(triangle_index_instance_map == prediction_id)[0]
            # For non-matching triangle indices
            # trimesh_indexes = np.asarray([stk_to_pygltk_triangle_index_map[str(stk_id)] for stk_id in stk_indexes])
            trimesh_indexes = np.asarray([stk_id for stk_id in stk_indexes])
            segmentation_map[str(instance_id)] = {"triangles": [], "semantic": None, "geometries": []}
            if prediction_id == -1:
                instance_color = color_map(instance_id / n_instances)
                face_instance_colors[trimesh_indexes] = np.asarray([np.asarray(instance_color) * 255] * len(trimesh_indexes))
                face_colors[trimesh_indexes] = np.asarray(S2O_COLOR_MAP_RGBA[3])
                segmentation_map[str(instance_id)]["semantic"] = str(3)
                segmentation_mask[trimesh_indexes] = instance_id
                label = 3
                base_part_id = instance_id
                motion = None
            else:
                mask_instance_map_data, _ = image_id_cache[str(prediction_id_image_id_map[str(prediction_id)])]
                label = mask_instance_map_data[str(prediction_id)]["label"]
                motion = mask_instance_map_data[str(prediction_id)]["motion"]
                face_colors[trimesh_indexes] = np.asarray(S2O_COLOR_MAP_RGBA[label - 1])
                instance_color = color_map(instance_id / n_instances)
                face_instance_colors[trimesh_indexes] = np.asarray([np.asarray(instance_color) * 255] * len(trimesh_indexes))
                segmentation_map[str(instance_id)]["semantic"] = str(label - 1)
                segmentation_mask[trimesh_indexes] = instance_id
            triangle_instance_mask[trimesh_indexes] = instance_id
            pred_inst_triangle_map[str(instance_id)] = {"semantic": label - 1, "instance": instance_id}
            if motion:
                with open(f"{args.export_path}/motion/{model_id}-{instance_id}.json", "w+") as f:
                    json.dump(motion, f)
        segmentation_mask[segmentation_mask == -1] = base_part_id
        colored_mesh = gltf.create_colored_trimesh(face_colors / 255)
        nonindexed_mesh = save_nonindexed_geometry(colored_mesh, f"{args.export_path}/obj/{model_id}.obj")
        os.makedirs(f"{args.export_path}/parts/{model_id}", exist_ok=True)
        for part_idx in np.unique(segmentation_mask):
            part_mesh = copy.deepcopy(nonindexed_mesh)
            part_mesh.update_faces(segmentation_mask == part_idx)
            part_mesh.remove_unreferenced_vertices()
            save_nonindexed_geometry(part_mesh, f"{args.export_path}/parts/{model_id}/{part_idx}.obj")

        _ = sample_and_export_points(temp_mesh, triangle_instance_mask, pred_inst_triangle_map, f"{args.export_path}/pcd", model_id)

        semantic_segmentation_map = np.zeros(len(temp_mesh.triangles), dtype=np.int_)
        for part_idx in np.unique(segmentation_mask):
            semantic_segmentation_map[segmentation_mask == part_idx] = int(segmentation_map[str(part_idx)]["semantic"])

        np.savez(f"{args.export_path}/pred/{model_id}.npz", instance=triangle_instance_mask, semantic=semantic_segmentation_map)
