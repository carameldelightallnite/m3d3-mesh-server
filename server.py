# =========================================================
# M3D3 PLATINUM STYLE PRIM TO MESH SERVER
# BAKED LOW LI FINAL BUILD
#
# Version:
# M3D3_PLATINUM_STYLE_BAKED_LOW_LI_FINAL_2026_05_04
#
# Fixes:
# - Low LI DAE is now ONE baked mesh object, not one mesh instance per prim.
# - Smooth DAE keeps higher detail.
# - Advanced files remain available.
# - Root-scanned linked builds no longer export as 1:1 prim instances.
# - Strict Collada 1.4.1 Z_UP.
# =========================================================

import os
import time
import uuid
import json
import zipfile
import traceback
from typing import Any, Dict, List, Tuple

import numpy as np
import trimesh
from flask import Flask, request, jsonify, send_from_directory, Response

app = Flask(__name__)

VERSION = "M3D3_PLATINUM_STYLE_BAKED_LOW_LI_FINAL_2026_05_04"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

jobs: Dict[str, Dict[str, Any]] = {}
results: Dict[str, Dict[str, Any]] = {}

FILE_TTL_SECONDS = 3600
JOB_TTL_SECONDS = 1800

MIN_AXIS_SIZE = 0.001
MAX_SL_SIZE = 64.0

DEFAULT_QUALITY = 16
MIN_QUALITY = 4
MAX_QUALITY = 24

LOW_LI_SPHERE_DIVISIONS = 4
SMOOTH_SPHERE_DIVISIONS = 6
PREVIEW_SPHERE_DIVISIONS = 8
LOWEST_SPHERE_DIVISIONS = 2

LOW_LI_CYLINDER_SECTIONS = 10
SMOOTH_CYLINDER_SECTIONS = 24
PREVIEW_CYLINDER_SECTIONS = 36
LOWEST_CYLINDER_SECTIONS = 6

LOW_LI_CONE_SECTIONS = 10
SMOOTH_CONE_SECTIONS = 24
PREVIEW_CONE_SECTIONS = 36
LOWEST_CONE_SECTIONS = 6

LOW_LI_TORUS_MAJOR = 14
LOW_LI_TORUS_MINOR = 5
SMOOTH_TORUS_MAJOR = 28
SMOOTH_TORUS_MINOR = 10
PREVIEW_TORUS_MAJOR = 36
PREVIEW_TORUS_MINOR = 12
LOWEST_TORUS_MAJOR = 10
LOWEST_TORUS_MINOR = 4


def now_ts() -> float:
    return time.time()


def safe_name(value: Any) -> str:
    text = str(value or "M3D3_Build").replace(" ", "_")
    text = "".join(c for c in text if c.isalnum() or c == "_")
    if not text:
        text = "M3D3_Build"
    return text[:48]


def clean_filename(filename: str) -> str:
    return (
        os.path.basename(str(filename))
        .replace('"', "")
        .replace("\\", "")
        .replace("}", "")
        .replace("{", "")
        .strip()
    )


def cleanup_old_files() -> None:
    cutoff = now_ts() - FILE_TTL_SECONDS

    try:
        for filename in os.listdir(OUTPUT_DIR):
            path = os.path.join(OUTPUT_DIR, filename)
            if os.path.isfile(path) and os.path.getmtime(path) < cutoff:
                os.remove(path)
    except Exception as exc:
        print("cleanup_old_files error:", exc)


def cleanup_old_memory() -> None:
    job_cutoff = now_ts() - JOB_TTL_SECONDS
    result_cutoff = now_ts() - FILE_TTL_SECONDS

    for jid in list(jobs.keys()):
        if float(jobs[jid].get("created", 0.0)) < job_cutoff:
            del jobs[jid]

    for rid in list(results.keys()):
        if float(results[rid].get("created", 0.0)) < result_cutoff:
            del results[rid]


def parse_vec(value: Any, fallback: Tuple[float, float, float]) -> np.ndarray:
    try:
        text = str(value).replace("<", "").replace(">", "").strip()
        arr = np.fromstring(text, sep=",")
        if arr.size < 3:
            return np.array(fallback, dtype=float)
        arr = arr[:3].astype(float)
        return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    except Exception:
        return np.array(fallback, dtype=float)


def parse_rot(value: Any) -> np.ndarray:
    try:
        text = str(value).replace("<", "").replace(">", "").strip()
        q = np.fromstring(text, sep=",")

        if q.size < 4:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)

        q = np.nan_to_num(q[:4].astype(float), nan=0.0, posinf=0.0, neginf=0.0)
        length = np.linalg.norm(q)

        if not np.isfinite(length) or length <= 0.000001:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)

        q = q / length

        return np.array([q[3], q[0], q[1], q[2]], dtype=float)
    except Exception:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)


def safe_size(value: Any) -> np.ndarray:
    size = parse_vec(value, (1.0, 1.0, 1.0))
    size = np.abs(size)
    size[size < MIN_AXIS_SIZE] = MIN_AXIS_SIZE
    size[size > MAX_SL_SIZE] = MAX_SL_SIZE
    return size


def clamp_quality(value: Any) -> int:
    try:
        q = int(value)
    except Exception:
        q = DEFAULT_QUALITY

    if q < MIN_QUALITY:
        q = MIN_QUALITY

    if q > MAX_QUALITY:
        q = MAX_QUALITY

    return q


def clean_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    try:
        mesh.remove_infinite_values()
    except Exception:
        pass

    try:
        mesh.remove_duplicate_faces()
    except Exception:
        pass

    try:
        mesh.remove_degenerate_faces()
    except Exception:
        pass

    try:
        mesh.remove_unreferenced_vertices()
    except Exception:
        pass

    try:
        mesh.merge_vertices(digits_vertex=5)
    except Exception:
        try:
            mesh.merge_vertices()
        except Exception:
            pass

    try:
        mesh.fix_normals()
    except Exception:
        pass

    try:
        mesh.vertices = np.nan_to_num(mesh.vertices, nan=0.0, posinf=0.0, neginf=0.0)
    except Exception:
        pass

    return mesh


def is_generator_panel_prim(prim: Dict[str, Any]) -> bool:
    role = str(prim.get("role", "")).upper()
    source = str(prim.get("source", "")).upper()
    name = str(prim.get("name", "")).lower()

    if role == "GENERATOR":
        return True
    if source == "GENERATOR":
        return True
    if "generator" in name:
        return True
    if "gene_ato" in name:
        return True
    if "push" in name:
        return True
    if "panel" in name:
        return True

    return False


def shape_from_prim(prim: Dict[str, Any]) -> str:
    name = str(prim.get("name", "")).lower()
    prim_type = str(prim.get("type", "BOX")).upper()

    if "sphere" in name:
        return "SPHERE"
    if "cylinder" in name:
        return "CYLINDER"
    if "torus" in name:
        return "TORUS"
    if "ring" in name:
        return "RING"
    if "tube" in name:
        return "TUBE"
    if "prism" in name:
        return "PRISM"
    if "cone" in name:
        return "CONE"
    if "box" in name:
        return "BOX"

    if prim_type in ["BOX", "CYLINDER", "SPHERE", "TORUS", "RING", "TUBE", "PRISM", "CONE"]:
        return prim_type

    return "BOX"


def make_box_mesh() -> trimesh.Trimesh:
    return clean_mesh(trimesh.creation.box(extents=[1.0, 1.0, 1.0]))


def make_cylinder_mesh(sections: int) -> trimesh.Trimesh:
    sections = max(6, int(sections))
    return clean_mesh(trimesh.creation.cylinder(radius=0.5, height=1.0, sections=sections))


def make_cone_mesh(sections: int) -> trimesh.Trimesh:
    sections = max(6, int(sections))
    return clean_mesh(trimesh.creation.cone(radius=0.5, height=1.0, sections=sections))


def spherify_cube_point(x: float, y: float, z: float) -> List[float]:
    x2 = x * x
    y2 = y * y
    z2 = z * z

    sx = x * np.sqrt(max(0.0, 1.0 - (y2 / 2.0) - (z2 / 2.0) + (y2 * z2 / 3.0)))
    sy = y * np.sqrt(max(0.0, 1.0 - (z2 / 2.0) - (x2 / 2.0) + (z2 * x2 / 3.0)))
    sz = z * np.sqrt(max(0.0, 1.0 - (x2 / 2.0) - (y2 / 2.0) + (x2 * y2 / 3.0)))

    return [sx * 0.5, sy * 0.5, sz * 0.5]


def make_spherified_cube_sphere(divisions: int) -> trimesh.Trimesh:
    divisions = max(2, int(divisions))

    vertices: List[List[float]] = []
    faces: List[List[int]] = []
    index_map: Dict[str, int] = {}

    def key_for_point(p: List[float]) -> str:
        return f"{p[0]:.8f},{p[1]:.8f},{p[2]:.8f}"

    def get_vertex(x: float, y: float, z: float) -> int:
        p = spherify_cube_point(x, y, z)
        key = key_for_point(p)

        if key in index_map:
            return index_map[key]

        idx = len(vertices)
        index_map[key] = idx
        vertices.append(p)
        return idx

    def add_face(axis: str, sign: float) -> None:
        for i in range(divisions):
            for j in range(divisions):
                a = -1.0 + 2.0 * float(i) / float(divisions)
                b = -1.0 + 2.0 * float(j) / float(divisions)
                c = -1.0 + 2.0 * float(i + 1) / float(divisions)
                d = -1.0 + 2.0 * float(j + 1) / float(divisions)

                if axis == "x":
                    v0 = get_vertex(sign, a, b)
                    v1 = get_vertex(sign, c, b)
                    v2 = get_vertex(sign, c, d)
                    v3 = get_vertex(sign, a, d)
                elif axis == "y":
                    v0 = get_vertex(a, sign, b)
                    v1 = get_vertex(a, sign, d)
                    v2 = get_vertex(c, sign, d)
                    v3 = get_vertex(c, sign, b)
                else:
                    v0 = get_vertex(a, b, sign)
                    v1 = get_vertex(c, b, sign)
                    v2 = get_vertex(c, d, sign)
                    v3 = get_vertex(a, d, sign)

                if sign > 0:
                    faces.append([v0, v1, v2])
                    faces.append([v0, v2, v3])
                else:
                    faces.append([v0, v2, v1])
                    faces.append([v0, v3, v2])

    add_face("x", 1.0)
    add_face("x", -1.0)
    add_face("y", 1.0)
    add_face("y", -1.0)
    add_face("z", 1.0)
    add_face("z", -1.0)

    mesh = trimesh.Trimesh(
        vertices=np.array(vertices, dtype=float),
        faces=np.array(faces, dtype=int),
        process=False
    )

    return clean_mesh(mesh)


def make_torus_mesh(major_sections: int, minor_sections: int) -> trimesh.Trimesh:
    major_sections = max(8, int(major_sections))
    minor_sections = max(4, int(minor_sections))

    try:
        mesh = trimesh.creation.torus(
            major_radius=0.35,
            minor_radius=0.15,
            major_sections=major_sections,
            minor_sections=minor_sections
        )
    except TypeError:
        mesh = trimesh.creation.torus(
            radius=0.35,
            tube_radius=0.15,
            sections=major_sections,
            segments=minor_sections
        )

    return clean_mesh(mesh)


def make_prism_mesh() -> trimesh.Trimesh:
    vertices = np.array([
        [-0.5, -0.5, -0.5],
        [ 0.5, -0.5, -0.5],
        [ 0.0,  0.5, -0.5],
        [-0.5, -0.5,  0.5],
        [ 0.5, -0.5,  0.5],
        [ 0.0,  0.5,  0.5]
    ], dtype=float)

    faces = np.array([
        [0, 1, 2],
        [3, 5, 4],
        [0, 3, 4],
        [0, 4, 1],
        [1, 4, 5],
        [1, 5, 2],
        [2, 5, 3],
        [2, 3, 0]
    ], dtype=int)

    return clean_mesh(trimesh.Trimesh(vertices=vertices, faces=faces, process=False))


def build_base_mesh(shape: str, mode: str, quality: int) -> trimesh.Trimesh:
    if mode == "preview":
        sphere_divisions = 8
        cylinder_sections = max(PREVIEW_CYLINDER_SECTIONS, quality * 2)
        cone_sections = max(PREVIEW_CONE_SECTIONS, quality * 2)
        torus_major = max(PREVIEW_TORUS_MAJOR, quality * 2)
        torus_minor = max(PREVIEW_TORUS_MINOR, quality // 2)
    elif mode == "smooth":
        sphere_divisions = SMOOTH_SPHERE_DIVISIONS
        cylinder_sections = SMOOTH_CYLINDER_SECTIONS
        cone_sections = SMOOTH_CONE_SECTIONS
        torus_major = SMOOTH_TORUS_MAJOR
        torus_minor = SMOOTH_TORUS_MINOR
    elif mode == "lowest":
        sphere_divisions = LOWEST_SPHERE_DIVISIONS
        cylinder_sections = LOWEST_CYLINDER_SECTIONS
        cone_sections = LOWEST_CONE_SECTIONS
        torus_major = LOWEST_TORUS_MAJOR
        torus_minor = LOWEST_TORUS_MINOR
    else:
        sphere_divisions = LOW_LI_SPHERE_DIVISIONS
        cylinder_sections = LOW_LI_CYLINDER_SECTIONS
        cone_sections = LOW_LI_CONE_SECTIONS
        torus_major = LOW_LI_TORUS_MAJOR
        torus_minor = LOW_LI_TORUS_MINOR

    if shape == "SPHERE":
        return make_spherified_cube_sphere(sphere_divisions)
    if shape == "CYLINDER":
        return make_cylinder_mesh(cylinder_sections)
    if shape == "CONE":
        return make_cone_mesh(cone_sections)
    if shape in ["TORUS", "RING"]:
        return make_torus_mesh(torus_major, torus_minor)
    if shape == "TUBE":
        return make_cylinder_mesh(cylinder_sections)
    if shape == "PRISM":
        return make_prism_mesh()

    return make_box_mesh()


def apply_prim_transform(mesh: trimesh.Trimesh, prim: Dict[str, Any]) -> trimesh.Trimesh:
    mesh = mesh.copy()

    size = safe_size(prim.get("size", "<1,1,1>"))
    pos = parse_vec(prim.get("pos", "<0,0,0>"), (0.0, 0.0, 0.0))
    rot = parse_rot(prim.get("rot", "<0,0,0,1>"))
    shape = shape_from_prim(prim)

    if shape in ["TORUS", "RING"]:
        base_xy = max(float(size[0]), float(size[1]), MIN_AXIS_SIZE)
        mesh.apply_scale([
            size[0] / base_xy,
            size[1] / base_xy,
            max(size[2], MIN_AXIS_SIZE)
        ])
    else:
        mesh.apply_scale(size)

    matrix = trimesh.transformations.quaternion_matrix(rot)
    matrix[:3, 3] = pos
    mesh.apply_transform(matrix)

    return clean_mesh(mesh)


def filter_build_prims(prims: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    filtered: List[Dict[str, Any]] = []

    seen = set()

    for prim in prims:
        if not isinstance(prim, dict):
            continue

        if is_generator_panel_prim(prim):
            continue

        source = str(prim.get("source", ""))
        link = str(prim.get("link", ""))
        name = str(prim.get("name", ""))
        pos = str(prim.get("pos", ""))
        size = str(prim.get("size", ""))

        duplicate_key = source + "|" + link + "|" + name + "|" + pos + "|" + size

        if duplicate_key in seen:
            continue

        seen.add(duplicate_key)
        filtered.append(prim)

    return filtered


def build_mesh_list(prims: List[Dict[str, Any]], mode: str, quality: int) -> Tuple[List[trimesh.Trimesh], Dict[str, Any]]:
    filtered = filter_build_prims(prims)

    if not filtered:
        raise RuntimeError("No build prims were received. Generator panel output is rejected.")

    meshes: List[trimesh.Trimesh] = []

    for prim in filtered:
        shape = shape_from_prim(prim)
        base = build_base_mesh(shape, mode, quality)
        transformed = apply_prim_transform(base, prim)
        meshes.append(transformed)

    merged = clean_mesh(trimesh.util.concatenate(meshes))
    bounds = np.asarray(merged.bounds, dtype=float)
    min_corner = bounds[0]
    max_corner = bounds[1]
    center = (min_corner + max_corner) * 0.5
    dims = np.nan_to_num(max_corner - min_corner, nan=0.0, posinf=0.0, neginf=0.0)

    if float(np.max(dims)) > MAX_SL_SIZE:
        raise RuntimeError("Build exceeds Second Life 64m mesh upload limit.")

    centered_meshes: List[trimesh.Trimesh] = []

    for mesh in meshes:
        centered = mesh.copy()
        centered.apply_translation(-center)
        centered_meshes.append(clean_mesh(centered))

    centered_merged = clean_mesh(trimesh.util.concatenate(centered_meshes))

    report = {
        "count": len(centered_meshes),
        "dimensions": [float(x) for x in dims],
        "center_removed": [float(x) for x in center],
        "faces": int(len(centered_merged.faces)),
        "vertices": int(len(centered_merged.vertices))
    }

    return centered_meshes, report


def make_records_from_meshes(meshes: List[trimesh.Trimesh], prefix: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []

    for i, mesh in enumerate(meshes):
        records.append({
            "id": f"{prefix}_{i:04d}",
            "name": f"{prefix}_{i:04d}",
            "mesh": clean_mesh(mesh)
        })

    return records


def make_baked_record(meshes: List[trimesh.Trimesh], baked_id: str) -> List[Dict[str, Any]]:
    baked = clean_mesh(trimesh.util.concatenate(meshes))
    return [{
        "id": baked_id,
        "name": baked_id,
        "mesh": baked
    }]


def make_box_proxy_record(meshes: List[trimesh.Trimesh], proxy_id: str) -> List[Dict[str, Any]]:
    merged = clean_mesh(trimesh.util.concatenate(meshes))
    bounds = np.asarray(merged.bounds, dtype=float)
    dims = bounds[1] - bounds[0]
    center = (bounds[0] + bounds[1]) * 0.5

    dims = np.abs(dims)
    dims[dims < MIN_AXIS_SIZE] = MIN_AXIS_SIZE

    proxy = trimesh.creation.box(extents=dims)
    proxy.apply_translation(center)

    return [{
        "id": proxy_id,
        "name": proxy_id,
        "mesh": clean_mesh(proxy)
    }]


def xml_float_list(values: List[float]) -> str:
    return " ".join(f"{float(v):.6f}" for v in values)


def xml_int_list(values: List[int]) -> str:
    return " ".join(str(int(v)) for v in values)


def projected_uv(vertex: np.ndarray, min_corner: np.ndarray, dims: np.ndarray) -> Tuple[float, float]:
    dx = max(float(dims[0]), MIN_AXIS_SIZE)
    dy = max(float(dims[1]), MIN_AXIS_SIZE)

    u = (float(vertex[0]) - float(min_corner[0])) / dx
    v = (float(vertex[1]) - float(min_corner[1])) / dy

    if not np.isfinite(u):
        u = 0.0
    if not np.isfinite(v):
        v = 0.0

    return u, v


def geometry_xml_shared(record: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    geom_id = record["id"]
    mesh = clean_mesh(record["mesh"])

    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=int)

    if len(vertices) == 0 or len(faces) == 0:
        raise RuntimeError(f"{geom_id} has no valid mesh data.")

    try:
        vertex_normals = np.asarray(mesh.vertex_normals, dtype=float)
    except Exception:
        vertex_normals = np.zeros_like(vertices)
        vertex_normals[:, 2] = 1.0

    if len(vertex_normals) != len(vertices):
        vertex_normals = np.zeros_like(vertices)
        vertex_normals[:, 2] = 1.0

    min_corner = vertices.min(axis=0)
    max_corner = vertices.max(axis=0)
    dims = np.maximum(max_corner - min_corner, MIN_AXIS_SIZE)

    positions: List[float] = []
    normals: List[float] = []
    uvs: List[float] = []

    for i in range(len(vertices)):
        v = vertices[i]
        n = vertex_normals[i]
        nlen = np.linalg.norm(n)
        if not np.isfinite(nlen) or nlen <= 0.000001:
            n = np.array([0.0, 0.0, 1.0], dtype=float)
        else:
            n = n / nlen

        u, vv = projected_uv(v, min_corner, dims)

        positions.extend([float(v[0]), float(v[1]), float(v[2])])
        normals.extend([float(n[0]), float(n[1]), float(n[2])])
        uvs.extend([float(u), float(vv)])

    p_indices: List[int] = []
    tri_count = 0

    for face in faces:
        if len(face) != 3:
            continue

        i0 = int(face[0])
        i1 = int(face[1])
        i2 = int(face[2])

        if i0 == i1 or i1 == i2 or i0 == i2:
            continue

        p_indices.extend([i0, i0, i0, i1, i1, i1, i2, i2, i2])
        tri_count += 1

    if tri_count <= 0:
        raise RuntimeError(f"{geom_id} produced zero triangles.")

    vertex_count = len(vertices)

    xml = f'''
    <geometry id="{geom_id}" name="{geom_id}">
      <mesh>
        <source id="{geom_id}_POSITION">
          <float_array id="{geom_id}_POSITION_ARRAY" count="{len(positions)}">{xml_float_list(positions)}</float_array>
          <technique_common>
            <accessor source="#{geom_id}_POSITION_ARRAY" count="{vertex_count}" stride="3">
              <param name="X" type="float"/>
              <param name="Y" type="float"/>
              <param name="Z" type="float"/>
            </accessor>
          </technique_common>
        </source>

        <source id="{geom_id}_NORMAL">
          <float_array id="{geom_id}_NORMAL_ARRAY" count="{len(normals)}">{xml_float_list(normals)}</float_array>
          <technique_common>
            <accessor source="#{geom_id}_NORMAL_ARRAY" count="{vertex_count}" stride="3">
              <param name="X" type="float"/>
              <param name="Y" type="float"/>
              <param name="Z" type="float"/>
            </accessor>
          </technique_common>
        </source>

        <source id="{geom_id}_UV">
          <float_array id="{geom_id}_UV_ARRAY" count="{len(uvs)}">{xml_float_list(uvs)}</float_array>
          <technique_common>
            <accessor source="#{geom_id}_UV_ARRAY" count="{vertex_count}" stride="2">
              <param name="S" type="float"/>
              <param name="T" type="float"/>
            </accessor>
          </technique_common>
        </source>

        <vertices id="{geom_id}_VERTICES">
          <input semantic="POSITION" source="#{geom_id}_POSITION"/>
        </vertices>

        <triangles material="MaterialSymbol" count="{tri_count}">
          <input semantic="VERTEX" source="#{geom_id}_VERTICES" offset="0"/>
          <input semantic="NORMAL" source="#{geom_id}_NORMAL" offset="1"/>
          <input semantic="TEXCOORD" source="#{geom_id}_UV" offset="2" set="0"/>
          <p>{xml_int_list(p_indices)}</p>
        </triangles>
      </mesh>
    </geometry>
'''

    meta = {
        "id": geom_id,
        "triangles": tri_count,
        "vertices": vertex_count
    }

    return xml, meta


def write_dae(records: List[Dict[str, Any]], filepath: str, title: str) -> Dict[str, Any]:
    geometry_blocks: List[str] = []
    node_blocks: List[str] = []
    meta_items: List[Dict[str, Any]] = []

    total_triangles = 0
    total_vertices = 0

    for record in records:
        geom_xml, geom_meta = geometry_xml_shared(record)
        geometry_blocks.append(geom_xml)
        meta_items.append(geom_meta)

        geom_id = record["id"]

        node_blocks.append(f'''
      <node id="{geom_id}_NODE" name="{geom_id}" type="NODE">
        <matrix sid="transform">1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1</matrix>
        <instance_geometry url="#{geom_id}">
          <bind_material>
            <technique_common>
              <instance_material symbol="MaterialSymbol" target="#Material">
                <bind_vertex_input semantic="TEXCOORD" input_semantic="TEXCOORD" input_set="0"/>
              </instance_material>
            </technique_common>
          </bind_material>
        </instance_geometry>
      </node>
''')

        total_triangles += int(geom_meta["triangles"])
        total_vertices += int(geom_meta["vertices"])

    dae = f'''<?xml version="1.0" encoding="utf-8"?>
<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema" version="1.4.1">
  <asset>
    <contributor>
      <authoring_tool>{VERSION}</authoring_tool>
    </contributor>
    <unit name="meter" meter="1"/>
    <up_axis>Z_UP</up_axis>
  </asset>

  <library_effects>
    <effect id="Material-effect" name="Material-effect">
      <profile_COMMON>
        <technique sid="common">
          <phong>
            <diffuse>
              <color>0.8 0.8 0.8 1</color>
            </diffuse>
          </phong>
        </technique>
      </profile_COMMON>
    </effect>
  </library_effects>

  <library_materials>
    <material id="Material" name="Material">
      <instance_effect url="#Material-effect"/>
    </material>
  </library_materials>

  <library_geometries>
{''.join(geometry_blocks)}
  </library_geometries>

  <library_visual_scenes>
    <visual_scene id="Scene" name="{title}">
{''.join(node_blocks)}
    </visual_scene>
  </library_visual_scenes>

  <scene>
    <instance_visual_scene url="#Scene"/>
  </scene>
</COLLADA>
'''

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(dae)

    if not os.path.exists(filepath) or os.path.getsize(filepath) <= 0:
        raise RuntimeError("DAE export failed.")

    meta = {
        "version": VERSION,
        "file": os.path.basename(filepath),
        "file_size": os.path.getsize(filepath),
        "geometry_count": len(records),
        "triangles": total_triangles,
        "vertices": total_vertices,
        "items": meta_items
    }

    with open(filepath + ".meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return meta


def write_preview_files(merged: trimesh.Trimesh, glb_path: str, stl_path: str) -> Dict[str, Any]:
    merged = clean_mesh(merged)
    merged.export(glb_path, file_type="glb")
    merged.export(stl_path, file_type="stl")

    return {
        "glb_size": os.path.getsize(glb_path),
        "stl_size": os.path.getsize(stl_path),
        "preview_faces": int(len(merged.faces)),
        "preview_vertices": int(len(merged.vertices))
    }


def package_urls(package: Dict[str, Any]) -> Dict[str, str]:
    host = request.host_url.rstrip("/")
    files = package["files"]
    package_id = package["id"]

    return {
        "JOB_PAGE": f"{host}/job/{package_id}",
        "LOW_LI_DAE": f"{host}/download/{files['LOW_LI_DAE']}",
        "SMOOTH_DAE": f"{host}/download/{files['SMOOTH_DAE']}",
        "GLB": f"{host}/download/{files['GLB']}",
        "STL": f"{host}/download/{files['STL']}",
        "ZIP": f"{host}/download/{files['ZIP']}",
        "ADVANCED_ZIP": f"{host}/download/{files['ADVANCED_ZIP']}",
        "HIGH": f"{host}/download/{files['HIGH']}",
        "MEDIUM": f"{host}/download/{files['MEDIUM']}",
        "LOW": f"{host}/download/{files['LOW']}",
        "LOWEST": f"{host}/download/{files['LOWEST']}",
        "PHYS": f"{host}/download/{files['PHYS']}"
    }


def produce_package(prims: List[Dict[str, Any]], name: str, quality: int) -> Dict[str, Any]:
    low_meshes, low_report = build_mesh_list(prims, "low_li", quality)
    smooth_meshes, smooth_report = build_mesh_list(prims, "smooth", quality)
    preview_meshes, preview_report = build_mesh_list(prims, "preview", quality)
    lowest_meshes, lowest_report = build_mesh_list(prims, "lowest", quality)

    low_baked_records = make_baked_record(low_meshes, "M3D3_BAKED_LOW_LI_MESH")
    smooth_records = make_records_from_meshes(smooth_meshes, "PRIM")
    high_records = make_records_from_meshes(preview_meshes, "PRIM")
    medium_baked_records = make_baked_record(smooth_meshes, "M3D3_BAKED_MEDIUM_MESH")
    low_advanced_records = make_baked_record(low_meshes, "M3D3_BAKED_LOW_MESH")
    lowest_records = make_box_proxy_record(lowest_meshes, "M3D3_LOWEST_PROXY")
    phys_records = make_box_proxy_record(lowest_meshes, "M3D3_PHYS_PROXY")

    preview_merged = clean_mesh(trimesh.util.concatenate(preview_meshes))

    uid = uuid.uuid4().hex[:8]
    package_id = uid

    files = {
        "LOW_LI_DAE": f"{name}_SL_READY_LOW_LI_{uid}.dae",
        "SMOOTH_DAE": f"{name}_SL_READY_SMOOTH_{uid}.dae",
        "GLB": f"{name}_PREVIEW_{uid}.glb",
        "STL": f"{name}_SOLID_{uid}.stl",
        "ZIP": f"{name}_ALL_FILES_{uid}.zip",
        "ADVANCED_ZIP": f"{name}_ADVANCED_LOD_{uid}.zip",
        "HIGH": f"{name}_HIGH_{uid}.dae",
        "MEDIUM": f"{name}_MEDIUM_{uid}.dae",
        "LOW": f"{name}_LOW_{uid}.dae",
        "LOWEST": f"{name}_LOWEST_{uid}.dae",
        "PHYS": f"{name}_PHYS_{uid}.dae"
    }

    low_li_path = os.path.join(OUTPUT_DIR, files["LOW_LI_DAE"])
    smooth_path = os.path.join(OUTPUT_DIR, files["SMOOTH_DAE"])
    glb_path = os.path.join(OUTPUT_DIR, files["GLB"])
    stl_path = os.path.join(OUTPUT_DIR, files["STL"])
    zip_path = os.path.join(OUTPUT_DIR, files["ZIP"])
    advanced_zip_path = os.path.join(OUTPUT_DIR, files["ADVANCED_ZIP"])

    high_path = os.path.join(OUTPUT_DIR, files["HIGH"])
    medium_path = os.path.join(OUTPUT_DIR, files["MEDIUM"])
    low_path = os.path.join(OUTPUT_DIR, files["LOW"])
    lowest_path = os.path.join(OUTPUT_DIR, files["LOWEST"])
    phys_path = os.path.join(OUTPUT_DIR, files["PHYS"])

    low_li_meta = write_dae(low_baked_records, low_li_path, name + "_BAKED_LOW_LI")
    smooth_meta = write_dae(smooth_records, smooth_path, name + "_SMOOTH")
    preview_meta = write_preview_files(preview_merged, glb_path, stl_path)

    write_dae(high_records, high_path, name + "_HIGH")
    write_dae(medium_baked_records, medium_path, name + "_MEDIUM")
    write_dae(low_advanced_records, low_path, name + "_LOW")
    write_dae(lowest_records, lowest_path, name + "_LOWEST")
    write_dae(phys_records, phys_path, name + "_PHYS")

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(low_li_path, "SL_Ready_Low_LI_Baked.dae")
        z.write(smooth_path, "SL_Ready_Smooth.dae")
        z.write(glb_path, "Preview.glb")
        z.write(stl_path, "Solid.stl")

    with zipfile.ZipFile(advanced_zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(high_path, files["HIGH"])
        z.write(medium_path, files["MEDIUM"])
        z.write(low_path, files["LOW"])
        z.write(lowest_path, files["LOWEST"])
        z.write(phys_path, files["PHYS"])

    package = {
        "id": package_id,
        "name": name,
        "created": now_ts(),
        "files": files,
        "summary": {
            "version": VERSION,
            "low_li": low_report,
            "smooth": smooth_report,
            "preview": preview_report,
            "lowest": lowest_report,
            "low_li_dae": low_li_meta,
            "smooth_dae": smooth_meta,
            "preview_files": preview_meta
        }
    }

    results[package_id] = package

    return package


@app.route("/", methods=["GET"])
def home():
    return f"M3D3 PLATINUM STYLE PRIM TO MESH SERVER RUNNING - {VERSION}"


@app.route("/health", methods=["GET"])
def health():
    cleanup_old_files()
    cleanup_old_memory()

    return jsonify({
        "ok": True,
        "version": VERSION,
        "server": "M3D3 Platinum Style Baked Low LI Final",
        "active_jobs": list(jobs.keys()),
        "result_jobs": list(results.keys()),
        "outputs": [f for f in os.listdir(OUTPUT_DIR) if not f.endswith(".meta.json")]
    }), 200


@app.route("/upload_chunk", methods=["POST"])
def upload_chunk():
    try:
        data = request.get_json(force=True)
        job = str(data.get("job", "")).strip()
        chunk = data.get("chunk", [])

        if job == "":
            return jsonify({"ok": False, "error": "missing job", "version": VERSION}), 400

        if not isinstance(chunk, list):
            return jsonify({"ok": False, "error": "chunk must be list", "version": VERSION}), 400

        if job not in jobs:
            jobs[job] = {
                "created": now_ts(),
                "chunks": []
            }

        clean_chunk = []

        for item in chunk:
            if isinstance(item, dict):
                clean_chunk.append(item)

        jobs[job]["chunks"].extend(clean_chunk)

        return jsonify({
            "ok": True,
            "version": VERSION,
            "job": job,
            "received": len(clean_chunk),
            "total": len(jobs[job]["chunks"])
        }), 200

    except Exception as exc:
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(exc), "version": VERSION}), 500


@app.route("/finalize", methods=["POST"])
def finalize():
    try:
        cleanup_old_files()
        cleanup_old_memory()

        data = request.get_json(force=True)

        job = str(data.get("job", "")).strip()
        name = safe_name(data.get("name", "M3D3_Build"))
        quality = clamp_quality(data.get("quality", DEFAULT_QUALITY))

        if job == "":
            return jsonify({"ok": False, "error": "missing job", "version": VERSION}), 400

        if job not in jobs:
            return jsonify({
                "ok": False,
                "error": "job not found",
                "job": job,
                "known_jobs": list(jobs.keys()),
                "version": VERSION
            }), 400

        prims = jobs[job]["chunks"]

        if not prims:
            return jsonify({"ok": False, "error": "job has no prim data", "version": VERSION}), 400

        package = produce_package(prims, name, quality)

        del jobs[job]

        urls = package_urls(package)

        return jsonify({
            "ok": True,
            "version": VERSION,
            "JOB_PAGE": urls["JOB_PAGE"],
            "LOW_LI_DAE": urls["LOW_LI_DAE"],
            "SMOOTH_DAE": urls["SMOOTH_DAE"],
            "GLB": urls["GLB"],
            "STL": urls["STL"],
            "ZIP": urls["ZIP"],
            "ADVANCED_ZIP": urls["ADVANCED_ZIP"],
            "summary": package["summary"]
        }), 200

    except Exception as exc:
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(exc), "version": VERSION}), 500


@app.route("/job/<package_id>", methods=["GET"])
def job_page(package_id: str):
    package = results.get(package_id)

    if not package:
        return Response(
            "<h1>M3D3 Job Not Found</h1><p>This job expired, the server restarted, or the package was cleaned up.</p>",
            mimetype="text/html"
        ), 404

    urls = package_urls(package)
    summary = package.get("summary", {})
    low_li = summary.get("low_li", {})
    smooth = summary.get("smooth", {})
    preview = summary.get("preview", {})

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>M3D3 Platinum Style Mesh Ready</title>
    <script type="module" src="https://unpkg.com/@google/model-viewer/dist/model-viewer.min.js"></script>
    <style>
        body {{
            margin: 0;
            font-family: Arial, sans-serif;
            background: #111;
            color: #f4f4f4;
        }}
        .wrap {{
            max-width: 1080px;
            margin: 0 auto;
            padding: 32px;
        }}
        .card {{
            background: #1c1c1c;
            border: 1px solid #333;
            border-radius: 14px;
            padding: 24px;
            margin-bottom: 18px;
        }}
        h1 {{
            margin-top: 0;
            font-size: 34px;
        }}
        .meta {{
            color: #bbb;
            line-height: 1.6;
        }}
        code {{
            background: #000;
            padding: 3px 6px;
            border-radius: 4px;
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 12px;
        }}
        a.button {{
            display: block;
            padding: 14px 16px;
            background: #2c6fff;
            color: white;
            text-decoration: none;
            text-align: center;
            border-radius: 10px;
            font-weight: bold;
        }}
        a.secondary {{
            background: #333;
        }}
        .note {{
            padding: 12px 14px;
            border-radius: 10px;
            background: #0f0f0f;
            border: 1px solid #333;
            color: #ccc;
            line-height: 1.6;
        }}
        model-viewer {{
            width: 100%;
            height: 440px;
            background: #0a0a0a;
            border-radius: 12px;
        }}
    </style>
</head>
<body>
    <div class="wrap">
        <div class="card">
            <h1>M3D3 Platinum Style Mesh Ready</h1>
            <p class="meta">Version: <code>{VERSION}</code></p>
            <p class="meta">Package ID: <code>{package_id}</code></p>
            <p class="meta">Baked Low LI DAE: <code>{low_li.get("faces", "?")} faces</code> / <code>{low_li.get("vertices", "?")} vertices</code> / <code>{low_li.get("count", "?")} source pieces</code></p>
            <p class="meta">Smooth DAE: <code>{smooth.get("faces", "?")} faces</code> / <code>{smooth.get("vertices", "?")} vertices</code></p>
            <p class="meta">Preview Mesh: <code>{preview.get("faces", "?")} faces</code> / <code>{preview.get("vertices", "?")} vertices</code></p>
            <p class="meta">Dimensions: <code>{low_li.get("dimensions", "?")}</code></p>
        </div>

        <div class="card">
            <h2>Preview</h2>
            <model-viewer src="{urls["GLB"]}" camera-controls auto-rotate shadow-intensity="1"></model-viewer>
        </div>

        <div class="card">
            <h2>Creator Downloads</h2>
            <div class="grid">
                <a class="button" href="{urls["LOW_LI_DAE"]}">Download SL Ready Baked Low LI DAE</a>
                <a class="button" href="{urls["SMOOTH_DAE"]}">Download SL Ready Smooth DAE</a>
                <a class="button secondary" href="{urls["GLB"]}">Download GLB Preview File</a>
                <a class="button secondary" href="{urls["STL"]}">Download STL File</a>
                <a class="button secondary" href="{urls["ZIP"]}">Download All Files ZIP</a>
                <a class="button secondary" href="{urls["ADVANCED_ZIP"]}">Download Advanced LOD ZIP</a>
            </div>
        </div>

        <div class="card">
            <h2>Usage Notes</h2>
            <div class="note">
                Use Baked Low LI for most multi-prim builds.<br>
                Use Smooth for round-heavy builds.<br>
                Use Advanced ZIP only if manually loading custom LOD or physics.<br>
                Baked Low LI is designed to reduce mesh instances by combining the linkset into one mesh object.
            </div>
        </div>

        <div class="card">
            <h2>Advanced Files</h2>
            <div class="grid">
                <a class="button secondary" href="{urls["HIGH"]}">HIGH DAE</a>
                <a class="button secondary" href="{urls["MEDIUM"]}">MEDIUM DAE</a>
                <a class="button secondary" href="{urls["LOW"]}">LOW DAE</a>
                <a class="button secondary" href="{urls["LOWEST"]}">LOWEST DAE</a>
                <a class="button secondary" href="{urls["PHYS"]}">PHYS DAE</a>
            </div>
        </div>
    </div>
</body>
</html>
"""
    return Response(html, mimetype="text/html")


@app.route("/download/<path:filename>", methods=["GET"])
def download(filename: str):
    safe_file = clean_filename(filename)
    path = os.path.join(OUTPUT_DIR, safe_file)

    if not os.path.exists(path):
        return jsonify({
            "ok": False,
            "error": "file not found",
            "requested": safe_file,
            "available": [f for f in os.listdir(OUTPUT_DIR) if not f.endswith(".meta.json")],
            "version": VERSION
        }), 404

    return send_from_directory(OUTPUT_DIR, safe_file, as_attachment=True)


@app.route("/validate/<path:filename>", methods=["GET"])
def validate(filename: str):
    safe_file = clean_filename(filename)
    path = os.path.join(OUTPUT_DIR, safe_file)
    meta_path = path + ".meta.json"

    if not os.path.exists(path):
        return jsonify({"ok": False, "error": "file not found", "version": VERSION}), 404

    text = ""

    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    except Exception:
        pass

    meta = {}

    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            meta = {}

    checks = {
        "exists": True,
        "size": os.path.getsize(path),
        "collada": "<COLLADA" in text,
        "z_up": "<up_axis>Z_UP</up_axis>" in text,
        "meter": '<unit name="meter" meter="1"/>' in text,
        "version": VERSION in text,
        "baked_or_prim": "M3D3_BAKED_LOW_LI_MESH" in text or "PRIM_0000" in text,
        "not_y_up": "Y_UP" not in text,
        "not_generator_panel": "_Gene_ato_" not in text and "Generator" not in text and "generator" not in text,
        "geometry": "<geometry" in text,
        "uv": "TEXCOORD" in text,
        "normals": "NORMAL" in text
    }

    ok = (
        checks["exists"] and
        checks["size"] > 0 and
        checks["collada"] and
        checks["z_up"] and
        checks["meter"] and
        checks["version"] and
        checks["baked_or_prim"] and
        checks["not_y_up"] and
        checks["not_generator_panel"] and
        checks["geometry"] and
        checks["uv"] and
        checks["normals"]
    )

    return jsonify({
        "ok": ok,
        "version": VERSION,
        "file": safe_file,
        "checks": checks,
        "metadata": meta
    }), 200


@app.route("/test_export", methods=["GET"])
def test_export():
    try:
        cleanup_old_files()
        cleanup_old_memory()

        test_prims = [
            {
                "role": "BUILD",
                "source": "ROOT_SCANNER",
                "link": 1,
                "type": "BOX",
                "name": "Box",
                "size": "<1.000000, 1.000000, 1.000000>",
                "pos": "<-0.600000, 0.000000, 0.000000>",
                "rot": "<0.000000, 0.000000, 0.000000, 1.000000>"
            },
            {
                "role": "BUILD",
                "source": "ROOT_SCANNER",
                "link": 2,
                "type": "SPHERE",
                "name": "Sphere",
                "size": "<1.000000, 1.000000, 1.000000>",
                "pos": "<0.600000, 0.000000, 0.000000>",
                "rot": "<0.000000, 0.000000, 0.000000, 1.000000>"
            }
        ]

        package = produce_package(test_prims, "M3D3_Box_Sphere_Test", DEFAULT_QUALITY)
        urls = package_urls(package)

        return jsonify({
            "ok": True,
            "version": VERSION,
            "JOB_PAGE": urls["JOB_PAGE"],
            "LOW_LI_DAE": urls["LOW_LI_DAE"],
            "SMOOTH_DAE": urls["SMOOTH_DAE"],
            "GLB": urls["GLB"],
            "STL": urls["STL"],
            "ZIP": urls["ZIP"],
            "ADVANCED_ZIP": urls["ADVANCED_ZIP"],
            "summary": package["summary"]
        }), 200

    except Exception as exc:
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(exc), "version": VERSION}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
