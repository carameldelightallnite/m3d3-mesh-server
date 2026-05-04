# =========================================================
# M3D3 PRIM TO MESH SERVER
# NN-STYLE MULTI-GEOMETRY Z_UP EXPORTER
#
# Version:
# M3D3_NN_STYLE_MULTI_GEOMETRY_ZUP_VERIFIED_2026_05_04
#
# Purpose:
# - Receives scripted build prim reports from Second Life.
# - Excludes generator panel geometry.
# - Creates one job page.
# - Default SL Ready DAE uses multi-geometry PRIM_0000 / PRIM_0001 blocks.
# - Default DAE is controlled-density and Z_UP.
# - Preview GLB can be higher quality.
# - Provides DAE, GLB, STL, ZIP, and Advanced ZIP.
#
# Verified from current user evidence:
# - Generator/receiver/server one-link flow reached working job page.
# - Sphere receiver reported correctly.
#
# Not yet verified by direct execution in this response:
# - Final multi-geometry DAE upload in SL viewer.
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

VERSION = "M3D3_NN_STYLE_MULTI_GEOMETRY_ZUP_VERIFIED_2026_05_04"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

jobs: Dict[str, Dict[str, Any]] = {}
results: Dict[str, Dict[str, Any]] = {}

FILE_TTL_SECONDS = 3600
JOB_TTL_SECONDS = 1800

MIN_AXIS_SIZE = 0.001
MAX_SL_SIZE = 64.0

DEFAULT_QUALITY = 20
MIN_QUALITY = 4
MAX_QUALITY = 24

UPLOAD_SPHERE_LAT = 10
UPLOAD_SPHERE_LON = 20

PREVIEW_SPHERE_LAT = 20
PREVIEW_SPHERE_LON = 40


# =========================================================
# GENERAL HELPERS
# =========================================================

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
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        return arr
    except Exception:
        return np.array(fallback, dtype=float)


def parse_rot(value: Any) -> np.ndarray:
    try:
        text = str(value).replace("<", "").replace(">", "").strip()
        q = np.fromstring(text, sep=",")
        if q.size < 4:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        q = q[:4].astype(float)
        q = np.nan_to_num(q, nan=0.0, posinf=0.0, neginf=0.0)

        length = np.linalg.norm(q)
        if not np.isfinite(length) or length <= 0.000001:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)

        q = q / length

        return np.array([q[3], q[0], q[1], q[2]], dtype=float)
    except Exception:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)


def safe_size(value: Any) -> np.ndarray:
    s = parse_vec(value, (1.0, 1.0, 1.0))
    s = np.abs(s)
    s[s < MIN_AXIS_SIZE] = MIN_AXIS_SIZE
    s[s > MAX_SL_SIZE] = MAX_SL_SIZE
    return s


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


def normalize_vector(v: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    try:
        v = np.asarray(v, dtype=float)
        v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        length = np.linalg.norm(v)
        if not np.isfinite(length) or length <= 0.000001:
            return fallback
        return v / length
    except Exception:
        return fallback


def mesh_bounds_and_dims(meshes: List[trimesh.Trimesh]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    merged = trimesh.util.concatenate(meshes)
    bounds = np.asarray(merged.bounds, dtype=float)
    min_corner = bounds[0]
    max_corner = bounds[1]
    center = (min_corner + max_corner) * 0.5
    dims = max_corner - min_corner
    dims = np.nan_to_num(dims, nan=0.0, posinf=0.0, neginf=0.0)
    return min_corner, max_corner, center, dims


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
    t = str(prim.get("type", "BOX")).upper()

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

    if t in ["BOX", "CYLINDER", "SPHERE", "TORUS", "RING", "TUBE", "PRISM", "CONE"]:
        return t

    return "BOX"


# =========================================================
# PRIMITIVE GEOMETRY
# =========================================================

def make_box_mesh() -> trimesh.Trimesh:
    return clean_mesh(trimesh.creation.box(extents=[1.0, 1.0, 1.0]))


def make_cylinder_mesh(sections: int) -> trimesh.Trimesh:
    sections = max(8, int(sections))
    return clean_mesh(trimesh.creation.cylinder(radius=0.5, height=1.0, sections=sections))


def make_cone_mesh(sections: int) -> trimesh.Trimesh:
    sections = max(8, int(sections))
    return clean_mesh(trimesh.creation.cone(radius=0.5, height=1.0, sections=sections))


def make_sphere_mesh(lon: int, lat: int) -> trimesh.Trimesh:
    lon = max(8, int(lon))
    lat = max(6, int(lat))
    return clean_mesh(trimesh.creation.uv_sphere(radius=0.5, count=[lon, lat]))


def make_torus_mesh(major_sections: int, minor_sections: int) -> trimesh.Trimesh:
    major_sections = max(12, int(major_sections))
    minor_sections = max(6, int(minor_sections))

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
    verts = np.array([
        [-0.5, -0.5, -0.5],
        [ 0.5, -0.5, -0.5],
        [ 0.0,  0.5, -0.5],
        [-0.5, -0.5,  0.5],
        [ 0.5, -0.5,  0.5],
        [ 0.0,  0.5,  0.5],
    ], dtype=float)

    faces = np.array([
        [0, 1, 2],
        [3, 5, 4],
        [0, 3, 4],
        [0, 4, 1],
        [1, 4, 5],
        [1, 5, 2],
        [2, 5, 3],
        [2, 3, 0],
    ], dtype=int)

    return clean_mesh(trimesh.Trimesh(vertices=verts, faces=faces, process=False))


def build_base_mesh(shape: str, mode: str, quality: int) -> trimesh.Trimesh:
    if mode == "preview":
        sphere_lon = max(PREVIEW_SPHERE_LON, quality * 2)
        sphere_lat = max(PREVIEW_SPHERE_LAT, quality)
        cyl_sections = max(32, quality * 2)
        torus_major = max(32, quality * 2)
        torus_minor = max(10, quality // 2)
    else:
        sphere_lon = UPLOAD_SPHERE_LON
        sphere_lat = UPLOAD_SPHERE_LAT
        cyl_sections = 20
        torus_major = 24
        torus_minor = 8

    if shape == "SPHERE":
        return make_sphere_mesh(sphere_lon, sphere_lat)

    if shape == "CYLINDER":
        return make_cylinder_mesh(cyl_sections)

    if shape == "CONE":
        return make_cone_mesh(cyl_sections)

    if shape in ["TORUS", "RING"]:
        return make_torus_mesh(torus_major, torus_minor)

    if shape == "TUBE":
        return make_cylinder_mesh(cyl_sections)

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
        mesh.apply_scale([size[0] / base_xy, size[1] / base_xy, max(size[2], MIN_AXIS_SIZE)])
    else:
        mesh.apply_scale(size)

    matrix = trimesh.transformations.quaternion_matrix(rot)
    matrix[:3, 3] = pos
    mesh.apply_transform(matrix)

    return clean_mesh(mesh)


def build_mesh_records(prims: List[Dict[str, Any]], mode: str, quality: int) -> Tuple[List[Dict[str, Any]], trimesh.Trimesh, Dict[str, Any]]:
    filtered: List[Dict[str, Any]] = []

    for prim in prims:
        if isinstance(prim, dict) and not is_generator_panel_prim(prim):
            filtered.append(prim)

    if not filtered:
        raise RuntimeError("No build receiver prims were received. Generator panel output is rejected.")

    meshes: List[trimesh.Trimesh] = []

    for prim in filtered:
        shape = shape_from_prim(prim)
        base = build_base_mesh(shape, mode, quality)
        transformed = apply_prim_transform(base, prim)
        meshes.append(transformed)

    min_corner, max_corner, center, dims = mesh_bounds_and_dims(meshes)

    if float(np.max(dims)) > MAX_SL_SIZE:
        raise RuntimeError("Build exceeds Second Life 64m mesh upload limit.")

    records: List[Dict[str, Any]] = []
    centered_meshes: List[trimesh.Trimesh] = []

    for i, mesh in enumerate(meshes):
        centered = mesh.copy()
        centered.apply_translation(-center)
        centered = clean_mesh(centered)

        records.append({
            "id": f"PRIM_{i:04d}",
            "name": safe_name(filtered[i].get("name", f"PRIM_{i:04d}")),
            "shape": shape_from_prim(filtered[i]),
            "mesh": centered
        })

        centered_meshes.append(centered)

    merged = clean_mesh(trimesh.util.concatenate(centered_meshes))

    report = {
        "count": len(records),
        "dimensions": [float(x) for x in dims],
        "center_removed": [float(x) for x in center],
        "faces": int(len(merged.faces)),
        "vertices": int(len(merged.vertices))
    }

    return records, merged, report


# =========================================================
# COLLADA WRITER
# =========================================================

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


def fallback_face_normal(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    n = np.cross(b - a, c - a)
    length = np.linalg.norm(n)

    if not np.isfinite(length) or length <= 0.000001:
        return np.array([0.0, 0.0, 1.0], dtype=float)

    return n / length


def geometry_xml(record: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
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

    tri_count = 0

    for face in faces:
        if len(face) != 3:
            continue

        i0 = int(face[0])
        i1 = int(face[1])
        i2 = int(face[2])

        if i0 == i1 or i1 == i2 or i0 == i2:
            continue

        a = vertices[i0]
        b = vertices[i1]
        c = vertices[i2]

        if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)) or not np.all(np.isfinite(c)):
            continue

        fallback = fallback_face_normal(a, b, c)

        for idx in [i0, i1, i2]:
            v = vertices[idx]
            n = normalize_vector(vertex_normals[idx], fallback)
            u, vv = projected_uv(v, min_corner, dims)

            positions.extend([float(v[0]), float(v[1]), float(v[2])])
            normals.extend([float(n[0]), float(n[1]), float(n[2])])
            uvs.extend([float(u), float(vv)])

        tri_count += 1

    if tri_count <= 0:
        raise RuntimeError(f"{geom_id} produced zero triangles.")

    vertex_count = tri_count * 3
    indices = list(range(vertex_count))

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
          <input semantic="NORMAL" source="#{geom_id}_NORMAL" offset="0"/>
          <input semantic="TEXCOORD" source="#{geom_id}_UV" offset="0" set="0"/>
          <p>{xml_int_list(indices)}</p>
        </triangles>
      </mesh>
    </geometry>
'''

    meta = {
        "id": geom_id,
        "shape": record.get("shape", "UNKNOWN"),
        "triangles": tri_count,
        "vertices": vertex_count
    }

    return xml, meta


def write_multi_geometry_dae(records: List[Dict[str, Any]], filepath: str, title: str) -> Dict[str, Any]:
    geometry_blocks: List[str] = []
    node_blocks: List[str] = []
    meta_items: List[Dict[str, Any]] = []

    total_triangles = 0
    total_vertices = 0

    for record in records:
        geom_xml, geom_meta = geometry_xml(record)
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
        "vertices_unrolled": total_vertices,
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


def produce_package(prims: List[Dict[str, Any]], name: str, quality: int) -> Dict[str, Any]:
    upload_records, upload_merged, upload_report = build_mesh_records(prims, "upload", quality)
    preview_records, preview_merged, preview_report = build_mesh_records(prims, "preview", quality)

    uid = uuid.uuid4().hex[:8]
    package_id = uid

    files = {
        "DAE": f"{name}_SL_READY_{uid}.dae",
        "GLB": f"{name}_PREVIEW_{uid}.glb",
        "STL": f"{name}_SOLID_{uid}.stl",
        "ZIP": f"{name}_PACKAGE_{uid}.zip",
        "ADVANCED_ZIP": f"{name}_ADVANCED_{uid}.zip",
        "HIGH": f"{name}_HIGH_{uid}.dae",
        "MEDIUM": f"{name}_MEDIUM_{uid}.dae",
        "LOW": f"{name}_LOW_{uid}.dae",
        "LOWEST": f"{name}_LOWEST_{uid}.dae",
        "PHYS": f"{name}_PHYS_{uid}.dae"
    }

    dae_path = os.path.join(OUTPUT_DIR, files["DAE"])
    glb_path = os.path.join(OUTPUT_DIR, files["GLB"])
    stl_path = os.path.join(OUTPUT_DIR, files["STL"])
    zip_path = os.path.join(OUTPUT_DIR, files["ZIP"])
    advanced_zip_path = os.path.join(OUTPUT_DIR, files["ADVANCED_ZIP"])

    dae_meta = write_multi_geometry_dae(upload_records, dae_path, name)
    preview_meta = write_preview_files(preview_merged, glb_path, stl_path)

    high_path = os.path.join(OUTPUT_DIR, files["HIGH"])
    medium_path = os.path.join(OUTPUT_DIR, files["MEDIUM"])
    low_path = os.path.join(OUTPUT_DIR, files["LOW"])
    lowest_path = os.path.join(OUTPUT_DIR, files["LOWEST"])
    phys_path = os.path.join(OUTPUT_DIR, files["PHYS"])

    write_multi_geometry_dae(preview_records, high_path, name + "_HIGH")
    write_multi_geometry_dae(upload_records, medium_path, name + "_MEDIUM")
    write_multi_geometry_dae(upload_records, low_path, name + "_LOW")
    write_multi_geometry_dae(upload_records, lowest_path, name + "_LOWEST")
    write_multi_geometry_dae(upload_records, phys_path, name + "_PHYS")

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(dae_path, "SL_Ready.dae")
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
            "upload": upload_report,
            "preview": preview_report,
            "dae": dae_meta,
            "preview_files": preview_meta
        }
    }

    results[package_id] = package

    return package


def package_urls(package: Dict[str, Any]) -> Dict[str, str]:
    host = request.host_url.rstrip("/")
    files = package["files"]
    pid = package["id"]

    return {
        "JOB_PAGE": f"{host}/job/{pid}",
        "DAE": f"{host}/download/{files['DAE']}",
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


# =========================================================
# ROUTES
# =========================================================

@app.route("/", methods=["GET"])
def home():
    return f"M3D3 PRIM TO MESH SERVER RUNNING - {VERSION}"


@app.route("/health", methods=["GET"])
def health():
    cleanup_old_files()
    cleanup_old_memory()

    return jsonify({
        "ok": True,
        "version": VERSION,
        "server": "M3D3 NN Style Multi Geometry Z_UP Exporter",
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
            "DAE": urls["DAE"],
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
    upload = summary.get("upload", {})
    preview = summary.get("preview", {})

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>M3D3 Mesh Ready</title>
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
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
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
            <h1>M3D3 Mesh Ready</h1>
            <p class="meta">Version: <code>{VERSION}</code></p>
            <p class="meta">Package ID: <code>{package_id}</code></p>
            <p class="meta">Upload Mesh: <code>{upload.get("faces", "?")} faces</code> / <code>{upload.get("vertices", "?")} vertices</code></p>
            <p class="meta">Preview Mesh: <code>{preview.get("faces", "?")} faces</code> / <code>{preview.get("vertices", "?")} vertices</code></p>
            <p class="meta">Dimensions: <code>{upload.get("dimensions", "?")}</code></p>
        </div>

        <div class="card">
            <h2>Preview</h2>
            <model-viewer src="{urls["GLB"]}" camera-controls auto-rotate shadow-intensity="1"></model-viewer>
        </div>

        <div class="card">
            <h2>Creator Downloads</h2>
            <div class="grid">
                <a class="button" href="{urls["DAE"]}">Download DAE File</a>
                <a class="button secondary" href="{urls["GLB"]}">Download GLB File</a>
                <a class="button secondary" href="{urls["STL"]}">Download STL File</a>
                <a class="button secondary" href="{urls["ZIP"]}">Download All Files ZIP</a>
            </div>
        </div>

        <div class="card">
            <h2>Advanced Files</h2>
            <div class="grid">
                <a class="button secondary" href="{urls["ADVANCED_ZIP"]}">Download Advanced ZIP</a>
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
        "z_up": "<up_axis>Z_UP</up_axis>" in text,
        "meter": '<unit name="meter" meter="1"/>' in text,
        "version": VERSION in text,
        "prim_0000": "PRIM_0000" in text,
        "not_y_up": "Y_UP" not in text,
        "not_generator": "_Gene_ato_" not in text and "Generator" not in text,
        "has_collada": "<COLLADA" in text,
        "has_geometry": "<geometry" in text,
        "has_uv": "TEXCOORD" in text,
        "has_normals": "NORMAL" in text
    }

    ok = (
        checks["exists"] and
        checks["size"] > 0 and
        checks["z_up"] and
        checks["meter"] and
        checks["version"] and
        checks["prim_0000"] and
        checks["not_y_up"] and
        checks["not_generator"] and
        checks["has_collada"] and
        checks["has_geometry"] and
        checks["has_uv"] and
        checks["has_normals"]
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
                "type": "BOX",
                "name": "Box",
                "size": "<1.000000, 1.000000, 1.000000>",
                "pos": "<-0.600000, 0.000000, 0.000000>",
                "rot": "<0.000000, 0.000000, 0.000000, 1.000000>"
            },
            {
                "role": "BUILD",
                "type": "SPHERE",
                "name": "Sphere",
                "size": "<1.000000, 1.000000, 1.000000>",
                "pos": "<0.600000, 0.000000, 0.000000>",
                "rot": "<0.000000, 0.000000, 0.000000, 1.000000>"
            }
        ]

        package = produce_package(test_prims, "M3D3_Box_Sphere_Test", 20)
        urls = package_urls(package)

        return jsonify({
            "ok": True,
            "version": VERSION,
            "JOB_PAGE": urls["JOB_PAGE"],
            "DAE": urls["DAE"],
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
