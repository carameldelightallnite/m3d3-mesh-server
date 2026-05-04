# =========================================================
# M3D3 PLATINUM PRIM TO MESH SERVER
# FINAL GENERAL BUILDER DELIVERY SYSTEM
#
# Render / Flask / LSL / Second Life
#
# Features:
# - LSL chunked upload
# - In-memory job store with single-worker gunicorn
# - One-link web job page
# - Single SL-ready DAE download
# - Advanced LOD ZIP download
# - OBJ download
# - GLB download
# - HIGH / MEDIUM / LOW / LOWEST / PHYS generation
# - Origin recentering to remove baked SL world coordinates
# - LOWEST and PHYS as 8-vertex / 12-triangle proxy boxes
# - Strict SL-safe Collada 1.4.1 writer
# - /health
# - /test_export
# - /validate/<filename>
# =========================================================

import os
import io
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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# IMPORTANT:
# These are in-memory by design for the current Render build.
# Procfile MUST use --workers 1 so upload chunks and finalize stay in the same process.
jobs: Dict[str, List[Dict[str, Any]]] = {}
results: Dict[str, Dict[str, Any]] = {}

FILE_TTL_SECONDS = 3600

MIN_AXIS_SIZE = 0.001
MAX_SL_SIZE = 64.0

MAX_HIGH_FACES = 3500
MAX_MEDIUM_FACES = 1200
MAX_LOW_FACES = 350


# =========================================================
# BASIC HELPERS
# =========================================================

def now_ts() -> float:
    return time.time()


def safe_name(name: Any) -> str:
    text = str(name or "Object").replace(" ", "_")
    text = "".join(c for c in text if c.isalnum() or c == "_")
    if not text:
        text = "Object"
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
    cutoff = now_ts() - FILE_TTL_SECONDS

    try:
        expired_results = []
        for package_id, package in results.items():
            if float(package.get("created", 0)) < cutoff:
                expired_results.append(package_id)

        for package_id in expired_results:
            del results[package_id]

    except Exception as exc:
        print("cleanup_old_memory error:", exc)


def parse_vec(value: Any, fallback: List[float]) -> np.ndarray:
    try:
        text = str(value).replace("<", "").replace(">", "").strip()
        arr = np.fromstring(text, sep=",")
        if arr.size < len(fallback):
            return np.array(fallback, dtype=float)
        return arr[:len(fallback)].astype(float)
    except Exception:
        return np.array(fallback, dtype=float)


def parse_rot(value: Any) -> np.ndarray:
    try:
        q = parse_vec(value, [0.0, 0.0, 0.0, 1.0])
        if q.size < 4:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)

        # LSL rotation = <x, y, z, s>
        # trimesh quaternion = <w, x, y, z>
        return np.array([q[3], q[0], q[1], q[2]], dtype=float)
    except Exception:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)


def sanitize_array(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def safe_size(size: np.ndarray) -> np.ndarray:
    s = sanitize_array(size)

    if s.size < 3:
        s = np.array([1.0, 1.0, 1.0], dtype=float)

    s = np.abs(s[:3])
    s[s < MIN_AXIS_SIZE] = MIN_AXIS_SIZE
    s[s > MAX_SL_SIZE] = MAX_SL_SIZE

    return s


def mesh_dimensions(mesh: trimesh.Trimesh) -> np.ndarray:
    bounds = np.asarray(mesh.bounds, dtype=float)
    dims = bounds[1] - bounds[0]
    dims = sanitize_array(dims)
    dims[dims < MIN_AXIS_SIZE] = MIN_AXIS_SIZE
    return dims


def mesh_report(mesh: trimesh.Trimesh) -> Dict[str, Any]:
    bounds = np.asarray(mesh.bounds, dtype=float)
    dims = bounds[1] - bounds[0]

    return {
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "bounds_min": sanitize_array(bounds[0]).tolist(),
        "bounds_max": sanitize_array(bounds[1]).tolist(),
        "dimensions": sanitize_array(dims).tolist(),
        "has_nan": bool(np.isnan(np.asarray(mesh.vertices)).any()),
        "has_inf": bool(np.isinf(np.asarray(mesh.vertices)).any())
    }


# =========================================================
# MESH CLEANING
# =========================================================

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
        mesh.merge_vertices(digits=5)
    except Exception:
        pass

    try:
        mesh.fix_normals()
    except Exception:
        pass

    try:
        mesh.vertices = sanitize_array(mesh.vertices)
    except Exception:
        pass

    return mesh


def center_mesh_to_origin(mesh: trimesh.Trimesh) -> Tuple[trimesh.Trimesh, Dict[str, Any]]:
    mesh = clean_mesh(mesh)

    bounds = np.asarray(mesh.bounds, dtype=float)
    min_corner = sanitize_array(bounds[0])
    max_corner = sanitize_array(bounds[1])
    dimensions = max_corner - min_corner
    center = (min_corner + max_corner) * 0.5

    mesh.apply_translation(-center)
    mesh = clean_mesh(mesh)

    new_bounds = np.asarray(mesh.bounds, dtype=float)
    new_min = sanitize_array(new_bounds[0])
    new_max = sanitize_array(new_bounds[1])
    new_dimensions = new_max - new_min

    report = {
        "original_min": min_corner.tolist(),
        "original_max": max_corner.tolist(),
        "original_dimensions": sanitize_array(dimensions).tolist(),
        "removed_center_offset": sanitize_array(center).tolist(),
        "centered_min": new_min.tolist(),
        "centered_max": new_max.tolist(),
        "centered_dimensions": sanitize_array(new_dimensions).tolist()
    }

    return mesh, report


# =========================================================
# PRIM GEOMETRY BUILDERS
# =========================================================

def build_box(size: np.ndarray) -> trimesh.Trimesh:
    return trimesh.creation.box(extents=size)


def build_cylinder(size: np.ndarray) -> trimesh.Trimesh:
    base = max(float(size[0]), float(size[1]), MIN_AXIS_SIZE)
    radius = base * 0.5
    height = max(float(size[2]), MIN_AXIS_SIZE)

    mesh = trimesh.creation.cylinder(radius=radius, height=height, sections=24)
    mesh.apply_scale([size[0] / base, size[1] / base, 1.0])
    return mesh


def build_sphere(size: np.ndarray) -> trimesh.Trimesh:
    mesh = trimesh.creation.uv_sphere(radius=0.5, count=[24, 24])
    mesh.apply_scale(size)
    return mesh


def build_torus(size: np.ndarray) -> trimesh.Trimesh:
    base = max(float(size[0]), float(size[1]), MIN_AXIS_SIZE)
    major = base * 0.35
    minor = max(min(float(size[0]), float(size[1])) * 0.10, float(size[2]) * 0.20, 0.01)

    try:
        mesh = trimesh.creation.torus(
            major_radius=major,
            minor_radius=minor,
            major_sections=32,
            minor_sections=12
        )
    except TypeError:
        mesh = trimesh.creation.torus(
            radius=major,
            tube_radius=minor,
            sections=32,
            segments=12
        )

    mesh.apply_scale([size[0] / base, size[1] / base, 1.0])
    return mesh


def build_cone(size: np.ndarray) -> trimesh.Trimesh:
    base = max(float(size[0]), float(size[1]), MIN_AXIS_SIZE)
    radius = base * 0.5
    height = max(float(size[2]), MIN_AXIS_SIZE)

    mesh = trimesh.creation.cone(radius=radius, height=height, sections=24)
    mesh.apply_scale([size[0] / base, size[1] / base, 1.0])
    return mesh


def build_prism(size: np.ndarray) -> trimesh.Trimesh:
    x = float(size[0]) * 0.5
    y = float(size[1]) * 0.5
    z = float(size[2]) * 0.5

    verts = np.array([
        [-x, -y, -z],
        [ x, -y, -z],
        [ 0,  y, -z],
        [-x, -y,  z],
        [ x, -y,  z],
        [ 0,  y,  z]
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

    return trimesh.Trimesh(vertices=verts, faces=faces, process=False)


def build_mesh_from_prim(prim: Dict[str, Any]) -> trimesh.Trimesh:
    size = safe_size(parse_vec(prim.get("size", "<1,1,1>"), [1.0, 1.0, 1.0]))
    prim_type = str(prim.get("type", "BOX")).upper()

    if prim_type == "CYLINDER":
        return build_cylinder(size)

    if prim_type == "SPHERE":
        return build_sphere(size)

    if prim_type in ["TORUS", "RING"]:
        return build_torus(size)

    if prim_type == "PRISM":
        return build_prism(size)

    if prim_type == "CONE":
        return build_cone(size)

    if prim_type == "TUBE":
        return build_cylinder(size)

    return build_box(size)


def build_from_prims(prims: List[Dict[str, Any]]) -> Tuple[trimesh.Trimesh, Dict[str, Any]]:
    meshes: List[trimesh.Trimesh] = []

    for index, prim in enumerate(prims):
        try:
            mesh = build_mesh_from_prim(prim)

            pos = parse_vec(prim.get("pos", "<0,0,0>"), [0.0, 0.0, 0.0])
            rot = parse_rot(prim.get("rot", "<0,0,0,1>"))

            transform = trimesh.transformations.quaternion_matrix(rot)
            transform[:3, 3] = pos
            mesh.apply_transform(transform)

            meshes.append(clean_mesh(mesh))

        except Exception as exc:
            print("Skipping prim", index, exc)
            traceback.print_exc()

    if not meshes:
        raise RuntimeError("No valid prims could be converted.")

    merged = trimesh.util.concatenate(meshes)
    merged = clean_mesh(merged)
    merged, origin_report = center_mesh_to_origin(merged)

    dims = mesh_dimensions(merged)
    if np.max(dims) > MAX_SL_SIZE:
        raise RuntimeError("Generated mesh exceeds Second Life 64m limit after recentering.")

    return merged, origin_report


# =========================================================
# LOD + PHYSICS
# =========================================================

def decimate_to_face_count(mesh: trimesh.Trimesh, target_faces: int) -> trimesh.Trimesh:
    mesh = clean_mesh(mesh)

    if len(mesh.faces) <= target_faces:
        return clean_mesh(mesh.copy())

    try:
        reduced = mesh.simplify_quadric_decimation(face_count=target_faces)
        return clean_mesh(reduced)
    except Exception:
        pass

    try:
        reduced = mesh.simplify_quadratic_decimation(target_faces)
        return clean_mesh(reduced)
    except Exception:
        pass

    faces = np.asarray(mesh.faces, dtype=int)
    vertices = np.asarray(mesh.vertices, dtype=float)

    step = max(1, int(np.ceil(len(faces) / target_faces)))
    reduced_faces = faces[::step][:target_faces]

    used = np.unique(reduced_faces.reshape(-1))
    remap = {old: new for new, old in enumerate(used)}

    new_vertices = vertices[used]
    new_faces = np.array([[remap[i] for i in face] for face in reduced_faces], dtype=int)

    fallback = trimesh.Trimesh(vertices=new_vertices, faces=new_faces, process=False)
    return clean_mesh(fallback)


def build_box_proxy(dimensions: np.ndarray) -> trimesh.Trimesh:
    d = np.asarray(dimensions, dtype=float)
    if d.size < 3:
        d = np.array([1.0, 1.0, 1.0], dtype=float)

    d = np.abs(d[:3])
    d[d < MIN_AXIS_SIZE] = MIN_AXIS_SIZE
    d[d > MAX_SL_SIZE] = MAX_SL_SIZE

    hx = d[0] * 0.5
    hy = d[1] * 0.5
    hz = d[2] * 0.5

    verts = np.array([
        [-hx, -hy, -hz],
        [ hx, -hy, -hz],
        [ hx,  hy, -hz],
        [-hx,  hy, -hz],
        [-hx, -hy,  hz],
        [ hx, -hy,  hz],
        [ hx,  hy,  hz],
        [-hx,  hy,  hz]
    ], dtype=float)

    faces = np.array([
        [0, 2, 1],
        [0, 3, 2],
        [4, 5, 6],
        [4, 6, 7],
        [0, 1, 5],
        [0, 5, 4],
        [1, 2, 6],
        [1, 6, 5],
        [2, 3, 7],
        [2, 7, 6],
        [3, 0, 4],
        [3, 4, 7]
    ], dtype=int)

    return clean_mesh(trimesh.Trimesh(vertices=verts, faces=faces, process=False))


def make_lod_package(high: trimesh.Trimesh) -> Dict[str, trimesh.Trimesh]:
    dims = mesh_dimensions(high)

    high_lod = decimate_to_face_count(high, MAX_HIGH_FACES)
    medium_lod = decimate_to_face_count(high, MAX_MEDIUM_FACES)
    low_lod = decimate_to_face_count(high, MAX_LOW_FACES)
    lowest_lod = build_box_proxy(dims)
    phys = build_box_proxy(dims)

    return {
        "HIGH": high_lod,
        "MEDIUM": medium_lod,
        "LOW": low_lod,
        "LOWEST": lowest_lod,
        "PHYS": phys
    }


# =========================================================
# SL-SAFE COLLADA WRITER
# =========================================================

def triangle_normal(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    n = np.cross(b - a, c - a)
    length = np.linalg.norm(n)

    if not np.isfinite(length) or length <= 0.000001:
        return np.array([0.0, 0.0, 1.0], dtype=float)

    return n / length


def projected_uv(vertex: np.ndarray, min_corner: np.ndarray, dimensions: np.ndarray) -> Tuple[float, float]:
    dx = max(float(dimensions[0]), MIN_AXIS_SIZE)
    dy = max(float(dimensions[1]), MIN_AXIS_SIZE)

    u = (float(vertex[0]) - float(min_corner[0])) / dx
    v = (float(vertex[1]) - float(min_corner[1])) / dy

    if not np.isfinite(u):
        u = 0.0
    if not np.isfinite(v):
        v = 0.0

    return u, v


def float_text(values: List[float]) -> str:
    return " ".join(f"{float(x):.6f}" for x in values)


def int_text(values: List[int]) -> str:
    return " ".join(str(int(x)) for x in values)


def write_sl_safe_dae(mesh: trimesh.Trimesh, filepath: str) -> Dict[str, Any]:
    mesh = clean_mesh(mesh)

    vertices = sanitize_array(np.asarray(mesh.vertices, dtype=float))
    faces = np.asarray(mesh.faces, dtype=int)

    if len(vertices) == 0 or len(faces) == 0:
        raise RuntimeError("Cannot export empty mesh.")

    min_corner = np.min(vertices, axis=0)
    max_corner = np.max(vertices, axis=0)
    dimensions = max_corner - min_corner

    unrolled_positions: List[float] = []
    unrolled_normals: List[float] = []
    unrolled_uvs: List[float] = []

    valid_triangle_count = 0

    for face in faces:
        if len(face) != 3:
            continue

        i0, i1, i2 = int(face[0]), int(face[1]), int(face[2])

        if i0 == i1 or i1 == i2 or i0 == i2:
            continue

        a = vertices[i0]
        b = vertices[i1]
        c = vertices[i2]

        if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)) or not np.all(np.isfinite(c)):
            continue

        n = triangle_normal(a, b, c)

        for idx in [i0, i1, i2]:
            vertex = vertices[idx]
            u, v = projected_uv(vertex, min_corner, dimensions)

            unrolled_positions.extend([float(vertex[0]), float(vertex[1]), float(vertex[2])])
            unrolled_normals.extend([float(n[0]), float(n[1]), float(n[2])])
            unrolled_uvs.extend([float(u), float(v)])

        valid_triangle_count += 1

    if valid_triangle_count == 0:
        raise RuntimeError("No valid triangles to export.")

    vertex_count = valid_triangle_count * 3
    p_values = list(range(vertex_count))

    pos_str = float_text(unrolled_positions)
    normal_str = float_text(unrolled_normals)
    uv_str = float_text(unrolled_uvs)
    p_str = int_text(p_values)

    dae = f'''<?xml version="1.0" encoding="utf-8"?>
<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema" version="1.4.1">
  <asset>
    <contributor>
      <authoring_tool>M3D3 Platinum SL-Safe Exporter</authoring_tool>
    </contributor>
    <created>2026-01-01T00:00:00Z</created>
    <modified>2026-01-01T00:00:00Z</modified>
    <unit name="meter" meter="1"/>
    <up_axis>Z_UP</up_axis>
  </asset>

  <library_effects>
    <effect id="Material-effect">
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
    <geometry id="SL_Mesh_Geom" name="SL_Mesh_Geom">
      <mesh>
        <source id="SL_Mesh_Geom-positions">
          <float_array id="SL_Mesh_Geom-positions-array" count="{vertex_count * 3}">
            {pos_str}
          </float_array>
          <technique_common>
            <accessor source="#SL_Mesh_Geom-positions-array" count="{vertex_count}" stride="3">
              <param name="X" type="float"/>
              <param name="Y" type="float"/>
              <param name="Z" type="float"/>
            </accessor>
          </technique_common>
        </source>

        <source id="SL_Mesh_Geom-normals">
          <float_array id="SL_Mesh_Geom-normals-array" count="{vertex_count * 3}">
            {normal_str}
          </float_array>
          <technique_common>
            <accessor source="#SL_Mesh_Geom-normals-array" count="{vertex_count}" stride="3">
              <param name="X" type="float"/>
              <param name="Y" type="float"/>
              <param name="Z" type="float"/>
            </accessor>
          </technique_common>
        </source>

        <source id="SL_Mesh_Geom-map-0">
          <float_array id="SL_Mesh_Geom-map-0-array" count="{vertex_count * 2}">
            {uv_str}
          </float_array>
          <technique_common>
            <accessor source="#SL_Mesh_Geom-map-0-array" count="{vertex_count}" stride="2">
              <param name="S" type="float"/>
              <param name="T" type="float"/>
            </accessor>
          </technique_common>
        </source>

        <vertices id="SL_Mesh_Geom-vertices">
          <input semantic="POSITION" source="#SL_Mesh_Geom-positions"/>
        </vertices>

        <triangles material="Material" count="{valid_triangle_count}">
          <input semantic="VERTEX" source="#SL_Mesh_Geom-vertices" offset="0"/>
          <input semantic="NORMAL" source="#SL_Mesh_Geom-normals" offset="0"/>
          <input semantic="TEXCOORD" source="#SL_Mesh_Geom-map-0" offset="0" set="0"/>
          <p>{p_str}</p>
        </triangles>
      </mesh>
    </geometry>
  </library_geometries>

  <library_visual_scenes>
    <visual_scene id="Scene" name="Scene">
      <node id="SL_Mesh_Node" name="SL_Mesh_Node" type="NODE">
        <matrix sid="transform">1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1</matrix>
        <instance_geometry url="#SL_Mesh_Geom">
          <bind_material>
            <technique_common>
              <instance_material symbol="Material" target="#Material"/>
            </technique_common>
          </bind_material>
        </instance_geometry>
      </node>
    </visual_scene>
  </library_visual_scenes>

  <scene>
    <instance_visual_scene url="#Scene"/>
  </scene>
</COLLADA>
'''

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(dae)

    if not os.path.exists(filepath):
        raise RuntimeError("DAE export failed.")

    if os.path.getsize(filepath) <= 0:
        raise RuntimeError("DAE export created empty file.")

    meta = {
        "file": os.path.basename(filepath),
        "file_size": os.path.getsize(filepath),
        "vertex_count_unrolled": vertex_count,
        "face_count": valid_triangle_count,
        "bounds_min": sanitize_array(min_corner).tolist(),
        "bounds_max": sanitize_array(max_corner).tolist(),
        "dimensions": sanitize_array(dimensions).tolist(),
        "has_nan": bool(np.isnan(vertices).any()),
        "has_inf": bool(np.isinf(vertices).any()),
        "node_name": "SL_Mesh_Node",
        "geometry_id": "SL_Mesh_Geom"
    }

    with open(filepath + ".meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return meta


def export_mesh(mesh: trimesh.Trimesh, filename: str) -> Dict[str, Any]:
    path = os.path.join(OUTPUT_DIR, filename)
    return write_sl_safe_dae(mesh, path)


def export_obj(mesh: trimesh.Trimesh, filename: str) -> Dict[str, Any]:
    path = os.path.join(OUTPUT_DIR, filename)
    mesh.export(path)
    return {
        "file": filename,
        "file_size": os.path.getsize(path),
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces))
    }


def export_glb(mesh: trimesh.Trimesh, filename: str) -> Dict[str, Any]:
    path = os.path.join(OUTPUT_DIR, filename)
    mesh.export(path)
    return {
        "file": filename,
        "file_size": os.path.getsize(path),
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces))
    }


def produce_package(high: trimesh.Trimesh, name: str) -> Dict[str, Any]:
    high = clean_mesh(high)
    high, origin_report = center_mesh_to_origin(high)

    lods = make_lod_package(high)

    uid = uuid.uuid4().hex[:8]

    files = {
        "DAE": f"{name}_SL_READY_{uid}.dae",
        "HIGH": f"{name}_HI_{uid}.dae",
        "MEDIUM": f"{name}_MED_{uid}.dae",
        "LOW": f"{name}_LOW_{uid}.dae",
        "LOWEST": f"{name}_LOWEST_{uid}.dae",
        "PHYS": f"{name}_PHYS_{uid}.dae",
        "OBJ": f"{name}_{uid}.obj",
        "GLB": f"{name}_{uid}.glb",
        "ZIP": f"{name}_LOD_PACKAGE_{uid}.zip"
    }

    metadata = {
        "DAE": export_mesh(lods["HIGH"], files["DAE"]),
        "HIGH": export_mesh(lods["HIGH"], files["HIGH"]),
        "MEDIUM": export_mesh(lods["MEDIUM"], files["MEDIUM"]),
        "LOW": export_mesh(lods["LOW"], files["LOW"]),
        "LOWEST": export_mesh(lods["LOWEST"], files["LOWEST"]),
        "PHYS": export_mesh(lods["PHYS"], files["PHYS"]),
        "OBJ": export_obj(lods["HIGH"], files["OBJ"]),
        "GLB": export_glb(lods["HIGH"], files["GLB"])
    }

    zip_path = os.path.join(OUTPUT_DIR, files["ZIP"])
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for key in ["HIGH", "MEDIUM", "LOW", "LOWEST", "PHYS"]:
            z.write(os.path.join(OUTPUT_DIR, files[key]), files[key])

    metadata["ZIP"] = {
        "file": files["ZIP"],
        "file_size": os.path.getsize(zip_path)
    }

    package_id = uid

    package = {
        "id": package_id,
        "name": name,
        "created": now_ts(),
        "files": files,
        "metadata": metadata,
        "origin": origin_report,
        "summary": mesh_report(lods["HIGH"])
    }

    results[package_id] = package
    return package


def package_urls(package: Dict[str, Any]) -> Dict[str, str]:
    host = request.host
    pid = package["id"]
    files = package["files"]

    return {
        "JOB_PAGE": f"https://{host}/job/{pid}",
        "DAE": f"https://{host}/download/{files['DAE']}",
        "OBJ": f"https://{host}/download/{files['OBJ']}",
        "GLB": f"https://{host}/download/{files['GLB']}",
        "LOD_ZIP": f"https://{host}/download/{files['ZIP']}",
        "HIGH": f"https://{host}/download/{files['HIGH']}",
        "MEDIUM": f"https://{host}/download/{files['MEDIUM']}",
        "LOW": f"https://{host}/download/{files['LOW']}",
        "LOWEST": f"https://{host}/download/{files['LOWEST']}",
        "PHYS": f"https://{host}/download/{files['PHYS']}"
    }


# =========================================================
# ROUTES
# =========================================================

@app.route("/")
def home():
    return "M3D3 PRIM TO MESH SERVER RUNNING"


@app.route("/health")
def health():
    cleanup_old_memory()

    return jsonify({
        "ok": True,
        "server": "M3D3 Platinum General Builder Delivery System",
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
            return jsonify({"error": "missing job"}), 400

        if not isinstance(chunk, list):
            return jsonify({"error": "chunk must be list"}), 400

        if job not in jobs:
            jobs[job] = []

        jobs[job].extend(chunk)

        return jsonify({
            "ok": True,
            "job": job,
            "received": len(chunk),
            "total": len(jobs[job])
        }), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/finalize", methods=["POST"])
def finalize():
    try:
        cleanup_old_files()
        cleanup_old_memory()

        data = request.get_json(force=True)
        job = str(data.get("job", "")).strip()
        name = safe_name(data.get("name", "Object"))

        if job == "":
            return jsonify({"error": "missing job"}), 400

        if job not in jobs:
            return jsonify({
                "error": "job not found",
                "job": job,
                "known_jobs": list(jobs.keys())
            }), 400

        prims = jobs.get(job, [])

        if not prims:
            return jsonify({"error": "job has no prim data"}), 400

        high, scan_origin = build_from_prims(prims)
        package = produce_package(high, name)

        del jobs[job]

        urls = package_urls(package)

        return jsonify({
            "ok": True,
            "JOB_PAGE": urls["JOB_PAGE"],
            "DAE": urls["DAE"],
            "OBJ": urls["OBJ"],
            "GLB": urls["GLB"],
            "LOD_ZIP": urls["LOD_ZIP"],
            "HIGH": urls["HIGH"],
            "MEDIUM": urls["MEDIUM"],
            "LOW": urls["LOW"],
            "LOWEST": urls["LOWEST"],
            "PHYS": urls["PHYS"],
            "summary": package["summary"],
            "scan_origin": scan_origin,
            "package_origin": package["origin"]
        }), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/test_export", methods=["GET"])
def test_export():
    try:
        cleanup_old_files()
        cleanup_old_memory()

        cube = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
        cylinder = trimesh.creation.cylinder(radius=0.25, height=0.35, sections=16)
        cylinder.apply_translation([0.0, 0.0, 0.675])

        high = trimesh.util.concatenate([cube, cylinder])
        high = clean_mesh(high)

        package = produce_package(high, "M3D3_Test")
        urls = package_urls(package)

        return jsonify({
            "ok": True,
            "JOB_PAGE": urls["JOB_PAGE"],
            "DAE": urls["DAE"],
            "OBJ": urls["OBJ"],
            "GLB": urls["GLB"],
            "LOD_ZIP": urls["LOD_ZIP"],
            "HIGH": urls["HIGH"],
            "MEDIUM": urls["MEDIUM"],
            "LOW": urls["LOW"],
            "LOWEST": urls["LOWEST"],
            "PHYS": urls["PHYS"],
            "summary": package["summary"],
            "metadata": package["metadata"]
        }), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/job/<package_id>", methods=["GET"])
def job_page(package_id: str):
    package = results.get(package_id)

    if not package:
        return Response(
            "<h1>M3D3 Job Not Found</h1><p>This job expired, the server restarted, or the package was cleaned up.</p>",
            mimetype="text/html"
        )

    urls = package_urls(package)

    summary = package.get("summary", {})
    dims = summary.get("dimensions", ["?", "?", "?"])

    html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>M3D3 Mesh Ready</title>
    <style>
        body {{
            margin: 0;
            font-family: Arial, sans-serif;
            background: #111;
            color: #f5f5f5;
        }}
        .wrap {{
            max-width: 980px;
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
        .meta {{
            color: #bbb;
            line-height: 1.6;
        }}
        code {{
            background: #000;
            padding: 3px 6px;
            border-radius: 4px;
        }}
    </style>
</head>
<body>
    <div class="wrap">
        <div class="card">
            <h1>M3D3 Mesh Ready</h1>
            <p class="meta">Package ID: <code>{package_id}</code></p>
            <p class="meta">Dimensions: <code>{dims}</code></p>
            <p class="meta">Faces: <code>{summary.get("faces", "?")}</code> | Vertices: <code>{summary.get("vertices", "?")}</code></p>
        </div>

        <div class="card">
            <h2>Creator Downloads</h2>
            <div class="grid">
                <a class="button" href="{urls["DAE"]}">Download SL Ready DAE</a>
                <a class="button secondary" href="{urls["OBJ"]}">Download OBJ</a>
                <a class="button secondary" href="{urls["GLB"]}">Download GLB</a>
                <a class="button secondary" href="{urls["LOD_ZIP"]}">Download Advanced LOD ZIP</a>
            </div>
        </div>

        <div class="card">
            <h2>Advanced LOD Files</h2>
            <div class="grid">
                <a class="button secondary" href="{urls["HIGH"]}">HIGH</a>
                <a class="button secondary" href="{urls["MEDIUM"]}">MEDIUM</a>
                <a class="button secondary" href="{urls["LOW"]}">LOW</a>
                <a class="button secondary" href="{urls["LOWEST"]}">LOWEST</a>
                <a class="button secondary" href="{urls["PHYS"]}">PHYS</a>
            </div>
        </div>

        <div class="card">
            <h2>Upload Modes</h2>
            <p class="meta"><b>Simple Mode:</b> Download <code>SL Ready DAE</code> and upload that one file.</p>
            <p class="meta"><b>Advanced Mode:</b> Download <code>LOD ZIP</code>, unzip it, then load HIGH / MEDIUM / LOW / LOWEST / PHYS into the SL uploader manually.</p>
        </div>
    </div>
</body>
</html>
"""
    return Response(html, mimetype="text/html")


@app.route("/validate/<path:filename>", methods=["GET"])
def validate(filename: str):
    safe_file = clean_filename(filename)
    path = os.path.join(OUTPUT_DIR, safe_file)
    meta_path = path + ".meta.json"

    if not os.path.exists(path):
        return jsonify({
            "ok": False,
            "error": "file not found",
            "requested": safe_file,
            "available": [f for f in os.listdir(OUTPUT_DIR) if not f.endswith(".meta.json")]
        }), 404

    dae_text = ""
    try:
        with open(path, "r", encoding="utf-8") as f:
            dae_text = f.read()
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
        "file_exists": True,
        "file_size": os.path.getsize(path),
        "contains_collada": "<COLLADA" in dae_text,
        "contains_z_up": "<up_axis>Z_UP</up_axis>" in dae_text,
        "contains_meter": '<unit name="meter" meter="1"/>' in dae_text,
        "contains_single_node": 'name="SL_Mesh_Node"' in dae_text,
        "contains_single_geometry": 'id="SL_Mesh_Geom"' in dae_text,
        "contains_triangles": "<triangles" in dae_text,
        "contains_texcoord": "TEXCOORD" in dae_text,
        "contains_normals": "normals" in dae_text.lower(),
        "contains_nan_text": "nan" in dae_text.lower(),
        "contains_inf_text": "inf" in dae_text.lower()
    }

    ok = (
        checks["file_exists"] and
        checks["file_size"] > 0 and
        checks["contains_collada"] and
        checks["contains_z_up"] and
        checks["contains_meter"] and
        checks["contains_single_node"] and
        checks["contains_single_geometry"] and
        checks["contains_triangles"] and
        checks["contains_texcoord"] and
        checks["contains_normals"] and
        not checks["contains_nan_text"] and
        not checks["contains_inf_text"]
    )

    return jsonify({
        "ok": ok,
        "file": safe_file,
        "checks": checks,
        "metadata": meta
    }), 200


@app.route("/download/<path:filename>", methods=["GET"])
def download(filename: str):
    safe_file = clean_filename(filename)
    path = os.path.join(OUTPUT_DIR, safe_file)

    if not os.path.exists(path):
        return jsonify({
            "error": "file not found",
            "requested": safe_file,
            "available": [f for f in os.listdir(OUTPUT_DIR) if not f.endswith(".meta.json")]
        }), 404

    return send_from_directory(
        OUTPUT_DIR,
        safe_file,
        as_attachment=True
    )


# =========================================================
# LOCAL RUN
# =========================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
