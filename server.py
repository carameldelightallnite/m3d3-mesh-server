# =========================================================
# M3D3 PLATINUM SERVER — FINAL PYCOLLADA SL-SAFE BUILD
# Fixes:
# - MAV_BLOCK_MISSING
# - Blank preview
# - Red-dot preview
# - Off-origin mesh export
# - LOD internal node mismatch
# - Bad physics hull density
# =========================================================

import os
import time
import uuid
import traceback
import numpy as np
import trimesh
import collada

from flask import Flask, request, jsonify, send_from_directory

app = Flask(__name__)

jobs = {}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

FILE_TTL_SECONDS = 1800


# =========================================================
# SAFE HELPERS
# =========================================================

def safe_name(name):
    name = str(name or "Object")
    name = name.replace(" ", "_")
    name = "".join(c for c in name if c.isalnum() or c == "_")
    if name == "":
        name = "Object"
    return name


def cleanup_old_files():
    now = time.time()
    try:
        for filename in os.listdir(OUTPUT_DIR):
            path = os.path.join(OUTPUT_DIR, filename)
            if os.path.isfile(path):
                age = now - os.path.getmtime(path)
                if age > FILE_TTL_SECONDS:
                    os.remove(path)
    except Exception as e:
        print("Cleanup error:", e)


def parse_vec(value, fallback):
    try:
        text = str(value).replace("<", "").replace(">", "").strip()
        arr = np.fromstring(text, sep=",")
        if arr.size < len(fallback):
            return np.array(fallback, dtype=float)
        return arr.astype(float)
    except Exception:
        return np.array(fallback, dtype=float)


def parse_rot(value):
    try:
        q = parse_vec(value, [0.0, 0.0, 0.0, 1.0])
        if q.size < 4:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)

        # LSL rotation = <x, y, z, s>
        # trimesh wants = <w, x, y, z>
        return np.array([q[3], q[0], q[1], q[2]], dtype=float)

    except Exception:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)


def safe_size(size):
    s = np.array(size, dtype=float)

    if s.size < 3:
        s = np.array([1.0, 1.0, 1.0], dtype=float)

    s = np.abs(s)
    s[s < 0.001] = 0.001

    return s


# =========================================================
# GEOMETRY BUILDERS
# =========================================================

def build_box(size):
    return trimesh.creation.box(extents=size)


def build_cylinder(size):
    base = max(size[0], size[1])
    if base <= 0.0:
        base = 1.0

    radius = base * 0.5
    height = size[2]

    mesh = trimesh.creation.cylinder(
        radius=radius,
        height=height,
        sections=32
    )

    mesh.apply_scale([
        size[0] / base,
        size[1] / base,
        1.0
    ])

    return mesh


def build_sphere(size):
    mesh = trimesh.creation.uv_sphere(
        radius=0.5,
        count=[32, 32]
    )

    mesh.apply_scale(size)

    return mesh


def build_torus(size):
    base = max(size[0], size[1])
    if base <= 0.0:
        base = 1.0

    major = base * 0.35
    minor = max(min(size[0], size[1]) * 0.12, size[2] * 0.25, 0.01)

    try:
        mesh = trimesh.creation.torus(
            major_radius=major,
            minor_radius=minor,
            major_sections=48,
            minor_sections=16
        )
    except TypeError:
        mesh = trimesh.creation.torus(
            radius=major,
            tube_radius=minor,
            sections=48,
            segments=16
        )

    mesh.apply_scale([
        size[0] / base,
        size[1] / base,
        1.0
    ])

    return mesh


def build_cone(size):
    base = max(size[0], size[1])
    if base <= 0.0:
        base = 1.0

    radius = base * 0.5
    height = size[2]

    mesh = trimesh.creation.cone(
        radius=radius,
        height=height,
        sections=32
    )

    mesh.apply_scale([
        size[0] / base,
        size[1] / base,
        1.0
    ])

    return mesh


def build_prism(size):
    x = size[0] * 0.5
    y = size[1] * 0.5
    z = size[2] * 0.5

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


def build_mesh_from_prim(prim):
    size = safe_size(parse_vec(
        prim.get("size", "<1,1,1>"),
        [1.0, 1.0, 1.0]
    ))

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


# =========================================================
# CLEAN / OPTIMIZE
# =========================================================

def clean_mesh(mesh):
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
        mesh.merge_vertices(digits=4)
    except Exception:
        pass

    try:
        mesh.fill_holes()
    except Exception:
        pass

    try:
        mesh.fix_normals()
    except Exception:
        pass

    vertices = np.asarray(mesh.vertices, dtype=float)

    if len(vertices) > 0:
        vertices = np.nan_to_num(vertices, nan=0.0, posinf=0.0, neginf=0.0)
        mesh.vertices = vertices

    return mesh


def center_mesh(mesh):
    mesh = clean_mesh(mesh)

    bounds = mesh.bounds
    center = (bounds[0] + bounds[1]) * 0.5

    mesh.apply_translation(-center)

    return clean_mesh(mesh)


def decimate_mesh(mesh, ratio):
    target_faces = max(8, int(len(mesh.faces) * ratio))

    try:
        out = mesh.simplify_quadric_decimation(face_count=target_faces)
        return center_mesh(out)
    except Exception:
        pass

    try:
        out = mesh.simplify_quadratic_decimation(target_faces)
        return center_mesh(out)
    except Exception:
        pass

    return center_mesh(mesh.copy())


def make_lods(high):
    high = center_mesh(high)
    medium = decimate_mesh(high, 0.50)
    low = decimate_mesh(high, 0.25)
    lowest = decimate_mesh(high, 0.10)

    return high, medium, low, lowest


def make_physics(high):
    try:
        bounds = high.bounds
        min_corner = bounds[0]
        max_corner = bounds[1]

        x0, y0, z0 = min_corner
        x1, y1, z1 = max_corner

        if abs(x1 - x0) < 0.001:
            x1 = x0 + 0.001
        if abs(y1 - y0) < 0.001:
            y1 = y0 + 0.001
        if abs(z1 - z0) < 0.001:
            z1 = z0 + 0.001

        verts = np.array([
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y1, z0],
            [x0, y1, z0],
            [x0, y0, z1],
            [x1, y0, z1],
            [x1, y1, z1],
            [x0, y1, z1]
        ], dtype=float)

        faces = np.array([
            [0, 1, 2],
            [0, 2, 3],
            [4, 6, 5],
            [4, 7, 6],
            [0, 4, 5],
            [0, 5, 1],
            [1, 5, 6],
            [1, 6, 2],
            [2, 6, 7],
            [2, 7, 3],
            [3, 7, 4],
            [3, 4, 0]
        ], dtype=int)

        phys = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        return center_mesh(phys)

    except Exception:
        return center_mesh(trimesh.creation.box(extents=[1.0, 1.0, 1.0]))


# =========================================================
# PYCOLLADA EXPORTER
# =========================================================

def mesh_to_collada(mesh, path, mesh_name):
    mesh = center_mesh(mesh)

    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int32)

    if len(vertices) == 0 or len(faces) == 0:
        raise RuntimeError("Cannot export empty mesh.")

    try:
        normals = np.asarray(mesh.vertex_normals, dtype=np.float32)
    except Exception:
        normals = np.zeros_like(vertices)
        normals[:, 2] = 1.0

    vertices = np.nan_to_num(vertices, nan=0.0, posinf=0.0, neginf=0.0)
    normals = np.nan_to_num(normals, nan=0.0, posinf=0.0, neginf=0.0)

    normal_lengths = np.linalg.norm(normals, axis=1)
    normal_lengths[normal_lengths == 0] = 1.0
    normals = normals / normal_lengths[:, None]

    uvs = np.zeros((len(vertices), 2), dtype=np.float32)

    name = safe_name(mesh_name)

    dae = collada.Collada()

    effect = collada.material.Effect(
        "MaterialEffect",
        [],
        "phong",
        diffuse=(0.8, 0.8, 0.8, 1.0),
        specular=(0.0, 0.0, 0.0, 1.0)
    )

    material = collada.material.Material(
        "Material",
        "Material",
        effect
    )

    dae.effects.append(effect)
    dae.materials.append(material)

    vert_source = collada.source.FloatSource(
        f"{name}_verts_array",
        vertices,
        ("X", "Y", "Z")
    )

    normal_source = collada.source.FloatSource(
        f"{name}_normals_array",
        normals,
        ("X", "Y", "Z")
    )

    uv_source = collada.source.FloatSource(
        f"{name}_uv_array",
        uvs,
        ("S", "T")
    )

    geom = collada.geometry.Geometry(
        dae,
        f"{name}_geometry",
        name,
        [vert_source, normal_source, uv_source]
    )

    input_list = collada.source.InputList()
    input_list.addInput(0, "VERTEX", f"#{name}_verts_array")
    input_list.addInput(1, "NORMAL", f"#{name}_normals_array")
    input_list.addInput(2, "TEXCOORD", f"#{name}_uv_array", set="0")

    indices = np.repeat(faces.reshape(-1), 3).astype(np.int32)

    tri_set = geom.createTriangleSet(
        indices,
        input_list,
        "MaterialRef"
    )

    geom.primitives.append(tri_set)
    dae.geometries.append(geom)

    mat_node = collada.scene.MaterialNode(
        "MaterialRef",
        material,
        inputs=[
            ("TEXCOORD", "TEXCOORD", "0")
        ]
    )

    geom_node = collada.scene.GeometryNode(
        geom,
        [mat_node]
    )

    node = collada.scene.Node(
        name,
        children=[geom_node]
    )

    scene = collada.scene.Scene(
        "Scene",
        [node]
    )

    dae.scenes.append(scene)
    dae.scene = scene

    dae.write(path)

    if not os.path.exists(path):
        raise RuntimeError("DAE export failed.")

    if os.path.getsize(path) <= 0:
        raise RuntimeError("DAE export created empty file.")

    return path


def export_mesh(mesh, filename, mesh_name):
    path = os.path.join(OUTPUT_DIR, filename)
    mesh_to_collada(mesh, path, mesh_name)
    return path


# =========================================================
# ROUTES
# =========================================================

@app.route("/")
def home():
    return "M3D3 SERVER RUNNING"


@app.route("/health")
def health():
    return jsonify({
        "ok": True,
        "server": "M3D3 Platinum PyCollada Exporter",
        "outputs": os.listdir(OUTPUT_DIR),
        "jobs": list(jobs.keys())
    })


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

        meshes = []

        for index, prim in enumerate(prims):
            try:
                mesh = build_mesh_from_prim(prim)

                pos = parse_vec(
                    prim.get("pos", "<0,0,0>"),
                    [0.0, 0.0, 0.0]
                )

                rot = parse_rot(
                    prim.get("rot", "<0,0,0,1>")
                )

                transform = trimesh.transformations.quaternion_matrix(rot)
                transform[:3, 3] = pos
                mesh.apply_transform(transform)

                meshes.append(clean_mesh(mesh))

            except Exception as prim_error:
                print("Skipping prim", index, prim_error)
                traceback.print_exc()

        if not meshes:
            return jsonify({"error": "mesh generation failed"}), 500

        high = trimesh.util.concatenate(meshes)
        high = center_mesh(high)

        high, medium, low, lowest = make_lods(high)
        phys = make_physics(high)

        uid = uuid.uuid4().hex[:8]

        files = {
            "HIGH": f"{name}_HI_{uid}.dae",
            "MEDIUM": f"{name}_MED_{uid}.dae",
            "LOW": f"{name}_LOW_{uid}.dae",
            "LOWEST": f"{name}_LOWEST_{uid}.dae",
            "PHYS": f"{name}_PHYS_{uid}.dae"
        }

        # All files use the exact same internal object name.
        export_mesh(high, files["HIGH"], name)
        export_mesh(medium, files["MEDIUM"], name)
        export_mesh(low, files["LOW"], name)
        export_mesh(lowest, files["LOWEST"], name)
        export_mesh(phys, files["PHYS"], name)

        del jobs[job]

        return jsonify({
            key: f"https://{request.host}/download/{filename}"
            for key, filename in files.items()
        }), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/download/<path:filename>", methods=["GET"])
def download(filename):
    safe_file = (
        os.path.basename(filename)
        .replace('"', "")
        .replace("\\", "")
        .replace("}", "")
        .replace("{", "")
        .strip()
    )

    file_path = os.path.join(OUTPUT_DIR, safe_file)

    if not os.path.exists(file_path):
        return jsonify({
            "error": "File not found",
            "requested": safe_file,
            "available": os.listdir(OUTPUT_DIR)
        }), 404

    return send_from_directory(
        OUTPUT_DIR,
        safe_file,
        as_attachment=True,
        mimetype="model/vnd.collada+xml"
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
