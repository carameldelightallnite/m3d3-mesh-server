# =========================================================
# M3D3 PLATINUM SERVER — FINAL COMMERCIAL BUILD
# LSL → Python → Trimesh → LOD → Physics → DAE Download
# =========================================================

import os
import re
import time
import uuid
import traceback
import numpy as np
import trimesh

from flask import Flask, request, jsonify, send_from_directory

app = Flask(__name__)

# =========================================================
# GLOBAL STORAGE
# =========================================================

jobs = {}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

FILE_TTL_SECONDS = 1800


# =========================================================
# SAFE HELPERS
# =========================================================

def safe_name(name):
    name = str(name or "M3D3_Export")
    name = name.replace(" ", "_")
    name = "".join(c for c in name if c.isalnum() or c == "_")
    if name == "":
        name = "M3D3_Export"
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
    ])

    return trimesh.Trimesh(
        vertices=verts,
        faces=faces,
        process=False
    )


def build_mesh_from_prim(prim):
    size = safe_size(parse_vec(
        prim.get("size", "<1,1,1>"),
        [1.0, 1.0, 1.0]
    ))

    prim_type = str(prim.get("type", "BOX")).upper()

    if prim_type == "CYLINDER":
        mesh = build_cylinder(size)

    elif prim_type == "SPHERE":
        mesh = build_sphere(size)

    elif prim_type == "TORUS":
        mesh = build_torus(size)

    elif prim_type == "PRISM":
        mesh = build_prism(size)

    elif prim_type == "CONE":
        mesh = build_cone(size)

    elif prim_type == "TUBE":
        mesh = build_cylinder(size)

    elif prim_type == "RING":
        mesh = build_torus(size)

    else:
        mesh = build_box(size)

    return mesh


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

    try:
        if getattr(mesh.visual, "uv", None) is None:
            uv = mesh.vertices[:, :2].copy()
            mesh.visual = trimesh.visual.TextureVisuals(uv=uv)
    except Exception:
        pass

    return mesh


def decimate_mesh(mesh, ratio):
    target_faces = max(8, int(len(mesh.faces) * ratio))

    try:
        out = mesh.simplify_quadric_decimation(face_count=target_faces)
        return clean_mesh(out)
    except Exception:
        pass

    try:
        out = mesh.simplify_quadratic_decimation(target_faces)
        return clean_mesh(out)
    except Exception:
        pass

    return clean_mesh(mesh.copy())


def make_lods(high):
    medium = decimate_mesh(high, 0.50)
    low = decimate_mesh(high, 0.25)
    lowest = decimate_mesh(high, 0.10)

    return high, medium, low, lowest


def make_physics(high):
    try:
        hull = high.convex_hull
        return decimate_mesh(hull, 0.25)
    except Exception:
        return decimate_mesh(high, 0.10)


def export_mesh(mesh, filename):
    path = os.path.join(OUTPUT_DIR, filename)

    mesh.export(path)

    if not os.path.exists(path):
        raise RuntimeError("Export failed: " + filename)

    if os.path.getsize(path) <= 0:
        raise RuntimeError("Export created empty file: " + filename)

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
        "server": "M3D3 Platinum",
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

                color = np.array([
                    (index * 45) % 255,
                    (index * 75) % 255,
                    (index * 115) % 255,
                    255
                ], dtype=np.uint8)

                mesh.visual.face_colors = np.tile(
                    color,
                    (len(mesh.faces), 1)
                )

                meshes.append(clean_mesh(mesh))

            except Exception as prim_error:
                print("Skipping prim", index, prim_error)
                traceback.print_exc()

        if not meshes:
            return jsonify({"error": "mesh generation failed"}), 500

        high = trimesh.util.concatenate(meshes)
        high = clean_mesh(high)

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

        export_mesh(high, files["HIGH"])
        export_mesh(medium, files["MEDIUM"])
        export_mesh(low, files["LOW"])
        export_mesh(lowest, files["LOWEST"])
        export_mesh(phys, files["PHYS"])

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
            "hint": "Check if the file expired, was deleted, or the link contains extra copied JSON characters.",
            "available": os.listdir(OUTPUT_DIR)
        }), 404

    return send_from_directory(
        OUTPUT_DIR,
        safe_file,
        as_attachment=True,
        mimetype="model/vnd.collada+xml"
    )


# =========================================================
# LOCAL RUN
# =========================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
