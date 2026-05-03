# =========================================================
# M3D3 MERGE ENGINE — CHUNKED (RENDER READY)
# =========================================================

import os
import uuid
import json
import numpy as np
import trimesh
from flask import Flask, request, jsonify, send_from_directory

app = Flask(__name__)

# In-memory job store (use Redis in production)
jobs = {}

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------
# HELPERS
# -------------------------

def parse_vec3(s):
    # accepts "<x, y, z>" or "x, y, z"
    s = s.strip().replace("<", "").replace(">", "")
    return np.fromstring(s, sep=',')

def parse_quat_lsl(s):
    # LSL: <x, y, z, s>  → trimesh expects [w, x, y, z]
    s = s.strip().replace("<", "").replace(">", "")
    q = np.fromstring(s, sep=',')
    if len(q) != 4:
        return np.array([1, 0, 0, 0])
    return np.array([q[3], q[0], q[1], q[2]])

def make_primitive(p):
    p_type = p.get("type", "BOX").upper()
    size = parse_vec3(p.get("size", "1,1,1"))
    size = np.maximum(size, 1e-6)

    if p_type == "BOX":
        return trimesh.creation.box(extents=size)

    if p_type == "CYLINDER":
        r = max(size[0], size[1]) * 0.5
        h = size[2]
        return trimesh.creation.cylinder(radius=r, height=h, sections=24)

    if p_type == "SPHERE":
        r = max(size) * 0.5
        return trimesh.creation.uv_sphere(radius=r, count=[24, 24])

    if p_type == "PRISM":
        # triangular prism
        r = max(size[0], size[1]) * 0.5
        h = size[2]
        return trimesh.creation.cone(radius=r, height=h, sections=3)

    if p_type in ("TORUS", "TUBE", "RING"):
        # approximate torus (major radius from size.x, minor from size.y)
        R = max(size[0], 1e-3)
        r = max(size[1], 1e-3) * 0.5
        return trimesh.creation.torus(radius=R, tube_radius=r, sections=32, segments=24)

    # fallback
    return trimesh.creation.box(extents=size)

def apply_transform(m, p):
    pos = parse_vec3(p.get("pos", "0,0,0"))
    quat = parse_quat_lsl(p.get("rot", "0,0,0,1"))

    T = trimesh.transformations.quaternion_matrix(quat)
    T[:3, 3] = pos
    m.apply_transform(T)

def apply_deformation(m, p):
    # NOTE: prim deformations (taper/twist/hollow/shear) are not native in trimesh.
    # This is a light approximation stage; exact SL parity requires custom vertex ops.

    # simple twist around Z
    twist = p.get("twist", "0,0,0")
    try:
        t = parse_vec3(twist)
        total = float(t[1]) if len(t) > 1 else 0.0
    except:
        total = 0.0

    if abs(total) > 0.0:
        z = m.vertices[:, 2]
        zmin, zmax = z.min(), z.max()
        span = max(zmax - zmin, 1e-6)
        ang = (z - zmin) / span * np.deg2rad(total)
        c, s = np.cos(ang), np.sin(ang)
        x, y = m.vertices[:, 0].copy(), m.vertices[:, 1].copy()
        m.vertices[:, 0] = x * c - y * s
        m.vertices[:, 1] = x * s + y * c

    # taper (scale XY along Z)
    taper = p.get("taper", "0,0,0")
    try:
        tv = parse_vec3(taper)
        tx = float(tv[0]); ty = float(tv[1])
    except:
        tx = ty = 0.0

    if abs(tx) > 0.0 or abs(ty) > 0.0:
        z = m.vertices[:, 2]
        zmin, zmax = z.min(), z.max()
        t = (z - zmin) / max(zmax - zmin, 1e-6)
        sx = 1.0 - tx * t
        sy = 1.0 - ty * t
        m.vertices[:, 0] *= sx
        m.vertices[:, 1] *= sy

    # shear (simple XY offset along Z)
    shear = p.get("shear", "0,0,0")
    try:
        sv = parse_vec3(shear)
        shx = float(sv[0]); shy = float(sv[1])
    except:
        shx = shy = 0.0

    if abs(shx) > 0.0 or abs(shy) > 0.0:
        z = m.vertices[:, 2]
        zmin, zmax = z.min(), z.max()
        t = (z - zmin) / max(zmax - zmin, 1e-6)
        m.vertices[:, 0] += shx * t
        m.vertices[:, 1] += shy * t

    # hollow omitted here (non-trivial boolean). Keep for later if needed.

# -------------------------
# ROUTES
# -------------------------

@app.route("/")
def home():
    return "M3D3 MERGE ENGINE RUNNING"

@app.route("/upload_chunk", methods=["POST"])
def upload_chunk():
    data = request.get_json(force=True)
    job_id = data.get("job")
    chunk = data.get("chunk", [])

    if not job_id:
        return jsonify({"error": "missing job id"}), 400

    if job_id not in jobs:
        jobs[job_id] = []

    jobs[job_id].extend(chunk)
    return jsonify({"status": "chunk received"}), 200

@app.route("/finalize", methods=["POST"])
def finalize():
    data = request.get_json(force=True)
    job_id = data.get("job")
    name = data.get("name", "M3D3_Export")

    if job_id not in jobs:
        return jsonify({"error": "job not found"}), 404

    prims = jobs[job_id]
    meshes = []

    for p in prims:
        try:
            m = make_primitive(p)
            apply_deformation(m, p)
            apply_transform(m, p)
            meshes.append(m)
        except Exception as e:
            # skip bad prim but continue
            continue

    if not meshes:
        return jsonify({"error": "no meshes built"}), 500

    final_mesh = trimesh.util.concatenate(meshes)

    filename = f"{name}_{uuid.uuid4().hex}.dae"
    filepath = os.path.join(OUTPUT_DIR, filename)

    # export Collada
    final_mesh.export(filepath)

    # cleanup
    del jobs[job_id]

    url = f"https://{request.host}/download/{filename}"

    return jsonify({
        "status": "complete",
        "url": url
    }), 200

@app.route("/download/<filename>")
def download(filename):
    return send_from_directory(OUTPUT_DIR, filename, as_attachment=True)

# -------------------------
# RUN (RENDER READY)
# -------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
