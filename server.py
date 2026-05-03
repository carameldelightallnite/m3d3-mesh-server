# =========================================================
# M3D3 PLATINUM SERVER — FINAL FIXED (SINGLE WORKER SAFE + AUTO DELETE)
# =========================================================

import os
import uuid
import numpy as np
import trimesh
from flask import Flask, request, jsonify, send_from_directory, after_this_request

app = Flask(__name__)
jobs = {}
OUTPUT = "outputs"
os.makedirs(OUTPUT, exist_ok=True)

# =========================
# PARSERS
# =========================
def v(s):
    return np.fromstring(s.replace("<","").replace(">",""), sep=',')

def r(s):
    q = v(s)
    return np.array([q[3], q[0], q[1], q[2]])

# =========================
# BUILDERS
# =========================
def box(s): return trimesh.creation.box(extents=s)
def cyl(s): return trimesh.creation.cylinder(radius=s[0]/2, height=s[2], sections=32)
def sph(s): return trimesh.creation.uv_sphere(radius=s[0]/2, count=[32,32])
def tor(s): return trimesh.creation.torus(radius=s[0]/2, tube_radius=s[1]/4, sections=48, segments=24)
def con(s): return trimesh.creation.cone(radius=s[0]/2, height=s[2], sections=32)

def pri(s):
    b = s[0]/2; h = s[2]
    verts = np.array([
        [-b,-b,0],[b,-b,0],[0,b,0],
        [-b,-b,h],[b,-b,h],[0,b,h]
    ])
    faces = np.array([
        [0,1,2],[3,5,4],
        [0,1,4],[0,4,3],
        [1,2,5],[1,5,4],
        [2,0,3],[2,3,5]
    ])
    return trimesh.Trimesh(vertices=verts, faces=faces)

# =========================
# CLEAN / OPTIMIZE
# =========================
def clean(m):
    m.remove_duplicate_faces()
    m.remove_degenerate_faces()
    m.remove_unreferenced_vertices()
    m.merge_vertices(digits=4)
    m.fill_holes()
    m.fix_normals()

    if not m.visual.uv:
        uv = m.vertices[:, :2]
        m.visual = trimesh.visual.TextureVisuals(uv=uv)

    return m

# =========================
# SAFE DECIMATION (OPEN3D FALLBACK)
# =========================
def safe_decimate(mesh, ratio):
    try:
        target = max(10, int(len(mesh.faces) * ratio))
        return clean(mesh.simplify_quadratic_decimation(target))
    except:
        return clean(mesh.copy())

# =========================
# LOD GENERATION
# =========================
def lods(m):
    return (
        clean(m.copy()),
        safe_decimate(m, 0.5),
        safe_decimate(m, 0.25),
        safe_decimate(m, 0.1)
    )

# =========================
# PHYSICS HULL
# =========================
def physics(m):
    try:
        h = m.convex_hull
        return safe_decimate(h, 0.25)
    except:
        return clean(m.copy())

# =========================
# ROOT
# =========================
@app.route("/")
def home():
    return "M3D3 SERVER RUNNING"

# =========================
# UPLOAD
# =========================
@app.route("/upload_chunk", methods=["POST"])
def upload():
    data = request.get_json(force=True)
    job = data.get("job")
    chunk = data.get("chunk", [])

    if not job:
        return jsonify({"error": "missing job"}), 400

    if job not in jobs:
        jobs[job] = []

    jobs[job].extend(chunk)
    return jsonify({"ok": True})

# =========================
# FINALIZE
# =========================
@app.route("/finalize", methods=["POST"])
def finalize():
    data = request.get_json(force=True)
    job = data.get("job")
    name = data.get("name", "M
