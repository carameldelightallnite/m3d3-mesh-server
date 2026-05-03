# =========================================================
# M3D3 MERGE ENGINE — V3 (LOD + PHYSICS + MATERIALS + WELD)
# =========================================================

import os
import uuid
import numpy as np
import trimesh
from flask import Flask, request, jsonify, send_from_directory

app = Flask(__name__)
jobs = {}
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# PARSERS
# =========================

def parse_vec(s):
    return np.fromstring(s.replace("<","").replace(">",""), sep=',')

def parse_rot(s):
    q = parse_vec(s)
    return np.array([q[3], q[0], q[1], q[2]])

# =========================
# BUILDERS
# =========================

def build_box(size):
    return trimesh.creation.box(extents=size)

def build_cylinder(size):
    return trimesh.creation.cylinder(radius=size[0]*0.5, height=size[2], sections=32)

def build_sphere(size):
    return trimesh.creation.uv_sphere(radius=size[0]*0.5, count=[32,32])

def build_torus(size):
    return trimesh.creation.torus(radius=size[0]*0.5, tube_radius=size[1]*0.25, sections=48, segments=24)

def build_prism(size):
    b = size[0]*0.5
    h = size[2]
    v = np.array([
        [-b,-b,0],[ b,-b,0],[0,b,0],
        [-b,-b,h],[ b,-b,h],[0,b,h]
    ])
    f = np.array([
        [0,1,2],[3,5,4],
        [0,1,4],[0,4,3],
        [1,2,5],[1,5,4],
        [2,0,3],[2,3,5]
    ])
    return trimesh.Trimesh(vertices=v, faces=f)

def build_cone(size):
    return trimesh.creation.cone(radius=size[0]*0.5, height=size[2], sections=32)

# =========================
# CLEAN + UV + NORMALS
# =========================

def finalize_mesh(mesh):
    mesh.remove_duplicate_faces()
    mesh.remove_degenerate_faces()
    mesh.remove_unreferenced_vertices()
    mesh.merge_vertices(digits=4)
    mesh.fix_normals()

    if not mesh.visual.uv:
        uv = mesh.vertices[:, :2]
        mesh.visual = trimesh.visual.TextureVisuals(uv=uv)

    return mesh

# =========================
# LOD GENERATION
# =========================

def generate_lods(mesh):
    vcount = len(mesh.faces)

    lod_high = mesh.copy()

    lod_medium = mesh.simplify_quadratic_decimation(int(vcount * 0.5))
    lod_low    = mesh.simplify_quadratic_decimation(int(vcount * 0.25))
    lod_lowest = mesh.simplify_quadratic_decimation(int(vcount * 0.1))

    return (
        finalize_mesh(lod_high),
        finalize_mesh(lod_medium),
        finalize_mesh(lod_low),
        finalize_mesh(lod_lowest)
    )

# =========================
# PHYSICS MODEL (LOW COST)
# =========================

def build_physics(mesh):
    hull = mesh.convex_hull
    hull = hull.simplify_quadratic_decimation(int(len(hull.faces) * 0.25))
    return finalize_mesh(hull)

# =========================
# ROUTES
# =========================

@app.route("/")
def home():
    return "M3D3 SERVER V3 LIVE"

@app.route("/upload_chunk", methods=["POST"])
def upload_chunk():
    data = request.json
    job = data["job"]
    chunk = data["chunk"]

    if job not in jobs:
        jobs[job] = []

    jobs[job].extend(chunk)

    return jsonify({"ok": True})

@app.route("/finalize", methods=["POST"])
def finalize():
    data = request.json
    job = data["job"]
    name = data.get("name", "M3D3")

    prims = jobs.get(job, [])
    meshes = []
    materials = []

    for idx, p in enumerate(prims):
        size = parse_vec(p["size"])
        pos  = parse_vec(p["pos"])
        rot  = parse_rot(p["rot"])
        t = p["type"]

        if t == "BOX":
            m = build_box(size)
        elif t == "CYLINDER":
            m = build_cylinder(size)
        elif t == "SPHERE":
            m = build_sphere(size)
        elif t == "TORUS":
            m = build_torus(size)
        elif t == "PRISM":
            m = build_prism(size)
        elif t == "CONE":
            m = build_cone(size)
        else:
            m = build_box(size)

        T = trimesh.transformations.quaternion_matrix(rot)
        T[:3,3] = pos
        m.apply_transform(T)

        m = finalize_mesh(m)

        # MATERIAL ID PER PRIM
        m.visual.face_colors = np.tile(
            np.array([idx % 255, (idx*3)%255, (idx*7)%255, 255]),
            (len(m.faces),1)
        )

        meshes.append(m)

    # =========================
    # MERGE + WELD
    # =========================

    final = trimesh.util.concatenate(meshes)
    final = finalize_mesh(final)

    # =========================
    # LODs
    # =========================

    high, medium, low, lowest = generate_lods(final)

    # =========================
    # PHYSICS
    # =========================

    physics = build_physics(final)

    uid = uuid.uuid4().hex

    f_high   = f"{name}_HIGH_{uid}.dae"
    f_med    = f"{name}_MED_{uid}.dae"
    f_low    = f"{name}_LOW_{uid}.dae"
    f_lowest = f"{name}_LOWEST_{uid}.dae"
    f_phys   = f"{name}_PHYS_{uid}.dae"

    high.export(os.path.join(OUTPUT_DIR, f_high))
    medium.export(os.path.join(OUTPUT_DIR, f_med))
    low.export(os.path.join(OUTPUT_DIR, f_low))
    lowest.export(os.path.join(OUTPUT_DIR, f_lowest))
    physics.export(os.path.join(OUTPUT_DIR, f_phys))

    del jobs[job]

    return jsonify({
        "HIGH":   f"https://{request.host}/download/{f_high}",
        "MEDIUM": f"https://{request.host}/download/{f_med}",
        "LOW":    f"https://{request.host}/download/{f_low}",
        "LOWEST": f"https://{request.host}/download/{f_lowest}",
        "PHYS":   f"https://{request.host}/download/{f_phys}"
    })

@app.route("/download/<filename>")
def download(filename):
    return send_from_directory(OUTPUT_DIR, filename, as_attachment=True)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
