# =========================================================
# M3D3 PLATINUM SERVER — FINAL STABLE (500 FIX + SAFE PARSERS + AUTO DELETE)
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
# BULLETPROOF PARSERS
# =========================
def v(s):
    try:
        clean_s = str(s).replace("<", "").replace(">", "").strip()
        arr = np.fromstring(clean_s, sep=',')
        if arr.size == 0:
            return np.array([1.0, 1.0, 1.0])
        return arr
    except:
        return np.array([1.0, 1.0, 1.0])

def r(s):
    try:
        q = v(s)
        if len(q) < 4:
            return np.array([1.0, 0.0, 0.0, 0.0])
        return np.array([q[3], q[0], q[1], q[2]])
    except:
        return np.array([1.0, 0.0, 0.0, 0.0])

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
# CLEAN
# =========================
def clean(m):
    try:
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
    except:
        return m

# =========================
# SAFE DECIMATION
# =========================
def safe_decimate(mesh, ratio):
    try:
        target = max(10, int(len(mesh.faces) * ratio))
        return clean(mesh.simplify_quadratic_decimation(target))
    except:
        return clean(mesh.copy())

def lods(m):
    return (
        clean(m.copy()),
        safe_decimate(m, 0.5),
        safe_decimate(m, 0.25),
        safe_decimate(m, 0.1)
    )

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
# FINALIZE (FIXED)
# =========================
@app.route("/finalize", methods=["POST"])
def finalize():
    data = request.get_json(force=True)
    job = data.get("job")
    name = data.get("name", "M3D3_Export").replace(" ", "_")

    if job not in jobs or not jobs[job]:
        return jsonify({"error": "No data found for this job ID"}), 400

    prims = jobs[job]
    meshes = []

    for i, p in enumerate(prims):
        try:
            size = v(p.get("size", "<1,1,1>"))
            pos  = v(p.get("pos", "<0,0,0>"))
            rot  = r(p.get("rot", "<0,0,0,1>"))
            t    = p.get("type", "BOX")

            if t == "CYLINDER": m = cyl(size)
            elif t == "SPHERE": m = sph(size)
            elif t == "TORUS":  m = tor(size)
            elif t == "PRISM":  m = pri(size)
            elif t == "CONE":   m = con(size)
            else:               m = box(size)

            T = trimesh.transformations.quaternion_matrix(rot)
            T[:3,3] = pos
            m.apply_transform(T)

            color = [ (i*45)%255, (i*75)%255, (i*115)%255, 255 ]
            m.visual.face_colors = color

            meshes.append(clean(m))

        except Exception as e:
            print(f"Skipping prim {i}: {e}")

    if not meshes:
        return jsonify({"error": "Mesh generation failed"}), 500

    final = clean(trimesh.util.concatenate(meshes))

    H, M, L, LO = lods(final)
    P = physics(final)

    uid = uuid.uuid4().hex[:8]

    files = {
        "HIGH": f"{name}_HI_{uid}.dae",
        "MEDIUM": f"{name}_MED_{uid}.dae",
        "LOW": f"{name}_LOW_{uid}.dae",
        "LOWEST": f"{name}_LOWEST_{uid}.dae",
        "PHYS": f"{name}_PHYS_{uid}.dae"
    }

    for key, fname in files.items():
        mesh_map = {"HIGH":H, "MEDIUM":M, "LOW":L, "LOWEST":LO, "PHYS":P}
        mesh_map[key].export(os.path.join(OUTPUT, fname))

    del jobs[job]

    return jsonify({
        k: f"https://{request.host}/download/{v}" for k, v in files.items()
    })

# =========================
# DOWNLOAD + AUTO DELETE
# =========================
@app.route("/download/<filename>")
def download(filename):
    file_path = os.path.join(OUTPUT, filename)

    if not os.path.exists(file_path):
        return "File not found", 404

    @after_this_request
    def cleanup(response):
        try:
            os.remove(file_path)
        except Exception as e:
            print(f"Cleanup failed: {e}")
        return response

    return send_from_directory(OUTPUT, filename, as_attachment=True)

# =========================
# RUN (LOCAL SAFE)
# =========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
