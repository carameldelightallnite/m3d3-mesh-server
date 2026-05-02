# =========================================================
# M3D3 SERVER — FULL FINAL (RENDER READY + ZIP DELIVERY)
# =========================================================

import os
import zipfile
import math
from flask import Flask, request, send_from_directory, jsonify

app = Flask(__name__)

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# PARSER
# =========================

def parse_m3(data_string):
    p = data_string.strip().split('|')
    if len(p) != 23 or p[0] != "M3FULL":
        raise ValueError("INVALID INPUT")

    def f(x): return float(x)
    def i(x): return int(x)
    def v(x): return [float(n) for n in x.strip('<>').split(',')]

    return {
        "type": p[1].upper(),
        "size": v(p[2]),
    }

# =========================
# ENGINE
# =========================

def build_mesh(p, seg, rings):
    verts, faces = [], []
    sx, sy, sz = p["size"]

    for i in range(rings):
        t = i / rings
        theta = 2 * math.pi * t

        for j in range(seg):
            ang = (j / seg) * 2 * math.pi
            px, py = math.cos(ang)*0.5, math.sin(ang)*0.5

            x = (sx*0.5 + px*sz*0.5) * math.cos(theta)
            y = (sx*0.5 + px*sz*0.5) * math.sin(theta)
            z = py * sy

            verts.append((x,y,z))

    for i in range(rings):
        for j in range(seg):
            a = i*seg + j
            b = i*seg + (j+1)%seg
            c = ((i+1)%rings)*seg + j
            d = ((i+1)%rings)*seg + (j+1)%seg

            faces.append((a,b,c))
            faces.append((b,d,c))

    return verts, faces

# =========================
# EXPORT
# =========================

def write_dae(mesh, path):
    v,f = mesh

    vs = " ".join(f"{x} {y} {z}" for x,y,z in v)

    idx=[]
    for face in f:
        for i in face:
            idx.append(i)

    ps = " ".join(map(str,idx))

    xml=f'''<?xml version="1.0"?>
<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema" version="1.4.1">
<asset><unit name="meter" meter="1"/><up_axis>Z_UP</up_axis></asset>
<library_geometries>
<geometry id="mesh"><mesh>
<source id="p"><float_array id="pa" count="{len(v)*3}">{vs}</float_array></source>
<vertices id="v"><input semantic="POSITION" source="#p"/></vertices>
<triangles count="{len(f)}">
<input semantic="VERTEX" source="#v" offset="0"/>
<p>{ps}</p>
</triangles>
</mesh></geometry>
</library_geometries>
</COLLADA>'''

    with open(path,"w") as f:
        f.write(xml)

# =========================
# GENERATE PACK
# =========================

def generate_pack(p, job_id):

    job_folder = os.path.join(OUTPUT_DIR, job_id)
    os.makedirs(job_folder, exist_ok=True)

    lods = {
        "high": (12,16),
        "med": (6,6),
        "low": (4,4),
        "lowest": (3,3)
    }

    files = []

    for name,(seg,rings) in lods.items():
        mesh = build_mesh(p, seg, rings)
        path = os.path.join(job_folder, f"{name}.dae")
        write_dae(mesh, path)
        files.append(path)

    zip_path = os.path.join(job_folder, "mesh_pack.zip")

    with zipfile.ZipFile(zip_path, 'w') as z:
        for f in files:
            z.write(f, os.path.basename(f))

    return job_id

# =========================
# ROUTES
# =========================

@app.route("/")
def home():
    return "M3D3 SERVER RUNNING"

@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json()
    m3 = data.get("m3")

    try:
        p = parse_m3(m3)
    except:
        return jsonify({"error": "INVALID M3 STRING"}), 400

    job_id = str(len(os.listdir(OUTPUT_DIR)))

    generate_pack(p, job_id)

    base_url = request.host_url.rstrip("/")

    url = f"{base_url}/download/{job_id}/mesh_pack.zip"

    return jsonify({"url": url})

@app.route("/download/<job>/<filename>")
def download(job, filename):
    return send_from_directory(
        os.path.join(OUTPUT_DIR, job),
        filename,
        as_attachment=True
    )

# =========================
# RUN (RENDER READY)
# =========================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
