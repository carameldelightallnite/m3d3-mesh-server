# =========================================================
# M3D3 PLATINUM SERVER
# =========================================================

import os
import uuid
import numpy as np
import trimesh
from flask import Flask, request, jsonify, send_from_directory

app = Flask(__name__)
jobs = {}
OUTPUT = "outputs"
os.makedirs(OUTPUT, exist_ok=True)

def v(s): return np.fromstring(s.replace("<","").replace(">",""), sep=',')
def r(s): q=v(s); return np.array([q[3],q[0],q[1],q[2]])

def box(s): return trimesh.creation.box(extents=s)
def cyl(s): return trimesh.creation.cylinder(radius=s[0]/2,height=s[2],sections=32)
def sph(s): return trimesh.creation.uv_sphere(radius=s[0]/2,count=[32,32])
def tor(s): return trimesh.creation.torus(radius=s[0]/2,tube_radius=s[1]/4)
def con(s): return trimesh.creation.cone(radius=s[0]/2,height=s[2])
def pri(s):
    b=s[0]/2;h=s[2]
    v=np.array([[-b,-b,0],[b,-b,0],[0,b,0],[-b,-b,h],[b,-b,h],[0,b,h]])
    f=np.array([[0,1,2],[3,5,4],[0,1,4],[0,4,3],[1,2,5],[1,5,4],[2,0,3],[2,3,5]])
    return trimesh.Trimesh(vertices=v,faces=f)

def clean(m):
    m.remove_duplicate_faces()
    m.remove_degenerate_faces()
    m.remove_unreferenced_vertices()
    m.merge_vertices(digits=4)
    m.fill_holes()
    m.fix_normals()
    if not m.visual.uv:
        uv=m.vertices[:,:2]
        m.visual=trimesh.visual.TextureVisuals(uv=uv)
    return m

def lods(m):
    f=len(m.faces)
    return (
        clean(m.copy()),
        clean(m.simplify_quadratic_decimation(int(f*0.5))),
        clean(m.simplify_quadratic_decimation(int(f*0.25))),
        clean(m.simplify_quadratic_decimation(int(f*0.1)))
    )

def physics(m):
    h=m.convex_hull
    return clean(h.simplify_quadratic_decimation(int(len(h.faces)*0.25)))

@app.route("/upload_chunk",methods=["POST"])
def up():
    d=request.json
    jobs.setdefault(d["job"],[]).extend(d["chunk"])
    return jsonify(ok=True)

@app.route("/finalize",methods=["POST"])
def fin():
    d=request.json
    job=d["job"]
    name=d["name"]

    meshes=[]

    for i,p in enumerate(jobs[job]):
        s=v(p["size"]);pos=v(p["pos"]);rot=r(p["rot"])
        t=p["type"]

        if t=="BOX": m=box(s)
        elif t=="CYLINDER": m=cyl(s)
        elif t=="SPHERE": m=sph(s)
        elif t=="TORUS": m=tor(s)
        elif t=="PRISM": m=pri(s)
        elif t=="CONE": m=con(s)
        else: m=box(s)

        T=trimesh.transformations.quaternion_matrix(rot)
        T[:3,3]=pos
        m.apply_transform(T)

        col=np.array([i%255,(i*3)%255,(i*7)%255,255])
        m.visual.face_colors=np.tile(col,(len(m.faces),1))

        meshes.append(clean(m))

    final=clean(trimesh.util.concatenate(meshes))

    H,M,L,LO=lods(final)
    P=physics(final)

    uid=uuid.uuid4().hex

    files={
        "HIGH":f"{name}_H_{uid}.dae",
        "MED":f"{name}_M_{uid}.dae",
        "LOW":f"{name}_L_{uid}.dae",
        "LOWEST":f"{name}_LO_{uid}.dae",
        "PHYS":f"{name}_P_{uid}.dae"
    }

    H.export(os.path.join(OUTPUT,files["HIGH"]))
    M.export(os.path.join(OUTPUT,files["MED"]))
    L.export(os.path.join(OUTPUT,files["LOW"]))
    LO.export(os.path.join(OUTPUT,files["LOWEST"]))
    P.export(os.path.join(OUTPUT,files["PHYS"]))

    del jobs[job]

    return jsonify({k:f"https://{request.host}/download/{v}" for k,v in files.items()})

@app.route("/download/<f>")
def dwn(f):
    return send_from_directory(OUTPUT,f,as_attachment=True)

if __name__=="__main__":
    app.run(host="0.0.0.0",port=int(os.environ.get("PORT",5000)))
