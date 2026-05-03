# =========================================================
# M3D3 PLATINUM SERVER — FINAL SL-SAFE BUILD
# Fixes:
# - MAV_BLOCK_MISSING
# - Blank preview
# - LOD node mismatch
# - Bad physics hull density
# - Degenerate physics triangles
# - JSON/URL ghost characters
# =========================================================

import os
import time
import uuid
import traceback
import numpy as np
import trimesh

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

    elif prim_type in ["TORUS", "RING"]:
        mesh = build_torus(size)

    elif prim_type == "PRISM":
        mesh = build_prism(size)

    elif prim_type == "CONE":
        mesh = build_cone(size)

    elif prim_type == "TUBE":
        mesh = build_cylinder(size)

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

        phys = trimesh.Trimesh(
            vertices=verts,
            faces=faces,
            process=False
        )

        return clean_mesh(phys)

    except Exception:
        return clean_mesh(trimesh.creation.box(extents=[1.0, 1.0, 1.0]))


# =========================================================
# SECOND LIFE SAFE COLLADA WRITER
# =========================================================

def float_list(values):
    return " ".join(f"{float(v):.6f}" for v in values)


def int_list(values):
    if isinstance(values, np.ndarray):
        return " ".join(values.astype(str))
    return " ".join(str(int(v)) for v in values)


def write_sl_safe_dae(mesh, path, mesh_name="Object"):
    mesh = clean_mesh(mesh)

    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=int)

    if len(vertices) == 0 or len(faces) == 0:
        raise RuntimeError("Cannot export empty mesh.")

    try:
        normals = np.asarray(mesh.vertex_normals, dtype=float)
    except Exception:
        normals = np.zeros_like(vertices)
        normals[:, 2] = 1.0

    uvs = np.zeros((len(vertices), 2), dtype=float)

    safe_mesh_name = safe_name(mesh_name)

    pos_values = float_list(vertices.reshape(-1))
    normal_values = float_list(normals.reshape(-1))
    uv_values = float_list(uvs.reshape(-1))

    p_values = np.repeat(faces.reshape(-1), 3)
    vcount_values = " ".join(["3"] * len(faces))

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
    <effect id="mat_effect">
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
    <material id="mat" name="mat">
      <instance_effect url="#mat_effect"/>
    </material>
  </library_materials>

  <library_geometries>
    <geometry id="{safe_mesh_name}_geometry" name="{safe_mesh_name}">
      <mesh>
        <source id="{safe_mesh_name}_positions">
          <float_array id="{safe_mesh_name}_positions_array" count="{len(vertices) * 3}">
            {pos_values}
          </float_array>
          <technique_common>
            <accessor source="#{safe_mesh_name}_positions_array" count="{len(vertices)}" stride="3">
              <param name="X" type="float"/>
              <param name="Y" type="float"/>
              <param name="Z" type="float"/>
            </accessor>
          </technique_common>
        </source>

        <source id="{safe_mesh_name}_normals">
          <float_array id="{safe_mesh_name}_normals_array" count="{len(normals) * 3}">
            {normal_values}
          </float_array>
          <technique_common>
            <accessor source="#{safe_mesh_name}_normals_array" count="{len(normals)}" stride="3">
              <param name="X" type="float"/>
              <param name="Y" type="float"/>
              <param name="Z" type="float"/>
            </accessor>
          </technique_common>
        </source>

        <source id="{safe_mesh_name}_uvs">
          <float_array id="{safe_mesh_name}_uvs_array" count="{len(uvs) * 2}">
            {uv_values}
          </float_array>
          <technique_common>
            <accessor source="#{safe_mesh_name}_uvs_array" count="{len(uvs)}" stride="2">
              <param name="S" type="float"/>
              <param name="T" type="float"/>
            </accessor>
          </technique_common>
        </source>

        <vertices id="{safe_mesh_name}_vertices">
          <input semantic="POSITION" source="#{safe_mesh_name}_positions"/>
        </vertices>

        <polylist material="mat" count="{len(faces)}">
          <input semantic="VERTEX" source="#{safe_mesh_name}_vertices" offset="0"/>
          <input semantic="NORMAL" source="#{safe_mesh_name}_normals" offset="1"/>
          <input semantic="TEXCOORD" source="#{safe_mesh_name}_uvs" offset="2" set="0"/>
          <vcount>{vcount_values}</vcount>
          <p>{int_list(p_values)}</p>
        </polylist>
      </mesh>
    </geometry>
  </library_geometries>

  <library_visual_scenes>
    <visual_scene id="Scene" name="Scene">
      <node id="{safe_mesh_name}_node" name="{safe_mesh_name}" type="NODE">
        <instance_geometry url="#{safe_mesh_name}_geometry">
          <bind_material>
            <technique_common>
              <instance_material symbol="mat" target="#mat">
                <bind_vertex_input semantic="TEXCOORD" input_semantic="TEXCOORD" input_set="0"/>
              </instance_material>
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

    with open(path, "w", encoding="utf-8") as f:
        f.write(dae)

    if not os.path.exists(path):
        raise RuntimeError("DAE export failed.")

    if os.path.getsize(path) <= 0:
        raise RuntimeError("DAE export created empty file.")

    return path


def export_mesh(mesh, filename, mesh_name):
    path = os.path.join(OUTPUT_DIR, filename)
    write_sl_safe_dae(mesh, path, mesh_name)
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
        "server": "M3D3 Platinum SL-Safe Exporter",
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
