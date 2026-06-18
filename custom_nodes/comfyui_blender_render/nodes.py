import os, re, uuid, subprocess, tempfile, textwrap, math
import torch, numpy as np
from PIL import Image
from comfy.utils import ProgressBar


_BLENDER_COMMON = """
import bpy, math, mathutils
from mathutils import Vector, Matrix

def get_center(meshes):
    pts = []
    for obj in meshes:
        for c in obj.bound_box:
            pts.append(obj.matrix_world @ Vector(c))
    return sum(pts, Vector()) / len(pts)

def set_plain_material(meshes, color=(0.45, 0.45, 0.40, 1.0)):
    mat = bpy.data.materials.new("Plain")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = color
    bsdf.inputs["Roughness"].default_value  = 0.75
    bsdf.inputs["Metallic"].default_value   = 0.1
    for obj in meshes:
        obj.data.materials.clear()
        obj.data.materials.append(mat)

def rotate_around_center(meshes, center, R4x4):
    T  = Matrix.Translation(center)
    Ti = Matrix.Translation(-center)
    xfm = T @ R4x4 @ Ti
    for obj in meshes:
        obj.matrix_world = xfm @ obj.matrix_world

def load_model(path):
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    p = path.lower()
    if p.endswith((".glb", ".gltf")): bpy.ops.import_scene.gltf(filepath=path)
    elif p.endswith(".fbx"):           bpy.ops.import_scene.fbx(filepath=path)
    else: raise Exception("Format non supporté: " + path)
    return [o for o in bpy.context.scene.objects if o.type == "MESH"]

def add_camera(target, distance, azimuth_deg, elevation_deg):
    cam_data = bpy.data.cameras.new("Cam")
    cam      = bpy.data.objects.new("Cam", cam_data)
    bpy.context.collection.objects.link(cam)
    az = math.radians(azimuth_deg)
    el = math.radians(elevation_deg)
    cam.location = (
        target.x + distance * math.cos(el) * math.cos(az),
        target.y + distance * math.cos(el) * math.sin(az),
        target.z + distance * math.sin(el),
    )
    d = (target - cam.location).normalized()
    cam.rotation_euler = d.to_track_quat("-Z","Y").to_euler()
    return cam

def add_lights(az_deg=0.0):
    # Les lumières tournent avec az_deg pour rester fixes PAR RAPPORT A LA CAMERA :
    # sinon, changer camera_azimuth ne change pas la géométrie (gauge libre) mais
    # change l'éclairage perçu, ce qui donne l'illusion d'un tank "différent".
    rad = math.radians(az_deg)
    ca, sa = math.cos(rad), math.sin(rad)
    def rot(x, y):
        return (x * ca - y * sa, x * sa + y * ca)
    for name, loc, pwr in [
        ("Key",   (0, -6, 8),  700),
        ("Left",  (-6, 4, 5),  400),
        ("Right", ( 6, 4, 5),  400),
    ]:
        x, y = rot(loc[0], loc[1])
        ld = bpy.data.lights.new(name, "AREA")
        lo = bpy.data.objects.new(name, ld)
        bpy.context.collection.objects.link(lo)
        lo.location = (x, y, loc[2]); ld.energy = pwr; ld.size = 10

def do_render(output_path, resolution, samples, transparent):
    scene = bpy.context.scene
    scene.render.engine                     = "CYCLES"
    scene.cycles.samples                    = samples
    scene.render.resolution_x               = resolution
    scene.render.resolution_y               = resolution
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode  = "RGBA"
    scene.render.film_transparent           = transparent
    scene.view_settings.exposure            = 1.0
    scene.render.filepath                   = output_path
    bpy.ops.render.render(write_still=True)
    print("Saved:", output_path)
"""


def _run_blender(blender_path, script_body, progress):
    script = textwrap.dedent(_BLENDER_COMMON + "\n" + script_body)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(script); script_path = f.name
    cmd = [blender_path, "-b", "--python", script_path]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, bufsize=1)
    logs, last_p = [], 10
    for line in process.stdout:
        logs.append(line)
        m = re.search(r"Sample\s+(\d+)/(\d+)", line)
        if m:
            val = 20 + int((int(m.group(1)) / int(m.group(2))) * 70)
            if val > last_p: progress.update_absolute(val); last_p = val
        elif "Fra:" in line or "Rendering" in line:
            progress.update_absolute(max(last_p, 20))
    process.wait(); os.remove(script_path)
    if process.returncode != 0:
        raise RuntimeError("Blender failed:\n" + "".join(logs))

def _load_image(path):
    img = Image.open(path).convert("RGBA")
    return torch.from_numpy(np.array(img).astype(np.float32) / 255.0)[None, ...]

def _output_path():
    out = os.path.join(tempfile.gettempdir(), "blender_renders")
    os.makedirs(out, exist_ok=True)
    return os.path.join(out, f"tank_{uuid.uuid4().hex}.png")

def _to_np(v):
    if isinstance(v, torch.Tensor): return v.cpu().float().numpy()
    return np.array(v, dtype=np.float32)


# ── Conversion road 2D → monde Blender (formule exacte, sans dé-rolling) ─────
#
# Pourquoi PAS de dé-rolling ici :
#   Un dé-rolling R_CCW(φ) déplace TOUJOURS la direction de φ degrés,
#   quelle que soit la direction de la route β.
#   Or les mesures montrent que pour une route droite (β≈0), l'offset est 0.
#   La correction doit être nulle pour β≈0 et croître avec sin(β).
#   C'est exactement ce que fait RoadGravityOffsetEstimator : φ × sin(β) × K.
#   Le dé-rolling géométrique ne respecte pas cette contrainte → il casse les routes droites.
#
# Correction du roll : utiliser road_direction_offset_deg branché sur
#   RoadGravityOffsetEstimator.offset_deg avec K ≈ 0.65.

def _road_vec_to_world(road, az, el):
    """
    Convertit road_vector_2d (rx, ry) en direction monde Blender horizontale.

    Formule exacte pour route plate :
      Cherche fw = (cos θ, sin θ, 0) tel que sa projection sur la caméra
      Blender (az, el) donne (nrx, nry) = (−rx, −ry).

        img_x = −cos(θ)·sin(az) + sin(θ)·cos(az)
        img_y =  sin(el)·(cos(θ)·cos(az) + sin(θ)·sin(az))

      Solution : θ = atan2(nrx·sin(el)·cos(az) + nry·sin(az),
                            nry·cos(az) − nrx·sin(el)·sin(az))

    Route droite devant (rx≈0, ry≈1) → offset ≈ 0 ✓
    """
    nrx = float(-road[0])
    nry = float(-road[1])
    sin_az, cos_az = np.sin(az), np.cos(az)
    sin_el         = np.sin(el)

    num = nrx * sin_el * cos_az + nry * sin_az
    den = nry * cos_az - nrx * sin_el * sin_az

    if abs(num) < 1e-9 and abs(den) < 1e-9:
        return np.array([-cos_az, -sin_az, 0.0], dtype=np.float64)

    theta = np.arctan2(num, den)
    return np.array([np.cos(theta), np.sin(theta), 0.0], dtype=np.float64)


# ═══════════════════════════════════════════════════════════════════════════════
# ① BlenderTankRender
# ═══════════════════════════════════════════════════════════════════════════════

class BlenderTankRender:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_3d_path":          ("STRING",  {"default": r"C:\Users\aurel\Downloads\t-14_armata.glb"}),
                "tank_rot_z":             ("FLOAT",   {"default": 0.0,  "min": -360.0, "max": 360.0, "step": 1.0}),
                "camera_azimuth":         ("FLOAT",   {"default": 45.0, "min": -360.0, "max": 360.0, "step": 1.0}),
                "camera_elevation":       ("FLOAT",   {"default": 15.0, "min": -89.0,  "max": 89.0,  "step": 1.0}),
                "camera_distance":        ("FLOAT",   {"default": 9.0,  "min": 0.1,    "max": 100.0, "step": 0.1}),
                "resolution":             ("INT",     {"default": 1024, "min": 128,    "max": 4096,  "step": 64}),
                "samples":                ("INT",     {"default": 128,  "min": 1,      "max": 1024,  "step": 1}),
                "transparent_background": ("BOOLEAN", {"default": True}),
                "plain_material":         ("BOOLEAN", {"default": True}),
                "blender_path":           ("STRING",  {"default": r"C:\Program Files\Blender Foundation\Blender 5.1\blender.exe"}),
            },
            "optional": {"orientation": ("STRING", {"forceInput": True})},
        }

    RETURN_TYPES = ("IMAGE",); RETURN_NAMES = ("image",)
    FUNCTION = "render"; CATEGORY = "Blender"

    def render(self, model_3d_path, tank_rot_z, camera_azimuth, camera_elevation,
               camera_distance, resolution, samples, transparent_background,
               plain_material, blender_path, orientation=""):
        progress = ProgressBar(100); out_path = _output_path(); progress.update_absolute(5)

        if orientation and orientation.strip():
            rb = f"""
import json as _j; _d=_j.loads({repr(orientation)}); _fwd=_d['forward']
_fx,_fy=_fwd[0],_fwd[1]; _az=math.radians({float(camera_azimuth)})
_wr_x=-_fx*math.sin(_az)-_fy*math.cos(_az)
_wr_y= _fx*math.cos(_az)-_fy*math.sin(_az)
_yaw=math.atan2(_wr_x,_wr_y)+math.radians({float(tank_rot_z)})
rotate_around_center(meshes,center,mathutils.Euler((0.,0.,_yaw),'XYZ').to_matrix().to_4x4())
"""
        else:
            rb = f"rotate_around_center(meshes,center,mathutils.Euler((0.,0.,math.radians({float(tank_rot_z)})),'XYZ').to_matrix().to_4x4())\n"

        script = f"""
MODEL_PATH={repr(model_3d_path)}; OUT_PATH={repr(out_path)}
meshes=load_model(MODEL_PATH)
if {plain_material}: set_plain_material(meshes)
center=get_center(meshes)
{rb}
center=get_center(meshes)
cam=add_camera(center,{float(camera_distance)},{float(camera_azimuth)},{float(camera_elevation)})
bpy.context.scene.camera=cam; add_lights({float(camera_azimuth)})
do_render(OUT_PATH,{int(resolution)},{int(samples)},{bool(transparent_background)})
"""
        _run_blender(blender_path, script, progress)
        progress.update_absolute(95); img=_load_image(out_path); progress.update_absolute(100)
        return (img,)


# ═══════════════════════════════════════════════════════════════════════════════
# ② BlenderTankEuler  – DEBUG
# ═══════════════════════════════════════════════════════════════════════════════

class BlenderTankEuler:

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model_3d_path":   ("STRING",  {"default": r"C:\Users\aurel\Downloads\t-14_armata.glb"}),
            "euler_z": ("FLOAT",{"default":0.0,"min":-360.,"max":360.,"step":1.,"tooltip":"Yaw (0=face +Y)"}),
            "euler_x": ("FLOAT",{"default":0.0,"min":-180.,"max":180.,"step":1.}),
            "euler_y": ("FLOAT",{"default":0.0,"min":-180.,"max":180.,"step":1.}),
            "camera_azimuth":  ("FLOAT",{"default":45.0,"min":-360.,"max":360.,"step":1.}),
            "camera_elevation":("FLOAT",{"default":15.0,"min":-89., "max":89., "step":1.}),
            "camera_distance": ("FLOAT",{"default":9.0, "min":0.1,  "max":100.,"step":0.1}),
            "resolution":      ("INT",  {"default":512, "min":128,  "max":4096,"step":64}),
            "samples":         ("INT",  {"default":32,  "min":1,    "max":1024,"step":1}),
            "transparent_background":("BOOLEAN",{"default":True}),
            "plain_material":  ("BOOLEAN",{"default":True}),
            "blender_path":    ("STRING",{"default":r"C:\Program Files\Blender Foundation\Blender 5.1\blender.exe"}),
        }}
    RETURN_TYPES=("IMAGE",); RETURN_NAMES=("image",); FUNCTION="render"; CATEGORY="Blender/Debug"

    def render(self, model_3d_path, euler_z, euler_x, euler_y,
               camera_azimuth, camera_elevation, camera_distance,
               resolution, samples, transparent_background, plain_material, blender_path):
        progress=ProgressBar(100); out_path=_output_path(); progress.update_absolute(5)
        script=f"""
MODEL_PATH={repr(model_3d_path)}; OUT_PATH={repr(out_path)}
meshes=load_model(MODEL_PATH)
if {plain_material}: set_plain_material(meshes)
center=get_center(meshes)
rotate_around_center(meshes,center,mathutils.Euler((math.radians({float(euler_x)}),math.radians({float(euler_y)}),math.radians({float(euler_z)})),'XYZ').to_matrix().to_4x4())
center=get_center(meshes)
cam=add_camera(center,{float(camera_distance)},{float(camera_azimuth)},{float(camera_elevation)})
bpy.context.scene.camera=cam; add_lights({float(camera_azimuth)})
do_render(OUT_PATH,{int(resolution)},{int(samples)},{bool(transparent_background)})
"""
        _run_blender(blender_path, script, progress)
        progress.update_absolute(95); img=_load_image(out_path); progress.update_absolute(100)
        return (img,)


# ═══════════════════════════════════════════════════════════════════════════════
# ③ BlenderTankVectors  – DEBUG
# ═══════════════════════════════════════════════════════════════════════════════

class BlenderTankVectors:

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model_3d_path": ("STRING",{"default":r"C:\Users\aurel\Downloads\t-14_armata.glb"}),
            "forward_x":("FLOAT",{"default":0.0,"min":-1.,"max":1.,"step":0.01}),
            "forward_y":("FLOAT",{"default":1.0,"min":-1.,"max":1.,"step":0.01}),
            "forward_z":("FLOAT",{"default":0.0,"min":-1.,"max":1.,"step":0.01}),
            "normal_x": ("FLOAT",{"default":0.0,"min":-1.,"max":1.,"step":0.01}),
            "normal_y": ("FLOAT",{"default":0.0,"min":-1.,"max":1.,"step":0.01}),
            "normal_z": ("FLOAT",{"default":1.0,"min":-1.,"max":1.,"step":0.01}),
            "extra_yaw_deg":  ("FLOAT",{"default":0.0,"min":-360.,"max":360.,"step":1.}),
            "camera_azimuth": ("FLOAT",{"default":45.0,"min":-360.,"max":360.,"step":1.}),
            "camera_elevation":("FLOAT",{"default":15.0,"min":-89.,"max":89.,"step":1.}),
            "camera_distance":("FLOAT",{"default":9.0,"min":0.1,"max":100.,"step":0.1}),
            "resolution":("INT",{"default":512,"min":128,"max":4096,"step":64}),
            "samples":("INT",{"default":32,"min":1,"max":1024,"step":1}),
            "transparent_background":("BOOLEAN",{"default":True}),
            "plain_material":("BOOLEAN",{"default":True}),
            "blender_path":("STRING",{"default":r"C:\Program Files\Blender Foundation\Blender 5.1\blender.exe"}),
        }}
    RETURN_TYPES=("IMAGE",); RETURN_NAMES=("image",); FUNCTION="render"; CATEGORY="Blender/Debug"

    def render(self, model_3d_path, forward_x, forward_y, forward_z,
               normal_x, normal_y, normal_z, extra_yaw_deg,
               camera_azimuth, camera_elevation, camera_distance,
               resolution, samples, transparent_background, plain_material, blender_path):
        progress=ProgressBar(100); out_path=_output_path(); progress.update_absolute(5)
        script=f"""
MODEL_PATH={repr(model_3d_path)}; OUT_PATH={repr(out_path)}
meshes=load_model(MODEL_PATH)
if {plain_material}: set_plain_material(meshes)
center=get_center(meshes)
_fwd_w=Vector(({float(forward_x)},{float(forward_y)},{float(forward_z)}))
_nrm_w=Vector(({float(normal_x)},{float(normal_y)},{float(normal_z)}))
_nrm_w=_nrm_w.normalized()
if _nrm_w.length<1e-9: _nrm_w=Vector((0,0,1))
_fwd_w=(_fwd_w-_fwd_w.dot(_nrm_w)*_nrm_w).normalized()
if _fwd_w.length<1e-9: _fwd_w=Vector((0,1,0))
_sid_w=_fwd_w.cross(_nrm_w).normalized()
_fwd_w=_nrm_w.cross(_sid_w).normalized()
_R=Matrix(((_sid_w.x,_fwd_w.x,_nrm_w.x),(_sid_w.y,_fwd_w.y,_nrm_w.y),(_sid_w.z,_fwd_w.z,_nrm_w.z)))
_R_extra=Matrix.Rotation(math.radians({float(extra_yaw_deg)}),3,_nrm_w)
rotate_around_center(meshes,center,(_R_extra@_R).to_4x4())
center=get_center(meshes)
cam=add_camera(center,{float(camera_distance)},{float(camera_azimuth)},{float(camera_elevation)})
bpy.context.scene.camera=cam; add_lights({float(camera_azimuth)})
do_render(OUT_PATH,{int(resolution)},{int(samples)},{bool(transparent_background)})
"""
        _run_blender(blender_path, script, progress)
        progress.update_absolute(95); img=_load_image(out_path); progress.update_absolute(100)
        return (img,)


# ═══════════════════════════════════════════════════════════════════════════════
# ④ BlenderPerspectiveRender
# ═══════════════════════════════════════════════════════════════════════════════

class BlenderPerspectiveRender:
    """
    Correction du roll caméra :
      NE PAS utiliser le dé-rolling géométrique (il casse les routes droites).
      Utiliser RoadGravityOffsetEstimator avec K≈0.65 branché sur road_direction_offset_deg.
      La formule φ×sin(β)×K donne naturellement 0 pour les routes droites.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model_3d_path":   ("STRING", {"default": r"C:\Users\aurel\Downloads\t-14_armata.glb"}),
            "up_vector_3d":    ("GRAVITY_FIELD",),
            "road_vector_2d":  ("GRAVITY_FIELD",),
            "latitude_deg":    ("FLOAT", {"forceInput": True,
                                "tooltip": "Depuis RoadGravitySampler. el = -latitude."}),
            "camera_distance": ("FLOAT", {"default":12.0,"min":0.5,"max":200.,"step":0.5}),
            "camera_azimuth":  ("FLOAT", {"default":45.0,"min":-360.,"max":360.,"step":1.0,
                                "tooltip": "À caler sur la direction réelle de la scène."}),
            "road_direction_offset_deg": ("FLOAT", {
                "default":0.0,"min":-180.,"max":180.,"step":0.5,
                "tooltip": "Brancher ici la sortie offset_deg de RoadGravityOffsetEstimator "
                           "(K=0.65). Donne 0 pour route droite, corrige les routes latérales."}),
            "extra_yaw_deg":   ("FLOAT", {"default":0.0,"min":-360.,"max":360.,"step":1.,
                                "tooltip": "Rotation du modèle sur lui-même. 180=sens inverse."}),
            "resolution":      ("INT",   {"default":1024,"min":128,"max":4096,"step":64}),
            "samples":         ("INT",   {"default":128,"min":1,"max":1024,"step":1}),
            "transparent_background":("BOOLEAN",{"default":True}),
            "plain_material":  ("BOOLEAN",{"default":False}),
            "blender_path":    ("STRING",{"default":r"C:\Program Files\Blender Foundation\Blender 5.1\blender.exe"}),
        }}
    RETURN_TYPES=("IMAGE",); RETURN_NAMES=("image",); FUNCTION="render"; CATEGORY="Blender"

    def render(self, model_3d_path,
               up_vector_3d, road_vector_2d, latitude_deg,
               camera_distance, camera_azimuth,
               road_direction_offset_deg, extra_yaw_deg,
               resolution, samples, transparent_background, plain_material, blender_path):

        up   = _to_np(up_vector_3d).flatten()[:3]
        road = _to_np(road_vector_2d).flatten()[:2]

        camera_elevation = float(np.clip(-float(latitude_deg), 5.0, 85.0))
        az = np.deg2rad(camera_azimuth)
        el = np.deg2rad(camera_elevation)

        R = np.array([
            [-np.sin(az),  np.sin(el)*np.cos(az), -np.cos(el)*np.cos(az)],
            [ np.cos(az),  np.sin(el)*np.sin(az), -np.cos(el)*np.sin(az)],
            [  0.0,       -np.cos(el),              -np.sin(el)          ],
        ], dtype=np.float64)

        normal_w = R @ up.astype(np.float64)
        normal_w /= np.linalg.norm(normal_w) + 1e-9

        # Formule exacte sans dé-rolling
        forward_w = _road_vec_to_world(road, az, el)

        forward_w -= np.dot(forward_w, normal_w) * normal_w
        nf = np.linalg.norm(forward_w)
        forward_w = forward_w/nf if nf > 1e-6 else np.array([-np.cos(az), -np.sin(az), 0.])

        # Correction roll via RoadGravityOffsetEstimator (Rodrigues autour de normal_w)
        if abs(road_direction_offset_deg) > 1e-4:
            theta = np.deg2rad(road_direction_offset_deg)
            forward_w = (forward_w*np.cos(theta)
                         + np.cross(normal_w, forward_w)*np.sin(theta))
            forward_w /= np.linalg.norm(forward_w) + 1e-9

        fx,fy,fz = forward_w
        nx,ny,nz = normal_w

        progress=ProgressBar(100); out_path=_output_path(); progress.update_absolute(5)
        script=f"""
MODEL_PATH={repr(model_3d_path)}; OUT_PATH={repr(out_path)}
meshes=load_model(MODEL_PATH)
if {plain_material}: set_plain_material(meshes)
center=get_center(meshes)
_fwd_w=Vector(({fx:.8f},{fy:.8f},{fz:.8f}))
_nrm_w=Vector(({nx:.8f},{ny:.8f},{nz:.8f}))
_nrm_w=_nrm_w.normalized()
_fwd_w=(_fwd_w-_fwd_w.dot(_nrm_w)*_nrm_w).normalized()
_sid_w=_fwd_w.cross(_nrm_w).normalized()
_fwd_w=_nrm_w.cross(_sid_w).normalized()
_R=Matrix(((_sid_w.x,_fwd_w.x,_nrm_w.x),(_sid_w.y,_fwd_w.y,_nrm_w.y),(_sid_w.z,_fwd_w.z,_nrm_w.z)))
_R_extra=Matrix.Rotation(math.radians({float(extra_yaw_deg)}),3,_nrm_w)
rotate_around_center(meshes,center,(_R_extra@_R).to_4x4())
center=get_center(meshes)
cam=add_camera(center,{float(camera_distance)},{float(camera_azimuth)},{float(camera_elevation)})
bpy.context.scene.camera=cam; add_lights({float(camera_azimuth)})
do_render(OUT_PATH,{int(resolution)},{int(samples)},{bool(transparent_background)})
"""
        _run_blender(blender_path, script, progress)
        progress.update_absolute(95); img=_load_image(out_path); progress.update_absolute(100)
        return (img,)


# ═══════════════════════════════════════════════════════════════════════════════
NODE_CLASS_MAPPINGS = {
    "BlenderViewGenerator":     BlenderTankRender,
    "BlenderTankRender":        BlenderTankRender,
    "BlenderTankEuler":         BlenderTankEuler,
    "BlenderTankVectors":       BlenderTankVectors,
    "BlenderPerspectiveRender": BlenderPerspectiveRender,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "BlenderViewGenerator":     "Blender View Generator",
    "BlenderTankRender":        "Blender Tank Render",
    "BlenderTankEuler":         "Blender Tank Euler (debug) 🔧",
    "BlenderTankVectors":       "Blender Tank Vectors (debug) 🔧",
    "BlenderPerspectiveRender": "Blender Perspective Render",
}
