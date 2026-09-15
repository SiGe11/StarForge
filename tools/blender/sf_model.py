"""sf_model.py -- Blender authoring layer for StarForge model packs.

Run with the `bpy` PyPI wheel (`pip install bpy`, needs CPython 3.11); no
Blender install and no GUI are required. Nothing here is compiled into the
game: this is an offline authoring tool whose only output is
`assets/models.bin`, which the game treats as optional.

Two decisions worth knowing before editing this file.

**Everything is authored in game space, Y-up.** bmesh is used purely as a
geometry kernel and never as a scene, so Blender's Z-up convention never
enters. That is deliberate: converting Z-up to Y-up as (x, z, y) is a
reflection, and a reflection flips triangle winding, which the renderer's
`frontFacingWinding: CCW` + `cullMode: back` turns into an invisible model
rather than an obviously broken one. Authoring in the destination space means
there is no conversion to get wrong, and the dimensions can be lifted straight
from the primitives they replace in MeshGen.cpp.

**Materials live on faces, not parts.** Each bmesh carries an int layer naming
a row of `MATERIALS`, so a face can be inset and recoloured without splitting
it into its own object. The pack stores one palette per mesh and one index per
vertex, which is what keeps the file small.
"""

import bmesh
import math
import struct
from mathutils import Matrix, Vector

# ---------------------------------------------------------------- materials
#
# Names and values track the palette at the top of MeshGen.cpp so the
# Blender-authored models sit beside the procedural fallback without a visible
# shift in albedo. Fields are the MeshVertex tail:
#   (r, g, b, roughness, metallic, team, emissive, ao)
# `team` is the blend weight toward the faction colour; `emissive` is a gain,
# so anything above 0 glows and, per the object shader, anything at or above
# 0.5 also suppresses the detail texture.
MATERIALS = {
    'armor':      (0.70, 0.69, 0.64, 0.55, 0.30, 0.0, 0.0, 1.0),
    'armor_dark': (0.46, 0.46, 0.44, 0.60, 0.25, 0.0, 0.0, 1.0),
    'dark':       (0.15, 0.16, 0.18, 0.88, 0.05, 0.0, 0.0, 1.0),
    'steel':      (0.55, 0.57, 0.61, 0.45, 0.75, 0.0, 0.0, 1.0),
    'rubber':     (0.11, 0.11, 0.12, 0.95, 0.00, 0.0, 0.0, 0.9),
    'team':       (0.70, 0.69, 0.64, 0.50, 0.20, 1.0, 0.0, 1.0),
    'team_dark':  (0.48, 0.47, 0.44, 0.58, 0.25, 1.0, 0.0, 1.0),
    'glow_warm':  (1.00, 0.82, 0.45, 0.25, 0.00, 0.0, 2.6, 1.0),
    'glow_dim':   (1.00, 0.80, 0.42, 0.30, 0.00, 0.0, 1.4, 1.0),
    'glow_cyan':  (0.30, 0.82, 1.00, 0.18, 0.00, 0.0, 2.4, 1.0),
    'glow_amber': (0.92, 0.38, 0.14, 0.30, 0.00, 0.0, 2.0, 1.0),
    'crystal':    (0.32, 0.78, 0.95, 0.16, 0.15, 0.0, 0.55, 1.0),
    'rock':       (0.34, 0.32, 0.30, 0.92, 0.00, 0.0, 0.0, 1.0),
    # Deep recesses. Baking the occlusion into ao costs nothing at runtime and
    # gives panel gaps and wheel wells a shadow the lighting alone will not
    # produce at this scale.
    'shadowed':   (0.30, 0.30, 0.30, 0.85, 0.10, 0.0, 0.0, 0.45),
}
MAT_NAMES = list(MATERIALS.keys())
MAT_INDEX = {n: i for i, n in enumerate(MAT_NAMES)}

MAT_LAYER = 'sfmat'

# Intended extents per part, keyed by id(bmesh), filled in by the primitives
# and checked at export. See check_bounds() for why this exists.
_EXPECT = {}


def expect_bounds(bm, lo, hi, label):
    _EXPECT[id(bm)] = (tuple(lo), tuple(hi), label)


def check_bounds(bm, slack=0.35):
    """Verify a part still occupies roughly the volume it was asked for.

    A modelling operation applied to the wrong set of faces does not produce
    invalid geometry -- it produces perfectly wound triangles in the wrong
    place. Winding audits pass, triangle counts look sane, and the only
    symptom is a spike sticking out of the model. Aspect-ratio heuristics
    cannot catch it either: bevelling a 5-metre plate by 2 cm legitimately
    produces triangles with aspect ratios in the hundreds.

    What does catch it is the one thing only the authoring side knows: how big
    the part was supposed to be. `slack` allows for bevels and insets pushing
    slightly outside the nominal box; a tear overshoots by multiples.

    Returns a complaint string, or None.
    """
    rec = _EXPECT.get(id(bm))
    if rec is None or not bm.verts:
        return None
    lo, hi, label = rec
    xs = [v.co.x for v in bm.verts]
    ys = [v.co.y for v in bm.verts]
    zs = [v.co.z for v in bm.verts]
    got_lo, got_hi = (min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs))
    for i, ax in enumerate('xyz'):
        size = max(hi[i] - lo[i], 1e-4)
        pad = size * slack + 0.05
        if got_lo[i] < lo[i] - pad or got_hi[i] > hi[i] + pad:
            return ('%s: %s extent %.3f..%.3f, expected %.3f..%.3f'
                    % (label, ax, got_lo[i], got_hi[i], lo[i], hi[i]))
    return None


# ---------------------------------------------------------------- helpers
def new_bm():
    """A bmesh with the material layer already present.

    Creating a custom-data layer reallocates bmesh element storage, which
    invalidates every BMVert/BMEdge/BMFace reference held in Python at the
    time. Adding the layer lazily therefore explodes with "BMesh data of type
    BMFace has been removed" the first time a caller does
    set_mat(bm, bm.faces[:], ...). Every primitive here starts from this
    function so the layer exists before any element is ever referenced.
    """
    bm = bmesh.new()
    bm.faces.layers.int.new(MAT_LAYER)
    return bm


def _layer(bm):
    lay = bm.faces.layers.int.get(MAT_LAYER)
    if lay is None:
        lay = bm.faces.layers.int.new(MAT_LAYER)
    return lay


def set_mat(bm, faces, mat):
    """Tag `faces` with a material name."""
    lay = _layer(bm)
    idx = MAT_INDEX[mat]
    for f in faces:
        f[lay] = idx


def xform(bm, matrix, verts=None):
    partial = verts is not None
    bmesh.ops.transform(bm, matrix=matrix, verts=verts if partial else bm.verts[:])
    if partial:
        # A partial transform deforms the part rather than moving it, so the
        # recorded box no longer describes anything; drop it instead of
        # reporting a bound the caller never asked for.
        _EXPECT.pop(id(bm), None)
        return
    rec = _EXPECT.get(id(bm))
    if rec is None:
        return
    # Carry the expectation through the transform, or a part that is built
    # axis-aligned and then rotated into place trips its own bounds check.
    # Taking the AABB of the transformed corners is a superset for a rotation,
    # which only makes the check more forgiving -- never less.
    lo, hi, label = rec
    pts = []
    for sx in (lo[0], hi[0]):
        for sy in (lo[1], hi[1]):
            for sz in (lo[2], hi[2]):
                pts.append(matrix @ Vector((sx, sy, sz)))
    _EXPECT[id(bm)] = (
        (min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)),
        (max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)),
        label)


def bevel(bm, offset, segments=2, profile=0.5, geom=None, angle_limit=None):
    """Bevel edges. This is the single biggest readability win over the
    primitives it replaces: a perfectly sharp edge catches no specular
    highlight, so a box-built hull reads as a flat silhouette, while a 2-3 cm
    bevel gives every edge a bright line that separates one plate from the
    next at RTS camera distance.

    `angle_limit` (radians) restricts the bevel to edges sharper than that,
    which keeps it off the seams inside a cylinder wall.
    """
    if geom is None:
        edges = bm.edges[:]
    else:
        edges = [e for e in geom if isinstance(e, bmesh.types.BMEdge)]
    if angle_limit is not None:
        sel = []
        for e in edges:
            if len(e.link_faces) != 2:
                continue
            if e.calc_face_angle(0.0) >= angle_limit:
                sel.append(e)
        edges = sel
    if not edges:
        return
    # A bevel wider than half the shortest edge it touches self-intersects and
    # produces inverted slivers, so clamp to the geometry actually present.
    shortest = min((e.calc_length() for e in edges), default=offset * 4.0)
    off = min(offset, shortest * 0.45)
    if off <= 1e-5:
        return
    bmesh.ops.bevel(bm, geom=edges, offset=off, segments=segments,
                    profile=profile, affect='EDGES', clamp_overlap=True,
                    material=-1, loop_slide=True)


def faces_where(bm, pred):
    return [f for f in bm.faces if pred(f)]


def face_facing(bm, axis, sign, max_deg=15.0):
    """Faces whose normal points along +/- an axis, within `max_deg`.

    The threshold is an angle, and a tight one, for a reason worth spelling
    out. These meshes are bevelled before anything is selected on them, and an
    n-segment bevel replaces each sharp edge with faces at evenly spaced
    angles -- 30 and 60 degrees for the two-segment bevel used throughout. A
    loose threshold therefore selects not just the plate but the ring of bevel
    strips wrapped around it, and running inset_region over a ring of corner
    strips does not recess a panel: it tears the part open. That failure
    inflated the trooper's torso from 0.76 units wide to 1.67 and put two
    spikes through its hips. Anything below 30 degrees excludes bevel faces
    with margin, while still admitting the mildly sloped sides of a wedge.
    """
    a = {'x': 0, 'y': 1, 'z': 2}[axis]
    lim = math.cos(math.radians(max_deg))
    out = []
    for f in bm.faces:
        n = f.normal
        if n.length < 1e-8:
            continue
        if (n[a] * sign) >= lim:
            out.append(f)
    return out


def inset(bm, faces, thickness, depth=0.0, mat=None):
    """Inset a face and optionally push it in, the standard way to read a
    panel, hatch or vent as recessed rather than painted on."""
    if not faces:
        return []
    res = bmesh.ops.inset_region(bm, faces=faces, thickness=thickness,
                                 depth=depth, use_even_offset=True,
                                 use_interpolate=True, use_boundary=True)
    if mat:
        set_mat(bm, faces, mat)
    return res.get('faces', [])


# ---------------------------------------------------------------- primitives
def box(center, half, mat='armor', bevel_w=0.03, bevel_seg=None):
    """Axis-aligned bevelled box, the workhorse. Same call shape as
    MeshBuilder::box so dimensions port across unchanged.

    `bevel_seg` defaults by size. A multi-segment bevel rounds an edge, which
    is worth paying for on a hull plate the camera sees end-on; on a 5 cm
    greeble the extra ring is a dozen triangles the player cannot resolve, and
    those greebles are where the triangle budget actually goes. One segment
    still produces a chamfer face, which is all that is needed to catch a
    specular highlight and separate two plates.
    """
    if bevel_seg is None:
        bevel_seg = 1 if min(half) < 0.13 else 2
    bm = new_bm()
    bmesh.ops.create_cube(bm, size=1.0)
    xform(bm, Matrix.Diagonal(Vector((half[0] * 2, half[1] * 2, half[2] * 2, 1.0))))
    xform(bm, Matrix.Translation(Vector(center)))
    set_mat(bm, bm.faces[:], mat)
    if bevel_w > 0:
        bevel(bm, bevel_w, bevel_seg)
    expect_bounds(bm,
                  [center[i] - half[i] for i in range(3)],
                  [center[i] + half[i] for i in range(3)], 'box%s' % (center,))
    return bm


def wedge(center, half, top_shrink_z, top_shrink_x, mat='armor',
          bevel_w=0.03, bevel_seg=2):
    """Box with an inset top face -- sloped armour. Mirrors MeshBuilder::wedge."""
    cx, cy, cz = center
    hx, hy, hz = half
    tx, tz = hx * (1.0 - top_shrink_x), hz * (1.0 - top_shrink_z)
    bm = new_bm()
    bmesh.ops.create_cube(bm, size=1.0)
    for v in bm.verts:
        sx = 1.0 if v.co.x > 0 else -1.0
        sy = 1.0 if v.co.y > 0 else -1.0
        sz = 1.0 if v.co.z > 0 else -1.0
        ax, az = (tx, tz) if sy > 0 else (hx, hz)
        v.co = Vector((cx + sx * ax, cy + sy * hy, cz + sz * az))
    bm.normal_update()
    set_mat(bm, bm.faces[:], mat)
    if bevel_w > 0:
        bevel(bm, bevel_w, bevel_seg)
    expect_bounds(bm,
                  [center[i] - half[i] for i in range(3)],
                  [center[i] + half[i] for i in range(3)], 'wedge%s' % (center,))
    return bm


def cyl(base, r0, r1, h, seg, mat='armor', axis='y', caps=True,
        bevel_w=0.02, bevel_seg=1):
    """Cylinder/cone with its base at `base`, growing along `axis`.
    bmesh builds cones centred on Z, so this recentres and reorients."""
    bm = new_bm()
    bmesh.ops.create_cone(bm, cap_ends=caps, cap_tris=False, segments=seg,
                          radius1=max(r0, 1e-5), radius2=max(r1, 1e-5), depth=h)
    # create_cone spans -h/2..+h/2 on Z; move the base to the origin.
    xform(bm, Matrix.Translation(Vector((0, 0, h * 0.5))))
    if axis == 'y':
        xform(bm, Matrix.Rotation(-math.pi / 2, 4, 'X'))
    elif axis == 'x':
        xform(bm, Matrix.Rotation(math.pi / 2, 4, 'Y'))
    # axis == 'z' needs no rotation
    xform(bm, Matrix.Translation(Vector(base)))
    set_mat(bm, bm.faces[:], mat)
    if bevel_w > 0 and caps:
        # Only the rim edges, never the seams running along the wall, or the
        # cylinder loses its round silhouette.
        bevel(bm, bevel_w, bevel_seg, angle_limit=math.radians(35))
    return bm


def cone_radius_at(r0, r1, h, y):
    """Radius of a cyl() at height `y` above its base.

    Placing a band or collar on a tapered drum means beating the taper at that
    exact height, and getting it wrong by a few centimetres buries the band
    inside the hull -- where it costs triangles and renders nothing, with no
    symptom beyond the detail silently not being there. Deriving the radius
    removes the arithmetic from the author.
    """
    t = 0.0 if h <= 1e-6 else max(0.0, min(1.0, y / h))
    return r0 + (r1 - r0) * t


def tube(base, r_out, r_in, h, seg, mat='steel', axis='y'):
    """Open-ended pipe with wall thickness -- exhaust stacks, barrel shrouds."""
    bm = new_bm()
    outer = bmesh.ops.create_cone(bm, cap_ends=False, cap_tris=False,
                                  segments=seg, radius1=r_out, radius2=r_out,
                                  depth=h)['verts']
    inner = bmesh.ops.create_cone(bm, cap_ends=False, cap_tris=False,
                                  segments=seg, radius1=r_in, radius2=r_in,
                                  depth=h)['verts']
    # Bridge the two rims so the wall reads as solid from any angle.
    for ring, flip in ((1.0, False), (-1.0, True)):
        ov = [v for v in outer if abs(v.co.z - ring * h * 0.5) < 1e-4]
        iv = [v for v in inner if abs(v.co.z - ring * h * 0.5) < 1e-4]
        ov.sort(key=lambda v: math.atan2(v.co.y, v.co.x))
        iv.sort(key=lambda v: math.atan2(v.co.y, v.co.x))
        n = min(len(ov), len(iv))
        for i in range(n):
            j = (i + 1) % n
            quad = [ov[i], ov[j], iv[j], iv[i]]
            if flip:
                quad.reverse()
            try:
                bm.faces.new(quad)
            except ValueError:
                pass
    bm.normal_update()
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    xform(bm, Matrix.Translation(Vector((0, 0, h * 0.5))))
    if axis == 'y':
        xform(bm, Matrix.Rotation(-math.pi / 2, 4, 'X'))
    elif axis == 'x':
        xform(bm, Matrix.Rotation(math.pi / 2, 4, 'Y'))
    xform(bm, Matrix.Translation(Vector(base)))
    set_mat(bm, bm.faces[:], mat)
    return bm


def sphere(center, r, u_seg=12, v_seg=8, mat='armor'):
    bm = new_bm()
    bmesh.ops.create_uvsphere(bm, u_segments=u_seg, v_segments=v_seg, radius=r)
    xform(bm, Matrix.Translation(Vector(center)))
    set_mat(bm, bm.faces[:], mat)
    return bm


def ring_flat(center, r_in, r_out, seg, mat='team', thickness=0.0):
    """Flat annulus, optionally given thickness so it is a raised collar."""
    bm = new_bm()
    verts_in, verts_out = [], []
    for i in range(seg):
        a = 2.0 * math.pi * i / seg
        c, s = math.cos(a), math.sin(a)
        verts_in.append(bm.verts.new((c * r_in, 0.0, s * r_in)))
        verts_out.append(bm.verts.new((c * r_out, 0.0, s * r_out)))
    bm.verts.ensure_lookup_table()
    for i in range(seg):
        j = (i + 1) % seg
        bm.faces.new([verts_in[i], verts_in[j], verts_out[j], verts_out[i]])
    bm.normal_update()
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    # recalc on an open sheet can settle either way; force the normal up.
    if bm.faces[0].normal.y < 0:
        bmesh.ops.reverse_faces(bm, faces=bm.faces[:])
    if thickness > 0:
        r = bmesh.ops.solidify(bm, geom=bm.faces[:], thickness=-thickness)
        bm.normal_update()
    xform(bm, Matrix.Translation(Vector(center)))
    set_mat(bm, bm.faces[:], mat)
    return bm


def blob(center, r, u_seg, v_seg, seed, amount, mat='rock'):
    """Irregular rock. A sphere displaced by value noise, then given a limited
    dissolve so the result reads as faceted stone rather than a lumpy ball."""
    import random
    rng = random.Random(seed)
    bm = new_bm()
    bmesh.ops.create_uvsphere(bm, u_segments=u_seg, v_segments=v_seg, radius=r)
    # Three low-frequency lobes give large faces; per-vertex jitter alone just
    # roughens the surface without changing the silhouette.
    lobes = [(Vector((rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-1, 1))).normalized(),
              rng.uniform(0.12, 0.34)) for _ in range(4)]
    for v in bm.verts:
        d = v.co.normalized()
        s = 1.0
        for axis, amp in lobes:
            s += amp * amount * max(0.0, d.dot(axis)) ** 1.5
        s += rng.uniform(-0.05, 0.05) * amount
        v.co = d * (r * s)
    bm.normal_update()
    bmesh.ops.dissolve_limit(bm, angle_limit=math.radians(12),
                             verts=bm.verts[:], edges=bm.edges[:])
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    xform(bm, Matrix.Translation(Vector(center)))
    set_mat(bm, bm.faces[:], mat)
    return bm


# ---------------------------------------------------------------- model
class Model:
    """One MeshId's worth of geometry: a list of bmesh parts merged at export."""

    def __init__(self, mesh_id, name):
        self.mesh_id = mesh_id
        self.name = name
        self.parts = []

    def add(self, bm):
        self.parts.append(bm)
        return bm

    def add_mirrored(self, bm, axis='x'):
        """Add a part and its mirror. Mirroring across an axis is a reflection,
        so the copy's faces have to be flipped or the whole half renders
        inside out -- this is the same handedness trap the module docstring
        warns about, in miniature."""
        self.parts.append(bm)
        m = bm.copy()
        a = {'x': 0, 'y': 1, 'z': 2}[axis]
        s = [1.0, 1.0, 1.0]
        s[a] = -1.0
        xform(m, Matrix.Diagonal(Vector(s + [1.0])))
        bmesh.ops.reverse_faces(m, faces=m.faces[:])
        m.normal_update()
        self.parts.append(m)
        return m


# ---------------------------------------------------------------- normals
SMOOTH_ANGLE = math.radians(38.0)


def _split_normals(bm):
    """Per-corner normals with a hard-edge threshold.

    Blender's own auto-smooth moved from a mesh flag to a modifier in 4.1 and
    the loop-normal API has shifted again since, so this computes the same
    thing directly and stops the pack depending on which Blender the author
    happened to have. For each face corner the normal is the area- and
    angle-weighted average of the faces meeting at that vertex whose normal is
    within SMOOTH_ANGLE of this face's -- so a bevel strip blends into the
    curve it rounds, while the flat plate on either side of it stays flat.

    Parts are never welded to each other, so a box bolted onto a hull cannot
    smear its shading into the hull: the boundary is two coincident but
    distinct vertices.
    """
    out = {}
    cos_lim = math.cos(SMOOTH_ANGLE)
    for f in bm.faces:
        fn = f.normal
        for loop in f.loops:
            acc = Vector((0.0, 0.0, 0.0))
            for nf in loop.vert.link_faces:
                if nf.normal.length < 1e-9:
                    continue
                if nf is not f and fn.dot(nf.normal) < cos_lim:
                    continue
                # Weight by the corner angle so a vertex where many small
                # triangles meet does not drag the average toward them.
                w = 0.0
                for nl in nf.loops:
                    if nl.vert is loop.vert:
                        try:
                            w = nl.calc_angle()
                        except ValueError:
                            w = 0.0
                        break
                acc += nf.normal * (w * nf.calc_area())
            if acc.length < 1e-9:
                acc = Vector(fn)
            out[loop] = acc.normalized()
    return out


# ---------------------------------------------------------------- export
MAGIC = b'SFMP'
VERSION = 1


def _encode_normal(n):
    def q(x):
        x = max(-1.0, min(1.0, x))
        return int(round(x * 32767.0))
    return q(n.x), q(n.y), q(n.z)


def tessellate(model):
    """Flatten a Model to (vertices, indices) in the pack's vertex layout.

    Returns (verts, idx, stats) where verts is a list of
    (px, py, pz, nx, ny, nz, mat) with normals already quantised.
    """
    verts = []
    idx = []
    lookup = {}
    for bm in model.parts:
        complaint = check_bounds(bm)
        if complaint:
            raise RuntimeError('%s: part escaped its intended bounds -- %s\n'
                               'This is almost always a modelling op hitting '
                               'the wrong faces; see check_bounds().'
                               % (model.name, complaint))
        bm.normal_update()
        bmesh.ops.triangulate(bm, faces=bm.faces[:], quad_method='BEAUTY',
                              ngon_method='BEAUTY')
        bm.normal_update()
        lay = _layer(bm)
        normals = _split_normals(bm)
        for f in bm.faces:
            if f.calc_area() < 1e-9:
                continue
            mat = f[lay]
            tri = []
            for loop in f.loops:
                co = loop.vert.co
                n = normals[loop]
                qn = _encode_normal(n)
                # Quantise the position for the dedupe key only; the stored
                # value stays full precision.
                key = (round(co.x, 5), round(co.y, 5), round(co.z, 5),
                       qn[0], qn[1], qn[2], mat)
                vi = lookup.get(key)
                if vi is None:
                    vi = len(verts)
                    lookup[key] = vi
                    verts.append((co.x, co.y, co.z, qn[0], qn[1], qn[2], mat))
                tri.append(vi)
            if tri[0] == tri[1] or tri[1] == tri[2] or tri[0] == tri[2]:
                continue
            idx.extend(tri)
    return verts, idx


def write_pack(path, models):
    """Serialise models to the on-disk pack.

    Layout (little-endian throughout; both Apple Silicon and x86 are LE, and
    the loader checks the magic and every count before trusting any of it):

        header      32 bytes
        mesh table  32 bytes x meshCount
        materials   32 bytes x matCount      (8 floats, one palette for all)
        vertices    20 bytes x vertexCount   (3 float pos, 3 int16 snorm normal,
                                              uint16 material index)
        indices      4 bytes x indexCount    (mesh-local, as MeshRange expects)

    Twenty bytes a vertex rather than the 64 the GPU wants: positions need the
    precision, normals do not (int16 snorm is ~0.003 degrees of error), and
    material is one index into a shared palette instead of eight floats
    repeated on every vertex. That is the whole reason the pack is under a
    megabyte instead of several.
    """
    mesh_rows = []
    all_verts = []
    all_idx = []
    report = []

    for mdl in models:
        v, i = tessellate(mdl)
        mesh_rows.append((mdl.mesh_id, len(all_verts), len(v), len(all_idx), len(i)))
        rad = 0.0
        hi = 0.0
        lo = 1e30
        for x, y, z, _, _, _, _ in v:
            rad = max(rad, math.hypot(x, z))
            hi = max(hi, y)
            lo = min(lo, y)
        report.append((mdl.name, len(v), len(i) // 3, rad, hi, lo))
        all_verts.extend(v)
        all_idx.extend(i)

    buf = bytearray()
    buf += struct.pack('<4sIIIIIII', MAGIC, VERSION, len(mesh_rows),
                       len(all_verts), len(all_idx), len(MAT_NAMES), 0, 0)
    for row in mesh_rows:
        buf += struct.pack('<iIIII12x', *row)
    for name in MAT_NAMES:
        buf += struct.pack('<8f', *MATERIALS[name])
    for x, y, z, nx, ny, nz, m in all_verts:
        buf += struct.pack('<3f3hH', x, y, z, nx, ny, nz, m)
    for i in all_idx:
        buf += struct.pack('<I', i)

    with open(path, 'wb') as fh:
        fh.write(buf)
    return report, len(buf)
