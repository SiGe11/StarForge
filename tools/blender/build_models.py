#!/usr/bin/env python3
"""build_models.py -- authors StarForge's unit and building meshes in Blender.

    pip install bpy            # CPython 3.11 only; no Blender install needed
    python3 tools/blender/build_models.py

Writes assets/models.bin. The game treats that file as optional: delete it and
MeshGen.cpp's primitives render instead, exactly as assets/ textures already
work. Validate the result with

    clang++ -std=c++20 -O2 -I src tools/mesh_check.cpp src/gfx/MeshGen.cpp -o /tmp/mc
    /tmp/mc assets/models.bin

Authoring notes
---------------
Everything is in game space: +Y up, +Z forward (the direction a unit faces),
origin on the ground at the model's centre. Dimensions are carried over from
the primitives in MeshGen.cpp, because MeshRange.radius drives the selection
ring and MeshRange.height positions the health bar and the build-in dissolve
cutoff -- a model that grows by a metre moves HUD elements with it.

What the extra triangles are spent on, in order of how much they change the
read at RTS camera distance:

1. Bevels on every silhouette edge. A sharp edge catches no specular
   highlight, so the low-poly originals read as flat blocks. This is most of
   the gain and most of the cost.
2. Recessed panels, hatches and vents via inset, which give large armour
   surfaces something to catch light on instead of reading as one tone.
3. Mechanical parts the primitives merely implied: wheels that differ from
   sprockets, tread blocks, barrel steps, jointed limbs.
"""

import os
import sys
import math

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bpy  # noqa: F401  -- importing bpy is what makes bmesh importable
import bmesh
from mathutils import Matrix, Vector

import sf_model as S
from sf_model import (Model, box, wedge, cyl, tube, sphere, ring_flat, blob,
                      inset, set_mat, xform, bevel, face_facing, cone_radius_at)

# MeshId values, mirroring the enum in src/gfx/RenderTypes.h.
MESH_WORKER, MESH_TROOPER, MESH_MAULER_HULL, MESH_MAULER_TURRET = 0, 1, 2, 3
MESH_FOUNDRY, MESH_GARRISON, MESH_WORKSHOP, MESH_BUNKHOUSE = 4, 5, 6, 7


def rot(bm, angle, axis, pivot=(0, 0, 0)):
    p = Vector(pivot)
    xform(bm, Matrix.Translation(-p))
    xform(bm, Matrix.Rotation(angle, 4, axis))
    xform(bm, Matrix.Translation(p))
    return bm


def face_angle(i, seg):
    """Angle of the centre of face `i` on a seg-sided cyl().

    bmesh's create_cone puts its first vertex at angle 0, so the flat faces of
    the octagonal drums here are centred half a segment round from there.
    Placing a rib at i*2pi/seg straddles a corner instead of sitting on a
    face, which is why the foundry's ribs floated off its edges.
    """
    return 2.0 * math.pi * i / seg + math.pi / seg


def orient_radial(bm, angle, pivot=(0, 0, 0)):
    """Turn a part built axis-aligned so its +Z faces radially outward.

    A rotation of theta about Y sends +Z to (sin theta, 0, cos theta), so
    pointing it at (cos a, 0, sin a) needs theta = pi/2 - a, not -a. Getting
    this wrong leaves ribs and louvres lying across their host wall rather
    than standing out of it, which is subtle enough to survive a glance.
    """
    return rot(bm, math.pi * 0.5 - angle, 'Y', pivot)


def panel(bm, axis, sign, thickness, depth, mat=None, max_deg=15.0):
    """Recess the faces pointing a given way. The workhorse for breaking a
    large plate into something that reads as panelled armour."""
    f = face_facing(bm, axis, sign, max_deg)
    if f:
        inset(bm, f, thickness, depth, mat)
    return bm


# ---------------------------------------------------------------- track unit
def track_unit(m, x, wheel_r, wheel_n, half_len, y_axle, width, tread=True):
    """One side's running gear: a bevelled track shoe, road wheels, a toothed
    drive sprocket at the back and a smooth idler at the front.

    The originals drew tracks as a single dark box. Distinguishing sprocket
    from idler from road wheel is what makes a tracked vehicle read as tracked
    rather than as a box on a plinth."""
    # The running gear defines where the vehicle meets the ground, so the
    # bottom of the track band is pinned to y=0. Geometry below the origin
    # sinks into the terrain, and on a slope it clips through it.
    y_axle = max(y_axle, wheel_r + 0.10)
    band = box((x, y_axle, 0.0), (width * 0.5, wheel_r + 0.10, half_len),
               'rubber', bevel_w=min(0.07, wheel_r * 0.35), bevel_seg=2)
    # Round the ends of the track so it reads as wrapping the wheels.
    for v in band.verts:
        if abs(v.co.z) > half_len * 0.78:
            v.co.y = y_axle + (v.co.y - y_axle) * 0.72
    band.normal_update()
    m.add_mirrored(band)

    if tread:
        # Tread blocks along the bottom run. The repetition is what the eye
        # reads as "track" in motion, so the count matters more than the
        # shape: these are 1-segment chamfers, not rounded blocks.
        n = max(4, int(half_len * 2.6))
        for i in range(n):
            t = (i + 0.5) / n * 2.0 - 1.0
            blk = box((x, 0.045, t * half_len * 0.92),
                      (width * 0.56, 0.045, half_len / n * 0.34),
                      'dark', bevel_w=0.015, bevel_seg=1)
            m.add_mirrored(blk)

    for i in range(wheel_n):
        t = (i / max(1, wheel_n - 1)) * 2.0 - 1.0
        # Road wheels sit largely behind the track band, so they are the
        # cheapest thing on the vehicle to coarsen: 8 sides, no rim bevel.
        w = cyl((x, y_axle, t * half_len * 0.72), wheel_r, wheel_r,
                width * 0.78, 8, 'steel', axis='x', bevel_w=0.0)
        xform(w, Matrix.Translation(Vector((-width * 0.39, 0, 0))))
        m.add_mirrored(w)

    # Drive sprocket (back, toothed) and idler (front, plain), both larger
    # than the road wheels so the running gear has a front and a back.
    # Both are seated so their outer edge lands exactly on +/- half_len.
    # MeshRange.radius is the max XZ distance over all vertices and drives the
    # selection ring, so a sprocket hanging off the back of the track silently
    # inflates the ring under every one of these units.
    spr_r = wheel_r * 1.25
    spr_z = half_len - spr_r * 1.04
    spr = cyl((x, y_axle + wheel_r * 0.30, -spr_z), spr_r, spr_r, width * 0.7, 10,
              'steel', axis='x', bevel_w=0.025)
    xform(spr, Matrix.Translation(Vector((-width * 0.35, 0, 0))))
    m.add_mirrored(spr)
    for k in range(6):
        a = 2.0 * math.pi * k / 6
        tooth = box((x, y_axle + wheel_r * 0.30 + math.sin(a) * spr_r * 1.02,
                     -spr_z + math.cos(a) * spr_r * 1.02),
                    (width * 0.32, 0.04, 0.04), 'dark', bevel_w=0.0)
        m.add_mirrored(tooth)

    idl_r = wheel_r * 1.1
    idl = cyl((x, y_axle + wheel_r * 0.26, half_len - idl_r * 1.04),
              idl_r, idl_r, width * 0.7, 10, 'steel', axis='x', bevel_w=0.025)
    xform(idl, Matrix.Translation(Vector((-width * 0.35, 0, 0))))
    m.add_mirrored(idl)


# ---------------------------------------------------------------- worker
def build_worker():
    m = Model(MESH_WORKER, 'WORKER')
    track_unit(m, 0.58, 0.22, 4, 0.82, 0.26, 0.34)

    # Chassis, panelled on top so the deck is not one flat tone.
    ch = box((0, 0.62, 0), (0.52, 0.20, 0.76), 'armor_dark', bevel_w=0.05)
    panel(ch, 'y', 1, 0.09, -0.035, 'shadowed')
    m.add(ch)

    # Sloped body. The top is recessed twice: once as a wide deck panel, once
    # as a narrow service hatch inside it.
    body = wedge((0, 1.00, -0.05), (0.46, 0.26, 0.62), 0.22, 0.18, 'armor',
                 bevel_w=0.05)
    top = face_facing(body, 'y', 1)
    if top:
        inner = inset(body, top, 0.07, -0.025, 'armor')
        if inner:
            inset(body, inner, 0.10, -0.03, 'shadowed')
    m.add(body)

    # Ore hopper: an open bin at the back, which gives the silhouette an
    # asymmetry the original box did not have.
    hop = box((0, 1.26, -0.50), (0.34, 0.24, 0.24), 'armor_dark', bevel_w=0.04)
    hf = face_facing(hop, 'y', 1)
    if hf:
        inset(hop, hf, 0.055, -0.30, 'shadowed')
    m.add(hop)

    # Cab, team-coloured, with a recessed window frame.
    cab = box((0, 1.30, 0.16), (0.34, 0.22, 0.38), 'team', bevel_w=0.045)
    front = face_facing(cab, 'z', 1)
    if front:
        inset(cab, front, 0.05, -0.03, 'dark')
    m.add(cab)
    m.add(box((0, 1.32, 0.55), (0.25, 0.13, 0.03), 'glow_warm', bevel_w=0.02, bevel_seg=1))

    # Roof light bar.
    m.add(box((0, 1.54, 0.20), (0.20, 0.035, 0.09), 'dark', bevel_w=0.015, bevel_seg=1))
    m.add(box((0, 1.575, 0.20), (0.15, 0.025, 0.07), 'glow_dim', bevel_w=0.01, bevel_seg=1))

    # Fusion cutter arms: upper arm, elbow, forearm, emitter. Jointing is what
    # separates a mining rig from two boxes stuck to the front.
    upper = box((0.46, 0.80, 0.50), (0.075, 0.075, 0.30), 'steel', bevel_w=0.025)
    m.add_mirrored(upper)
    elbow = cyl((0.46, 0.80, 0.80), 0.10, 0.10, 0.17, 8, 'dark', axis='x', bevel_w=0.02)
    xform(elbow, Matrix.Translation(Vector((-0.085, 0, 0))))
    m.add_mirrored(elbow)
    fore = box((0.46, 0.78, 0.96), (0.065, 0.065, 0.19), 'steel', bevel_w=0.02)
    rot(fore, math.radians(-8), 'X', (0.46, 0.80, 0.80))
    m.add_mirrored(fore)
    tip = cyl((0.46, 0.755, 1.10), 0.055, 0.035, 0.10, 8, 'dark', axis='z', bevel_w=0.015)
    m.add_mirrored(tip)
    m.add_mirrored(sphere((0.46, 0.755, 1.175), 0.045, 8, 6, 'glow_warm'))

    # Antenna and its base.
    m.add(cyl((0.30, 1.48, -0.30), 0.055, 0.045, 0.06, 8, 'dark', bevel_w=0.012))
    m.add(cyl((0.30, 1.52, -0.30), 0.022, 0.012, 0.48, 6, 'steel', bevel_w=0.0))
    return m


# ---------------------------------------------------------------- trooper
def build_trooper():
    m = Model(MESH_TROOPER, 'TROOPER')

    # Legs: thigh, knee, shin, boot. The original was one box per leg.
    for side, zoff in ((1, 0.06), (-1, -0.06)):
        x = 0.26 * side
        thigh = box((x, 0.62, zoff), (0.15, 0.26, 0.18), 'armor_dark', bevel_w=0.04)
        m.add(thigh)
        knee = cyl((x, 0.36, zoff), 0.14, 0.14, 0.26, 8, 'dark', axis='x', bevel_w=0.025)
        xform(knee, Matrix.Translation(Vector((-0.13, 0, 0))))
        m.add(knee)
        shin = box((x, 0.20, zoff + 0.02), (0.13, 0.20, 0.16), 'armor_dark', bevel_w=0.035)
        m.add(shin)
        boot = box((x, 0.07, zoff + 0.08), (0.18, 0.08, 0.26), 'dark', bevel_w=0.035)
        m.add(boot)
        # Shin plate, team-coloured.
        m.add(box((x, 0.22, zoff + 0.16), (0.11, 0.16, 0.035), 'team_dark', bevel_w=0.02))

    # Torso with a recessed chest plate and an abdomen segment.
    torso = wedge((0, 1.30, 0), (0.38, 0.40, 0.25), 0.15, 0.10, 'armor', bevel_w=0.045)
    chest = face_facing(torso, 'z', 1)
    if chest:
        inner = inset(torso, chest, 0.07, -0.028, 'armor')
        if inner:
            inset(torso, inner, 0.05, -0.02, 'shadowed')
    m.add(torso)
    m.add(box((0, 0.94, 0.02), (0.28, 0.13, 0.20), 'dark', bevel_w=0.03))

    # Pauldrons: a bevelled cap plus a rim, rather than a plain box.
    pl = box((0.50, 1.56, 0), (0.15, 0.17, 0.23), 'team', bevel_w=0.06, bevel_seg=3)
    m.add_mirrored(pl)
    m.add_mirrored(box((0.50, 1.72, 0), (0.13, 0.03, 0.20), 'armor', bevel_w=0.02))
    # Upper arms tucked under the pauldrons.
    m.add_mirrored(box((0.46, 1.26, 0.04), (0.095, 0.20, 0.11), 'armor_dark', bevel_w=0.03))
    m.add_mirrored(box((0.44, 1.02, 0.18), (0.085, 0.13, 0.10), 'armor_dark', bevel_w=0.028))

    # Backpack with vent slots and a power indicator.
    pack = box((0, 1.36, -0.34), (0.26, 0.29, 0.13), 'dark', bevel_w=0.04)
    back = face_facing(pack, 'z', -1)
    if back:
        inset(pack, back, 0.05, -0.035, 'shadowed')
    m.add(pack)
    for i in (-1, 0, 1):
        m.add(box((i * 0.13, 1.16, -0.47), (0.045, 0.06, 0.02), 'shadowed',
                  bevel_w=0.012, bevel_seg=1))
    m.add(box((0, 1.62, -0.45), (0.15, 0.045, 0.03), 'glow_warm', bevel_w=0.015, bevel_seg=1))
    # Air tanks.
    m.add_mirrored(cyl((0.17, 1.12, -0.40), 0.065, 0.065, 0.40, 8, 'steel', bevel_w=0.02))

    # Neck, helmet and visor. The helmet is a squashed sphere rather than a
    # round one, which reads far more like a helmet in silhouette.
    m.add(cyl((0, 1.66, 0.01), 0.11, 0.10, 0.10, 8, 'dark', bevel_w=0.02))
    # Helmet: a sphere squashed vertically and drawn out along Z, which is
    # what separates a helmet from a ball in silhouette. The visor then has to
    # sit *proud of* that stretched surface -- a box at the sphere's nominal
    # radius disappears inside it, which is how the first version ended up
    # with a visor that read as a small patch on the side of a bald head.
    SQ = (1.04, 0.94, 1.18)
    HR, HC = 0.26, (0.0, 1.90, 0.02)
    helm = sphere(HC, HR, 14, 9, 'armor')
    xform(helm, Matrix.Translation(-Vector(HC)))
    xform(helm, Matrix.Diagonal(Vector(SQ + (1.0,))))
    xform(helm, Matrix.Translation(Vector(HC)))
    m.add(helm)
    # Brow ridge above the visor.
    m.add(box((0, 1.98, HC[2] + HR * SQ[2] * 0.80), (0.17, 0.035, 0.07),
              'armor_dark', bevel_w=0.02))
    # Visor, straddling the front surface so it reads as set into the shell.
    m.add(box((0, 1.89, HC[2] + HR * SQ[2] * 0.93), (0.155, 0.062, 0.055),
              'glow_cyan', bevel_w=0.022, bevel_seg=2))
    # Jaw guard and rebreather under it.
    m.add(box((0, 1.76, HC[2] + HR * SQ[2] * 0.62), (0.115, 0.065, 0.10),
              'dark', bevel_w=0.025))
    # Low crest fin rather than a spike on top.
    m.add(box((0, 2.04, -0.05), (0.028, 0.055, 0.15), 'team', bevel_w=0.018))

    # Gauss rifle: receiver, magazine, grip, barrel with a stepped muzzle.
    m.add(box((0.40, 1.20, 0.34), (0.085, 0.10, 0.34), 'dark', bevel_w=0.028))
    m.add(box((0.40, 1.33, 0.30), (0.055, 0.045, 0.22), 'steel', bevel_w=0.02))
    m.add(box((0.40, 1.01, 0.22), (0.055, 0.11, 0.07), 'dark', bevel_w=0.025))
    m.add(box((0.40, 1.04, 0.50), (0.06, 0.09, 0.05), 'armor_dark', bevel_w=0.02))
    m.add(cyl((0.40, 1.24, 0.68), 0.05, 0.042, 0.36, 8, 'steel', axis='z', bevel_w=0.015))
    m.add(cyl((0.40, 1.24, 1.04), 0.062, 0.062, 0.08, 8, 'dark', axis='z', bevel_w=0.015))
    return m


# ---------------------------------------------------------------- mauler
def build_mauler_hull():
    m = Model(MESH_MAULER_HULL, 'MAULER_HULL')
    track_unit(m, 1.32, 0.40, 5, 1.95, 0.42, 0.68)

    # Lower hull, with a recessed sponson strip along each flank.
    low = box((0, 0.72, 0), (1.20, 0.30, 1.85), 'armor_dark', bevel_w=0.07)
    panel(low, 'x', 1, 0.14, -0.05, 'shadowed')
    panel(low, 'x', -1, 0.14, -0.05, 'shadowed')
    m.add(low)

    # Upper hull. The glacis is the face a tank is read by, so it gets a
    # two-step inset: a wide deck panel and a driver's hatch inside it.
    up = wedge((0, 1.14, -0.15), (1.15, 0.30, 1.70), 0.20, 0.12, 'armor', bevel_w=0.07)
    deck = face_facing(up, 'y', 1)
    if deck:
        inner = inset(up, deck, 0.16, -0.035, 'armor')
        if inner:
            inset(up, inner, 0.28, -0.04, 'shadowed')
    m.add(up)

    # Glacis plate: a separate sloped slab across the nose.
    gl = box((0, 1.05, 1.62), (1.08, 0.26, 0.20), 'armor', bevel_w=0.055)
    rot(gl, math.radians(34), 'X', (0, 1.05, 1.62))
    m.add(gl)

    # Team side skirts, slotted so they read as bolted-on plate.
    for zc in (-0.95, -0.10, 0.75):
        sk = box((1.19, 1.05, zc), (0.055, 0.19, 0.40), 'team', bevel_w=0.03)
        m.add_mirrored(sk)

    # Fenders over the tracks, and stowage boxes on them.
    m.add_mirrored(box((1.32, 1.34, 0.30), (0.46, 0.05, 1.55), 'armor_dark', bevel_w=0.03))
    m.add_mirrored(box((1.32, 1.47, -1.05), (0.34, 0.11, 0.42), 'dark', bevel_w=0.035))
    m.add_mirrored(box((1.32, 1.45, 0.95), (0.30, 0.09, 0.34), 'armor_dark', bevel_w=0.03))

    # Exhaust over the engine deck at the rear. Kept below the hull roof:
    # MeshRange.height is the max Y of the mesh and positions the health bar
    # and the build-in dissolve cutoff, so a stack poking above the hull
    # visibly lifts both.
    m.add_mirrored(tube((0.62, 1.20, -1.52), 0.12, 0.08, 0.22, 8, 'dark', axis='y'))
    # Engine deck louvres.
    for i in range(4):
        m.add(box((0, 1.44, -1.05 + i * 0.17), (0.72, 0.035, 0.05), 'shadowed',
                  bevel_w=0.012, bevel_seg=1))

    # Headlights with guards.
    m.add_mirrored(cyl((0.85, 1.30, 1.54), 0.11, 0.11, 0.05, 10, 'dark', axis='z', bevel_w=0.02))
    m.add_mirrored(cyl((0.85, 1.30, 1.58), 0.085, 0.085, 0.03, 10, 'glow_dim', axis='z', bevel_w=0.01))
    # Tow hooks.
    m.add_mirrored(box((0.55, 0.62, 1.86), (0.07, 0.07, 0.10), 'steel', bevel_w=0.02))
    return m


def build_mauler_turret():
    m = Model(MESH_MAULER_TURRET, 'MAULER_TURRET')

    # Turret shell with a sloped roof and a recessed roof panel.
    sh = wedge((0, 0.30, -0.15), (0.82, 0.30, 0.95), 0.25, 0.20, 'armor', bevel_w=0.065)
    roof = face_facing(sh, 'y', 1)
    if roof:
        inset(sh, roof, 0.13, -0.03, 'armor')
    m.add(sh)
    # Cheek armour, angled, which is what gives a turret its faceted read.
    for s in (1, -1):
        ck = box((0.62 * s, 0.30, 0.52), (0.22, 0.25, 0.34), 'armor', bevel_w=0.05)
        rot(ck, math.radians(-22 * s), 'Y', (0, 0.30, 0.52))
        m.add(ck)

    # Mantlet: rounded, so the barrel appears to pivot in it.
    m.add(cyl((0, 0.30, 0.72), 0.30, 0.30, 0.22, 12, 'armor_dark', axis='z', bevel_w=0.05))
    m.add(box((0, 0.30, 0.80), (0.34, 0.24, 0.16), 'armor_dark', bevel_w=0.05))

    # 120mm barrel: a stepped tube, thicker at the breech, with a fume
    # extractor bulge. A single tapered cylinder reads as a stick.
    m.add(cyl((0, 0.30, 0.88), 0.145, 0.125, 0.55, 12, 'steel', axis='z', bevel_w=0.02))
    m.add(cyl((0, 0.30, 1.43), 0.165, 0.165, 0.30, 12, 'armor_dark', axis='z', bevel_w=0.035))
    m.add(cyl((0, 0.30, 1.73), 0.115, 0.098, 0.92, 12, 'steel', axis='z', bevel_w=0.02))
    # Muzzle brake with side ports.
    m.add(cyl((0, 0.30, 2.65), 0.185, 0.185, 0.26, 12, 'dark', axis='z', bevel_w=0.03))
    for s in (1, -1):
        m.add(box((0.17 * s, 0.30, 2.72), (0.05, 0.075, 0.07), 'shadowed',
                  bevel_w=0.015, bevel_seg=1))

    # Team recognition band, commander's cupola with a rim and periscopes.
    m.add(box((0, 0.58, -0.50), (0.68, 0.055, 0.28), 'team', bevel_w=0.025))
    m.add(cyl((0.32, 0.58, -0.28), 0.23, 0.22, 0.13, 10, 'armor_dark', bevel_w=0.03))
    m.add(cyl((0.32, 0.71, -0.28), 0.19, 0.19, 0.04, 10, 'dark', bevel_w=0.015))
    for k in range(3):
        a = math.radians(-40 + k * 40)
        m.add(box((0.32 + math.sin(a) * 0.20, 0.68, -0.28 + math.cos(a) * 0.20),
                  (0.035, 0.03, 0.02), 'glow_cyan', bevel_w=0.008, bevel_seg=1))

    # Coaxial mount and smoke launchers -- small, but they break the roofline.
    m.add(box((-0.34, 0.36, 0.78), (0.06, 0.06, 0.22), 'dark', bevel_w=0.02))
    for s in (1, -1):
        m.add(box((s * 0.58, 0.50, -0.46), (0.07, 0.10, 0.24), 'dark', bevel_w=0.02))
        for k in range(3):
            m.add(box((s * 0.58, 0.61, -0.64 + k * 0.17), (0.05, 0.02, 0.05),
                      'shadowed', bevel_w=0.0))
    return m


# ---------------------------------------------------------------- buildings
def build_foundry():
    m = Model(MESH_FOUNDRY, 'FOUNDRY')

    base = cyl((0, 0, 0), 4.6, 4.3, 1.10, 8, 'armor_dark', bevel_w=0.10, bevel_seg=2)
    m.add(base)
    # Buttresses around the foundation.
    for i in range(8):
        a = face_angle(i, 8)
        bt = box((math.cos(a) * 4.05, 0.55, math.sin(a) * 4.05), (0.28, 0.55, 0.45),
                 'armor_dark', bevel_w=0.06)
        orient_radial(bt, a)
        m.add(bt)

    drum = cyl((0, 1.10, 0), 3.7, 3.5, 2.40, 8, 'armor', bevel_w=0.09, bevel_seg=2)
    m.add(drum)
    # Vertical ribs: eight plates standing proud of the drum wall. On a smooth
    # cylinder at this size the lighting has nothing to break up, and the
    # building reads as a barrel.
    for i in range(8):
        a = face_angle(i, 8)
        # Seat each rib on the drum's surface at its own height so it stands
        # proud of the taper rather than sinking into it.
        rr = cone_radius_at(3.7, 3.5, 2.40, 2.30 - 1.10) + 0.12
        rb = box((math.cos(a) * rr, 2.30, math.sin(a) * rr), (0.16, 1.12, 0.26),
                 'armor_dark', bevel_w=0.045)
        orient_radial(rb, a)
        m.add(rb)

    # Window band, sized from the drum's own taper at this height plus a
    # small proud offset. Hardcoding 3.57 here put the band 5 cm inside a
    # drum that is 3.62 wide at y=2.05, and it rendered nothing at all.
    band_r = cone_radius_at(3.7, 3.5, 2.40, 2.05 - 1.10) + 0.04
    m.add(cyl((0, 2.05, 0), band_r, band_r, 0.42, 8, 'glow_warm', caps=False, bevel_w=0.0))
    m.add(cyl((0, 3.50, 0), 3.5, 2.5, 1.00, 8, 'armor', bevel_w=0.08, bevel_seg=2))

    m.add(ring_flat((0, 4.50, 0), 1.65, 2.45, 24, 'team', thickness=0.10))
    tower = cyl((0, 4.50, 0), 1.6, 1.4, 0.55, 8, 'armor_dark', bevel_w=0.06)
    m.add(tower)
    glz_r = cone_radius_at(1.6, 1.4, 0.55, 4.72 - 4.50) + 0.04
    m.add(cyl((0, 4.72, 0), glz_r, glz_r, 0.26, 8, 'glow_cyan', caps=False, bevel_w=0.0))
    m.add(cyl((0, 4.90, 0), 1.35, 1.15, 0.15, 8, 'armor_dark', bevel_w=0.04))

    # Corner pylons, capped and lit.
    for i in range(4):
        a = math.pi * 0.25 + i * math.pi * 0.5
        px, pz = math.cos(a) * 4.15, math.sin(a) * 4.15
        py = box((px, 1.5, pz), (0.26, 1.5, 0.26), 'steel', bevel_w=0.045)
        rot(py, -a, 'Y', (0, 0, 0))
        m.add(py)
        m.add(box((px, 3.05, pz), (0.32, 0.10, 0.32), 'armor_dark', bevel_w=0.035))
        m.add(box((px, 3.20, pz), (0.15, 0.09, 0.15), 'glow_dim', bevel_w=0.02, bevel_seg=1))

    # Intake pipes running up the drum -- the greeble that says "industry".
    for i in (2, 6):
        a = face_angle(i, 8) + math.pi / 8
        pr = cone_radius_at(3.7, 3.5, 2.40, 1.10) + 0.10
        px, pz = math.cos(a) * pr, math.sin(a) * pr
        m.add(cyl((px, 1.20, pz), 0.20, 0.20, 2.10, 8, 'steel', bevel_w=0.03))
        m.add(cyl((px, 3.30, pz), 0.26, 0.20, 0.22, 8, 'dark', bevel_w=0.03))
    return m


def build_garrison():
    m = Model(MESH_GARRISON, 'GARRISON')

    slab = box((0, 0.30, 0), (3.20, 0.30, 3.70), 'armor_dark', bevel_w=0.08)
    m.add(slab)
    main = wedge((0, 1.70, -0.2), (2.80, 1.10, 3.30), 0.18, 0.14, 'armor', bevel_w=0.09,
                 bevel_seg=2)
    # Recess the long flanks into wall panels.
    panel(main, 'x', 1, 0.30, -0.07, 'shadowed')
    panel(main, 'x', -1, 0.30, -0.07, 'shadowed')
    m.add(main)

    # Buttress ribs down each side.
    for zc in (-2.3, -0.9, 0.5, 1.9):
        rb = box((2.72, 1.55, zc), (0.20, 1.20, 0.26), 'armor_dark', bevel_w=0.04)
        m.add_mirrored(rb)

    roof = box((0, 3.05, -0.9), (1.90, 0.35, 1.70), 'armor_dark', bevel_w=0.06)
    rf = face_facing(roof, 'y', 1)
    if rf:
        inset(roof, rf, 0.22, -0.06, 'shadowed')
    m.add(roof)
    # Roof vents.
    for s in (1, -1):
        m.add(box((s * 1.05, 3.48, -0.9), (0.45, 0.10, 1.20), 'steel', bevel_w=0.03))

    # Bay door, set into a recessed frame with a lit surround.
    frame = box((0, 1.25, 3.05), (1.52, 1.12, 0.22), 'armor_dark', bevel_w=0.05)
    df = face_facing(frame, 'z', 1)
    if df:
        inset(frame, df, 0.14, -0.10, 'dark')
    m.add(frame)
    m.add(box((0, 1.22, 3.20), (1.22, 0.88, 0.06), 'glow_amber', bevel_w=0.02, bevel_seg=1))
    # Door rails.
    for s in (1, -1):
        m.add(box((s * 1.44, 1.25, 3.22), (0.08, 1.10, 0.08), 'steel', bevel_w=0.02))

    m.add(box((0, 2.72, 0.6), (2.74, 0.17, 2.50), 'team', bevel_w=0.04))

    # Antennae with dishes.
    for s, h in ((-1, 1.7), (1, 1.4)):
        m.add(cyl((s * 2.3, 3.40, -2.60), 0.085, 0.05, h, 8, 'steel', bevel_w=0.02))
        m.add(box((s * 2.3, 3.40 + h, -2.60), (0.075, 0.075, 0.075), 'glow_warm',
                  bevel_w=0.02, bevel_seg=1))
    m.add(cyl((2.3, 3.55, -2.60), 0.42, 0.42, 0.07, 12, 'armor_dark', bevel_w=0.03))
    return m


def build_workshop():
    m = Model(MESH_WORKSHOP, 'WORKSHOP')

    m.add(box((0, 0.32, 0), (3.70, 0.32, 4.20), 'armor_dark', bevel_w=0.08))

    # Arched hangar: a cylinder on its side. Bevelling its rim would round the
    # arch away, so it is left sharp and the ribs below provide the relief.
    arch = cyl((0, 1.15, -3.6), 2.55, 2.55, 7.2, 14, 'armor', axis='z', bevel_w=0.0)
    m.add(arch)
    # Structural ribs banding the arch.
    for zc in (-2.6, -0.9, 0.8, 2.5):
        rb = tube((0, 1.15, zc), 2.66, 2.50, 0.26, 14, 'armor_dark', axis='z')
        m.add(rb)

    m.add(box((0, 1.15, -3.75), (2.60, 1.45, 0.22), 'armor_dark', bevel_w=0.06))
    # Rear service door.
    m.add(box((0, 0.95, -3.92), (1.10, 1.05, 0.06), 'dark', bevel_w=0.02))

    # Hangar mouth: recessed frame plus the lit opening.
    mouth = box((0, 1.20, 3.70), (2.34, 1.58, 0.20), 'armor_dark', bevel_w=0.06)
    mf = face_facing(mouth, 'z', 1)
    if mf:
        inset(mouth, mf, 0.22, -0.12, 'dark')
    m.add(mouth)
    m.add(box((0, 1.05, 3.84), (1.85, 1.22, 0.05), 'glow_amber', bevel_w=0.02, bevel_seg=1))
    # Approach lights on the apron.
    for s in (1, -1):
        m.add(box((s * 2.10, 0.70, 4.05), (0.12, 0.12, 0.10), 'glow_dim',
                  bevel_w=0.02, bevel_seg=1))

    m.add(box((0, 3.62, -0.4), (1.30, 0.15, 3.00), 'team', bevel_w=0.04))
    # Gantry crane rail down the roof line.
    m.add(box((0, 3.80, -0.4), (0.22, 0.10, 2.80), 'steel', bevel_w=0.025))

    # Exhaust stacks with caps.
    for s in (1, -1):
        m.add(cyl((s * 2.55, 1.60, -2.50), 0.42, 0.36, 2.50, 12, 'steel', bevel_w=0.04))
        m.add(tube((s * 2.55, 4.02, -2.50), 0.40, 0.29, 0.20, 12, 'dark', axis='y'))
        m.add(box((s * 2.55, 2.30, -2.50), (0.50, 0.09, 0.50), 'armor_dark', bevel_w=0.03))
    return m


def build_bunkhouse():
    m = Model(MESH_BUNKHOUSE, 'BUNKHOUSE')

    m.add(cyl((0, 0, 0), 2.35, 2.20, 0.45, 8, 'armor_dark', bevel_w=0.07, bevel_seg=2))
    body = cyl((0, 0.45, 0), 2.05, 1.75, 0.95, 8, 'armor', bevel_w=0.07, bevel_seg=2)
    m.add(body)

    # Wall panels between the vents.
    for i in range(8):
        a = face_angle(i, 8)
        pr = cone_radius_at(2.05, 1.75, 0.95, 0.45) + 0.04
        pl = box((math.cos(a) * pr, 0.90, math.sin(a) * pr), (0.42, 0.30, 0.06),
                 'armor_dark', bevel_w=0.03)
        orient_radial(pl, a)
        m.add(pl)

    m.add(ring_flat((0, 1.40, 0), 1.05, 1.72, 16, 'team', thickness=0.09))
    m.add(cyl((0, 1.38, 0), 1.02, 0.82, 0.30, 8, 'glow_cyan', bevel_w=0.03))
    # Lit slit around the collar, proud of the body's taper at the rim.
    slit_r = cone_radius_at(2.05, 1.75, 0.95, 0.86) + 0.03
    m.add(cyl((0, 1.24, 0), slit_r, slit_r, 0.10, 8, 'glow_cyan', caps=False, bevel_w=0.0))
    m.add(cyl((0, 1.66, 0), 0.80, 0.58, 0.06, 8, 'armor_dark', bevel_w=0.02))

    # Vents, with a louvred face.
    for i in range(4):
        a = math.pi * 0.25 + i * math.pi * 0.5
        vx, vz = math.cos(a) * 1.95, math.sin(a) * 1.95
        vt = box((vx, 0.85, vz), (0.22, 0.42, 0.22), 'steel', bevel_w=0.035)
        orient_radial(vt, a)
        m.add(vt)
        for k in (-1, 0, 1):
            lv = box((vx * 1.10, 0.85 + k * 0.14, vz * 1.10), (0.17, 0.035, 0.05),
                     'shadowed', bevel_w=0.01, bevel_seg=1)
            orient_radial(lv, a)
            m.add(lv)
    return m


# ---------------------------------------------------------------- main
BUILDERS = [
    build_worker, build_trooper, build_mauler_hull, build_mauler_turret,
    build_foundry, build_garrison, build_workshop, build_bunkhouse,
]


def main():
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    out = os.path.join(root, 'assets', 'models.bin')

    models = []
    for fn in BUILDERS:
        models.append(fn())

    report, nbytes = S.write_pack(out, models)

    print('%-16s %8s %8s %8s %8s %8s' % ('mesh', 'verts', 'tris', 'radius', 'height', 'minY'))
    print('-' * 62)
    tot_v = tot_t = 0
    for name, nv, nt, rad, hi, lo in report:
        print('%-16s %8d %8d %8.2f %8.2f %8.2f' % (name, nv, nt, rad, hi, lo))
        tot_v += nv
        tot_t += nt
    print('-' * 62)
    print('%-16s %8d %8d   %.1f KB' % ('TOTAL', tot_v, tot_t, nbytes / 1024.0))
    print('\nwrote %s' % out)


if __name__ == '__main__':
    main()
