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
from sf_model import (Model, box, wedge, frustum, cyl, tube, sphere, ring_flat,
                      blob, inset, set_mat, xform, bevel, face_facing,
                      cone_radius_at, track_loop, face_plate,
                      mark_flat)

# MeshId values, mirroring the enum in src/gfx/RenderTypes.h.
MESH_WORKER, MESH_TROOPER, MESH_MAULER_HULL, MESH_MAULER_TURRET = 0, 1, 2, 3
MESH_FOUNDRY, MESH_GARRISON, MESH_WORKSHOP, MESH_BUNKHOUSE = 4, 5, 6, 7
MESH_ORE, MESH_BOULDER = 8, 9


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


def panel(bm, axis, sign, thickness, depth, mat=None, max_deg=42.0):
    """Recess the plate facing a given way. The workhorse for breaking a
    large surface into something that reads as panelled armour.

    Uses face_plate rather than face_facing: these parts are bevelled before
    anything is selected on them, and on a sloped plate an angle test cannot
    tell the plate from its own trim."""
    f = face_plate(bm, axis, sign, max_deg)
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
    # bottom of the track loop is pinned to y=0. Geometry below the origin
    # sinks into the terrain, and on a slope it clips through it.
    outer_r = wheel_r + 0.10
    y_axle = max(y_axle, outer_r)
    # Inner radius set to the road-wheel radius, so the wheels meet the track
    # exactly instead of floating inside it or cutting through.
    m.add_mirrored(track_loop(x, y_axle, half_len, outer_r, width,
                              outer_r - wheel_r, seg_arc=max(5, int(wheel_r * 18))))

    if tread:
        # Tread shoes on both straight runs. The repetition is what the eye
        # reads as "track" in motion, so the count matters more than the shape:
        # these are 1-segment chamfers, not rounded blocks. The top run matters
        # as much as the bottom because the game's camera looks down at the
        # vehicle -- the bottom run is the one the player never sees.
        n = max(4, int(half_len * 2.6))
        run = half_len - wheel_r
        for i in range(n):
            t = (i + 0.5) / n * 2.0 - 1.0
            for y_run in (0.04, y_axle * 2.0 - 0.04):
                blk = box((x, y_run, t * run * 0.98),
                          (width * 0.53, 0.04, run / n * 0.40),
                          'dark', bevel_w=0.012, bevel_seg=1)
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
    """Digger: a heavy mining crawler.

    Silhouette first. The shape reads as a low forward-raked wedge with one
    heavy cutter arm reaching out in front and a big ore drum slung across the
    back -- asymmetric front to back, and unmistakable from a top-down camera
    at a hundred metres. The version this replaces was a box on tracks with
    detail added to it, which is a box.
    """
    m = Model(MESH_WORKER, 'WORKER')
    track_unit(m, 0.60, 0.22, 4, 0.80, 0.26, 0.32)

    # Track shrouds, flared outward at the top so the stance widens upward and
    # the vehicle stops reading as a rectangular prism.
    shroud = frustum((0.60, 0.62, -0.05), (0.19, 0.16, 0.74),
                     top=(1.55, 0.92), shift=(0.07, 0.0), mat='armor_dark',
                     bevel_w=0.035)
    m.add_mirrored(shroud)

    # Hull: narrow at the base, wider and raked at the deck.
    hull = frustum((0, 0.68, -0.02), (0.40, 0.26, 0.70), top=(1.18, 0.92),
                   shift=(0.0, -0.05), mat='armor', bevel_w=0.05)
    panel(hull, 'y', 1, 0.09, -0.035, 'shadowed')
    m.add(hull)

    # Raked prow, a single sloped plate rather than a flat nose.
    prow = frustum((0, 0.50, 0.72), (0.38, 0.22, 0.16), top=(0.78, 1.0),
                   shift=(0.0, -0.13), mat='armor_lit', bevel_w=0.04)
    rot(prow, math.radians(-34), 'X', (0, 0.50, 0.72))
    m.add(prow)
    m.add(box((0, 0.36, 0.80), (0.30, 0.05, 0.10), 'steel', bevel_w=0.02))

    # Ore drum across the back. This is the mass that carries the silhouette,
    # so it is deliberately oversized and sits high.
    drum = cyl((-0.50, 1.28, -0.46), 0.42, 0.42, 1.00, 12, 'armor_dark',
               axis='x', bevel_w=0.05)
    m.add(drum)
    for k in (-1, 1):
        m.add(cyl((k * 0.50, 1.28, -0.46), 0.45, 0.45, 0.07, 12, 'steel',
                  axis='x', bevel_w=0.025))
    # Hopper mouth on top of the drum, open and dark.
    mouth = box((0, 1.64, -0.46), (0.36, 0.10, 0.30), 'armor_dark', bevel_w=0.03)
    mf = face_plate(mouth, 'y', 1)
    if mf:
        inset(mouth, mf, 0.055, -0.20, 'shadowed')
    m.add(mouth)
    # Discharge chute angled off the back.
    chute = frustum((0, 1.06, -0.86), (0.22, 0.24, 0.12), top=(0.6, 1.0),
                    shift=(0.0, -0.10), mat='steel', bevel_w=0.03)
    rot(chute, math.radians(24), 'X', (0, 1.06, -0.86))
    m.add(chute)

    # Cab: small, canted, and pushed off-centre. A centred cab reads as
    # symmetrical machinery; an offset one reads as a vehicle with a driver.
    cab = frustum((-0.30, 1.16, 0.26), (0.28, 0.24, 0.28), top=(0.80, 0.82),
                  shift=(0.03, -0.02), mat='team', bevel_w=0.04)
    m.add(cab)
    rot(cab, math.radians(-7), 'Z', (-0.30, 1.16, 0.26))
    m.add(box((-0.30, 1.22, 0.54), (0.21, 0.13, 0.03), 'glow_warm',
              bevel_w=0.018, bevel_seg=1))
    m.add(box((-0.30, 1.44, 0.22), (0.17, 0.035, 0.09), 'dark', bevel_w=0.015))
    m.add(box((-0.30, 1.48, 0.22), (0.13, 0.025, 0.07), 'glow_dim',
              bevel_w=0.01, bevel_seg=1))

    # Cutter arm: one heavy limb, forward and down, ending in a toothed disc.
    # A single big arm beats two small ones -- it breaks the symmetry and
    # gives the front of the model something to be.
    m.add(box((0.34, 0.86, 0.34), (0.13, 0.15, 0.13), 'steel', bevel_w=0.03))
    upper = box((0.34, 0.80, 0.60), (0.10, 0.10, 0.26), 'armor_dark', bevel_w=0.028)
    rot(upper, math.radians(14), 'X', (0.34, 0.86, 0.34))
    m.add(upper)
    m.add(cyl((0.34, 0.66, 0.84), 0.12, 0.12, 0.20, 8, 'dark', axis='x',
              bevel_w=0.022))
    fore = box((0.34, 0.55, 1.00), (0.085, 0.085, 0.22), 'steel', bevel_w=0.025)
    rot(fore, math.radians(-26), 'X', (0.34, 0.66, 0.84))
    m.add(fore)
    # Hydraulic ram alongside the arm, a cheap read of "this thing moves".
    m.add(cyl((0.50, 0.92, 0.48), 0.045, 0.045, 0.34, 6, 'steel', axis='z',
              bevel_w=0.0))

    # Rotary cutter head.
    head_z = 1.14
    m.add(cyl((0.34, 0.44, head_z), 0.24, 0.24, 0.11, 14, 'armor_dark',
              axis='z', bevel_w=0.03))
    m.add(cyl((0.34, 0.44, head_z + 0.06), 0.11, 0.09, 0.07, 10, 'steel',
              axis='z', bevel_w=0.015))
    for k in range(8):
        a = 2.0 * math.pi * k / 8
        t = box((0.34 + math.cos(a) * 0.255, 0.44 + math.sin(a) * 0.255, head_z),
                (0.045, 0.045, 0.07), 'steel', bevel_w=0.0)
        m.add(t)

    # Exhaust stack and antenna, both behind the cab so the roofline is broken.
    m.add(tube((0.30, 1.30, -0.02), 0.085, 0.055, 0.42, 8, 'dark', axis='y'))
    m.add(cyl((0.44, 1.10, 0.06), 0.05, 0.04, 0.06, 8, 'dark', bevel_w=0.012))
    m.add(cyl((0.44, 1.14, 0.06), 0.02, 0.012, 0.84, 6, 'steel', bevel_w=0.0))
    return m


# ---------------------------------------------------------------- trooper

def build_trooper():
    """Trooper: powered infantry.

    Read at distance comes from proportion, not detail. The shoulders are
    deliberately enormous and team-coloured, the waist is pinched, the boots
    are heavy, and the head is small and sunk between the pauldrons so there
    is no neck -- a caricature, because a realistically proportioned figure
    twenty metres from the camera is a grey smudge.
    """
    m = Model(MESH_TROOPER, 'TROOPER')

    # Legs: splayed, heavy thighs, armoured shins, wide boots.
    for side in (1, -1):
        x = 0.27 * side
        m.add(box((x, 0.78, -0.02), (0.14, 0.13, 0.17), 'dark', bevel_w=0.03))
        thigh = frustum((x, 0.58, 0.0), (0.155, 0.20, 0.19), top=(1.12, 1.05),
                        mat='armor', bevel_w=0.04)
        rot(thigh, math.radians(-5 * side), 'Z', (x, 0.78, 0))
        m.add(thigh)
        m.add(cyl((x, 0.36, 0.0), 0.135, 0.135, 0.24, 8, 'dark', axis='x',
                  bevel_w=0.025))
        xform(m.parts[-1], Matrix.Translation(Vector((-0.12, 0, 0))))
        shin = frustum((x, 0.21, 0.02), (0.145, 0.17, 0.16), top=(0.88, 1.0),
                       mat='armor_dark', bevel_w=0.035)
        m.add(shin)
        # Shin plate, team, on the silhouette edge where it will be seen.
        m.add(box((x, 0.24, 0.17), (0.115, 0.15, 0.04), 'team', bevel_w=0.02))
        boot = frustum((x, 0.07, 0.07), (0.175, 0.07, 0.25), top=(0.85, 0.88),
                       mat='dark', bevel_w=0.03)
        m.add(boot)
        m.add(box((x, 0.03, 0.22), (0.15, 0.03, 0.08), 'steel', bevel_w=0.015))

    # Waist, pinched, then a chest that flares out and up.
    m.add(frustum((0, 0.98, 0.0), (0.20, 0.13, 0.16), top=(1.25, 1.2),
                  mat='dark', bevel_w=0.03))
    chest = frustum((0, 1.32, -0.01), (0.27, 0.23, 0.20), top=(1.42, 1.18),
                    shift=(0.0, 0.02), mat='armor', bevel_w=0.05)
    cf = face_plate(chest, 'z', 1)
    if cf:
        inner = inset(chest, cf, 0.06, -0.03, 'armor_lit')
        if inner:
            inset(chest, inner, 0.05, -0.022, 'shadowed')
    m.add(chest)
    # Collar the head sinks into.
    m.add(frustum((0, 1.60, -0.02), (0.30, 0.08, 0.22), top=(0.80, 0.80),
                  mat='armor_dark', bevel_w=0.035))

    # Pauldrons. These are the silhouette: oversized, canted outward, and the
    # one place the faction colour is guaranteed to be visible from any angle.
    for side in (1, -1):
        pl = frustum((0.46 * side, 1.56, -0.01), (0.20, 0.17, 0.24),
                     top=(0.82, 0.80), shift=(0.05 * side, 0.0),
                     mat='team', bevel_w=0.055, bevel_seg=2)
        rot(pl, math.radians(-16 * side), 'Z', (0.30 * side, 1.52, 0))
        m.add(pl)
        m.add(box((0.52 * side, 1.40, -0.01), (0.14, 0.05, 0.21),
                  'armor_dark', bevel_w=0.022))
        # Arms hang inboard and slightly forward.
        m.add(frustum((0.47 * side, 1.20, 0.05), (0.10, 0.16, 0.115),
                      top=(0.9, 0.9), mat='armor_dark', bevel_w=0.028))
        m.add(box((0.45 * side, 0.99, 0.16), (0.095, 0.115, 0.105),
                  'armor', bevel_w=0.025))

    # Backpack with two stacks rising above the shoulder line, which gives the
    # top of the silhouette something other than a dome.
    pack = frustum((0, 1.36, -0.32), (0.24, 0.26, 0.12), top=(0.86, 1.0),
                   mat='armor_dark', bevel_w=0.04)
    bf = face_plate(pack, 'z', -1)
    if bf:
        inset(pack, bf, 0.05, -0.03, 'shadowed')
    m.add(pack)
    for side in (1, -1):
        st = tube((0.15 * side, 1.58, -0.34), 0.062, 0.040, 0.42, 8, 'dark',
                  axis='y')
        rot(st, math.radians(-13), 'X', (0.15 * side, 1.58, -0.34))
        m.add(st)
    m.add(box((0, 1.20, -0.45), (0.15, 0.04, 0.03), 'glow_warm',
              bevel_w=0.012, bevel_seg=1))

    # Head: small, low, and mostly visor.
    helm = frustum((0, 1.74, 0.02), (0.145, 0.115, 0.15), top=(0.80, 0.78),
                   shift=(0.0, -0.01), mat='armor_dark', bevel_w=0.04)
    m.add(helm)
    m.add(box((0, 1.74, 0.16), (0.115, 0.055, 0.035), 'glow_cyan',
              bevel_w=0.018, bevel_seg=1))
    m.add(box((0, 1.88, -0.03), (0.10, 0.035, 0.12), 'team', bevel_w=0.02))

    # Gauss rifle: chunky, with a drum magazine and a heavy muzzle.
    m.add(box((0.30, 1.14, 0.30), (0.075, 0.095, 0.30), 'dark', bevel_w=0.025))
    m.add(cyl((0.30, 1.06, 0.26), 0.115, 0.115, 0.12, 10, 'armor_dark',
              axis='x', bevel_w=0.022))
    m.add(box((0.30, 1.28, 0.24), (0.05, 0.05, 0.18), 'steel', bevel_w=0.018))
    m.add(cyl((0.30, 1.17, 0.60), 0.046, 0.040, 0.30, 8, 'steel', axis='z',
              bevel_w=0.014))
    m.add(cyl((0.30, 1.17, 0.90), 0.070, 0.070, 0.10, 8, 'dark', axis='z',
              bevel_w=0.016))
    return m


# ---------------------------------------------------------------- mauler

def build_mauler_hull():
    """Mauler hull: an arrow, not a brick.

    The prow comes to a vertical centre edge and the hull widens toward the
    deck, so the shape has a front and a direction even in plan view. The
    engine deck steps up at the rear with the stacks on it, which stops the
    roofline being one flat rectangle -- the single biggest tell that a model
    is a box.
    """
    m = Model(MESH_MAULER_HULL, 'MAULER_HULL')
    track_unit(m, 1.30, 0.40, 5, 1.90, 0.50, 0.66)

    # Flared track shrouds: narrow at the bottom, overhanging at the top.
    for side in (1, -1):
        sh = frustum((1.30 * side, 1.12, -0.10), (0.36, 0.20, 1.72),
                     top=(1.30, 0.94), shift=(0.10 * side, 0.0),
                     mat='armor_dark', bevel_w=0.05)
        m.add(sh)

    # Lower hull, tucked in between the tracks.
    low = frustum((0, 0.66, 0), (1.02, 0.30, 1.76), top=(1.12, 1.0),
                  mat='dark', bevel_w=0.06)
    m.add(low)

    # Upper hull, widening toward the deck.
    up = frustum((0, 1.14, -0.18), (1.04, 0.24, 1.48), top=(1.06, 0.90),
                 shift=(0.0, -0.06), mat='armor', bevel_w=0.06)
    deck = face_plate(up, 'y', 1)
    if deck:
        inner = inset(up, deck, 0.16, -0.035, 'armor_lit')
        if inner:
            inset(up, inner, 0.30, -0.045, 'shadowed')
    m.add(up)

    # The prow: two plates meeting at a centre edge, so the nose is a wedge in
    # plan as well as in profile.
    for side in (1, -1):
        gl = frustum((0.50 * side, 1.00, 1.46), (0.56, 0.30, 0.34),
                     top=(0.80, 0.55), shift=(-0.10 * side, -0.14),
                     mat='armor_lit', bevel_w=0.05)
        rot(gl, math.radians(30), 'X', (0.50 * side, 1.00, 1.46))
        rot(gl, math.radians(-17 * side), 'Y', (0, 0, 1.70))
        m.add(gl)
    m.add(box((0, 0.74, 1.62), (0.26, 0.16, 0.22), 'dark', bevel_w=0.04))
    # Dozer lugs on the nose.
    for side in (1, -1):
        m.add(box((0.72 * side, 0.70, 1.52), (0.09, 0.10, 0.14), 'steel',
                  bevel_w=0.022))

    # Engine deck, stepped up at the rear.
    eng = frustum((0, 1.50, -1.16), (0.86, 0.15, 0.60), top=(0.88, 0.84),
                  mat='armor_dark', bevel_w=0.045)
    m.add(eng)
    for i in range(5):
        m.add(box((0, 1.66, -1.52 + i * 0.18), (0.66, 0.035, 0.05),
                  'shadowed', bevel_w=0.0))
    for side in (1, -1):
        st = tube((0.62 * side, 1.58, -0.58), 0.130, 0.086, 0.26, 8, 'dark',
                  axis='y')
        rot(st, math.radians(-16), 'X', (0.62 * side, 1.62, -0.58))
        m.add(st)

    # Team skirts, angled with the shroud so the colour sits on the
    # silhouette edge rather than flat on a side panel.
    for side in (1, -1):
        for zc in (-1.02, -0.12, 0.78):
            sk = frustum((1.56 * side, 1.06, zc), (0.05, 0.22, 0.40),
                         top=(1.0, 0.88), shift=(0.07 * side, 0.0),
                         mat='team', bevel_w=0.028)
            m.add(sk)

    # Stowage and lights, asymmetric on purpose.
    m.add(box((1.34, 1.42, -1.30), (0.30, 0.13, 0.40), 'rust', bevel_w=0.035))
    m.add(box((-1.34, 1.40, 0.90), (0.26, 0.11, 0.32), 'armor_dark', bevel_w=0.03))
    for side in (1, -1):
        m.add(cyl((0.78 * side, 1.26, 1.30), 0.10, 0.10, 0.06, 10, 'dark',
                  axis='z', bevel_w=0.018))
        m.add(cyl((0.78 * side, 1.26, 1.35), 0.078, 0.078, 0.03, 10,
                  'glow_dim', axis='z', bevel_w=0.0))
    return m



def build_mauler_turret():
    """Mauler turret: a faceted wedge with an overhanging bustle.

    Narrow and sloped at the front, wide and square at the back, with the
    ammunition bustle cantilevered off the rear. That asymmetry is what makes
    a turret readable as pointing somewhere, which matters because this mesh
    is drawn rotating independently of the hull.
    """
    m = Model(MESH_MAULER_TURRET, 'MAULER_TURRET')

    # Shell: wide at the base, narrowing and sloping up, and tapered toward
    # the muzzle end so the whole turret is a wedge in plan.
    shell = frustum((0, 0.30, -0.20), (0.86, 0.30, 0.92), top=(0.66, 0.74),
                    shift=(0.0, -0.10), mat='armor', bevel_w=0.06)
    roof = face_plate(shell, 'y', 1)
    if roof:
        inset(shell, roof, 0.13, -0.03, 'armor_lit')
    m.add(shell)

    # Cheeks: angled plates each side of the mantlet, the classic faceted read.
    for side in (1, -1):
        ck = frustum((0.50 * side, 0.30, 0.52), (0.30, 0.27, 0.40),
                     top=(0.70, 0.80), shift=(-0.06 * side, -0.04),
                     mat='armor_lit', bevel_w=0.05)
        rot(ck, math.radians(-26 * side), 'Y', (0, 0.30, 0.20))
        m.add(ck)

    # Overhanging rear bustle.
    bus = frustum((0, 0.40, -1.02), (0.74, 0.24, 0.34), top=(0.92, 1.12),
                  shift=(0.0, -0.08), mat='armor_dark', bevel_w=0.05)
    m.add(bus)
    for i in range(4):
        m.add(box((-0.48 + i * 0.32, 0.40, -1.34), (0.11, 0.16, 0.04),
                  'shadowed', bevel_w=0.0))
    m.add(box((0, 0.66, -1.02), (0.60, 0.04, 0.26), 'team', bevel_w=0.022))

    # Mantlet and barrel. The barrel is stepped and ends in a big slotted
    # brake, because a plain tapered tube reads as a stick.
    m.add(cyl((0, 0.30, 0.66), 0.31, 0.31, 0.24, 12, 'armor_dark', axis='z',
              bevel_w=0.05))
    m.add(box((0, 0.30, 0.76), (0.36, 0.25, 0.16), 'armor_dark', bevel_w=0.05))
    m.add(cyl((0, 0.30, 0.86), 0.150, 0.128, 0.56, 12, 'steel', axis='z',
              bevel_w=0.02))
    m.add(cyl((0, 0.30, 1.42), 0.172, 0.172, 0.30, 12, 'armor_dark', axis='z',
              bevel_w=0.035))
    m.add(cyl((0, 0.30, 1.72), 0.116, 0.099, 0.90, 12, 'steel', axis='z',
              bevel_w=0.02))
    m.add(cyl((0, 0.30, 2.62), 0.196, 0.196, 0.28, 12, 'dark', axis='z',
              bevel_w=0.03))
    for side in (1, -1):
        for k in range(2):
            m.add(box((0.18 * side, 0.30, 2.66 + k * 0.13),
                      (0.055, 0.080, 0.042), 'shadowed', bevel_w=0.0))

    # Offset cupola with periscope blocks, and a coaxial mount.
    m.add(cyl((0.34, 0.60, -0.30), 0.24, 0.215, 0.14, 10, 'armor_dark',
              bevel_w=0.03))
    m.add(cyl((0.34, 0.74, -0.30), 0.19, 0.19, 0.035, 10, 'dark', bevel_w=0.014))
    for k in range(3):
        a = math.radians(-46 + k * 46)
        m.add(box((0.34 + math.sin(a) * 0.20, 0.70, -0.30 + math.cos(a) * 0.20),
                  (0.04, 0.032, 0.025), 'glow_cyan', bevel_w=0.0))
    m.add(box((-0.40, 0.40, 0.74), (0.065, 0.065, 0.24), 'dark', bevel_w=0.02))
    # Smoke launcher banks on the cheeks.
    for side in (1, -1):
        m.add(box((0.62 * side, 0.54, 0.10), (0.075, 0.105, 0.24), 'dark',
                  bevel_w=0.02))
        for k in range(3):
            m.add(box((0.62 * side, 0.66, -0.06 + k * 0.16),
                      (0.052, 0.022, 0.052), 'shadowed', bevel_w=0.0))
    return m


# ---------------------------------------------------------------- buildings

def build_foundry():
    """Foundry: a stepped tower, not a dome.

    The version this replaces was a drum with a cone on top, which from the
    game's camera reads as a dome -- a silhouette with no direction, no scale
    cue and nothing to distinguish it from the supply depot. This steps in
    three stages from a buttressed base to a narrow control head, hangs a
    cantilevered landing ring off the middle, and runs ducting up one side so
    the outline is not rotationally symmetric.
    """
    m = Model(MESH_FOUNDRY, 'FOUNDRY')

    # Stage one: a low battered plinth with heavy angled buttresses.
    m.add(cyl((0, 0, 0), 4.30, 4.05, 0.80, 8, 'armor_dark', bevel_w=0.09))
    for i in range(8):
        a = face_angle(i, 8)
        bt = frustum((math.cos(a) * 3.95, 0.62, math.sin(a) * 3.95),
                     (0.34, 0.62, 0.52), top=(0.55, 0.62), shift=(0.0, -0.20),
                     mat='armor_dark', bevel_w=0.055)
        orient_radial(bt, a)
        m.add(bt)
    m.add(cyl((0, 0.80, 0), 3.95, 3.80, 0.34, 8, 'dark', bevel_w=0.05))

    # Stage two: the main drum, tapering hard so the eye reads height.
    m.add(cyl((0, 1.14, 0), 3.40, 2.75, 1.80, 8, 'armor', bevel_w=0.08))
    for i in range(8):
        a = face_angle(i, 8)
        rr = cone_radius_at(3.40, 2.75, 1.80, 0.90) + 0.14
        rb = frustum((math.cos(a) * rr, 2.04, math.sin(a) * rr),
                     (0.19, 0.90, 0.30), top=(0.70, 0.72),
                     mat='armor_dark', bevel_w=0.04)
        orient_radial(rb, a)
        m.add(rb)
    band_r = cone_radius_at(3.40, 2.75, 1.80, 0.34) + 0.05
    m.add(cyl((0, 1.48, 0), band_r, band_r, 0.40, 8, 'glow_warm', caps=False,
              bevel_w=0.0))

    # Cantilevered landing ring: the one horizontal in an otherwise vertical
    # shape, which is what gives the tower its scale.
    m.add(ring_flat((0, 2.94, 0), 2.08, 3.02, 24, 'team', thickness=0.22))
    for i in range(8):
        a = face_angle(i, 8)
        br = frustum((math.cos(a) * 2.66, 2.70, math.sin(a) * 2.66),
                     (0.16, 0.22, 0.52), top=(1.0, 0.45), shift=(0.0, 0.22),
                     mat='steel', bevel_w=0.03)
        orient_radial(br, a)
        m.add(br)
    for i in range(4):
        a = face_angle(i * 2, 8)
        m.add(box((math.cos(a) * 2.80, 3.22, math.sin(a) * 2.80),
                  (0.12, 0.10, 0.12), 'glow_dim', bevel_w=0.02))

    # Stage three: the control head, narrow and glazed.
    m.add(cyl((0, 3.16, 0), 1.86, 1.30, 1.34, 8, 'armor', bevel_w=0.06))
    gl_r = cone_radius_at(1.86, 1.30, 1.34, 0.96) + 0.05
    m.add(cyl((0, 4.12, 0), gl_r, gl_r, 0.30, 8, 'glow_cyan', caps=False,
              bevel_w=0.0))
    m.add(cyl((0, 4.50, 0), 1.42, 0.98, 0.30, 8, 'armor_dark', bevel_w=0.05))
    m.add(cyl((0, 4.80, 0), 0.86, 0.70, 0.14, 8, 'dark', bevel_w=0.03))

    # Ducting up one flank, and a mast: both break the rotational symmetry.
    duct_a = face_angle(2, 8)
    dx, dz = math.cos(duct_a), math.sin(duct_a)
    for (yb, hh, rr2) in ((1.10, 2.00, 0.26), (3.10, 1.05, 0.20)):
        pr = cone_radius_at(3.40, 2.75, 1.80, min(yb - 1.14, 1.80)) + 0.26
        m.add(cyl((dx * pr, yb, dz * pr), rr2, rr2, hh, 8, 'steel', bevel_w=0.03))
    m.add(cyl((dx * 3.40, 3.16, dz * 3.40), 0.34, 0.24, 0.30, 8, 'dark',
              bevel_w=0.04))
    m.add(cyl((-dx * 1.30, 4.70, -dz * 1.30), 0.075, 0.045, 0.34, 6, 'steel',
              bevel_w=0.0))
    m.add(box((-dx * 1.30, 5.00, -dz * 1.30), (0.075, 0.05, 0.075),
              'glow_warm', bevel_w=0.0))
    return m



def build_garrison():
    """Garrison: a fortress with an asymmetric watchtower.

    Battered walls -- wider at the base than the top -- read as fortification
    rather than as a shed, and the tower on one corner gives the building an
    orientation. A symmetrical box with a stripe on the roof does not.
    """
    m = Model(MESH_GARRISON, 'GARRISON')

    m.add(box((0, 0.26, 0), (3.20, 0.26, 3.70), 'dark', bevel_w=0.07))
    m.add(frustum((0, 0.62, 0), (3.02, 0.36, 3.52), top=(0.95, 0.96),
                  mat='armor_dark', bevel_w=0.06))

    # Main block, battered.
    main = frustum((0, 1.72, -0.16), (2.78, 1.10, 3.24), top=(0.80, 0.88),
                   shift=(0.0, 0.06), mat='armor', bevel_w=0.08)
    panel(main, 'x', 1, 0.34, -0.08, 'shadowed')
    panel(main, 'x', -1, 0.34, -0.08, 'shadowed')
    m.add(main)

    # Buttress ribs, leaning with the wall.
    for side in (1, -1):
        for zc in (-2.30, -0.90, 0.50, 1.86):
            rb = frustum((2.72 * side, 1.66, zc), (0.26, 1.16, 0.30),
                         top=(0.62, 0.86), shift=(-0.16 * side, 0.0),
                         mat='armor_dark', bevel_w=0.04)
            m.add(rb)

    # Angled roof plates meeting at a ridge, rather than a flat lid.
    for side in (1, -1):
        rp = frustum((1.05 * side, 2.94, -0.10), (1.18, 0.14, 3.00),
                     top=(0.92, 0.94), mat='armor_lit', bevel_w=0.05)
        rot(rp, math.radians(-9 * side), 'Z', (0, 2.88, 0))
        m.add(rp)
    m.add(box((0, 3.10, -0.10), (0.34, 0.11, 3.02), 'team', bevel_w=0.04))

    # Watchtower on one corner. This is the silhouette.
    tx, tz = -1.90, -2.42
    m.add(frustum((tx, 2.30, tz), (0.98, 2.30, 0.98), top=(0.82, 0.82),
                  mat='armor', bevel_w=0.06))
    m.add(frustum((tx, 4.72, tz), (0.92, 0.22, 0.92), top=(1.18, 1.18),
                  mat='armor_dark', bevel_w=0.05))
    m.add(box((tx, 4.44, tz + 0.84), (0.60, 0.22, 0.05), 'glow_amber',
              bevel_w=0.02))
    m.add(box((tx + 0.84, 4.44, tz), (0.05, 0.22, 0.60), 'glow_amber',
              bevel_w=0.02))
    m.add(frustum((tx, 5.04, tz), (0.72, 0.13, 0.72), top=(0.55, 0.55),
                  mat='team', bevel_w=0.04))
    # Vertical team banner down the tower face, on the silhouette edge.
    m.add(box((tx + 0.96, 2.60, tz), (0.05, 1.30, 0.42), 'team', bevel_w=0.025))

    # Armoured bay door in a heavy angled frame.
    frame = frustum((0, 1.32, 3.06), (1.66, 1.20, 0.24), top=(0.86, 1.0),
                    mat='armor_dark', bevel_w=0.055)
    df = face_plate(frame, 'z', 1)
    if df:
        inset(frame, df, 0.16, -0.12, 'dark')
    m.add(frame)
    m.add(box((0, 1.24, 3.22), (1.24, 0.90, 0.05), 'glow_amber', bevel_w=0.02))
    for side in (1, -1):
        m.add(frustum((1.58 * side, 1.34, 3.20), (0.13, 1.20, 0.13),
                      top=(0.7, 0.7), mat='steel', bevel_w=0.025))
    # Blast deflectors flanking the door.
    for side in (1, -1):
        bd = frustum((2.26 * side, 0.72, 3.30), (0.30, 0.72, 0.60),
                     top=(0.45, 0.70), mat='armor_dark', bevel_w=0.045)
        rot(bd, math.radians(11 * side), 'Z', (2.26 * side, 0.0, 3.30))
        m.add(bd)

    m.add(cyl((2.40, 2.96, -2.60), 0.09, 0.05, 1.70, 8, 'steel', bevel_w=0.0))
    m.add(box((2.40, 4.70, -2.60), (0.07, 0.05, 0.07), 'glow_warm', bevel_w=0.0))
    return m



def build_workshop():
    """Workshop: a hangar under a gantry crane.

    The arch alone was the most readable of the old buildings, so it stays.
    What it lacked was anything above the roofline: a straddle crane on rails
    gives the building a tall open structure that is unmistakable from above
    and still reads in silhouette from the side, which no amount of surface
    panelling on a shed will do.
    """
    m = Model(MESH_WORKSHOP, 'WORKSHOP')

    m.add(box((0, 0.26, 0), (3.70, 0.26, 4.20), 'dark', bevel_w=0.07))
    m.add(frustum((0, 0.60, 0), (3.52, 0.34, 4.02), top=(0.96, 0.97),
                  mat='armor_dark', bevel_w=0.055))

    # Arched hangar. Its rim is left unbevelled so the arch keeps its curve.
    m.add(cyl((0, 0.94, -3.42), 2.46, 2.46, 6.90, 14, 'armor', axis='z',
              bevel_w=0.0))
    for zc in (-2.45, -0.85, 0.75, 2.35):
        m.add(tube((0, 0.94, zc), 2.58, 2.42, 0.24, 14, 'armor_dark', axis='z'))
    m.add(frustum((0, 1.10, -3.58), (2.52, 1.50, 0.20), top=(0.86, 1.0),
                  mat='armor_dark', bevel_w=0.055))
    m.add(box((0, 0.90, -3.76), (1.05, 1.00, 0.06), 'dark', bevel_w=0.02))

    # Hangar mouth: heavy angled frame with the lit opening set into it.
    mouth = frustum((0, 1.16, 3.56), (2.34, 1.56, 0.22), top=(0.88, 1.0),
                    mat='armor_dark', bevel_w=0.06)
    mf = face_plate(mouth, 'z', 1)
    if mf:
        inset(mouth, mf, 0.22, -0.12, 'dark')
    m.add(mouth)
    m.add(box((0, 1.00, 3.72), (1.82, 1.20, 0.05), 'glow_amber', bevel_w=0.02))
    for side in (1, -1):
        m.add(frustum((2.20 * side, 1.20, 3.68), (0.20, 1.50, 0.20),
                      top=(0.6, 0.6), mat='steel', bevel_w=0.03))

    # Gantry crane straddling the hangar, offset down the pad so the shape is
    # not mirror-symmetric front to back.
    gz = -0.60
    for side in (1, -1):
        leg = frustum((3.16 * side, 1.92, gz), (0.34, 1.92, 0.30),
                      top=(0.52, 0.70), shift=(-0.22 * side, 0.0),
                      mat='steel', bevel_w=0.04)
        m.add(leg)
        m.add(box((3.16 * side, 0.98, gz), (0.44, 0.13, 0.46), 'dark',
                  bevel_w=0.03))
        # Diagonal brace.
        br = box((2.30 * side, 2.30, gz), (0.78, 0.07, 0.10), 'steel',
                 bevel_w=0.02)
        rot(br, math.radians(26 * side), 'Z', (2.94 * side, 2.30, gz))
        m.add(br)
    m.add(box((0, 3.90, gz), (3.10, 0.17, 0.26), 'armor_dark', bevel_w=0.04))
    m.add(box((0, 4.06, gz), (3.16, 0.09, 0.34), 'team', bevel_w=0.03))
    # Trolley and hook hanging off the beam, deliberately off-centre.
    m.add(box((-1.15, 3.62, gz), (0.36, 0.16, 0.32), 'armor_dark', bevel_w=0.03))
    m.add(cyl((-1.15, 2.90, gz), 0.035, 0.035, 0.72, 6, 'dark', bevel_w=0.0))
    m.add(box((-1.15, 2.78, gz), (0.13, 0.13, 0.13), 'steel', bevel_w=0.02))
    for side in (1, -1):
        m.add(box((3.16 * side, 4.02, gz), (0.11, 0.09, 0.11), 'glow_dim',
                  bevel_w=0.0))

    # Exhaust stacks, canted outward.
    for side in (1, -1):
        st = frustum((2.92 * side, 1.70, -2.80), (0.34, 1.10, 0.34),
                     top=(0.74, 0.74), shift=(0.16 * side, 0.0),
                     mat='steel', bevel_w=0.04)
        m.add(st)
        m.add(tube((3.08 * side, 2.80, -2.80), 0.30, 0.21, 0.22, 10, 'dark',
                   axis='y'))
    # Apron markings.
    for side in (1, -1):
        m.add(box((1.50 * side, 0.54, 3.30), (0.16, 0.03, 0.70), 'glow_dim',
                  bevel_w=0.0))
    return m



def build_bunkhouse():
    """Bunkhouse: a supply silo with radiator fins.

    Squat and round is the hardest silhouette to make distinct, so the fins do
    the work: six plates standing well proud of the drum give the outline
    teeth in plan view, which is the view that matters for a building this
    low, and separate it at a glance from the foundry's round base.
    """
    m = Model(MESH_BUNKHOUSE, 'BUNKHOUSE')

    m.add(cyl((0, 0, 0), 2.00, 1.90, 0.34, 6, 'dark', bevel_w=0.05))
    body = cyl((0, 0.34, 0), 1.76, 1.44, 0.92, 6, 'armor', bevel_w=0.06)
    m.add(body)

    # Radiator fins, canted and reaching past the drum.
    for i in range(6):
        a = face_angle(i, 6)
        fr = cone_radius_at(1.76, 1.44, 0.92, 0.42)
        fin = frustum((math.cos(a) * (fr + 0.42), 0.78, math.sin(a) * (fr + 0.42)),
                      (0.58, 0.60, 0.10), top=(0.72, 0.72), shift=(-0.10, 0.0),
                      mat='armor_dark', bevel_w=0.035)
        orient_radial(fin, a)
        m.add(fin)
        # Fin ribs, cheap and they catch the light along the outer edge.
        for k in (-1, 0, 1):
            rb = box((math.cos(a) * (fr + 0.58), 0.78 + k * 0.30,
                      math.sin(a) * (fr + 0.58)), (0.42, 0.045, 0.055),
                     'shadowed', bevel_w=0.0)
            orient_radial(rb, a)
            m.add(rb)
        m.add(cyl((math.cos(a) * (fr + 0.05), 0.40, math.sin(a) * (fr + 0.05)),
                  0.11, 0.09, 0.80, 6, 'steel', bevel_w=0.0))

    # Lit collar and a stepped cap, so the top is not a flat lid.
    slit_r = cone_radius_at(1.76, 1.44, 0.92, 0.84) + 0.03
    m.add(cyl((0, 1.14, 0), slit_r, slit_r, 0.12, 6, 'glow_cyan', caps=False,
              bevel_w=0.0))
    m.add(ring_flat((0, 1.26, 0), 0.96, 1.54, 18, 'team', thickness=0.10))
    m.add(cyl((0, 1.26, 0), 0.94, 0.78, 0.26, 6, 'armor_dark', bevel_w=0.04))
    m.add(cyl((0, 1.52, 0), 0.62, 0.44, 0.12, 6, 'steel', bevel_w=0.025))
    # Beacon.
    m.add(cyl((0, 1.62, 0), 0.13, 0.13, 0.08, 8, 'dark', bevel_w=0.018))
    m.add(box((0, 1.70, 0), (0.09, 0.05, 0.09), 'glow_warm', bevel_w=0.0))
    return m


def build_ore():
    """Ore seam: a crystal spire, not a bush of cones.

    These sit on the map for the whole match next to everything else, so
    leaving them on the old palette made them the palest objects on screen
    once the units were rebuilt. Shape matters too: one dominant shard with
    smaller ones leaning off it reads far better at distance than six cones
    of similar height fanned out evenly.
    """
    m = Model(MESH_ORE, 'ORE')

    # Rock socket the crystals grow out of. Kept low and wide: a tall base
    # swallows the shards, and the shards are the thing a player looks for.
    m.add(blob((0, 0.02, 0), 0.76, 8, 5, 3, 0.66, 'rock'))
    m.add(blob((0.48, -0.02, -0.32), 0.36, 6, 4, 11, 0.8, 'rock'))

    # One tall shard, then a supporting cast at decreasing height. Each is a
    # tapered prism rather than a cone, so the facets catch light separately.
    shards = [(0.00, 0.02, 1.86, 0.40, 0.05, 0.08),
              (-0.44, -0.20, 1.30, 0.32, 0.22, -0.14),
              (0.40, 0.30, 1.14, 0.29, -0.20, 0.13),
              (0.14, -0.48, 0.86, 0.25, -0.12, -0.26),
              (-0.30, 0.42, 0.72, 0.22, 0.24, 0.14),
              (0.54, -0.12, 0.58, 0.19, -0.28, 0.06)]
    for (sx, sz, h, r, lz, lx) in shards:
        # Six-sided and only lightly tapered: a thin spike reads as an
        # antenna, a chunky prism reads as a crystal. The taper is carried by
        # a second, shorter section so the tip still comes to a point.
        body = cyl((sx, 0.04, sz), r, r * 0.74, h * 0.66, 6, 'crystal',
                   bevel_w=0.0)
        tip = cyl((sx, 0.04 + h * 0.66, sz), r * 0.74, r * 0.10, h * 0.34, 6,
                  'crystal', bevel_w=0.0)
        core = cyl((sx, 0.04 + h * 0.18, sz), r * 0.42, r * 0.30, h * 0.42, 6,
                   'glow_cyan', bevel_w=0.0)
        for part in (body, tip, core):
            mark_flat(part)
            rot(part, lz, 'Z', (sx, 0.04, sz))
            rot(part, lx, 'X', (sx, 0.04, sz))
            m.add(part)
    return m


def build_boulder():
    """Boulder: angular rock.

    The procedural version is a noise-displaced sphere, which reads as a
    potato. Large flat facets with hard edges read as stone, and they also
    catch the cliff texture the renderer assigns this mesh far better than a
    smoothly undulating surface does.
    """
    m = Model(MESH_BOULDER, 'BOULDER')
    m.add(blob((0, 0.18, 0), 0.80, 8, 5, 7, 0.72, 'rock'))
    m.add(blob((0.44, 0.10, -0.30), 0.38, 6, 4, 21, 0.85, 'rock'))
    m.add(blob((-0.36, 0.08, 0.34), 0.31, 6, 4, 33, 0.85, 'rock'))
    return m


# ---------------------------------------------------------------- main
BUILDERS = [
    build_worker, build_trooper, build_mauler_hull, build_mauler_turret,
    build_foundry, build_garrison, build_workshop, build_bunkhouse,
    build_ore, build_boulder,
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
