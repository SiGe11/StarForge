#include "Game.h"
#include "../gfx/RenderTypes.h"
#include <algorithm>

namespace sf {

// name          mesh              bld  neu  rad    hp   spd  turn  rng  dmg   cd   spl  sight bt   cost sup give
const UnitDef kDefs[UT_COUNT] = {
    {"Digger",    MESH_WORKER,       0, 0, 0.70f,   60, 6.6f, 7.0f, 2.0f,  5, 1.0f, 0,   22, 12,  50, 1,  0},
    {"Trooper",   MESH_TROOPER,      0, 0, 0.58f,   55, 6.0f, 9.0f,16.0f,  7, 0.70f,0,   30, 14,  50, 1,  0},
    {"Mauler",    MESH_MAULER_HULL,  0, 0, 1.50f,  180, 4.1f, 2.6f,24.0f, 32, 2.20f,4.5f,34, 30, 150, 3,  0},
    {"Foundry",   MESH_FOUNDRY,      1, 0, 5.00f, 1500, 0, 0,     0,  0, 0,   0,   32, 55, 400, 0, 10},
    {"Garrison",  MESH_GARRISON,     1, 0, 4.30f, 1000, 0, 0,     0,  0, 0,   0,   26, 30, 150, 0,  0},
    {"Workshop",  MESH_WORKSHOP,     1, 0, 5.00f, 1250, 0, 0,     0,  0, 0,   0,   26, 40, 200, 0,  0},
    {"Bunkhouse", MESH_BUNKHOUSE,    1, 0, 2.40f,  400, 0, 0,     0,  0, 0,   0,   20, 20, 100, 0,  8},
    {"Ore Seam",  MESH_ORE,          0, 1, 1.60f,    1, 0, 0,     0,  0, 0,   0,    0,  0,   0, 0,  0},
    {"Boulder",   MESH_BOULDER,      0, 1, 1.10f,    1, 0, 0,     0,  0, 0,   0,    0,  0,   0, 0,  0},
};

static constexpr int   kOrePerTrip = 8;
static constexpr float kHarvestTime     = 1.6f;
static constexpr int   kNodeCapacity    = 1500;

// ---------------------------------------------------------------- lifecycle
EntId Game::spawn(UnitType t, int team, v2 pos, bool complete) {
    const UnitDef& D = kDefs[t];
    int idx;
    if (!freeSlots_.empty()) { idx = freeSlots_.back(); freeSlots_.pop_back(); }
    else { ents.push_back(Entity{}); idx = (int)ents.size() - 1; }

    Entity& e = ents[idx];
    uint32_t g = e.gen + 1;
    e = Entity{};
    e.gen = g;
    e.alive = true;
    e.type = t;
    e.team = team;
    e.pos = pos;
    e.lastPos = pos;
    e.yaw = rng_.range(-PI, PI);
    e.turretYaw = e.yaw;
    e.buildProgress = complete ? 1.0f : 0.0f;
    e.hp = complete ? D.hp : D.hp * 0.08f;
    e.rally = pos + v2{0, D.radius + 4.0f};
    if (t == UT_ORE) e.oreLeft = kNodeCapacity;
    if (D.building || D.neutral) nav.addBlocker(pos, D.radius * 0.92f);
    if (D.building && complete && team < 2) fac[team].supplyCap += D.supplyGive;
    return EntId::make(idx, g);
}

void Game::killEntity(Entity& e) {
    const UnitDef& D = kDefs[e.type];
    if (D.building || D.neutral) nav.removeBlocker(e.pos, D.radius * 0.92f);
    if (e.team < 2) {
        if (D.building && e.buildProgress >= 1.0f) fac[e.team].supplyCap -= D.supplyGive;
        fac[e.team].lost++;
        fac[1 - e.team].killed++;
    }
    e.deathTimer = 0.0f;
    e.order = ORD_IDLE;
    e.path.clear();
    e.vel = {0, 0};
}

void Game::init(uint32_t seed) {
    rng_ = Rng(seed);
    ents.clear(); ents.reserve(4096);
    freeSlots_.clear();
    projectiles.clear(); projectiles.reserve(512);
    particles.clear(); particles.reserve(8192);
    pings.clear();
    winner = -1; time = 0;
    fac[0] = Faction{}; fac[1] = Faction{};
    fac[0].ore = fac[1].ore = 150;

    basePos[0] = v2{0.22f * Terrain::SIZE, 0.26f * Terrain::SIZE};
    basePos[1] = v2{0.78f * Terrain::SIZE, 0.74f * Terrain::SIZE};
    terrain.generate(seed, basePos[0], basePos[1]);
    nav.init(&terrain);

    for (int t = 0; t < 2; t++) {
        vis_[t].assign(VIS * VIS, 0);
        explored_[t].assign(VIS * VIS, 0);
        lastSeen_[t].assign(VIS * VIS, -1.0f);
    }

    for (int t = 0; t < 2; t++) {
        spawn(UT_FOUNDRY, t, basePos[t]);
        // Ore seam arranged in an arc facing away from the map centre.
        v2 toCentre = normalize(v2{Terrain::SIZE * 0.5f, Terrain::SIZE * 0.5f} - basePos[t]);
        float baseAng = std::atan2(-toCentre.y, -toCentre.x);
        for (int i = 0; i < 8; i++) {
            float a = baseAng + (i - 3.5f) * 0.20f;
            v2 p = basePos[t] + v2{std::cos(a), std::sin(a)} * 13.5f;
            p = nav.nearestWalkable(p);
            spawn(UT_ORE, 2, p);
        }
        for (int i = 0; i < 5; i++) {
            float a = TAU * i / 5 + 0.4f;
            v2 p = basePos[t] + v2{std::cos(a), std::sin(a)} * 7.5f;
            EntId h = spawn(UT_WORKER, t, nav.nearestWalkable(p));
            if (Entity* w = get(h)) {
                w->order = ORD_HARVEST;
                w->harvestNode = nearestFreeNode(w->pos, t);
            }
        }
        fac[t].supplyUsed = 5;
    }

    // Neutral expansion clusters plus scattered boulders for cover and colour.
    for (int c = 0; c < 4; c++) {
        for (int tries = 0; tries < 60; tries++) {
            v2 p{rng_.range(35.0f, Terrain::SIZE - 35.0f), rng_.range(35.0f, Terrain::SIZE - 35.0f)};
            if (!nav.walkableWorld(p)) continue;
            if (length(p - basePos[0]) < 60.0f || length(p - basePos[1]) < 60.0f) continue;
            for (int i = 0; i < 6; i++) {
                float a = TAU * i / 6;
                v2 q = nav.nearestWalkable(p + v2{std::cos(a), std::sin(a)} * 5.0f);
                spawn(UT_ORE, 2, q);
            }
            break;
        }
    }
    for (int i = 0; i < 55; i++) {
        for (int tries = 0; tries < 20; tries++) {
            v2 p{rng_.range(10.0f, Terrain::SIZE - 10.0f), rng_.range(10.0f, Terrain::SIZE - 10.0f)};
            if (!nav.walkableWorld(p)) continue;
            if (length(p - basePos[0]) < 30.0f || length(p - basePos[1]) < 30.0f) continue;
            if (pick(p, 3.0f).valid()) continue;
            spawn(UT_BOULDER, 2, p);
            break;
        }
    }
    updateVisibility();
}

// ---------------------------------------------------------------- queries
EntId Game::pick(v2 p, float extra) const {
    EntId best; float bestScore = 1e30f;
    for (size_t i = 0; i < ents.size(); i++) {
        const Entity& e = ents[i];
        if (!e.alive || e.deathTimer >= 0) continue;
        const UnitDef& D = kDefs[e.type];
        float d = length(e.pos - p);
        if (d > D.radius + extra) continue;
        // Prefer the tightest fit so a unit standing on a building still wins.
        float score = d - D.radius;
        if (score < bestScore) { bestScore = score; best = EntId::make((int)i, e.gen); }
    }
    return best;
}

EntId Game::nearestEnemy(const Entity& e, float radius) const {
    EntId best; float bestD = radius;
    for (size_t i = 0; i < ents.size(); i++) {
        const Entity& o = ents[i];
        if (!o.alive || o.deathTimer >= 0 || o.team == e.team || o.team == 2) continue;
        float d = length(o.pos - e.pos);
        if (d < bestD) { bestD = d; best = EntId::make((int)i, o.gen); }
    }
    return best;
}

EntId Game::nearestDropoff(const Entity& e) const {
    EntId best; float bestD = 1e30f;
    for (size_t i = 0; i < ents.size(); i++) {
        const Entity& o = ents[i];
        if (!o.alive || o.deathTimer >= 0 || o.team != e.team) continue;
        if (o.type != UT_FOUNDRY || o.buildProgress < 1.0f) continue;
        float d = length(o.pos - e.pos);
        if (d < bestD) { bestD = d; best = EntId::make((int)i, o.gen); }
    }
    return best;
}

EntId Game::nearestFreeNode(v2 near, int team) const {
    // Bias toward nodes with fewer assigned harvesters so the line spreads out.
    int load[64] = {0};
    std::vector<std::pair<float, int>> cand;
    for (size_t i = 0; i < ents.size(); i++) {
        const Entity& o = ents[i];
        if (!o.alive || o.deathTimer >= 0 || o.type != UT_ORE || o.oreLeft <= 0) continue;
        float d = length(o.pos - near);
        if (d < 40.0f) cand.push_back({d, (int)i});
    }
    if (cand.empty()) {
        for (size_t i = 0; i < ents.size(); i++) {
            const Entity& o = ents[i];
            if (!o.alive || o.type != UT_ORE || o.oreLeft <= 0) continue;
            cand.push_back({length(o.pos - near), (int)i});
        }
    }
    if (cand.empty()) return EntId{};
    std::sort(cand.begin(), cand.end());
    if (cand.size() > 64) cand.resize(64);
    for (const Entity& w : ents) {
        if (!w.alive || w.team != team || w.type != UT_WORKER) continue;
        if (!w.harvestNode.valid()) continue;
        for (size_t k = 0; k < cand.size(); k++)
            if (cand[k].second == w.harvestNode.index()) load[k]++;
    }
    int bestK = 0; float bestScore = 1e30f;
    for (size_t k = 0; k < cand.size(); k++) {
        float score = cand[k].first + load[k] * 6.0f;
        if (score < bestScore) { bestScore = score; bestK = (int)k; }
    }
    return EntId::make(cand[bestK].second, ents[cand[bestK].second].gen);
}

bool Game::canPlace(UnitType what, v2 where, EntId ignore) const {
    const UnitDef& D = kDefs[what];
    float r = D.radius;
    if (where.x < r + 2 || where.y < r + 2 ||
        where.x > Terrain::SIZE - r - 2 || where.y > Terrain::SIZE - r - 2) return false;
    // Footprint must be flat, passable ground.
    for (float a = 0; a < TAU; a += TAU / 12.0f)
        for (float rr = 0; rr <= r; rr += Terrain::CELL * 0.7f) {
            v2 p = where + v2{std::cos(a), std::sin(a)} * rr;
            if (!terrain.passableWorld(p.x, p.y)) return false;
        }
    for (size_t i = 0; i < ents.size(); i++) {
        const Entity& e = ents[i];
        if (!e.alive || e.deathTimer >= 0) continue;
        if (ignore.valid() && (int)i == ignore.index()) continue;
        const UnitDef& OD = kDefs[e.type];
        float need = r + OD.radius + (OD.building || OD.neutral ? 1.0f : -0.35f);
        if (length(e.pos - where) < need) return false;
    }
    return true;
}

// ---------------------------------------------------------------- commands
void Game::repath(Entity& e, v2 dest) {
    e.path.clear();
    e.pathIdx = 0;
    nav.findPath(e.pos, dest, e.path);
    e.repathTimer = rng_.range(1.4f, 2.2f);
}

void Game::cmdMove(const std::vector<EntId>& sel, v2 dest, bool attackMove) {
    // Spread the destination over a ring so a group does not stack on one point.
    int n = 0;
    for (EntId h : sel) { const Entity* e = get(h); if (e && !kDefs[e->type].building) n++; }
    int i = 0;
    float spread = std::sqrt((float)std::max(1, n)) * 1.15f;
    for (EntId h : sel) {
        Entity* e = get(h);
        if (!e || kDefs[e->type].building || e->deathTimer >= 0) continue;
        v2 goal = dest;
        if (n > 1) {
            float a = TAU * i / n + 0.6f;
            float rr = spread * (0.4f + 0.6f * std::sqrt((float)(i + 1) / n));
            goal = nav.nearestWalkable(dest + v2{std::cos(a), std::sin(a)} * rr);
        }
        e->order = attackMove ? ORD_ATTACKMOVE : ORD_MOVE;
        e->orderPos = goal;
        e->target = EntId{};
        e->harvestNode = EntId{};
        e->buildTarget = EntId{};
        repath(*e, goal);
        i++;
    }
}

void Game::cmdAttack(const std::vector<EntId>& sel, EntId target) {
    for (EntId h : sel) {
        Entity* e = get(h);
        if (!e || e->deathTimer >= 0) continue;
        if (kDefs[e->type].building) continue;
        e->order = ORD_ATTACK;
        e->target = target;
        e->harvestNode = EntId{};
        if (const Entity* t = get(target)) repath(*e, t->pos);
    }
}

void Game::cmdHarvest(const std::vector<EntId>& sel, EntId node) {
    for (EntId h : sel) {
        Entity* e = get(h);
        if (!e || e->type != UT_WORKER || e->deathTimer >= 0) continue;
        e->order = ORD_HARVEST;
        e->harvestNode = node;
        e->target = EntId{};
        e->buildTarget = EntId{};
        if (const Entity* n = get(node)) repath(*e, n->pos);
    }
}

void Game::cmdStop(const std::vector<EntId>& sel) {
    for (EntId h : sel) {
        Entity* e = get(h);
        if (!e || e->deathTimer >= 0) continue;
        e->order = ORD_IDLE; e->path.clear(); e->target = EntId{};
        e->harvestNode = EntId{}; e->buildTarget = EntId{}; e->vel = {0, 0};
        e->queue.clear();
    }
}

void Game::cmdHold(const std::vector<EntId>& sel) {
    for (EntId h : sel) {
        Entity* e = get(h);
        if (!e || e->deathTimer >= 0 || kDefs[e->type].building) continue;
        e->order = ORD_HOLD; e->path.clear(); e->vel = {0, 0};
    }
}

void Game::cmdRally(const std::vector<EntId>& sel, v2 dest) {
    for (EntId h : sel) {
        Entity* e = get(h);
        if (!e || !kDefs[e->type].building) continue;
        e->rally = dest; e->rallySet = true;
    }
}

bool Game::cmdBuild(const std::vector<EntId>& sel, UnitType what, v2 where) {
    const UnitDef& D = kDefs[what];
    Entity* worker = nullptr; EntId wh;
    for (EntId h : sel) {
        Entity* e = get(h);
        if (e && e->type == UT_WORKER && e->deathTimer < 0) { worker = e; wh = h; break; }
    }
    if (!worker) return false;
    int team = worker->team;
    if (fac[team].ore < D.cost) return false;
    if (!canPlace(what, where)) return false;

    fac[team].ore -= D.cost;
    EntId bh = spawn(what, team, where, false);
    worker = get(wh);                        // spawn() may have reallocated
    if (!worker) return false;
    worker->order = ORD_BUILD;
    worker->buildTarget = bh;
    worker->harvestNode = EntId{};
    repath(*worker, where);
    return true;
}

bool Game::cmdTrain(EntId building, UnitType what) {
    Entity* b = get(building);
    if (!b || !kDefs[b->type].building || b->buildProgress < 1.0f) return false;
    const UnitDef& D = kDefs[what];
    int team = b->team;
    if (fac[team].ore < D.cost) return false;
    if (fac[team].supplyUsed + D.supplyCost > fac[team].supplyCap) return false;
    if (b->queue.size() >= 5) return false;
    fac[team].ore -= D.cost;
    // Reserve supply immediately so a queue cannot oversubscribe the cap.
    fac[team].supplyUsed += D.supplyCost;
    if (b->queue.empty()) b->queueTimer = D.buildTime;
    b->queue.push_back(what);
    return true;
}

void Game::cmdSmart(const std::vector<EntId>& sel, v2 worldPos, EntId hovered) {
    if (const Entity* h = get(hovered)) {
        if (h->team == 2 && h->type == UT_ORE) { cmdHarvest(sel, hovered); 
            // Non-workers in the selection should still move there.
            std::vector<EntId> rest;
            for (EntId s : sel) { const Entity* e = get(s); if (e && e->type != UT_WORKER) rest.push_back(s); }
            if (!rest.empty()) cmdMove(rest, worldPos, false);
            return;
        }
        if (h->team != 2) {
            bool enemy = false;
            for (EntId s : sel) { const Entity* e = get(s); if (e && e->team != h->team) { enemy = true; break; } }
            if (enemy) { cmdAttack(sel, hovered); return; }
        }
    }
    cmdMove(sel, worldPos, false);
}

// ---------------------------------------------------------------- combat
void Game::applyDamage(Entity& e, float dmg, v2 from) {
    if (e.deathTimer >= 0) return;
    e.hp -= dmg;
    e.damageFlash = 1.0f;
    if (e.team == 0) {
        bool near = false;
        for (const auto& p : pings) if (length(p.pos - e.pos) < 25.0f) { near = true; break; }
        if (!near) pings.push_back({e.pos, 0.0f, 0});
    }
    // Idle defenders retaliate against whatever just hit them.
    if (e.order == ORD_IDLE && !kDefs[e.type].building && kDefs[e.type].range > 0) {
        EntId t = nearestEnemy(e, kDefs[e.type].sight);
        if (t.valid()) { e.order = ORD_ATTACK; e.target = t; }
    }
    if (e.hp <= 0) {
        const UnitDef& D = kDefs[e.type];
        float y = groundY(e.pos);
        explosion(v3{e.pos.x, y + D.radius * 0.5f, e.pos.y},
                  D.building ? 2.6f : 1.0f,
                  D.building ? v3{1.0f, 0.6f, 0.25f} : v3{1.0f, 0.55f, 0.2f});
        killEntity(e);
    }
}

void Game::fire(Entity& e, Entity& target) {
    const UnitDef& D = kDefs[e.type];
    e.cooldown = D.cooldown;
    float y = groundY(e.pos);
    bool mauler = (e.type == UT_MAULER);
    float muzzleH = mauler ? 1.75f : (e.type == UT_TROOPER ? 1.25f : 0.8f);
    float fwd = mauler ? 3.1f : 0.7f;
    float ang = mauler ? e.turretYaw : e.yaw;
    v3 muzzle{e.pos.x + std::sin(ang) * fwd, y + muzzleH, e.pos.y + std::cos(ang) * fwd};

    float ty = groundY(target.pos) + kDefs[target.type].radius * 0.6f;
    v3 aim{target.pos.x, ty, target.pos.y};

    Projectile p{};
    p.alive = true;
    p.pos = muzzle;
    p.target = EntId::make((int)(&target - ents.data()), target.gen);
    p.aimPos = target.pos;
    p.dmg = D.dmg;
    p.splash = D.splash;
    p.team = e.team;
    if (mauler) {
        // Ballistic arc: solve the vertical velocity for a fixed flight time.
        p.kind = 1;
        float dist = length(aim - muzzle);
        float t = clampf(dist / 55.0f, 0.25f, 2.0f);
        p.life = t;
        v3 d = aim - muzzle;
        p.vel = v3{d.x / t, d.y / t + 0.5f * 42.0f * t, d.z / t};
    } else {
        p.kind = 0;
        p.life = 2.0f;
        p.vel = normalize(aim - muzzle) * 115.0f;
    }
    projectiles.push_back(p);

    // Muzzle flash and smoke.
    v3 dir = normalize(v3{std::sin(ang), 0, std::cos(ang)});
    Particle f{};
    f.alive = true; f.pos = muzzle + dir * 0.2f; f.vel = dir * 3.0f;
    f.life = 0; f.maxLife = mauler ? 0.13f : 0.06f;
    f.size0 = mauler ? 1.5f : 0.5f; f.size1 = mauler ? 0.5f : 0.15f;
    f.col0 = v4{2.6f, 1.7f, 0.8f, 1.0f}; f.col1 = v4{1.2f, 0.4f, 0.1f, 0.0f};
    f.kind = 1;
    spawnParticle(f);
    if (mauler) {
        for (int i = 0; i < 6; i++) {
            Particle s{};
            s.alive = true; s.pos = muzzle;
            s.vel = dir * rng_.range(4.0f, 12.0f) + v3{rng_.range(-2.f,2.f), rng_.range(0.f,2.f), rng_.range(-2.f,2.f)};
            s.life = 0; s.maxLife = rng_.range(0.4f, 0.8f);
            s.size0 = 0.7f; s.size1 = 2.4f;
            s.col0 = v4{0.55f, 0.5f, 0.45f, 0.55f}; s.col1 = v4{0.3f, 0.29f, 0.28f, 0.0f};
            s.drag = 1.8f; s.kind = 0;
            spawnParticle(s);
        }
    }
}

void Game::updateCombat(Entity& e, float dt) {
    const UnitDef& D = kDefs[e.type];
    if (D.range <= 0 || e.buildProgress < 1.0f) return;

    Entity* t = get(e.target);
    if (!t || t->deathTimer >= 0 || t->team == e.team) {
        e.target = EntId{};
        t = nullptr;
        // Auto-acquire unless explicitly told to sit still without a target.
        if (e.order != ORD_MOVE) {
            EntId nt = nearestEnemy(e, D.sight);
            if (nt.valid()) { e.target = nt; t = get(nt); }
        }
        if (e.order == ORD_ATTACK && !e.target.valid()) e.order = ORD_IDLE;
    }
    if (!t) return;

    float d = length(t->pos - e.pos) - kDefs[t->type].radius;
    if (d <= D.range) {
        float want = std::atan2(t->pos.x - e.pos.x, t->pos.y - e.pos.y);
        if (e.type == UT_MAULER) {
            e.turretYaw = approachAngle(e.turretYaw, want, 2.6f * dt);
            if (std::fabs(wrapAngle(want - e.turretYaw)) < 0.10f && e.cooldown <= 0) fire(e, *t);
        } else {
            e.yaw = approachAngle(e.yaw, want, D.turn * dt);
            if (std::fabs(wrapAngle(want - e.yaw)) < 0.35f && e.cooldown <= 0) fire(e, *t);
        }
        if (e.order == ORD_ATTACK || e.order == ORD_ATTACKMOVE) { e.path.clear(); e.vel = {0, 0}; }
    }
}

// ---------------------------------------------------------------- economy
void Game::updateHarvest(Entity& e, float dt) {
    if (e.order == ORD_HARVEST) {
        Entity* node = get(e.harvestNode);
        if (!node || node->oreLeft <= 0) {
            e.harvestNode = nearestFreeNode(e.pos, e.team);
            if (!e.harvestNode.valid()) { e.order = ORD_IDLE; return; }
            node = get(e.harvestNode);
            if (node) repath(e, node->pos);
            return;
        }
        float d = length(node->pos - e.pos);
        if (d <= kDefs[UT_ORE].radius + kDefs[UT_WORKER].radius + 0.9f) {
            e.path.clear(); e.vel = {0, 0};
            e.yaw = approachAngle(e.yaw, std::atan2(node->pos.x - e.pos.x, node->pos.y - e.pos.y), 8.0f * dt);
            e.harvestTimer += dt;
            if (rng_.f01() < dt * 22.0f) {
                Particle s{};
                float y = groundY(node->pos);
                s.alive = true;
                s.pos = v3{node->pos.x, y + 1.0f, node->pos.y} +
                        v3{rng_.range(-0.8f,0.8f), rng_.range(0.f,1.0f), rng_.range(-0.8f,0.8f)};
                s.vel = v3{rng_.range(-1.f,1.f), rng_.range(1.5f,4.f), rng_.range(-1.f,1.f)};
                s.life = 0; s.maxLife = rng_.range(0.3f, 0.6f);
                s.size0 = 0.22f; s.size1 = 0.02f;
                s.col0 = v4{0.5f, 1.6f, 2.2f, 1.0f}; s.col1 = v4{0.2f, 0.6f, 0.9f, 0.0f};
                s.gravity = -6.0f; s.kind = 1;
                spawnParticle(s);
            }
            if (e.harvestTimer >= kHarvestTime) {
                e.harvestTimer = 0;
                int take = std::min(kOrePerTrip, node->oreLeft);
                node->oreLeft -= take;
                e.carrying = take;
                if (node->oreLeft <= 0) killEntity(*node);
                e.order = ORD_RETURN;
                EntId dh = nearestDropoff(e);
                if (const Entity* dp = get(dh)) repath(e, dp->pos);
            }
        } else if (e.path.empty() && (e.repathTimer -= dt) <= 0) {
            repath(e, node->pos);
        }
    } else if (e.order == ORD_RETURN) {
        EntId dh = nearestDropoff(e);
        Entity* dp = get(dh);
        if (!dp) { e.order = ORD_HARVEST; return; }
        float d = length(dp->pos - e.pos);
        if (d <= kDefs[UT_FOUNDRY].radius + kDefs[UT_WORKER].radius + 1.2f) {
            fac[e.team].ore += e.carrying;
            e.carrying = 0;
            e.order = ORD_HARVEST;
            Entity* node = get(e.harvestNode);
            if (!node || node->oreLeft <= 0) e.harvestNode = nearestFreeNode(e.pos, e.team);
            if (Entity* n2 = get(e.harvestNode)) repath(e, n2->pos);
        } else if (e.path.empty() && (e.repathTimer -= dt) <= 0) {
            repath(e, dp->pos);
        }
    }
}

void Game::updateProduction(Entity& e, float dt) {
    if (e.buildProgress < 1.0f || e.queue.empty()) return;
    e.queueTimer -= dt;
    if (e.queueTimer > 0) return;
    UnitType what = e.queue.front();
    const UnitDef& D = kDefs[what];
    // Find a free spot on the building's perimeter to place the new unit.
    v2 spot = e.pos; bool ok = false;
    for (int i = 0; i < 24 && !ok; i++) {
        float a = TAU * i / 24.0f + e.yaw;
        v2 p = e.pos + v2{std::cos(a), std::sin(a)} * (kDefs[e.type].radius + D.radius + 1.4f);
        if (nav.walkableWorld(p) && !pick(p, D.radius * 0.5f).valid()) { spot = p; ok = true; }
    }
    if (!ok) { e.queueTimer = 0.5f; return; }   // blocked in: retry shortly

    e.queue.erase(e.queue.begin());
    if (!e.queue.empty()) e.queueTimer = kDefs[e.queue.front()].buildTime;

    int team = e.team;
    v2 rally = e.rallySet ? e.rally : (e.pos + normalize(basePos[1 - team] - e.pos) * 8.0f);
    bool wantHarvest = (what == UT_WORKER && !e.rallySet);
    EntId nh = spawn(what, team, spot);            // may reallocate `ents`
    if (Entity* n = get(nh)) {
        n->yaw = std::atan2(spot.x - basePos[team].x, spot.y - basePos[team].y);
        if (wantHarvest) {
            n->order = ORD_HARVEST;
            n->harvestNode = nearestFreeNode(n->pos, team);
            if (Entity* node = get(n->harvestNode)) repath(*n, node->pos);
        } else {
            n->order = ORD_MOVE;
            n->orderPos = rally;
            repath(*n, rally);
        }
    }
}

// ---------------------------------------------------------------- unit tick
void Game::updateUnit(Entity& e, int idx, float dt) {
    const UnitDef& D = kDefs[e.type];
    e.damageFlash = std::max(0.0f, e.damageFlash - dt * 6.5f);
    e.cooldown = std::max(0.0f, e.cooldown - dt);

    if (D.building) {
        if (e.buildProgress < 1.0f) return;      // assembled by its builder
        // A badly damaged structure burns. This is the clearest long-range cue
        // that a base is under attack, and it reads from across the map.
        float frac = e.hp / D.hp;
        if (frac < 0.55f) {
            e.smokeTimer -= dt;
            if (e.smokeTimer <= 0.0f) {
                e.smokeTimer = 0.30f + frac * 0.75f;   // burns harder as it fails
                Particle m{};
                m.alive = true;
                float r = D.radius * 0.55f;
                m.pos = v3{e.pos.x + rng_.range(-r, r),
                           groundY(e.pos) + D.radius * 0.9f,
                           e.pos.y + rng_.range(-r, r)};
                m.vel = v3{rng_.range(-0.4f, 0.4f), rng_.range(1.6f, 2.6f), rng_.range(-0.4f, 0.4f)};
                m.life = 0; m.maxLife = rng_.range(2.6f, 3.8f);
                m.size0 = D.radius * 0.9f; m.size1 = D.radius * 2.6f;
                m.col0 = v4{0.34f, 0.32f, 0.31f, 0.62f};
                m.col1 = v4{0.26f, 0.26f, 0.27f, 0.00f};
                m.drag = 0.8f;
                m.kind = 6;
                spawnParticle(m);
            }
        }
        updateProduction(e, dt);
        return;
    }
    if (D.neutral) return;

    // Worker assembling a structure.
    if (e.order == ORD_BUILD) {
        Entity* b = get(e.buildTarget);
        if (!b) { e.order = ORD_IDLE; e.buildTarget = EntId{}; }
        else {
            float need = kDefs[b->type].radius + D.radius + 1.0f;
            if (length(b->pos - e.pos) <= need) {
                e.path.clear(); e.vel = {0, 0};
                e.yaw = approachAngle(e.yaw, std::atan2(b->pos.x - e.pos.x, b->pos.y - e.pos.y), 8.0f * dt);
                float rate = dt / std::max(0.5f, kDefs[b->type].buildTime);
                b->buildProgress = std::min(1.0f, b->buildProgress + rate);
                b->hp = kDefs[b->type].hp * (0.08f + 0.92f * b->buildProgress);
                if (rng_.f01() < dt * 30.0f) {
                    Particle s{};
                    float by = groundY(b->pos);
                    s.alive = true;
                    s.pos = v3{b->pos.x, by + kDefs[b->type].radius * b->buildProgress, b->pos.y} +
                            v3{rng_.range(-2.f,2.f), rng_.range(0.f,1.f), rng_.range(-2.f,2.f)};
                    s.vel = v3{rng_.range(-1.f,1.f), rng_.range(2.f,5.f), rng_.range(-1.f,1.f)};
                    s.life = 0; s.maxLife = rng_.range(0.25f, 0.5f);
                    s.size0 = 0.3f; s.size1 = 0.05f;
                    s.col0 = v4{1.8f, 1.4f, 0.5f, 1.0f}; s.col1 = v4{0.8f, 0.4f, 0.1f, 0.0f};
                    s.gravity = -4.0f; s.kind = 1;
                    spawnParticle(s);
                }
                if (b->buildProgress >= 1.0f) {
                    fac[b->team].supplyCap += kDefs[b->type].supplyGive;
                    e.order = ORD_HARVEST;
                    e.buildTarget = EntId{};
                    e.harvestNode = nearestFreeNode(e.pos, e.team);
                    if (Entity* n = get(e.harvestNode)) repath(e, n->pos);
                }
            } else if (e.path.empty() && (e.repathTimer -= dt) <= 0) {
                repath(e, b->pos);
            }
        }
    }

    if (e.order == ORD_HARVEST || e.order == ORD_RETURN) updateHarvest(e, dt);
    updateCombat(e, dt);

    // Close on the target for explicit attack orders AND for attack-move once a
    // target has been acquired -- otherwise an attack-moving unit that spots a
    // distant enemy would stand still forever, never entering weapons range.
    if ((e.order == ORD_ATTACK || e.order == ORD_ATTACKMOVE) && e.target.valid()) {
        if (Entity* t = get(e.target)) {
            float d = length(t->pos - e.pos) - kDefs[t->type].radius;
            if (d > D.range * 0.92f && (e.repathTimer -= dt) <= 0.0f) {
                repath(e, t->pos);
                e.repathTimer = rng_.range(0.45f, 0.75f);   // chases retarget faster
            }
        }
    }
    if (e.order == ORD_ATTACKMOVE && !e.target.valid() && e.path.empty()) {
        if (length(e.orderPos - e.pos) > 2.0f && (e.repathTimer -= dt) <= 0) repath(e, e.orderPos);
    }

    // --- path following ---------------------------------------------------
    v2 desired{0, 0};
    if (!e.path.empty() && e.pathIdx < e.path.size()) {
        v2 wp = e.path[e.pathIdx];
        v2 d = wp - e.pos;
        float dist = length(d);
        bool last = (e.pathIdx + 1 == e.path.size());
        float arrive = last ? 0.6f : Terrain::CELL * 0.75f;
        if (dist < arrive) {
            e.pathIdx++;
            if (e.pathIdx >= e.path.size()) {
                e.path.clear();
                if (e.order == ORD_MOVE || e.order == ORD_ATTACKMOVE) e.order = ORD_IDLE;
            }
        } else {
            desired = d / dist;
            // Ease off near the final waypoint so units settle instead of jitter.
            if (last && dist < 2.5f) desired *= (dist / 2.5f);
        }
    }

    float speed = D.speed;
    v2 target = desired * speed;
    float accel = speed * 4.5f;
    v2 dv = target - e.vel;
    float dvl = length(dv);
    if (dvl > accel * dt) dv = dv * (accel * dt / dvl);
    e.vel += dv;

    float vl = length(e.vel);
    if (vl > 0.02f) {
        v2 np = e.pos + e.vel * dt;
        if (nav.walkableWorld(np)) {
            e.pos = np;
        } else {
            // Slide along whichever axis is still open.
            v2 nx{np.x, e.pos.y}, nz{e.pos.x, np.y};
            if (nav.walkableWorld(nx)) { e.pos = nx; e.vel.y *= 0.4f; }
            else if (nav.walkableWorld(nz)) { e.pos = nz; e.vel.x *= 0.4f; }
            else { e.vel *= 0.2f; }
            if ((e.stuckTimer += dt) > 0.9f) { e.stuckTimer = 0; if (!e.path.empty()) repath(e, e.path.back()); }
        }
        float want = std::atan2(e.vel.x, e.vel.y);
        e.yaw = approachAngle(e.yaw, want, D.turn * dt);
        e.bob += vl * dt * 3.2f;
        if (e.type != UT_MAULER) e.turretYaw = e.yaw;
    } else {
        e.vel *= std::max(0.0f, 1.0f - dt * 8.0f);
        if (e.type == UT_MAULER && !e.target.valid())
            e.turretYaw = approachAngle(e.turretYaw, e.yaw, 1.2f * dt);
    }
    e.pos.x = clampf(e.pos.x, 1.0f, Terrain::SIZE - 1.0f);
    e.pos.y = clampf(e.pos.y, 1.0f, Terrain::SIZE - 1.0f);
}

// Push overlapping units apart so formations spread instead of interpenetrating.
void Game::separate(float dt) {
    const int n = (int)ents.size();
    for (int i = 0; i < n; i++) {
        Entity& a = ents[i];
        if (!a.alive || a.deathTimer >= 0) continue;
        const UnitDef& DA = kDefs[a.type];
        if (DA.building || DA.neutral) continue;
        v2 push{0, 0};
        for (int j = 0; j < n; j++) {
            if (i == j) continue;
            Entity& b = ents[j];
            if (!b.alive || b.deathTimer >= 0) continue;
            const UnitDef& DB = kDefs[b.type];
            v2 d = a.pos - b.pos;
            float min = DA.radius + DB.radius;
            float l2 = length2(d);
            if (l2 >= min * min || l2 < 1e-6f) continue;
            float l = std::sqrt(l2);
            // Static obstacles push at full strength; units share the correction.
            float w = (DB.building || DB.neutral) ? 1.0f : 0.5f;
            push += (d / l) * ((min - l) * w);
        }
        if (length2(push) > 1e-8f) {
            v2 np = a.pos + push * std::min(1.0f, dt * 14.0f);
            if (nav.walkableWorld(np)) a.pos = np;
        }
    }
}

// ---------------------------------------------------------------- effects
void Game::spawnParticle(const Particle& p) {
    Particle q = p;
    // Smoke and spark kinds pick a sprite from the 4x4 particle sheet. Assigning
    // it here rather than at each of the dozen call sites keeps every effect --
    // muzzle flashes, impacts, harvest dust, engine trails, debris -- varied
    // without any of them having to remember to ask for it.
    if (q.kind < 1.5f) q.param = rng_.f01();
    if (particles.size() < 8000) { particles.push_back(q); return; }
    // Pool is full: recycle the first dead slot, otherwise drop the request.
    for (auto& slot : particles) if (!slot.alive) { slot = q; return; }
}

void Game::explosion(v3 at, float scale, v3 tint) {
    // Kind 4 plays the 16-frame explosion sheet; it carries the fireball and its
    // own smoke, so fewer procedural puffs are needed underneath it.
    Particle f{};
    f.alive = true; f.pos = at; f.life = 0; f.maxLife = 0.85f * std::sqrt(scale);
    f.size0 = 2.1f * scale; f.size1 = 4.0f * scale;
    f.col0 = v4{1.15f, 1.10f, 1.05f, 1.0f};
    f.col1 = v4{0.85f, 0.80f, 0.78f, 0.85f};
    f.vel = v3{0.0f, 0.9f * scale, 0.0f};
    f.kind = 4;
    spawnParticle(f);

    Particle w{};
    w.alive = true; w.pos = v3{at.x, at.y - scale * 0.4f, at.z};
    w.life = 0; w.maxLife = 0.45f * scale;
    w.size0 = 0.8f * scale; w.size1 = 4.2f * scale;
    w.col0 = v4{tint.x * 0.9f, tint.y * 0.6f, tint.z * 0.35f, 0.40f};
    w.col1 = v4{tint.x * 0.3f, tint.y * 0.15f, 0.05f, 0.0f};
    w.kind = 3;
    spawnParticle(w);

    int puffs = (int)(4 * scale);
    for (int i = 0; i < puffs; i++) {
        Particle s{};
        s.alive = true;
        s.pos = at + v3{rng_.range(-1.f,1.f), rng_.range(-0.4f,1.f), rng_.range(-1.f,1.f)} * scale;
        s.vel = v3{rng_.range(-4.f,4.f), rng_.range(1.f,7.f), rng_.range(-4.f,4.f)} * scale;
        s.life = 0; s.maxLife = rng_.range(0.7f, 1.6f) * scale;
        s.size0 = 0.9f * scale; s.size1 = 3.0f * scale;
        s.col0 = v4{0.55f, 0.50f, 0.45f, 0.45f};
        s.col1 = v4{0.20f, 0.19f, 0.18f, 0.0f};
        s.drag = 1.5f; s.gravity = 1.2f; s.kind = 0;
        s.spin = rng_.range(-2.f, 2.f);
        spawnParticle(s);
    }
    // Ground scorch. A burn mark is the only thing a fight leaves behind, so it
    // outlives everything else here by two orders of magnitude; the alpha ramp
    // from col0 to col1 is what fades it back out over the next minute.
    float gy = groundY(v2{at.x, at.z});
    if (at.y - gy < 3.5f * scale) {
        Particle d{};
        d.alive = true;
        d.pos = v3{at.x, gy + 0.10f, at.z};
        d.life = 0; d.maxLife = 55.0f;
        d.size0 = d.size1 = 2.0f * scale;
        d.col0 = v4{0.045f, 0.038f, 0.032f, 0.80f};
        d.col1 = v4{0.060f, 0.052f, 0.045f, 0.00f};
        d.rot = rng_.range(0.0f, 6.283f);
        d.param = rng_.f01();          // which of the four marks on the sheet
        d.kind = 5;
        spawnParticle(d);
    }

    // Smoke plumes. Kind 6 is alpha-blended rather than additive -- smoke has to
    // be able to darken the sky behind it, which an additive sprite cannot.
    int plumes = (scale > 1.6f) ? 2 : 1;
    for (int i = 0; i < plumes; i++) {
        Particle m{};
        m.alive = true;
        m.pos = at + v3{rng_.range(-0.6f, 0.6f), 0.2f, rng_.range(-0.6f, 0.6f)} * scale;
        m.vel = v3{rng_.range(-0.5f, 0.5f), rng_.range(1.4f, 2.4f), rng_.range(-0.5f, 0.5f)};
        m.life = 0; m.maxLife = rng_.range(2.1f, 3.0f);
        m.size0 = 1.6f * scale; m.size1 = 5.0f * scale;
        m.col0 = v4{0.42f, 0.40f, 0.39f, 0.66f};
        m.col1 = v4{0.28f, 0.28f, 0.29f, 0.00f};
        m.drag = 0.9f;
        m.kind = 6;
        spawnParticle(m);
    }

    int sparks = (int)(10 * scale);
    for (int i = 0; i < sparks; i++) {
        Particle s{};
        s.alive = true; s.pos = at;
        s.vel = v3{rng_.range(-1.f,1.f), rng_.range(-0.2f,1.f), rng_.range(-1.f,1.f)} * (14.0f * scale);
        s.life = 0; s.maxLife = rng_.range(0.3f, 0.9f);
        s.size0 = 0.34f * scale; s.size1 = 0.04f;
        s.col0 = v4{tint.x * 2.6f, tint.y * 1.8f, tint.z * 0.7f, 1.0f};
        s.col1 = v4{tint.x * 0.8f, tint.y * 0.2f, 0.0f, 0.0f};
        s.gravity = -22.0f; s.drag = 0.7f; s.kind = 1;
        spawnParticle(s);
    }
}

void Game::updateProjectiles(float dt) {
    for (auto& p : projectiles) {
        if (!p.alive) continue;
        p.life -= dt;
        Entity* t = get(p.target);

        if (p.kind == 0) {
            // Gauss tracer: lightly homing so it does not miss a moving target.
            if (t && t->deathTimer < 0) {
                v3 tp{t->pos.x, groundY(t->pos) + kDefs[t->type].radius * 0.6f, t->pos.y};
                v3 want = normalize(tp - p.pos) * length(p.vel);
                p.vel = normalize(lerp(p.vel, want, std::min(1.0f, dt * 14.0f))) * length(p.vel);
            }
        } else {
            p.vel.y -= 42.0f * dt;              // ballistic shell
        }

        v3 np = p.pos + p.vel * dt;
        bool hit = false;
        v3 hitAt = np;

        if (p.kind == 0) {
            if (t && t->deathTimer < 0) {
                v3 tp{t->pos.x, groundY(t->pos) + kDefs[t->type].radius * 0.6f, t->pos.y};
                if (length(np - tp) < kDefs[t->type].radius + 0.7f) { hit = true; hitAt = tp; }
            } else if (p.life > 0.12f) {
                p.life = 0.12f;                 // target gone: fizzle out
            }
        }
        float gy = Terrain::inBoundsWorld(np.x, np.z) ? groundY(v2{np.x, np.z}) : 0.0f;
        if (!hit && np.y <= gy) { hit = true; hitAt = v3{np.x, gy, np.z}; }

        if (!hit && p.life <= 0.0f) { p.alive = false; continue; }
        if (!hit) {
            p.pos = np;
            if (p.kind == 1 && rng_.f01() < dt * 45.0f) {
                Particle s{};
                s.alive = true; s.pos = p.pos;
                s.life = 0; s.maxLife = 0.4f;
                s.size0 = 0.35f; s.size1 = 1.1f;
                s.col0 = v4{0.5f, 0.47f, 0.44f, 0.5f}; s.col1 = v4{0.25f, 0.24f, 0.23f, 0.0f};
                s.drag = 2.0f; s.kind = 0;
                spawnParticle(s);
            }
            continue;
        }

        // --- impact --------------------------------------------------------
        p.alive = false;
        if (p.splash > 0.0f) {
            explosion(hitAt, 1.35f, v3{1.0f, 0.62f, 0.22f});
            v2 c{hitAt.x, hitAt.z};
            for (auto& e : ents) {
                if (!e.alive || e.deathTimer >= 0 || e.team == p.team || e.team == 2) continue;
                float d = length(e.pos - c);
                if (d > p.splash + kDefs[e.type].radius) continue;
                float falloff = 1.0f - saturate((d - kDefs[e.type].radius) / p.splash) * 0.6f;
                applyDamage(e, p.dmg * falloff, c);
            }
        } else {
            if (t && t->deathTimer < 0) applyDamage(*t, p.dmg, v2{hitAt.x, hitAt.z});
            for (int i = 0; i < 5; i++) {
                Particle s{};
                s.alive = true; s.pos = hitAt;
                s.vel = v3{rng_.range(-1.f,1.f), rng_.range(0.f,1.f), rng_.range(-1.f,1.f)} * 7.0f;
                s.life = 0; s.maxLife = rng_.range(0.15f, 0.35f);
                s.size0 = 0.20f; s.size1 = 0.02f;
                s.col0 = v4{2.4f, 1.5f, 0.6f, 1.0f}; s.col1 = v4{1.0f, 0.3f, 0.0f, 0.0f};
                s.gravity = -18.0f; s.kind = 1;
                spawnParticle(s);
            }
        }
    }
    projectiles.erase(std::remove_if(projectiles.begin(), projectiles.end(),
                                     [](const Projectile& p) { return !p.alive; }),
                      projectiles.end());
}

void Game::updateParticles(float dt) {
    for (auto& p : particles) {
        if (!p.alive) continue;
        p.life += dt;
        if (p.life >= p.maxLife) { p.alive = false; continue; }
        p.vel.y += p.gravity * dt;
        if (p.drag > 0.0f) p.vel *= std::max(0.0f, 1.0f - p.drag * dt);
        p.pos += p.vel * dt;
        p.rot += p.spin * dt;
    }
    if (particles.size() > 4000) {
        particles.erase(std::remove_if(particles.begin(), particles.end(),
                                       [](const Particle& p) { return !p.alive; }),
                        particles.end());
    }
}

void Game::updateVisibility() {
    // Both sides are fogged. The AI reads this exactly as the player's HUD does,
    // so it can only reason about what its own units have actually seen.
    for (int t = 0; t < 2; t++) std::fill(vis_[t].begin(), vis_[t].end(), 0);
    for (const auto& e : ents) {
        if (!e.alive || e.deathTimer >= 0 || e.team > 1) continue;
        float sight = kDefs[e.type].sight;
        if (sight <= 0) continue;
        const int t = e.team;
        int r = (int)(sight / VIS_CELL) + 1;
        int cx = (int)(e.pos.x / VIS_CELL), cz = (int)(e.pos.y / VIS_CELL);
        for (int z = std::max(0, cz - r); z <= std::min(VIS - 1, cz + r); z++)
            for (int x = std::max(0, cx - r); x <= std::min(VIS - 1, cx + r); x++) {
                float dx = (x + 0.5f) * VIS_CELL - e.pos.x;
                float dz = (z + 0.5f) * VIS_CELL - e.pos.y;
                if (dx * dx + dz * dz <= sight * sight) {
                    int i = z * VIS + x;
                    vis_[t][i] = 1;
                    explored_[t][i] = 1;
                    lastSeen_[t][i] = (float)time;
                }
            }
    }
}

// ---------------------------------------------------------------- frame tick
void Game::update(float dt) {
    dt = clampf(dt, 0.0f, 0.05f);
    time += dt;

    for (size_t i = 0; i < ents.size(); i++) {
        Entity& e = ents[i];
        if (!e.alive) continue;
        if (e.deathTimer >= 0.0f) {
            e.deathTimer += dt;
            float ttl = kDefs[e.type].building ? 2.4f : 1.2f;
            if (e.deathTimer > ttl) { e.alive = false; freeSlots_.push_back((int)i); }
            continue;
        }
        // updateUnit may spawn (production), which can reallocate `ents`; it is
        // written to stop touching `e` after any spawn, so the reference is safe.
        updateUnit(e, (int)i, dt);
    }

    separate(dt);
    updateProjectiles(dt);
    updateParticles(dt);

    visTimer_ -= dt;
    if (visTimer_ <= 0.0f) { updateVisibility(); visTimer_ = 0.12f; }

    // Recount supply from live units plus everything still in a production queue.
    for (int t = 0; t < 2; t++) fac[t].supplyUsed = 0;
    for (const auto& e : ents) {
        if (!e.alive || e.deathTimer >= 0 || e.team > 1) continue;
        fac[e.team].supplyUsed += kDefs[e.type].supplyCost;
        for (UnitType q : e.queue) fac[e.team].supplyUsed += kDefs[q].supplyCost;
    }

    for (auto& p : pings) p.t += dt;
    pings.erase(std::remove_if(pings.begin(), pings.end(),
                               [](const Ping& p) { return p.t > 4.0f; }), pings.end());

    if (winner < 0) {
        int alive[2] = {0, 0};
        for (const auto& e : ents) {
            if (!e.alive || e.deathTimer >= 0 || e.team > 1) continue;
            // Losing every structure and worker means you can no longer rebuild.
            if (kDefs[e.type].building || e.type == UT_WORKER) alive[e.team]++;
        }
        if (!alive[0] && alive[1]) winner = 1;
        else if (!alive[1] && alive[0]) winner = 0;
    }
}

} // namespace sf
