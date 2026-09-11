// Game.h — the simulation. Owns the world, all entities and the enemy AI.
// Deliberately free of any rendering or platform dependency.
#pragma once
#include <vector>
#include <string>
#include "Terrain.h"
#include "Nav.h"
#include "../core/Random.h"

namespace sf {

enum UnitType {
    UT_WORKER = 0, UT_TROOPER, UT_MAULER,
    UT_FOUNDRY, UT_GARRISON, UT_WORKSHOP, UT_BUNKHOUSE,
    UT_ORE, UT_BOULDER,
    UT_COUNT
};

struct UnitDef {
    const char* name;
    int   meshId;
    bool  building;
    bool  neutral;
    float radius, hp, speed, turn;
    float range, dmg, cooldown, splash, sight;
    float buildTime;
    int   cost, supplyCost, supplyGive;
};
extern const UnitDef kDefs[UT_COUNT];

enum Order {
    ORD_IDLE = 0, ORD_MOVE, ORD_ATTACKMOVE, ORD_ATTACK,
    ORD_HARVEST, ORD_RETURN, ORD_BUILD, ORD_HOLD
};

// EntId packs a slot index with a generation counter so references to dead
// entities are detectable after a slot is recycled. (Named EntId rather than
// Handle because Apple's MacTypes.h already defines a global `Handle`.)
struct EntId {
    int32_t v = -1;
    bool valid() const { return v >= 0; }
    int  index() const { return v & 0xFFFFF; }
    uint32_t gen() const { return (uint32_t)v >> 20; }
    static EntId make(int idx, uint32_t gen) { EntId h; h.v = (int32_t)(idx | (gen << 20)); return h; }
};

struct Entity {
    bool  alive = false;
    uint32_t gen = 0;
    UnitType type = UT_WORKER;
    int   team = 0;              // 0 player, 1 enemy, 2 neutral

    v2    pos{}, vel{};
    float yaw = 0, turretYaw = 0;
    float hp = 1;
    float bob = 0;               // walk-cycle phase
    float damageFlash = 0;
    float deathTimer = -1.0f;    // >=0 while dissolving
    float smokeTimer = 0;        // structures burn visibly once badly damaged

    Order order = ORD_IDLE;
    v2    orderPos{};
    EntId target;
    std::vector<v2> path;
    size_t pathIdx = 0;
    float repathTimer = 0;
    float stuckTimer = 0;
    v2    lastPos{};

    float cooldown = 0;
    float buildProgress = 1.0f;  // <1 while under construction
    EntId buildTarget;          // building this worker is assembling

    // economy
    int   carrying = 0;
    EntId harvestNode;
    float harvestTimer = 0;
    int   oreLeft = 0;      // for UT_ORE

    // production
    std::vector<UnitType> queue;
    float queueTimer = 0;
    v2    rally{};
    bool  rallySet = false;
};

struct Projectile {
    bool  alive = false;
    v3    pos{}, vel{};
    EntId target;
    v2    aimPos{};
    float dmg = 0, splash = 0, life = 0;
    int   team = 0;
    int   kind = 0;              // 0 gauss tracer, 1 mauler shell
};

struct Particle {
    bool  alive = false;
    v3    pos{}, vel{};
    float life = 0, maxLife = 1;
    float size0 = 1, size1 = 1;
    v4    col0{}, col1{};
    float rot = 0, spin = 0, gravity = 0, drag = 0;
    float param = 0;             // per-kind extra; the scorch picks a mark with it
    int   kind = 0;              // matches the billboard shader's kind
};

struct Faction {
    int ore = 50;
    int supplyUsed = 0, supplyCap = 0;
    int lost = 0, killed = 0;
};

class Game {
public:
    static constexpr int VIS = 64;               // visibility grid resolution
    static constexpr float VIS_CELL = Terrain::SIZE / VIS;

    void init(uint32_t seed);
    void update(float dt);

    // --- entity access ----------------------------------------------------
    Entity* get(EntId h) {
        if (!h.valid() || h.index() >= (int)ents.size()) return nullptr;
        Entity& e = ents[h.index()];
        return (e.alive && e.gen == h.gen()) ? &e : nullptr;
    }
    const Entity* get(EntId h) const { return const_cast<Game*>(this)->get(h); }
    EntId handleOf(int idx) const { return EntId::make(idx, ents[idx].gen); }

    // --- player commands --------------------------------------------------
    void cmdMove(const std::vector<EntId>& sel, v2 dest, bool attackMove);
    void cmdAttack(const std::vector<EntId>& sel, EntId target);
    void cmdHarvest(const std::vector<EntId>& sel, EntId node);
    void cmdStop(const std::vector<EntId>& sel);
    void cmdHold(const std::vector<EntId>& sel);
    void cmdRally(const std::vector<EntId>& sel, v2 dest);
    bool cmdBuild(const std::vector<EntId>& sel, UnitType what, v2 where);
    bool cmdTrain(EntId building, UnitType what);
    // Contextual right-click: attack enemies, harvest crystals, otherwise move.
    void cmdSmart(const std::vector<EntId>& sel, v2 worldPos, EntId hovered);

    bool canPlace(UnitType what, v2 where, EntId ignore = EntId{}) const;
    EntId pick(v2 worldPos, float extra = 0.0f) const;
    float groundY(v2 p) const { return terrain.heightAt(p.x, p.y); }

    // --- visibility (per team; both sides are subject to fog of war) -------
    bool visible(int team, v2 p) const {
        int x = (int)(p.x / VIS_CELL), z = (int)(p.y / VIS_CELL);
        if (x < 0 || z < 0 || x >= VIS || z >= VIS || team < 0 || team > 1) return false;
        return vis_[team][z * VIS + x] > 0;
    }
    bool explored(int team, v2 p) const {
        int x = (int)(p.x / VIS_CELL), z = (int)(p.y / VIS_CELL);
        if (x < 0 || z < 0 || x >= VIS || z >= VIS || team < 0 || team > 1) return false;
        return explored_[team][z * VIS + x] != 0;
    }
    // Seconds since a cell was last observed by `team`; large when never seen.
    float staleness(int team, v2 p) const {
        int x = (int)(p.x / VIS_CELL), z = (int)(p.y / VIS_CELL);
        if (x < 0 || z < 0 || x >= VIS || z >= VIS || team < 0 || team > 1) return 1e9f;
        float t = lastSeen_[team][z * VIS + x];
        return t < 0.0f ? 1e9f : (float)time - t;
    }
    bool visibleCell(int team, int x, int z) const {
        if (x < 0 || z < 0 || x >= VIS || z >= VIS) return false;
        return vis_[team][z * VIS + x] > 0;
    }
    bool exploredCell(int team, int x, int z) const {
        if (x < 0 || z < 0 || x >= VIS || z >= VIS) return false;
        return explored_[team][z * VIS + x] != 0;
    }

    Terrain terrain;
    Nav     nav;
    std::vector<Entity>     ents;
    std::vector<Projectile> projectiles;
    std::vector<Particle>   particles;
    Faction fac[2];
    v2 basePos[2];
    float time = 0;
    int   winner = -1;           // -1 none, 0 player, 1 enemy

    // Presentation-only feedback consumed by the app layer.
    struct Ping { v2 pos; float t; int kind; };
    std::vector<Ping> pings;

    EntId spawn(UnitType t, int team, v2 pos, bool complete = true);
    void   spawnParticle(const Particle& p);
    void   explosion(v3 at, float scale, v3 tint);

private:
    void updateUnit(Entity& e, int idx, float dt);
    void updateProduction(Entity& e, float dt);
    void updateCombat(Entity& e, float dt);
    void updateHarvest(Entity& e, float dt);
    void applyDamage(Entity& e, float dmg, v2 from);
    void fire(Entity& e, Entity& target);
    void separate(float dt);
    void updateProjectiles(float dt);
    void updateParticles(float dt);
    void updateVisibility();
    void killEntity(Entity& e);
    EntId nearestEnemy(const Entity& e, float radius) const;
    EntId nearestDropoff(const Entity& e) const;
    EntId nearestFreeNode(v2 near, int team) const;
    void   repath(Entity& e, v2 dest);

    std::vector<int> freeSlots_;
    std::vector<uint8_t> vis_[2], explored_[2];
    std::vector<float>   lastSeen_[2];   // game time each cell was last observed
    Rng rng_{1};
    float visTimer_ = 0;
};

} // namespace sf
