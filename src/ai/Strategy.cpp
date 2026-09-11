// Strategy.cpp — spatial influence fields and the online strategy selector.
#include "AI.h"
#include <cmath>
#include <algorithm>

namespace sf::ai {

// ---------------------------------------------------------------- InfluenceMap
void InfluenceMap::computeCPU(const InfluenceUnit* u, int n) {
    std::fill(field_.begin(), field_.end(), 0.0f);
    for (int i = 0; i < n; i++) {
        const InfluenceUnit& a = u[i];
        float reach = a.range * 2.0f;
        int x0 = std::max(0, (int)((a.x - reach) / CELL));
        int x1 = std::min(N - 1, (int)((a.x + reach) / CELL));
        int z0 = std::max(0, (int)((a.z - reach) / CELL));
        int z1 = std::min(N - 1, (int)((a.z + reach) / CELL));
        int ch = (a.team > 0.0f) ? 0 : 1;
        for (int z = z0; z <= z1; z++)
            for (int x = x0; x <= x1; x++) {
                float dx = (x + 0.5f) * CELL - a.x;
                float dz = (z + 0.5f) * CELL - a.z;
                float d2 = dx * dx + dz * dz;
                if (d2 > reach * reach) continue;
                float f = a.strength / (1.0f + d2 / (a.range * a.range));
                int idx = (z * N + x) * CHANNELS;
                field_[idx + ch] += f;
                if (ch == 1) field_[idx + 2] += f;   // enemy influence is threat
            }
    }
}

void InfluenceMap::build(const Perception& p, Game& g, int team) {
    units_.clear();
    for (const Entity& e : g.ents) {
        if (!e.alive || e.deathTimer >= 0 || e.team != team) continue;
        const UnitDef& D = kDefs[e.type];
        if (D.range <= 0.0f && e.type != UT_WORKER) continue;
        InfluenceUnit u{};
        u.x = e.pos.x; u.z = e.pos.y;
        u.strength = (e.type == UT_MAULER) ? 3.0f : (e.type == UT_TROOPER ? 1.0f : 0.25f);
        u.range = std::max(12.0f, D.range * 1.4f);
        u.team = 1.0f;
        units_.push_back(u);
    }
    for (const Remembered& r : p.enemies()) {
        if (kDefs[r.type].building) continue;
        // Stale sightings contribute less: we are less sure they are still there.
        float age = (float)g.time - r.lastSeen;
        float conf = std::exp(-age / 25.0f);
        if (conf < 0.05f) continue;
        InfluenceUnit u{};
        u.x = r.pos.x; u.z = r.pos.y;
        u.strength = ((r.type == UT_MAULER) ? 3.0f : (r.type == UT_TROOPER ? 1.0f : 0.25f)) * conf;
        u.range = std::max(12.0f, kDefs[r.type].range * 1.4f);
        u.team = -1.0f;
        units_.push_back(u);
    }

    bool done = false;
    if (backend_ && !units_.empty())
        done = backend_->compute(units_.data(), (int)units_.size(), N, CELL, field_.data());
    if (!done) computeCPU(units_.data(), (int)units_.size());

    // Channel 3 is information age, which only the game can answer.
    for (int z = 0; z < N; z++)
        for (int x = 0; x < N; x++) {
            v2 w{(x + 0.5f) * CELL, (z + 0.5f) * CELL};
            float st = g.staleness(team, w);
            field_[(z * N + x) * CHANNELS + 3] = (st > 1e8f) ? 1.0f : saturate(st / 90.0f);
        }
}

float InfluenceMap::sample(int c, v2 w) const {
    int x = (int)clampf(w.x / CELL, 0, N - 1);
    int z = (int)clampf(w.y / CELL, 0, N - 1);
    return field_[(z * N + x) * CHANNELS + c];
}

v2 InfluenceMap::safestNear(v2 around, float radius) const {
    v2 best = around;
    float bestScore = -1e30f;
    int r = (int)(radius / CELL) + 1;
    int cx = (int)clampf(around.x / CELL, 0, N - 1);
    int cz = (int)clampf(around.y / CELL, 0, N - 1);
    for (int z = std::max(0, cz - r); z <= std::min(N - 1, cz + r); z++)
        for (int x = std::max(0, cx - r); x <= std::min(N - 1, cx + r); x++) {
            int i = (z * N + x) * CHANNELS;
            float score = field_[i + 0] - field_[i + 2] * 1.6f;
            if (score > bestScore) {
                bestScore = score;
                best = v2{(x + 0.5f) * CELL, (z + 0.5f) * CELL};
            }
        }
    return best;
}

// ---------------------------------------------------------------- StrategySelector
// Feature layout. Kept explicit because the weights below are hand-authored and
// have to stay readable.
enum {
    F_TIME = 0, F_ARMY_RATIO, F_SUPPLY_FILL, F_ORE, F_WORKERS, F_ARMY,
    F_EARMY, F_P_RUSH, F_P_MACRO, F_P_TURTLE, F_P_HARASS, F_P_EXPAND, F_P_TECH,
    F_AGGRESSION, F_EXPANSION, F_DEFENSIVE, F_SCOUTCONF, F_ENTROPY,
    F_ARMY_AWAY, F_BIAS
};

void StrategySelector::init(uint32_t seed) {
    rng_ = Rng(seed);
    for (int s = 0; s < STRAT_COUNT; s++)
        for (int f = 0; f < NFEAT; f++) w_[s][f] = 0.0f;

    // --- hand-authored doctrine prior -------------------------------------
    // These make the AI competent from the first second; the online update then
    // moves them toward whatever actually works against this particular player.
    auto W = [&](Strategy s, int f, float v) { w_[s][f] = v; };

    // Economy: good early, good when they are not threatening us.
    W(STRAT_ECO, F_BIAS, 0.35f);  W(STRAT_ECO, F_TIME, -0.9f);
    W(STRAT_ECO, F_P_RUSH, -1.4f); W(STRAT_ECO, F_P_TURTLE, 0.5f);
    W(STRAT_ECO, F_WORKERS, -0.8f); W(STRAT_ECO, F_EARMY, -0.6f);

    // Trooper rush: only early, and best against a greedy or teching opponent.
    W(STRAT_TROOPER_RUSH, F_BIAS, -0.20f); W(STRAT_TROOPER_RUSH, F_TIME, -1.8f);
    W(STRAT_TROOPER_RUSH, F_P_MACRO, 1.1f); W(STRAT_TROOPER_RUSH, F_P_EXPAND, 1.3f);
    W(STRAT_TROOPER_RUSH, F_P_TECH, 0.9f);  W(STRAT_TROOPER_RUSH, F_P_TURTLE, -1.0f);
    W(STRAT_TROOPER_RUSH, F_ARMY, 0.7f);    W(STRAT_TROOPER_RUSH, F_P_RUSH, -0.6f);

    // Harass: punish economic play; pointless against someone sitting on units.
    W(STRAT_HARASS, F_BIAS, 0.05f);  W(STRAT_HARASS, F_P_MACRO, 1.2f);
    W(STRAT_HARASS, F_P_EXPAND, 1.1f); W(STRAT_HARASS, F_DEFENSIVE, -1.0f);
    W(STRAT_HARASS, F_ARMY, 0.5f);   W(STRAT_HARASS, F_P_RUSH, -0.7f);
    W(STRAT_HARASS, F_ARMY_AWAY, 0.6f);

    // Timing push: commit when we are actually ahead in army.
    W(STRAT_TIMING_PUSH, F_BIAS, -0.15f); W(STRAT_TIMING_PUSH, F_ARMY_RATIO, 2.2f);
    W(STRAT_TIMING_PUSH, F_ARMY, 0.9f);   W(STRAT_TIMING_PUSH, F_SUPPLY_FILL, 0.7f);
    W(STRAT_TIMING_PUSH, F_TIME, 0.4f);   W(STRAT_TIMING_PUSH, F_SCOUTCONF, 0.5f);

    // Turtle/tech: the answer to aggression, and to being behind.
    W(STRAT_TURTLE_TECH, F_BIAS, -0.05f); W(STRAT_TURTLE_TECH, F_P_RUSH, 2.0f);
    W(STRAT_TURTLE_TECH, F_P_HARASS, 1.0f); W(STRAT_TURTLE_TECH, F_ARMY_RATIO, -1.5f);
    W(STRAT_TURTLE_TECH, F_AGGRESSION, 1.2f);

    // Expand: only when we are safe and have money.
    W(STRAT_EXPAND, F_BIAS, -0.35f); W(STRAT_EXPAND, F_ORE, 1.6f);
    W(STRAT_EXPAND, F_P_RUSH, -1.8f); W(STRAT_EXPAND, F_AGGRESSION, -1.3f);
    W(STRAT_EXPAND, F_ARMY_RATIO, 0.8f); W(STRAT_EXPAND, F_P_TURTLE, 0.7f);

    // Counter-attack: their army is out of position.
    W(STRAT_COUNTER_ATTACK, F_BIAS, -0.5f); W(STRAT_COUNTER_ATTACK, F_ARMY_AWAY, 2.6f);
    W(STRAT_COUNTER_ATTACK, F_ARMY, 0.6f);  W(STRAT_COUNTER_ATTACK, F_SCOUTCONF, 0.8f);

    // Feint: worth it against a player who reacts hard to what they see.
    W(STRAT_FEINT, F_BIAS, -0.75f); W(STRAT_FEINT, F_DEFENSIVE, 1.4f);
    W(STRAT_FEINT, F_ARMY, 0.8f);   W(STRAT_FEINT, F_ARMY_RATIO, 0.6f);
    W(STRAT_FEINT, F_TIME, 0.5f);

    cur_ = pending_ = STRAT_ECO;
    conf_ = 0.0f; commitT_ = 0.0f; lastAdv_ = 0.0f; evalT_ = 0.0f; switches_ = 0;
    for (int i = 0; i < NFEAT; i++) lastFeat_[i] = 0.0f;
    snprintf(reason_, sizeof(reason_), "opening");
}

void StrategySelector::features(const Snapshot& s, const OpponentModel& om,
                                float t, float* f) const {
    for (int i = 0; i < NFEAT; i++) f[i] = 0.0f;
    f[F_TIME]        = saturate(t / 600.0f);
    f[F_ARMY_RATIO]  = (s.armyValue - s.eArmyEstimate) / (s.armyValue + s.eArmyEstimate + 200.0f);
    f[F_SUPPLY_FILL] = saturate((float)s.supplyUsed / std::max(1.0f, (float)s.supplyCap));
    f[F_ORE]    = saturate((float)s.ore / 900.0f);
    f[F_WORKERS]     = saturate((float)s.workers / 20.0f);
    f[F_ARMY]        = saturate(s.armyValue / 900.0f);
    f[F_EARMY]       = saturate(s.eArmyEstimate / 900.0f);
    f[F_P_RUSH]      = om.belief(PS_RUSHING);
    f[F_P_MACRO]     = om.belief(PS_MACRO);
    f[F_P_TURTLE]    = om.belief(PS_TURTLING);
    f[F_P_HARASS]    = om.belief(PS_HARASSING);
    f[F_P_EXPAND]    = om.belief(PS_EXPANDING);
    f[F_P_TECH]      = om.belief(PS_TECHING);
    f[F_AGGRESSION]  = om.profile().aggression;
    f[F_EXPANSION]   = om.profile().expansion;
    f[F_DEFENSIVE]   = om.profile().defensive;
    f[F_SCOUTCONF]   = s.scoutConfidence;
    f[F_ENTROPY]     = om.entropy() / std::log2((float)PS_COUNT);
    f[F_ARMY_AWAY]   = saturate(s.eArmyAwayFromHome / 400.0f);
    f[F_BIAS]        = 1.0f;
}

bool StrategySelector::legal(Strategy s, const Snapshot& snap) const {
    switch (s) {
        case STRAT_TROOPER_RUSH:    return snap.garrison >= 1 && snap.troopers >= 4;
        case STRAT_HARASS:         return snap.troopers >= 3;
        case STRAT_TIMING_PUSH:    return snap.armyValue >= 300.0f;
        case STRAT_COUNTER_ATTACK: return snap.armyValue >= 200.0f && snap.eArmyAwayFromHome > 60.0f;
        case STRAT_FEINT:          return snap.armyValue >= 400.0f && snap.enemyBaseKnown;
        case STRAT_EXPAND:         return snap.ore >= 350 && snap.workers >= 10;
        default:                   return true;
    }
}

void StrategySelector::reward(float r) {
    // Bandit-style gradient step on the strategy that was actually running.
    const float lr = 0.05f;
    float g = clampf(r, -1.0f, 1.0f);
    for (int i = 0; i < NFEAT; i++) {
        w_[cur_][i] = clampf(w_[cur_][i] + lr * g * lastFeat_[i], -4.0f, 4.0f);
    }
}

void StrategySelector::update(const Snapshot& s, const OpponentModel& om,
                              float t, float dt) {
    float f[NFEAT];
    features(s, om, t, f);

    // --- online credit assignment for the plan we have been running ---------
    evalT_ += dt;
    if (evalT_ >= 18.0f) {
        float adv = 0.0020f * (s.armyValue - s.eArmyValue)
                  + 0.030f * (float)s.workers
                  + 0.0008f * (float)s.ore;
        reward(adv - lastAdv_);
        lastAdv_ = adv;
        evalT_ = 0.0f;
    }

    for (int i = 0; i < NFEAT; i++) lastFeat_[i] = f[i];

    // --- score every legal candidate ---------------------------------------
    float best = -1e30f, second = -1e30f;
    Strategy bestS = STRAT_ECO;
    for (int si = 0; si < STRAT_COUNT; si++) {
        Strategy s2 = (Strategy)si;
        if (!legal(s2, s)) { score_[si] = -1e30f; continue; }
        float acc = 0.0f;
        for (int i = 0; i < NFEAT; i++) acc += w_[si][i] * f[i];
        // A little optimism keeps the AI trying things it has not measured yet,
        // and it anneals away as the match goes on.
        acc += rng_.range(-1.0f, 1.0f) * 0.10f * (1.0f - saturate(t / 420.0f));
        score_[si] = acc;
        if (acc > best) { second = best; best = acc; bestS = s2; }
        else if (acc > second) second = acc;
    }
    conf_ = saturate((best - second) * 1.5f);

    // --- hysteresis: commit for a while so the AI does not thrash -----------
    commitT_ -= dt;
    if (bestS != cur_) {
        if (bestS == pending_) {
            if (commitT_ <= 0.0f) {
                cur_ = bestS;
                switches_++;
                commitT_ = 14.0f;
                snprintf(reason_, sizeof(reason_), "%s -> they look %s",
                         strategyName(cur_), playerStratName(om.mostLikely()));
            }
        } else {
            pending_ = bestS;
            if (commitT_ <= 0.0f) commitT_ = 2.5f;   // must persist to take effect
        }
    } else {
        pending_ = bestS;
    }
}

} // namespace sf::ai
