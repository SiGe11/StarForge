// Commander.cpp — orchestration: macro, scouting and tactics.
//
// Every order in this file goes through Game's public cmd* API (the same
// entry points the mouse drives) and is paid for out of ActionBudget, so the
// AI's click rate is bounded exactly like a human's.
#include "AI.h"
#include <cmath>
#include <algorithm>

namespace sf::ai {

namespace {

struct MacroPlan {
    int   workers;
    int   garrison;
    int   workshops;
    float trooperBias;    // share of production spent on troopers vs maulers
    bool  wantExpand;
    float pushThreshold; // army value before the main squad commits
};

MacroPlan planFor(Strategy s) {
    switch (s) {
        case STRAT_TROOPER_RUSH:    return {12, 3, 0, 1.00f, false, 200.0f};
        case STRAT_HARASS:         return {16, 2, 1, 0.80f, false, 500.0f};
        case STRAT_TIMING_PUSH:    return {17, 2, 2, 0.55f, false, 420.0f};
        case STRAT_TURTLE_TECH:    return {17, 2, 2, 0.35f, false, 900.0f};
        case STRAT_EXPAND:         return {22, 2, 1, 0.55f, true,  700.0f};
        case STRAT_COUNTER_ATTACK: return {16, 2, 2, 0.60f, false, 180.0f};
        case STRAT_FEINT:          return {18, 2, 2, 0.55f, false, 450.0f};
        case STRAT_ECO:
        default:                   return {20, 1, 1, 0.60f, true,  800.0f};
    }
}

// Orders are only worth re-issuing when something actually changed; re-clicking
// the same destination every tick would burn the whole APM budget.
bool worthReissuing(v2 want, v2 have, float slack) {
    return length(want - have) > slack;
}

} // namespace

void Commander::init(Game* g, int team, uint32_t seed, IInfluenceBackend* gpu) {
    g_ = g;
    team_ = team;
    rng_ = Rng(seed ^ 0x9E3779B9u);
    per_.init(g, team);
    opp_.init(seed);
    strat_.init(seed);
    infl_.setBackend(gpu);
    budget_ = ActionBudget{};
    dbg_ = AIDebug{};
    dbg_.backend = infl_.backendName();
    main_.clear(); harass_.clear(); scouts_.clear();
    rally_ = per_.home();
    committed_ = false;
    scoutGuess_ = 0;
    thinkT_ = macroT_ = scoutT_ = tacticT_ = 0.0f;
}

// ---------------------------------------------------------------- order paths
std::vector<EntId> Commander::ownOf(UnitType t, bool idleOnly) const {
    std::vector<EntId> v;
    for (size_t i = 0; i < g_->ents.size(); i++) {
        const Entity& e = g_->ents[i];
        if (!e.alive || e.deathTimer >= 0 || e.team != team_ || e.type != t) continue;
        if (e.buildProgress < 1.0f) continue;
        if (idleOnly && e.order != ORD_IDLE) continue;
        v.push_back(g_->handleOf((int)i));
    }
    return v;
}

EntId Commander::nearestOwnBuilding(UnitType t) const {
    for (size_t i = 0; i < g_->ents.size(); i++) {
        const Entity& e = g_->ents[i];
        if (!e.alive || e.deathTimer >= 0 || e.team != team_ || e.type != t) continue;
        if (e.buildProgress < 1.0f) continue;
        return g_->handleOf((int)i);
    }
    return EntId{};
}

bool Commander::order(const std::vector<EntId>& sel, int kind, v2 pos, EntId target) {
    if (sel.empty()) return false;
    // A human pays for the selection and then for the order. Re-ordering the
    // same group costs one action; switching groups costs two.
    float h = 0.0f;
    for (EntId e : sel) h += (float)(e.v & 0xFFFF);
    h += (float)sel.size() * 7919.0f;
    int cost = (std::fabs(h - lastSelHash_) > 0.5f) ? 2 : 1;
    if (!budget_.spend(cost)) return false;
    budget_.note((float)g_->time);
    lastSelHash_ = h;

    switch (kind) {
        case 0: g_->cmdMove(sel, pos, false); break;
        case 1: g_->cmdMove(sel, pos, true);  break;   // attack-move
        case 2: g_->cmdAttack(sel, target);   break;
        case 3: g_->cmdStop(sel);             break;
        case 4: g_->cmdHarvest(sel, target);  break;
        default: break;
    }
    return true;
}

bool Commander::train(EntId building, UnitType what) {
    if (!budget_.spend(2)) return false;          // select structure + hotkey
    budget_.note((float)g_->time);
    lastSelHash_ = -1.0f;                         // selection moved off the army
    return g_->cmdTrain(building, what);
}

bool Commander::build(EntId worker, UnitType what, v2 where) {
    if (!budget_.spend(3)) return false;          // select Digger + hotkey + click
    budget_.note((float)g_->time);
    lastSelHash_ = -1.0f;
    std::vector<EntId> sel{worker};
    return g_->cmdBuild(sel, what, where);
}

bool Commander::reserved(EntId h) const {
    for (EntId o : scouts_) if (o.v == h.v) return true;
    for (EntId o : harass_) if (o.v == h.v) return true;
    for (EntId o : main_)   if (o.v == h.v) return true;
    return false;
}

bool Commander::placeNear(UnitType what, v2 around, float rmin, float rmax, v2& out) const {
    for (int i = 0; i < 26; i++) {
        float a = rng_.f01() * TAU;
        float r = rmin + (rmax - rmin) * std::sqrt(rng_.f01());
        v2 p = around + v2{std::cos(a) * r, std::sin(a) * r};
        if (g_->canPlace(what, p)) { out = p; return true; }
    }
    return false;
}

// ---------------------------------------------------------------- think
void Commander::think(float dt) {
    opp_.update(per_, dt, (float)g_->time);
    infl_.build(per_, *g_, team_);
    strat_.update(per_.snap(), opp_, (float)g_->time, dt);

    dbg_.strategy = strat_.current();
    dbg_.believed = opp_.mostLikely();
    for (int i = 0; i < PS_COUNT; i++) dbg_.beliefs[i] = opp_.belief((PlayerStrat)i);
    for (int i = 0; i < STRAT_COUNT; i++) dbg_.stratScores[i] = strat_.scoreOf((Strategy)i);
    dbg_.entropy = opp_.entropy();
    dbg_.scoutConfidence = per_.snap().scoutConfidence;
    dbg_.apm = budget_.measuredAPM();
    dbg_.apmPeak = budget_.peakAPM();
    dbg_.actions = budget_.lifetime;
    dbg_.denied = budget_.denied;
    dbg_.reason = strat_.reason();
    dbg_.switches = strat_.switches();
    dbg_.enemyBaseKnown = per_.snap().enemyBaseKnown;
    dbg_.enemyBase = per_.snap().enemyBase;
    dbg_.squads = (int)main_.size();
}

// ---------------------------------------------------------------- macro
void Commander::runMacro(float dt) {
    const Snapshot& s = per_.snap();
    const MacroPlan plan = planFor(strat_.current());
    Faction& F = g_->fac[team_];

    // Free workers: everything not committed to a squad or out scouting.
    std::vector<EntId> freeWorkers;
    for (EntId h : ownOf(UT_WORKER)) if (!reserved(h)) freeWorkers.push_back(h);
    auto builder = [&]() -> EntId {
        if (freeWorkers.empty()) return EntId{};
        return freeWorkers[rng_.irange(0, (int)freeWorkers.size())];
    };

    // A human macros in bursts, issuing several things back to back, and is
    // limited by hands rather than by a one-action-per-tick rule. The budget is
    // the real constraint, so this runs a prioritised pass instead of returning
    // after the first success -- previously worker production starved the army
    // entirely, because it returned before ever reaching the garrison.
    int issued = 0;
    const int kMaxPerPass = 4;

    // 1. Supply. Gated on bunkhouses specifically, not on all construction.
    if (issued < kMaxPerPass && s.supplyCap - s.supplyUsed < 6 &&
        F.ore >= kDefs[UT_BUNKHOUSE].cost && s.pendingType[UT_BUNKHOUSE] < 2) {
        EntId w = builder(); v2 spot;
        if (w.valid() && placeNear(UT_BUNKHOUSE, per_.home(), 10.0f, 26.0f, spot))
            if (build(w, UT_BUNKHOUSE, spot)) issued++;
    }

    // 2. Production structures (at most two going up at once).
    const int extraProd = (F.ore > 550) ? 2 : (F.ore > 320 ? 1 : 0);
    const int wantRax = std::min(5, plan.garrison + extraProd);
    const int wantFac = std::min(3, plan.workshops + (plan.workshops > 0 ? extraProd : 0));
    if (issued < kMaxPerPass && s.garrison + s.pendingType[UT_GARRISON] < wantRax &&
        F.ore >= kDefs[UT_GARRISON].cost && s.pending < 2) {
        EntId w = builder(); v2 spot;
        if (w.valid() && placeNear(UT_GARRISON, per_.home(), 12.0f, 28.0f, spot))
            if (build(w, UT_GARRISON, spot)) issued++;
    }
    if (issued < kMaxPerPass && s.garrison >= 1 &&
        s.workshops + s.pendingType[UT_WORKSHOP] < wantFac &&
        F.ore >= kDefs[UT_WORKSHOP].cost && s.pending < 2) {
        EntId w = builder(); v2 spot;
        if (w.valid() && placeNear(UT_WORKSHOP, per_.home(), 13.0f, 29.0f, spot))
            if (build(w, UT_WORKSHOP, spot)) issued++;
    }

    // 3. Expansion, only onto ground we have actually explored and that is quiet.
    if (issued < kMaxPerPass && plan.wantExpand && F.ore >= kDefs[UT_FOUNDRY].cost &&
        s.ccs + s.pendingType[UT_FOUNDRY] < 2 && s.pending < 2) {
        v2 best{}; float bestScore = -1e30f;
        for (const Entity& e : g_->ents) {
            if (!e.alive || e.type != UT_ORE) continue;
            if (!g_->explored(team_, e.pos)) continue;
            float dHome = length(e.pos - per_.home());
            if (dHome < 26.0f || dHome > 110.0f) continue;
            float score = -dHome * 0.02f - infl_.sample(2, e.pos) * 3.0f;
            if (score > bestScore) { bestScore = score; best = e.pos; }
        }
        v2 spot;
        if (bestScore > -1e29f && placeNear(UT_FOUNDRY, best, 8.0f, 16.0f, spot)) {
            EntId w = builder();
            if (w.valid() && build(w, UT_FOUNDRY, spot)) issued++;
        }
    }

    // 4. Workers -- never allowed to block army production below.
    if (issued < kMaxPerPass && s.workers + s.pendingType[UT_WORKER] < plan.workers &&
        F.ore >= kDefs[UT_WORKER].cost) {
        for (size_t i = 0; i < g_->ents.size() && issued < kMaxPerPass; i++) {
            const Entity& e = g_->ents[i];
            if (!e.alive || e.deathTimer >= 0 || e.team != team_) continue;
            if (e.type != UT_FOUNDRY || e.buildProgress < 1.0f || !e.queue.empty()) continue;
            if (train(g_->handleOf((int)i), UT_WORKER)) issued++;
            break;
        }
    }

    // 5. Army from every idle production structure. Floating ore is the
    //    single most common way for an RTS bot to lose a game it should win.
    for (size_t i = 0; i < g_->ents.size() && issued < kMaxPerPass; i++) {
        const Entity& e = g_->ents[i];
        if (!e.alive || e.deathTimer >= 0 || e.team != team_ || e.buildProgress < 1.0f) continue;
        if (e.queue.size() >= 2) continue;
        EntId h = g_->handleOf((int)i);
        bool rich = F.ore > 400;   // when banking, build everything available
        if (e.type == UT_GARRISON && F.ore >= kDefs[UT_TROOPER].cost) {
            if (rich || s.workshops == 0 || rng_.f01() < plan.trooperBias)
                if (train(h, UT_TROOPER)) issued++;
        } else if (e.type == UT_WORKSHOP && F.ore >= kDefs[UT_MAULER].cost) {
            if (train(h, UT_MAULER)) issued++;
        }
    }

    // 6. Idle workers back to ore (scouts and squad members excluded).
    std::vector<EntId> idle;
    for (EntId h : ownOf(UT_WORKER, true)) if (!reserved(h)) idle.push_back(h);
    if (!idle.empty() && issued < kMaxPerPass) {
        for (size_t i = 0; i < g_->ents.size(); i++) {
            const Entity& n = g_->ents[i];
            if (!n.alive || n.type != UT_ORE || n.oreLeft <= 0) continue;
            if (length(n.pos - per_.home()) > 50.0f) continue;
            order(idle, 4, n.pos, g_->handleOf((int)i));
            break;
        }
    }
}

// ---------------------------------------------------------------- scouting
void Commander::runScouts(float dt) {
    const Snapshot& s = per_.snap();

    scouts_.erase(std::remove_if(scouts_.begin(), scouts_.end(),
        [&](EntId h) { return g_->get(h) == nullptr; }), scouts_.end());

    // Keep exactly one scout alive whenever our picture has gone stale. A stale
    // model is worse than none: the strategy layer acts confidently on it.
    bool wantScout = (!s.enemyBaseKnown && g_->time > 8.0) ||
                     (s.scoutConfidence < 0.40f && g_->time > 25.0);
    if (wantScout && scouts_.empty()) {
        auto troopers = ownOf(UT_TROOPER);
        if (troopers.size() >= 6) scouts_.push_back(troopers.back());
        else {
            auto ws = ownOf(UT_WORKER);
            if (ws.size() >= 6) scouts_.push_back(ws.back());
        }
    }
    if (!wantScout && !scouts_.empty() && s.scoutConfidence > 0.8f) {
        scouts_.clear();          // release it back to the economy
        return;
    }

    for (EntId h : scouts_) {
        const Entity* e = g_->get(h);
        if (!e) continue;
        // Re-target when it has arrived, gone idle, or been pulled into a fight.
        bool arrived = length(e->pos - dbg_.scoutTarget) < 9.0f;
        bool busy = (e->order == ORD_MOVE || e->order == ORD_ATTACKMOVE);
        if (busy && !arrived) continue;

        if (!s.enemyBaseKnown) {
            scoutGuess_++;
            v2 t = per_.baseGuesses()[scoutGuess_ % (int)per_.baseGuesses().size()];
            if (order({h}, 0, t, EntId{})) dbg_.scoutTarget = t;
            continue;
        }

        // Uncertainty-driven: sample around whatever we are least sure about,
        // preferring stale ground and avoiding known threat.
        int q = opp_.mostValuableQuestion();
        v2 anchor = s.enemyBase;
        if (q == 2) {
            float bestD = 1e30f;
            for (const Entity& n : g_->ents) {
                if (!n.alive || n.type != UT_ORE) continue;
                float d = length(n.pos - s.enemyBase);
                if (d > 25.0f && d < bestD) { bestD = d; anchor = n.pos; }
            }
        }
        v2 best = anchor; float bestScore = -1e30f;
        for (int i = 0; i < 28; i++) {
            float a = rng_.f01() * TAU, r = rng_.range(0.0f, 30.0f);
            v2 c = g_->nav.nearestWalkable(anchor + v2{std::cos(a) * r, std::sin(a) * r});
            float score = infl_.sample(3, c) * 2.0f - infl_.sample(2, c) * 1.2f;
            if (score > bestScore) { bestScore = score; best = c; }
        }
        if (order({h}, 0, best, EntId{})) dbg_.scoutTarget = best;
    }
}

// ---------------------------------------------------------------- tactics
void Commander::runTactics(float dt) {
    const Snapshot& s = per_.snap();
    const MacroPlan plan = planFor(strat_.current());
    const Strategy st = strat_.current();

    auto prune = [&](std::vector<EntId>& v) {
        v.erase(std::remove_if(v.begin(), v.end(),
            [&](EntId h) { const Entity* e = g_->get(h); return !e || e->deathTimer >= 0; }), v.end());
    };
    prune(main_); prune(harass_); prune(scouts_);

    // Assign fresh army units to a squad.
    auto inSquad = [&](EntId h) {
        for (EntId o : main_)   if (o.v == h.v) return true;
        for (EntId o : harass_) if (o.v == h.v) return true;
        for (EntId o : scouts_) if (o.v == h.v) return true;
        return false;
    };
    size_t harassWant = (st == STRAT_HARASS) ? 4 : 0;
    for (UnitType t : {UT_TROOPER, UT_MAULER}) {
        for (EntId h : ownOf(t)) {
            if (inSquad(h)) continue;
            if (t == UT_TROOPER && harass_.size() < harassWant) harass_.push_back(h);
            else main_.push_back(h);
        }
    }
    dbg_.squads = (int)main_.size();

    // --- home defence -----------------------------------------------------
    // Only *fresh* sightings count. Remembered attackers linger for a while, and
    // counting them here previously pinned the whole army at home for the entire
    // game against an opponent that trickled units in.
    float threatHome = 0.0f;
    v2 threatAt = per_.home();
    for (const Remembered& r : per_.enemies()) {
        if (kDefs[r.type].building) continue;
        if (!per_.reactable(r, reactionDelay_)) continue;   // no instant reactions
        if ((float)g_->time - r.lastSeen > 10.0f) continue; // stale: not a live threat
        float d = length(r.pos - per_.home());
        if (d < 55.0f) {
            float v = (r.type == UT_MAULER) ? 150.0f : (r.type == UT_TROOPER ? 50.0f : 15.0f);
            threatHome += v;
            if (d < length(threatAt - per_.home()) || threatAt.x == per_.home().x) threatAt = r.pos;
        }
    }

    v2 mainCentre{0, 0};
    int n = 0;
    for (EntId h : main_) if (const Entity* e = g_->get(h)) { mainCentre += e->pos; n++; }
    if (n) mainCentre = mainCentre / (float)n;

    // Divert only if the raid is meaningful relative to what we field. A pair of
    // troopers should not turn a 3000-value army around.
    const float divertAt = std::max(45.0f, s.armyValue * 0.14f);
    if (threatHome > divertAt && !main_.empty()) {
        if (worthReissuing(threatAt, rally_, 12.0f)) {
            if (order(main_, 1, threatAt, EntId{})) rally_ = threatAt;
        }
        committed_ = false;
        return;
    }

    // --- retreat when the local balance is bad ---------------------------
    if (n > 0 && committed_) {
        float mine = infl_.sample(0, mainCentre);
        float theirs = infl_.sample(1, mainCentre);
        if (theirs > mine * 1.5f + 0.5f && g_->time > retreatUntil_) {
            v2 safe = infl_.safestNear(per_.home(), 30.0f);
            if (order(main_, 0, safe, EntId{})) {
                rally_ = safe;
                committed_ = false;
                retreatUntil_ = (float)g_->time + 8.0f;   // don't oscillate
            }
            return;
        }
    }

    // --- harass squad: go for workers, never trade with the army ----------
    if (!harass_.empty()) {
        v2 target = s.enemyBase;
        EntId victim{};
        float bestD = 1e30f;
        // Target a worker we can currently see, so the attack order sticks.
        for (size_t i = 0; i < g_->ents.size(); i++) {
            const Entity& e = g_->ents[i];
            if (!e.alive || e.deathTimer >= 0 || e.team == team_ || e.team == 2) continue;
            if (e.type != UT_WORKER || !g_->visible(team_, e.pos)) continue;
            float d = length(e.pos - s.enemyBase);
            if (d < bestD) { bestD = d; victim = g_->handleOf((int)i); target = e.pos; }
        }
        v2 hc{0, 0}; int hn = 0;
        for (EntId h : harass_) if (const Entity* e = g_->get(h)) { hc += e->pos; hn++; }
        if (hn) {
            hc = hc / (float)hn;
            float theirs = infl_.sample(1, hc);
            float mine = infl_.sample(0, hc);
            if (theirs > mine * 2.0f + 0.8f) {
                v2 safe = infl_.safestNear(per_.home(), 40.0f);
                if (worthReissuing(safe, harassTarget_, 14.0f))
                    if (order(harass_, 0, safe, EntId{})) harassTarget_ = safe;
            } else if (victim.valid()) {
                if (worthReissuing(target, harassTarget_, 9.0f))
                    if (order(harass_, 2, target, victim)) harassTarget_ = target;
            } else if (worthReissuing(target, harassTarget_, 14.0f)) {
                if (order(harass_, 1, target, EntId{})) harassTarget_ = target;
            }
        }
    }

    // --- main army --------------------------------------------------------
    if (main_.empty()) return;

    bool wantAttack = false;
    v2 attackAt = s.enemyBase;

    switch (st) {
        case STRAT_TROOPER_RUSH:
        case STRAT_TIMING_PUSH:
            wantAttack = s.armyValue >= plan.pushThreshold;
            break;
        case STRAT_COUNTER_ATTACK:
            wantAttack = s.eArmyAwayFromHome > 60.0f && s.armyValue >= plan.pushThreshold;
            break;
        case STRAT_FEINT: {
            // Show up somewhere obvious, then swing. Against a player who reacts
            // to what they see, the reposition is the whole point.
            wantAttack = s.armyValue >= plan.pushThreshold;
            float phase = std::fmod((float)g_->time - commitT_, 44.0f);
            if (phase < 22.0f) {
                v2 dir = normalize(s.enemyBase - per_.home());
                v2 side = perp(dir) * 42.0f;
                attackAt = g_->nav.nearestWalkable(s.enemyBase + side);
                feintTarget_ = attackAt;
            }
            break;
        }
        case STRAT_HARASS:
        case STRAT_ECO:
        case STRAT_EXPAND:
        case STRAT_TURTLE_TECH:
        default:
            wantAttack = s.armyValue >= plan.pushThreshold;
            break;
    }

    if (wantAttack) {
        if (!committed_) { committed_ = true; commitT_ = (float)g_->time; }
        // Once the base is razed, keep hunting whatever structures we remember.
        if (st != STRAT_FEINT) {
            float bestD = 1e30f;
            for (const Remembered& r : per_.enemies()) {
                if (!kDefs[r.type].building) continue;
                float d = length(r.pos - mainCentre);
                if (d < bestD) { bestD = d; attackAt = r.pos; }
            }
        }
        // Units that finish an attack-move go idle; re-issue so the push presses on.
        bool anyIdle = false;
        for (EntId h : main_) {
            const Entity* e = g_->get(h);
            if (e && e->order == ORD_IDLE) { anyIdle = true; break; }
        }
        if (anyIdle || worthReissuing(attackAt, rally_, 12.0f))
            if (order(main_, 1, attackAt, EntId{})) rally_ = attackAt;
    } else {
        committed_ = false;
        // Hold a defensible spot between home and the likely approach.
        v2 face = s.enemyBaseKnown ? s.enemyBase : per_.baseGuesses()[0];
        v2 guard = per_.home() + normalize(face - per_.home()) * 16.0f;
        guard = infl_.safestNear(g_->nav.nearestWalkable(guard), 14.0f);
        bool anyIdle = false;
        for (EntId h : main_) {
            const Entity* e = g_->get(h);
            if (e && e->order == ORD_IDLE && length(e->pos - guard) > 14.0f) { anyIdle = true; break; }
        }
        if (anyIdle || worthReissuing(guard, rally_, 16.0f))
            if (order(main_, 1, guard, EntId{})) rally_ = guard;
    }
}

// ---------------------------------------------------------------- micro
void Commander::runMicro(float dt) {
    // Focus fire. Spreading damage across full-health units is the most common
    // way to lose a fight you should win, and concentrating it is what a human
    // spends most of their actions on during an engagement -- which is why the
    // AI's APM spikes in combat and idles out of it, like a real player's.
    if (main_.empty()) return;
    if (!budget_.can(2)) return;

    v2 c{0, 0}; int n = 0;
    for (EntId h : main_) if (const Entity* e = g_->get(h)) { c += e->pos; n++; }
    if (!n) return;
    c = c / (float)n;

    EntId best{}; float bestScore = 1e30f;
    for (size_t i = 0; i < g_->ents.size(); i++) {
        const Entity& e = g_->ents[i];
        if (!e.alive || e.deathTimer >= 0 || e.team == team_ || e.team == 2) continue;
        if (kDefs[e.type].building) continue;
        if (!g_->visible(team_, e.pos)) continue;      // must actually see it
        float d = length(e.pos - c);
        if (d > 28.0f) continue;
        // Prefer the weakest, nearest thing that can shoot back.
        float score = e.hp + d * 2.5f - (kDefs[e.type].range > 0.0f ? 45.0f : 0.0f);
        if (score < bestScore) { bestScore = score; best = g_->handleOf((int)i); }
    }
    if (!best.valid()) { focusTarget_ = EntId{}; return; }
    if (focusTarget_.valid() && best.v == focusTarget_.v && g_->get(focusTarget_)) return;
    if (order(main_, 2, v2{}, best)) focusTarget_ = best;
}

// ---------------------------------------------------------------- tick
void Commander::update(float dt) {
    if (!g_ || g_->winner >= 0) return;
    budget_.tick(dt);
    per_.update(dt);

    thinkT_ -= dt;
    if (thinkT_ <= 0.0f) { float used = 1.0f / 6.0f; thinkT_ = used; think(used); }

    macroT_ -= dt;
    if (macroT_ <= 0.0f) { macroT_ = 0.28f; runMacro(macroT_); }

    scoutT_ -= dt;
    if (scoutT_ <= 0.0f) { scoutT_ = 1.1f; runScouts(scoutT_); }

    tacticT_ -= dt;
    if (tacticT_ <= 0.0f) { tacticT_ = 0.42f; runTactics(tacticT_); }

    microT_ -= dt;
    if (microT_ <= 0.0f) { microT_ = focusTarget_.valid() ? 0.16f : 0.34f; runMicro(microT_); }

    dbg_.apm = budget_.measuredAPM();
    dbg_.apmPeak = budget_.peakAPM();
}

} // namespace sf::ai
