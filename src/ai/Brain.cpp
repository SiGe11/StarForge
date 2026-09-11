// Brain.cpp — action budget, fog-limited perception, and the opponent model.
#include "AI.h"
#include <cmath>
#include <algorithm>

namespace sf::ai {

const char* strategyName(Strategy s) {
    switch (s) {
        case STRAT_ECO:            return "Economy";
        case STRAT_TROOPER_RUSH:    return "Trooper Rush";
        case STRAT_HARASS:         return "Harass";
        case STRAT_TIMING_PUSH:    return "Timing Push";
        case STRAT_TURTLE_TECH:    return "Turtle / Tech";
        case STRAT_EXPAND:         return "Expand";
        case STRAT_COUNTER_ATTACK: return "Counter-Attack";
        case STRAT_FEINT:          return "Feint";
        default:                   return "?";
    }
}
const char* playerStratName(PlayerStrat s) {
    switch (s) {
        case PS_RUSHING:   return "rushing";
        case PS_MACRO:     return "macro";
        case PS_TURTLING:  return "turtling";
        case PS_HARASSING: return "harassing";
        case PS_EXPANDING: return "expanding";
        case PS_TECHING:   return "teching";
        default:           return "?";
    }
}

// ---------------------------------------------------------------- ActionBudget
void ActionBudget::tick(float dt) {
    tokens = std::min(cap, tokens + regen * dt);
}
bool ActionBudget::spend(int n) {
    if (tokens < (float)n) { denied++; return false; }
    tokens -= (float)n;
    lifetime += n;
    return true;
}
void ActionBudget::note(float t) {
    for (int i = 0; i < 1; i++) stamps_.push_back(t);
    while (!stamps_.empty() && t - stamps_.front() > 60.0f) stamps_.pop_front();
}
float ActionBudget::measuredAPM() const {
    if (stamps_.size() < 2) return 0.0f;
    float span = stamps_.back() - stamps_.front();
    if (span < 1.0f) return 0.0f;
    return (float)stamps_.size() / span * 60.0f;
}

float ActionBudget::peakAPM(float window) const {
    // Densest `window` seconds anywhere in the last minute, scaled to per-minute.
    // Averaging over a full minute hides the bursts an APM meter actually shows.
    if (stamps_.size() < 2) return 0.0f;
    size_t best = 0, lo = 0;
    for (size_t hi = 0; hi < stamps_.size(); hi++) {
        while (stamps_[hi] - stamps_[lo] > window) lo++;
        best = std::max(best, hi - lo + 1);
    }
    return (float)best * (60.0f / window);
}

// ---------------------------------------------------------------- Perception
static float unitValue(UnitType t) {
    switch (t) {
        case UT_TROOPER: return 50.0f;
        case UT_MAULER:   return 150.0f;
        case UT_WORKER: return 50.0f;
        default:        return 0.0f;
    }
}

void Perception::init(Game* g, int team) {
    g_ = g;
    team_ = team;
    home_ = g->basePos[team];
    mem_.clear();
    aggrT_ = -1e9f;
    aggrScore_ = 0.0f;
    armyEst_ = 0.0f;
    commitPeak_ = 0.0f;

    // Reading a symmetric map is something a human does before scouting too --
    // but a guess is all it is. Nothing is treated as known until a unit of ours
    // actually lays eyes on an enemy structure.
    const float S = Terrain::SIZE;
    guesses_.clear();
    guesses_.push_back(v2{S - home_.x, S - home_.y});   // rotational mirror
    guesses_.push_back(v2{S - home_.x, home_.y});       // horizontal mirror
    guesses_.push_back(v2{home_.x, S - home_.y});       // vertical mirror
    for (auto& q : guesses_) q = g->nav.nearestWalkable(q);
}

void Perception::update(float dt) {
    observe();
    forget();
    // Units we kill vanish from memory immediately, so an instantaneous count
    // badly underestimates them. Track a decaying peak instead: what a human
    // would say if asked "roughly how much army do they have?".
    if (snap_.eArmyValue > armyEst_) armyEst_ = snap_.eArmyValue;
    else armyEst_ = lerpf(armyEst_, snap_.eArmyValue, 1.0f - std::exp(-dt / 60.0f));
    snap_.eArmyEstimate = armyEst_;
    // Same reasoning for how hard they committed: a wave that we killed still
    // tells us they are the kind of player who sends waves.
    if (snap_.eArmyOurHalf > commitPeak_) commitPeak_ = snap_.eArmyOurHalf;
    else commitPeak_ = lerpf(commitPeak_, snap_.eArmyOurHalf, 1.0f - std::exp(-dt / 120.0f));
    snap_.eCommitPeak = commitPeak_;
    // Cumulative aggression. Recency alone is not enough: a rusher rebuilding
    // between waves looks "calm" for 45 s and would otherwise read as a turtle.
    float target = saturate(snap_.eArmyOurHalf / 200.0f);
    if (target > aggrScore_) aggrScore_ = lerpf(aggrScore_, target, 1.0f - std::exp(-dt / 3.0f));
    else                     aggrScore_ = lerpf(aggrScore_, target, 1.0f - std::exp(-dt / 240.0f));
    snap_.pressure = aggrScore_;
}

void Perception::observe() {
    Snapshot s{};
    const int enemy = 1 - team_;
    s.ore   = g_->fac[team_].ore;
    s.supplyUsed = g_->fac[team_].supplyUsed;
    s.supplyCap  = g_->fac[team_].supplyCap;

    for (const Entity& e : g_->ents) {
        if (!e.alive || e.deathTimer >= 0) continue;

        if (e.team == team_) {                     // our own side: fully known
            if (e.buildProgress < 1.0f) { s.pending++; s.pendingType[e.type]++; continue; }
            switch (e.type) {
                case UT_WORKER:   s.workers++;   break;
                case UT_TROOPER:   s.troopers++;   s.armyValue += 50.0f;  break;
                case UT_MAULER:     s.maulers++;     s.armyValue += 150.0f; break;
                case UT_FOUNDRY:  s.ccs++;       break;
                case UT_GARRISON: s.garrison++;  break;
                case UT_WORKSHOP:  s.workshops++; break;
                case UT_BUNKHOUSE:    s.bunkhouses++;    break;
                default: break;
            }
            continue;
        }
        if (e.team != enemy) continue;

        // Enemy: only if one of our units can currently see that cell.
        if (!g_->visible(team_, e.pos)) continue;

        bool merged = false;
        for (auto& r : mem_) {
            if (r.type != e.type) continue;
            // Buildings are matched tightly; mobile units loosely, since they
            // move between sightings and we must not duplicate them.
            float tol = kDefs[e.type].building ? 2.0f : 7.0f;
            if (length(r.pos - e.pos) <= tol) {
                r.pos = e.pos;
                r.lastSeen = (float)g_->time;
                r.stillVisible = true;
                merged = true;
                break;
            }
        }
        if (!merged) {
            Remembered r;
            r.type = e.type;
            r.pos = e.pos;
            r.firstSeen = r.lastSeen = (float)g_->time;
            r.stillVisible = true;
            mem_.push_back(r);
        }
    }

    // Derive believed enemy state from memory alone. Resolve the base guess
    // first so distance comparisons below have something to work with.
    v2 believedBase = guesses_.empty() ? home_ : guesses_[0];
    float bestBaseScore = -1.0f;
    for (const auto& r : mem_) {
        if (kDefs[r.type].building) {
            float w = (r.type == UT_FOUNDRY) ? 3.0f : 1.0f;
            if (w > bestBaseScore) { bestBaseScore = w; believedBase = r.pos; s.enemyBaseKnown = true; }
        }
    }
    bestBaseScore = -1.0f;
    for (const auto& r : mem_) {
        switch (r.type) {
            case UT_WORKER:   s.eWorkers++; break;
            case UT_TROOPER:   s.eTroopers++; s.eArmyValue += 50.0f;  break;
            case UT_MAULER:     s.eMaulers++;   s.eArmyValue += 150.0f; break;
            case UT_FOUNDRY:  s.eFoundries++;     break;
            case UT_GARRISON: s.eGarrisons++;   break;
            case UT_WORKSHOP:  s.eWorkshops++;  break;
            case UT_BUNKHOUSE:    s.eBunkhouses++;     break;
            default: break;
        }
        if (!kDefs[r.type].building && r.type != UT_WORKER) {
            float dHome = length(r.pos - home_);
            if (dHome < 70.0f) s.eArmyNearOurBase += unitValue(r.type);
            // Anything of theirs on our side of the map is aggression, wherever
            // the fight actually happens -- engagements rarely occur in the base.
            float dThem = s.enemyBaseKnown ? length(r.pos - believedBase)
                                           : length(r.pos - guesses_[0]);
            if (dHome < dThem) {
                s.eArmyOurHalf += unitValue(r.type);
                if ((float)g_->time - r.lastSeen < 6.0f) aggrT_ = (float)g_->time;
            }
        }
    }
    s.enemyBase = s.enemyBaseKnown ? believedBase : guesses_[0];

    if (s.enemyBaseKnown) {
        for (const auto& r : mem_) {
            if (kDefs[r.type].building || r.type == UT_WORKER) continue;
            if (length(r.pos - s.enemyBase) > 45.0f) s.eArmyAwayFromHome += unitValue(r.type);
        }
    }

    // How current is our picture of them? Purely a function of when we last
    // looked at the place we believe they live.
    s.sinceAggression = (aggrT_ < -1e8f) ? 1e9f : (float)g_->time - aggrT_;
    float stale = g_->staleness(team_, s.enemyBase);
    s.scoutConfidence = s.enemyBaseKnown ? std::exp(-stale / 40.0f) : 0.0f;

    snap_ = s;
}

void Perception::forget() {
    const float now = (float)g_->time;
    for (auto& r : mem_) r.stillVisible = false;

    // Re-mark anything currently in view (observe() already refreshed lastSeen).
    for (const Entity& e : g_->ents) {
        if (!e.alive || e.deathTimer >= 0 || e.team == team_ || e.team == 2) continue;
        if (!g_->visible(team_, e.pos)) continue;
        for (auto& r : mem_) {
            float tol = kDefs[e.type].building ? 2.0f : 7.0f;
            if (r.type == e.type && length(r.pos - e.pos) <= tol) { r.stillVisible = true; break; }
        }
    }

    // Evidence of absence: if we can see where we last saw something and it is
    // not there, drop it. This is the scouting-effort idea from Hostetler et al.
    // -- "I looked and it's gone" is information, not just missing data.
    mem_.erase(std::remove_if(mem_.begin(), mem_.end(), [&](const Remembered& r) {
        if (r.stillVisible) return false;
        if (g_->visible(team_, r.pos)) return true;
        // Buildings do not move: keep them until disproven by observation.
        if (kDefs[r.type].building) return false;
        return (now - r.lastSeen) > 50.0f;
    }), mem_.end());
}

// ---------------------------------------------------------------- OpponentModel
void OpponentModel::init(uint32_t seed) {
    for (int i = 0; i < PS_COUNT; i++) post_[i] = 1.0f / (float)PS_COUNT;
    prof_ = PlayerProfile{};
    question_ = 0;
    earlyAggr_ = 0.0f;
    lastArmy_ = 0.0f;
    lastSwitchT_ = 0.0f;
    lastTop_ = PS_MACRO;
}

static inline float softIndicator(float x, float lo, float hi) {
    return saturate((x - lo) / std::max(1e-3f, hi - lo));
}

void OpponentModel::update(const Perception& p, float dt, float t) {
    const Snapshot& s = p.snap();

    // --- HMM-style predict step: beliefs decay toward uniform so the model can
    // change its mind when the player changes plan.
    const float drift = 1.0f - std::exp(-dt / 45.0f);
    for (int i = 0; i < PS_COUNT; i++)
        post_[i] = post_[i] * (1.0f - drift) + drift / (float)PS_COUNT;

    // --- likelihoods from what we have actually observed
    float earlyGame  = 1.0f - softIndicator(t, 180.0f, 420.0f);
    float armySeen   = softIndicator(s.eArmyEstimate, 50.0f, 600.0f);
    float nearUs     = softIndicator(std::max(s.eArmyNearOurBase, s.eCommitPeak), 25.0f, 300.0f);
    // A rush commits its whole army; a raid is three units. Size at our door is
    // the cleanest discriminator between the two.
    float bigCommit  = softIndicator(s.eCommitPeak, 150.0f, 400.0f);
    float ecoHeavy   = softIndicator((float)s.eWorkers, 6.0f, 18.0f);
    float techSeen   = softIndicator((float)s.eWorkshops, 0.0f, 2.0f);
    float expandSeen = softIndicator((float)s.eFoundries, 1.0f, 2.0f);
    float depotHeavy = softIndicator((float)s.eBunkhouses, 2.0f, 6.0f);
    float smallRaid  = nearUs * (1.0f - bigCommit);
    // How recently they pressured us. Absence of aggression is itself evidence:
    // it separates a turtle sitting on an army from someone who is coming.
    float aggrRecent = (s.sinceAggression > 1e8f) ? 0.0f
                                                  : std::exp(-s.sinceAggression / 30.0f);
    const float pressure = s.pressure;          // cumulative, slow-decaying
    const float peace = (1.0f - pressure) * (1.0f - pressure);
    float prodSeen = softIndicator((float)(s.eGarrisons + s.eWorkshops), 0.0f, 2.0f);
    // What fraction of everything we believe they own is thrown at us? A rush
    // commits nearly all of it; a raider keeps the bulk of the army at home.
    // Only meaningful if we have actually looked at their base recently -- the
    // ratio is uninformative when the raiders are all we have ever seen.
    float commitFrac = saturate(s.eCommitPeak / std::max(120.0f, s.eArmyEstimate));
    commitFrac = lerpf(0.5f, commitFrac, s.scoutConfidence);

    float L[PS_COUNT];
    // Aggression is judged on how much they commit and how persistently, not on
    // the clock -- a player attacking continuously stays classified as aggressive.
    L[PS_RUSHING]   = 0.05f + 2.0f * bigCommit + 1.0f * pressure * earlyGame
                            + 0.5f * nearUs + 0.5f * earlyGame * armySeen
                            + 1.6f * pressure * commitFrac;
    // Turtling and macro both require quiet, and quiet is now cumulative.
    L[PS_TURTLING]  = 0.05f + (1.2f * prodSeen + 1.6f * armySeen * (1.0f - nearUs)
                               + 0.9f * depotHeavy) * peace;
    L[PS_MACRO]     = 0.05f + (2.2f * ecoHeavy + 0.30f * (1.0f - armySeen)) * peace;
    L[PS_HARASSING] = 0.05f + 2.2f * smallRaid * (1.0f - bigCommit)
                            + 1.8f * pressure * (1.0f - commitFrac)
                            + 0.4f * aggrRecent * (1.0f - bigCommit);
    L[PS_EXPANDING] = 0.05f + 2.4f * expandSeen + 0.5f * ecoHeavy * peace;
    L[PS_TECHING]   = 0.05f + 2.2f * techSeen + 0.4f * (1.0f - earlyGame) * peace;

    // Down-weight evidence when our picture of them is stale. Raising the
    // likelihood to a fractional power flattens it toward "uninformative",
    // which is the honest thing to do when we have not scouted recently.
    const float conf = clampf(0.25f + 0.75f * s.scoutConfidence, 0.05f, 1.0f);
    float sum = 0.0f;
    for (int i = 0; i < PS_COUNT; i++) {
        post_[i] *= std::pow(std::max(1e-4f, L[i]), conf);
        sum += post_[i];
    }
    if (sum > 1e-9f) for (int i = 0; i < PS_COUNT; i++) post_[i] /= sum;
    // Never let a hypothesis reach certainty. A saturated posterior cannot be
    // moved by new evidence, which is exactly when a player switches plans.
    const float floorP = 0.02f;
    sum = 0.0f;
    for (int i = 0; i < PS_COUNT; i++) { post_[i] = std::max(post_[i], floorP); sum += post_[i]; }
    for (int i = 0; i < PS_COUNT; i++) post_[i] /= sum;

    // --- continuous behaviour profile (EMA; independent of the discrete label)
    const float a = 1.0f - std::exp(-dt / 12.0f);
    prof_.aggression = lerpf(prof_.aggression, nearUs, a);
    prof_.expansion  = lerpf(prof_.expansion, expandSeen, a);
    prof_.defensive  = lerpf(prof_.defensive, armySeen * (1.0f - pressure), a);
    prof_.harass     = lerpf(prof_.harass, smallRaid, a);
    prof_.teching    = lerpf(prof_.teching, techSeen, a);
    float growth = (s.eArmyEstimate - lastArmy_) / std::max(1e-3f, dt) * 60.0f;
    prof_.armyTrend = lerpf(prof_.armyTrend, clampf(growth, -600.0f, 600.0f), a * 0.5f);
    lastArmy_ = s.eArmyEstimate;

    PlayerStrat top = mostLikely();
    if (top != lastTop_) {
        float since = t - lastSwitchT_;
        // Frequent label changes mean a volatile player -- worth knowing, because
        // a volatile opponent is the one worth feinting.
        prof_.volatility = lerpf(prof_.volatility, saturate(30.0f / std::max(4.0f, since)), 0.35f);
        lastSwitchT_ = t;
        lastTop_ = top;
    }

    earlyAggr_ = saturate(post_[PS_RUSHING] * 1.4f + post_[PS_HARASSING] * 0.7f) *
                 (0.4f + 0.6f * earlyGame);

    // --- which question would sharpen the picture the most?
    // Prefer whichever axis we are least sure about *and* have least data on.
    float needArmy   = 1.0f - saturate(s.eArmyEstimate / 300.0f);
    float needTech   = 1.0f - saturate((float)(s.eWorkshops + s.eGarrisons) / 2.0f);
    float needExpand = 1.0f - saturate((float)s.eFoundries / 2.0f);
    needArmy   += post_[PS_RUSHING] * 0.6f;
    needTech   += post_[PS_TECHING] * 0.6f;
    needExpand += post_[PS_EXPANDING] * 0.6f;
    question_ = (needTech > needArmy && needTech > needExpand) ? 1
              : ((needExpand > needArmy) ? 2 : 0);
}

PlayerStrat OpponentModel::mostLikely() const {
    int best = 0;
    for (int i = 1; i < PS_COUNT; i++) if (post_[i] > post_[best]) best = i;
    return (PlayerStrat)best;
}

float OpponentModel::entropy() const {
    float h = 0.0f;
    for (int i = 0; i < PS_COUNT; i++)
        if (post_[i] > 1e-6f) h -= post_[i] * std::log2(post_[i]);
    return h;
}

} // namespace sf::ai
