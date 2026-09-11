// ai_eval.cpp — headless evaluation of the adaptive AI (team 1) against scripted
// opponents (team 0) with distinct, recognisable styles.
//
// Reports win rate, how quickly and how often the AI correctly identifies the
// opponent's archetype, which counter-strategies it selects, and its APM.
#include "sim/Game.h"
#include "ai/AI.h"
#include <cstdio>
#include <cstring>
#include <chrono>
#include <vector>

using namespace sf;

// ---------------------------------------------------------------- opponents
struct Script {
    enum Kind { RUSH = 0, MACRO, TURTLE, HARASS, KIND_COUNT };
    Kind kind = RUSH;
    Game* g = nullptr;
    int team = 0;
    float acc = 0.0f;
    std::vector<EntId> army, raid;
    Rng rng{99};

    static const char* name(Kind k) {
        switch (k) { case RUSH: return "rusher"; case MACRO: return "macro";
                     case TURTLE: return "turtle"; default: return "harasser"; }
    }
    ai::PlayerStrat truth() const {
        switch (kind) {
            case RUSH:   return ai::PS_RUSHING;
            case MACRO:  return ai::PS_MACRO;
            case TURTLE: return ai::PS_TURTLING;
            default:     return ai::PS_HARASSING;
        }
    }

    std::vector<EntId> own(UnitType t) const {
        std::vector<EntId> v;
        for (size_t i = 0; i < g->ents.size(); i++) {
            const Entity& e = g->ents[i];
            if (e.alive && e.deathTimer < 0 && e.team == team && e.type == t &&
                e.buildProgress >= 1.0f) v.push_back(g->handleOf((int)i));
        }
        return v;
    }
    int count(UnitType t) const { return (int)own(t).size(); }

    void update(float dt) {
        acc += dt;
        if (acc < 0.5f) return;
        acc = 0.0f;
        Faction& F = g->fac[team];
        v2 home = g->basePos[team], foe = g->basePos[1 - team];

        int workers = count(UT_WORKER), rax = count(UT_GARRISON);
        int fac = count(UT_WORKSHOP), ccs = count(UT_FOUNDRY);
        int pending = 0;
        for (const Entity& e : g->ents)
            if (e.alive && e.team == team && e.buildProgress < 1.0f) pending++;

        int wantWorkers = (kind == MACRO) ? 22 : (kind == RUSH ? 10 : 16);
        int wantRax     = (kind == RUSH) ? 3 : (kind == MACRO ? 1 : 2);
        int wantFac     = (kind == TURTLE) ? 2 : (kind == MACRO ? 1 : 0);

        auto placeAndBuild = [&](UnitType what) {
            auto ws = own(UT_WORKER);
            if (ws.empty()) return false;
            for (int i = 0; i < 24; i++) {
                float a = rng.f01() * TAU, r = rng.range(10.0f, 26.0f);
                v2 p = home + v2{std::cos(a) * r, std::sin(a) * r};
                if (g->canPlace(what, p)) {
                    std::vector<EntId> sel{ws[0]};
                    return g->cmdBuild(sel, what, p);
                }
            }
            return false;
        };

        if (F.supplyCap - F.supplyUsed < 5 && F.ore >= 100 && pending == 0) { placeAndBuild(UT_BUNKHOUSE); return; }
        if (rax < wantRax && F.ore >= 150 && pending == 0) { placeAndBuild(UT_GARRISON); return; }
        if (rax >= 1 && fac < wantFac && F.ore >= 200 && pending == 0) { placeAndBuild(UT_WORKSHOP); return; }
        if (kind == MACRO && ccs < 2 && F.ore >= 400 && pending == 0) { placeAndBuild(UT_FOUNDRY); return; }

        for (size_t i = 0; i < g->ents.size(); i++) {
            const Entity& e = g->ents[i];
            if (!e.alive || e.deathTimer >= 0 || e.team != team || e.buildProgress < 1.0f) continue;
            EntId h = g->handleOf((int)i);
            if (e.type == UT_FOUNDRY && workers < wantWorkers && e.queue.empty()) g->cmdTrain(h, UT_WORKER);
            else if (e.type == UT_GARRISON && e.queue.size() < 2) g->cmdTrain(h, UT_TROOPER);
            else if (e.type == UT_WORKSHOP && e.queue.size() < 2) g->cmdTrain(h, UT_MAULER);
        }

        // Idle workers back to ore.
        std::vector<EntId> idle;
        for (size_t i = 0; i < g->ents.size(); i++) {
            const Entity& e = g->ents[i];
            if (e.alive && e.deathTimer < 0 && e.team == team && e.type == UT_WORKER &&
                e.order == ORD_IDLE) idle.push_back(g->handleOf((int)i));
        }
        if (!idle.empty())
            for (size_t i = 0; i < g->ents.size(); i++) {
                const Entity& n = g->ents[i];
                if (n.alive && n.type == UT_ORE && n.oreLeft > 0 &&
                    length(n.pos - home) < 45.0f) { g->cmdHarvest(idle, g->handleOf((int)i)); break; }
            }

        // --- army behaviour: this is what makes each archetype recognisable.
        army.clear();
        for (UnitType t : {UT_TROOPER, UT_MAULER}) for (EntId h : own(t)) army.push_back(h);
        if (army.empty()) return;

        switch (kind) {
            case RUSH:
                if ((int)army.size() >= 5) g->cmdMove(army, foe, true);
                break;
            case MACRO:
                if ((int)army.size() >= 14) g->cmdMove(army, foe, true);
                else g->cmdMove(army, home + normalize(foe - home) * 14.0f, true);
                break;
            case TURTLE:
                g->cmdMove(army, home + normalize(foe - home) * 12.0f, true);
                break;
            case HARASS: {
                // Keep a small raiding party permanently in their economy.
                raid.erase(std::remove_if(raid.begin(), raid.end(),
                    [&](EntId h) { return g->get(h) == nullptr; }), raid.end());
                for (EntId h : army) {
                    if (raid.size() >= 3) break;
                    bool in = false;
                    for (EntId o : raid) if (o.v == h.v) in = true;
                    if (!in) raid.push_back(h);
                }
                if (!raid.empty()) g->cmdMove(raid, foe, true);
                std::vector<EntId> rest;
                for (EntId h : army) {
                    bool in = false;
                    for (EntId o : raid) if (o.v == h.v) in = true;
                    if (!in) rest.push_back(h);
                }
                if (!rest.empty()) g->cmdMove(rest, home + normalize(foe - home) * 12.0f, true);
                break;
            }
            default: break;
        }
    }
};

// ---------------------------------------------------------------- one match
struct Result {
    int winner = -1;
    float duration = 0;
    float beliefAcc = 0;     // fraction of samples where top belief == truth
    float firstCorrect = -1; // seconds until first correct identification
    float peakAPM = 0, avgAPM = 0;
    int   switches = 0;
    int   actions = 0, denied = 0;
    int   stratUse[ai::STRAT_COUNT] = {0};
    float scoutConf = 0;
    int   labelHist[ai::PS_COUNT] = {0};
    float maxCommit = 0, avgCommit = 0, maxEst = 0, avgPressure = 0;
};

Result playMatch(Script::Kind kind, uint32_t seed, float maxSeconds) {
    Game g;
    g.init(seed);
    Script sc; sc.kind = kind; sc.g = &g; sc.team = 0;
    ai::Commander ai;
    ai.init(&g, 1, seed);

    Result r;
    const float dt = 1.0f / 30.0f;   // 30 Hz is plenty for evaluation
    int samples = 0, correct = 0, apmSamples = 0;
    float apmSum = 0, confSum = 0;

    while (g.time < maxSeconds && g.winner < 0) {
        sc.update(dt);
        g.update(dt);
        ai.update(dt);

        if (g.time > 30.0) {   // ignore the opening, nothing is observable yet
            samples++;
            if (ai.debug().believed == sc.truth()) {
                correct++;
                if (r.firstCorrect < 0) r.firstCorrect = (float)g.time;
            }
            r.stratUse[ai.debug().strategy]++;
            confSum += ai.debug().scoutConfidence;
            r.labelHist[ai.debug().believed]++;
            const auto& sn2 = ai.perception().snap();
            r.maxCommit = std::max(r.maxCommit, sn2.eCommitPeak);
            r.avgCommit += sn2.eCommitPeak;
            r.maxEst = std::max(r.maxEst, sn2.eArmyEstimate);
            r.avgPressure += sn2.pressure;
        }
        float apm = ai.debug().apm;
        if (apm > 0) { apmSum += apm; apmSamples++; }
        r.peakAPM = std::max(r.peakAPM, ai.debug().apmPeak);
    }
    r.winner = g.winner;
    r.duration = (float)g.time;
    r.beliefAcc = samples ? (float)correct / samples : 0.0f;
    r.scoutConf = samples ? confSum / samples : 0.0f;
    if (samples) { r.avgCommit /= samples; r.avgPressure /= samples; }
    r.avgAPM = apmSamples ? apmSum / apmSamples : 0.0f;
    r.switches = ai.debug().switches;
    r.actions = ai.debug().actions;
    r.denied = ai.debug().denied;
    return r;
}

// Verbose single-match trace, for diagnosing where a game is being lost.
static void trace(Script::Kind kind, uint32_t seed, float maxSec) {
    Game g; g.init(seed);
    Script sc; sc.kind = kind; sc.g = &g; sc.team = 0;
    ai::Commander ai; ai.init(&g, 1, seed);
    printf("=== trace vs %s (seed %u) ===\n", Script::name(kind), seed);
    printf("%5s | %-28s | %-28s | %s\n", "t",
           "AI  w/m/t rax/fac/cc  min sup", "FOE w/m/t rax/fac/cc  min sup",
           "strategy / belief / scout / apm");
    const float dt = 1.0f / 30.0f;
    float next = 0.0f;
    while (g.time < maxSec && g.winner < 0) {
        sc.update(dt); g.update(dt); ai.update(dt);
        if (g.time >= next) {
            next += 45.0f;
            int c[2][UT_COUNT] = {};
            for (const Entity& e : g.ents)
                if (e.alive && e.deathTimer < 0 && e.team < 2 && e.buildProgress >= 1.0f) c[e.team][e.type]++;
            const auto& d = ai.debug();
            printf("%4.0fs | %2d/%2d/%2d  %d/%d/%d  %4d %2d/%-2d | %2d/%2d/%2d  %d/%d/%d  %4d %2d/%-2d | %-14s %-10s %.2f %3.0f\n",
                g.time,
                c[1][UT_WORKER], c[1][UT_TROOPER], c[1][UT_MAULER], c[1][UT_GARRISON], c[1][UT_WORKSHOP], c[1][UT_FOUNDRY],
                g.fac[1].ore, g.fac[1].supplyUsed, g.fac[1].supplyCap,
                c[0][UT_WORKER], c[0][UT_TROOPER], c[0][UT_MAULER], c[0][UT_GARRISON], c[0][UT_WORKSHOP], c[0][UT_FOUNDRY],
                g.fac[0].ore, g.fac[0].supplyUsed, g.fac[0].supplyCap,
                ai::strategyName(d.strategy), ai::playerStratName(d.believed),
                d.scoutConfidence, d.apm);
            const auto& sn = ai.perception().snap();
            printf("      | pressure=%.2f commitPeak=%.0f ourHalf=%.0f est=%.0f nearBase=%.0f eW=%d eRax=%d eFac=%d eCC=%d eDep=%d sinceAggr=%.0f\n",
                   sn.pressure, sn.eCommitPeak, sn.eArmyOurHalf, sn.eArmyEstimate, sn.eArmyNearOurBase, sn.eWorkers,
                   sn.eGarrisons, sn.eWorkshops, sn.eFoundries, sn.eBunkhouses,
                   sn.sinceAggression > 1e8f ? -1.0f : sn.sinceAggression);
        }
    }
    printf("winner=%d at %.0fs\n\n", g.winner, g.time);
}

int main(int argc, char** argv) {
    if (argc > 1 && !strcmp(argv[1], "--trace")) {
        int k = (argc > 2) ? atoi(argv[2]) : 0;
        trace((Script::Kind)k, 1000, (argc > 3) ? (float)atof(argv[3]) : 420.0f);
        return 0;
    }
    int games = (argc > 1) ? atoi(argv[1]) : 3;
    float maxSec = (argc > 2) ? (float)atof(argv[2]) : 600.0f;

    printf("Adaptive AI evaluation — %d games per opponent, %.0fs cap\n\n", games, maxSec);
    printf("%-10s %6s %7s %9s %9s %6s %7s %8s  %s\n",
           "opponent", "wins", "avg len", "belief acc", "1st ident", "scout", "APM", "switches", "top strategies");
    printf("%s\n", std::string(104, '-').c_str());

    auto t0 = std::chrono::high_resolution_clock::now();
    int totalWins = 0, totalGames = 0;

    for (int k = 0; k < Script::KIND_COUNT; k++) {
        int wins = 0; float len = 0, acc = 0, ident = 0, apm = 0, peak = 0, conf = 0; int sw = 0;
        int use[ai::STRAT_COUNT] = {0};
        int hist[ai::PS_COUNT] = {0};
        int identN = 0;
        float mxc = 0, avc = 0, mxe = 0, avp = 0;
        long acts = 0, dn = 0;
        for (int i = 0; i < games; i++) {
            Result r = playMatch((Script::Kind)k, 1000u + i * 37u, maxSec);
            if (r.winner == 1) wins++;
            len += r.duration; acc += r.beliefAcc; apm += r.avgAPM; conf += r.scoutConf;
            peak = std::max(peak, r.peakAPM); sw += r.switches;
            if (r.firstCorrect >= 0) { ident += r.firstCorrect; identN++; }
            for (int s = 0; s < ai::STRAT_COUNT; s++) use[s] += r.stratUse[s];
            for (int s = 0; s < ai::PS_COUNT; s++) hist[s] += r.labelHist[s];
            acts += r.actions; dn += r.denied;
            mxc = std::max(mxc, r.maxCommit); avc += r.avgCommit;
            mxe = std::max(mxe, r.maxEst); avp += r.avgPressure;
            totalGames++;
        }
        totalWins += wins;
        // Name the two most-used counter-strategies.
        int a = 0, b = 1;
        for (int s = 0; s < ai::STRAT_COUNT; s++) if (use[s] > use[a]) a = s;
        for (int s = 0; s < ai::STRAT_COUNT; s++) if (s != a && use[s] > use[b]) b = s;
        char top[80];
        snprintf(top, sizeof(top), "%s, %s", ai::strategyName((ai::Strategy)a),
                 ai::strategyName((ai::Strategy)b));
        printf("%-10s %3d/%-2d %6.0fs %8.0f%% %8.0fs %5.2f %3.0f/%-3.0f %8.1f  %s\n",
               Script::name((Script::Kind)k), wins, games, len / games,
               acc / games * 100.0f, identN ? ident / identN : -1.0f, conf / games,
               apm / games, peak, (float)sw / games, top);
        printf("%12s actions %ld total, %ld refused by the APM cap (%.0f%% of intents)\n",
               "", acts / games, dn / games, 100.0 * dn / std::max(1L, acts + dn));
        printf("%12s signals: commitPeak max %.0f avg %.0f | armyEst max %.0f | pressure %.2f\n",
               "", mxc, avc / games, mxe, avp / games);
        int tot = 0; for (int s = 0; s < ai::PS_COUNT; s++) tot += hist[s];
        if (tot) {
            printf("%12s believed: ", "");
            for (int s = 0; s < ai::PS_COUNT; s++)
                if (hist[s] * 100 / tot >= 5)
                    printf("%s %d%%  ", ai::playerStratName((ai::PlayerStrat)s), hist[s] * 100 / tot);
            printf("\n");
        }
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    printf("\noverall: %d/%d wins (%.0f%%)   wall time %.1fs\n", totalWins, totalGames,
           100.0f * totalWins / std::max(1, totalGames),
           std::chrono::duration<double>(t1 - t0).count());
    return 0;
}
