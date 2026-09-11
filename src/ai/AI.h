// AI.h — adaptive opponent for Starforge.
//
// The AI plays under the same restrictions as the human: it only sees what its
// own units see (Game::visible), it issues orders exclusively through the same
// cmd* API the mouse drives, and every order is paid for out of an APM budget
// sized to a top StarCraft II professional.
//
// Structure follows the modular pipeline from ai-research.md:
//
//   Perception -> OpponentModel -> StrategySelector -> {Macro, Scouts, Tactics}
//                      ^                                        |
//                      +------------- observations -------------+
//
// Hardware mapping (hardware.md): symbolic reasoning, Bayesian inference and
// orchestration on the CPU; the spatial influence fields on the GPU.
#pragma once
#include <vector>
#include <string>
#include <deque>
#include "../sim/Game.h"

namespace sf::ai {

// ---------------------------------------------------------------- strategies
enum Strategy {
    STRAT_ECO = 0,        // workers, expand, tech up; minimal army
    STRAT_TROOPER_RUSH,    // early garrison, commit troopers before maulers exist
    STRAT_HARASS,         // small squads into the worker line, refuse fights
    STRAT_TIMING_PUSH,    // mass to a threshold, then commit everything
    STRAT_TURTLE_TECH,    // hold the ramp, tech to maulers, punish overextension
    STRAT_EXPAND,         // take a second ore line
    STRAT_COUNTER_ATTACK, // hit the base while their army is away from it
    STRAT_FEINT,          // show force one place, strike another
    STRAT_COUNT
};
const char* strategyName(Strategy s);

// What we believe the *player* is doing.
enum PlayerStrat {
    PS_RUSHING = 0, PS_MACRO, PS_TURTLING, PS_HARASSING, PS_EXPANDING, PS_TECHING,
    PS_COUNT
};
const char* playerStratName(PlayerStrat s);

// ---------------------------------------------------------------- APM budget
// Token bucket. `regen` is the sustained rate; `cap` allows short bursts, which
// is how real APM behaves (a pro idles, then spikes during an engagement).
struct ActionBudget {
    float tokens = 0.0f;
    // Sized against a top StarCraft II professional: ~330 sustained actions per
    // minute, with a burst reserve that lets a five-second engagement spike to
    // roughly 500 APM, then settle. Idle periods cost nothing.
    float regen  = 5.5f;     // actions/sec sustained (=330 APM)
    float cap    = 14.0f;    // burst depth
    int   lifetime = 0;
    int   denied = 0;        // times an intended action was refused by the cap

    void  tick(float dt);
    bool  spend(int n = 1);
    bool  can(int n = 1) const { return tokens >= (float)n; }
    float measuredAPM() const;                 // rolling 60 s average
    float peakAPM(float window = 5.0f) const;  // best short window, as APM meters report
    void  note(float gameTime);
private:
    std::deque<float> stamps_;   // action timestamps inside a 60 s window
};

// ---------------------------------------------------------------- perception
// One remembered enemy object. The AI keeps these after they leave vision, with
// decaying confidence, and clears them when it looks and finds nothing there.
struct Remembered {
    UnitType type = UT_TROOPER;
    v2       pos{};
    float    firstSeen = 0.0f;
    float    lastSeen  = 0.0f;
    bool     stillVisible = false;
};

struct Snapshot {
    // Own side (always fully known -- you can see your own units).
    int workers = 0, troopers = 0, maulers = 0;
    int ccs = 0, garrison = 0, workshops = 0, bunkhouses = 0, pending = 0;
    int pendingType[UT_COUNT] = {0};   // per-type, so a garrison cannot block a bunkhouse
    int ore = 0, supplyUsed = 0, supplyCap = 0;
    float armyValue = 0.0f;

    // Believed enemy state, derived only from what we have actually observed.
    int   eWorkers = 0, eTroopers = 0, eMaulers = 0;
    int   eFoundries = 0, eGarrisons = 0, eWorkshops = 0, eBunkhouses = 0;
    float eArmyValue = 0.0f;       // believed present right now
    float eArmyEstimate = 0.0f;    // decaying peak: what we think they have overall
    float eArmyOurHalf = 0.0f;     // enemy combat units on our side of the map
    float eCommitPeak = 0.0f;      // decaying peak of force they have committed at us
    float eArmyNearOurBase = 0.0f;   // believed enemy army value close to home
    float eArmyAwayFromHome = 0.0f;  // ... and far from their own base
    float sinceAggression = 1e9f;    // seconds since we last saw them at our door
    float pressure = 0.0f;           // cumulative aggression, decays slowly
    bool  enemyBaseKnown = false;
    v2    enemyBase{};
    float scoutConfidence = 0.0f;    // 0..1, how fresh our picture of them is
};

class Perception {
public:
    void init(Game* g, int team);
    void update(float dt);

    const std::vector<Remembered>& enemies() const { return mem_; }
    const Snapshot& snap() const { return snap_; }
    // Candidate enemy main-base sites, best guess first.
    const std::vector<v2>& baseGuesses() const { return guesses_; }
    v2   home() const { return home_; }
    // Threats we have seen for at least `reactionDelay` -- the AI is not allowed
    // to react to something it only just laid eyes on.
    bool reactable(const Remembered& r, float reactionDelay) const {
        return (float)g_->time - r.firstSeen >= reactionDelay;
    }

private:
    void observe();
    void forget();

    Game* g_ = nullptr;
    int   team_ = 1;
    v2    home_{};
    std::vector<Remembered> mem_;
    std::vector<v2> guesses_;
    Snapshot snap_;
    float aggrT_ = -1e9f;   // last time we saw them pressuring our base
    float aggrScore_ = 0.0f; // cumulative aggression: a rusher stays high between waves
    float armyEst_ = 0.0f;   // decaying estimate of their total army
    float commitPeak_ = 0.0f;// decaying peak of committed force
};

// ---------------------------------------------------------------- opponent model
// Bayesian posterior over a small strategy set, fused with a continuously
// updated behaviour profile. Both are cheap and interpretable, which is what
// ai-research.md recommends over an opaque end-to-end model at this scale.
struct PlayerProfile {
    float aggression = 0.45f;    // pushes out / attacks
    float expansion  = 0.30f;    // takes extra bases
    float defensive  = 0.40f;    // keeps army home
    float harass     = 0.25f;    // sends small squads at workers
    float teching    = 0.35f;    // invests in workshops/maulers
    float armyTrend  = 0.0f;     // observed army growth per minute
    float volatility = 0.3f;     // how often they switch behaviour
};

class OpponentModel {
public:
    void init(uint32_t seed);
    void update(const Perception& p, float dt, float gameTime);

    float belief(PlayerStrat s) const { return post_[s]; }
    PlayerStrat mostLikely() const;
    float entropy() const;                  // uncertainty over their plan, in bits
    const PlayerProfile& profile() const { return prof_; }
    // Which enemy fact would most reduce our uncertainty right now?
    // 0 = their army, 1 = their tech, 2 = their expansions.
    int   mostValuableQuestion() const { return question_; }
    float threatOfEarlyAggression() const { return earlyAggr_; }

private:
    float post_[PS_COUNT];
    PlayerProfile prof_;
    int   question_ = 0;
    float earlyAggr_ = 0.0f;
    float lastArmy_ = 0.0f;
    float lastSwitchT_ = 0.0f;
    PlayerStrat lastTop_ = PS_MACRO;
};

// ---------------------------------------------------------------- influence maps
struct InfluenceUnit {   // 32 B, matches the Metal struct
    float x, z, strength, range;
    float team, pad0, pad1, pad2;
};

// Backend interface so the simulation-only build links without Metal.
struct IInfluenceBackend {
    virtual ~IInfluenceBackend() = default;
    virtual bool compute(const InfluenceUnit* units, int n,
                         int gridN, float cellSize, float* out) = 0;
    virtual const char* name() const = 0;
};
// Implemented in InfluenceMapGPU.mm; returns nullptr (and fills err) on failure.
IInfluenceBackend* createMetalInfluenceBackend(std::string& err);

class InfluenceMap {
public:
    static constexpr int   N = 64;
    static constexpr int   CHANNELS = 4;   // 0 friendly, 1 enemy, 2 threat, 3 unseen
    static constexpr float CELL = Terrain::SIZE / N;

    void setBackend(IInfluenceBackend* b) { backend_ = b; }
    const char* backendName() const { return backend_ ? backend_->name() : "cpu"; }
    void build(const Perception& p, Game& g, int team);

    float sample(int channel, v2 world) const;
    const float* data() const { return field_.data(); }
    // Best position to fight from: high friendly, low enemy threat.
    v2 safestNear(v2 around, float radius) const;

private:
    void computeCPU(const InfluenceUnit* u, int n);
    IInfluenceBackend* backend_ = nullptr;
    std::vector<InfluenceUnit> units_;
    std::vector<float> field_ = std::vector<float>(N * N * CHANNELS, 0.0f);
};

// ---------------------------------------------------------------- strategy
// Linear contextual bandit. Each strategy scores a shared feature vector with
// its own weight row; weights start from a hand-designed prior so the AI is
// competent from the first game, then move online toward whatever is actually
// working against this particular player.
class StrategySelector {
public:
    static constexpr int NFEAT = 20;

    void init(uint32_t seed);
    void update(const Snapshot& s, const OpponentModel& om, float gameTime, float dt);

    Strategy current() const { return cur_; }
    float    scoreOf(Strategy s) const { return score_[s]; }
    float    confidence() const { return conf_; }
    const char* reason() const { return reason_; }
    int      switches() const { return switches_; }

private:
    void   features(const Snapshot& s, const OpponentModel& om, float t, float* f) const;
    bool   legal(Strategy s, const Snapshot& snap) const;
    void   reward(float r);

    float w_[STRAT_COUNT][NFEAT];
    float score_[STRAT_COUNT];
    float lastFeat_[NFEAT];
    Strategy cur_ = STRAT_ECO;
    Strategy pending_ = STRAT_ECO;
    float  commitT_ = 0.0f;     // hysteresis: don't thrash between plans
    float  conf_ = 0.0f;
    float  lastAdv_ = 0.0f;
    float  evalT_ = 0.0f;
    int    switches_ = 0;
    char   reason_[96] = {0};
    Rng    rng_{7};
};

// ---------------------------------------------------------------- commander
struct AIDebug {
    Strategy strategy = STRAT_ECO;
    PlayerStrat believed = PS_MACRO;
    float beliefs[PS_COUNT] = {0};
    float stratScores[STRAT_COUNT] = {0};
    float entropy = 0, scoutConfidence = 0, apm = 0, apmPeak = 0;
    int   squads = 0, switches = 0;
    int   actions = 0, denied = 0;
    const char* reason = "";
    const char* backend = "cpu";
    v2    scoutTarget{};
    float sinceAggression = 1e9f;    // seconds since we last saw them at our door
    float pressure = 0.0f;           // cumulative aggression, decays slowly
    bool  enemyBaseKnown = false;
    v2    enemyBase{};
};

class Commander {
public:
    void init(Game* g, int team, uint32_t seed, IInfluenceBackend* gpu = nullptr);
    void update(float dt);
    const AIDebug& debug() const { return dbg_; }
    const Perception& perception() const { return per_; }

private:
    // Each of these may spend from the shared action budget.
    void think(float dt);
    void runMacro(float dt);
    void runScouts(float dt);
    void runTactics(float dt);
    void runMicro(float dt);

    // Order issuing -- the only path to the game, and it costs actions.
    bool order(const std::vector<EntId>& sel, int kind, v2 pos, EntId target);
    bool train(EntId building, UnitType what);
    bool build(EntId worker, UnitType what, v2 where);
    std::vector<EntId> ownOf(UnitType t, bool idleOnly = false) const;
    EntId nearestOwnBuilding(UnitType t) const;
    bool  placeNear(UnitType what, v2 around, float rmin, float rmax, v2& out) const;
    bool  reserved(EntId h) const;   // unit is committed to a squad, not free labour

    Game* g_ = nullptr;
    int   team_ = 1;
    Perception per_;
    OpponentModel opp_;
    StrategySelector strat_;
    InfluenceMap infl_;
    ActionBudget budget_;
    AIDebug dbg_;
    mutable Rng rng_{11};

    float thinkT_ = 0, macroT_ = 0, scoutT_ = 0, tacticT_ = 0, microT_ = 0;
    EntId focusTarget_;
    float reactionDelay_ = 0.22f;

    // squads
    std::vector<EntId> main_, harass_, scouts_;
    v2    rally_{}, harassTarget_{}, feintTarget_{};
    bool  committed_ = false;
    float commitT_ = 0.0f;
    float lastSelHash_ = 0.0f;
    int   scoutGuess_ = 0;
    float retreatUntil_ = 0.0f;
};

} // namespace sf::ai
