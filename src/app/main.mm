// main.mm — AppKit shell, RTS camera, input handling and HUD assembly.
// Owns the game/renderer pair and translates input into simulation commands.
#import <Cocoa/Cocoa.h>
#import <QuartzCore/CAMetalLayer.h>
#import <QuartzCore/CADisplayLink.h>
#include <cstdio>
#include <cstring>
#include <string>
#include "../sim/Game.h"
#include "../gfx/Renderer.h"
#include "../ai/AI.h"

using namespace sf;

// ---------------------------------------------------------------- app state
namespace {

struct Input {
    bool keys[256] = {false};
    bool keyPressed[256] = {false};      // edge-triggered, cleared each frame
    bool shift = false, ctrl = false, cmd = false, alt = false;
    float mx = 0, my = 0;                // drawable pixels, origin top-left
    // Where the cursor sits inside the view, 0..1. The drawable is not the view
    // -- adaptive resolution resizes it underneath a stationary cursor -- so the
    // fraction is the durable quantity and mx/my are re-derived from it.
    float mnx = 0.5f, mny = 0.5f;
    float scroll = 0;
    bool  lDown = false, rDown = false;
    bool  lPressed = false, lReleased = false, rPressed = false;
    bool  dragging = false;
    bool  mouseInside = false;   // gates edge-scroll; false until the cursor enters
    float dragX = 0, dragY = 0;
    double lastClickTime = 0;
    bool  doubleClick = false;
};

struct Camera {
    v2 focus{60, 60};
    float dist = 62.0f;
    float yaw = 0.7f;
    float smoothY = 0.0f;
};

enum UIMode { MODE_NORMAL = 0, MODE_ATTACK_TARGET, MODE_PLACE_BUILDING };

struct Button {
    float x, y, w, h;
    char  label[24];
    char  hotkey;
    int   action;        // >=0 : UnitType to train/build ; -1 none
    bool  isBuild;
    bool  enabled;
};

struct App {
    Game     game;
    ai::Commander enemyAI;
    ai::IInfluenceBackend* gpuInfluence = nullptr;
    bool     showAI = false;
    Renderer renderer;
    RenderFrame frame;
    Camera   cam;
    Input    in;
    std::vector<EntId> selection;
    std::vector<EntId> groups[10];
    std::vector<Button> buttons;
    UIMode   mode = MODE_NORMAL;
    UnitType buildType = UT_BUNKHOUSE;
    int   pixelW = 1, pixelH = 1;
    float scale = 2.0f;
    double lastTime = 0;
    float fps = 60.0f;
    bool  paused = false;
    bool  showHelp = true;
    float speed = 1.0f;
    uint32_t seed = 1;
    float    aiMs = 0.0f;          // rolling cost of one AI update
    v3   cursorWorld{0, 0, 0};
    bool cursorValid = false;
    EntId hovered;
    char  toast[96] = {0};
    float toastTimer = 0;
    bool  menuOpen = false;      // Esc menu; modal, and freezes the simulation
    int   menuIndex = 0;         // keyboard cursor within it
    std::vector<float> fowVis, fowExp;   // eased copies of the sim's visibility grid
};

App* gApp = nullptr;

void say(const char* msg) {
    if (!gApp) return;
    snprintf(gApp->toast, sizeof(gApp->toast), "%s", msg);
    gApp->toastTimer = 2.6f;
}

// ---------------------------------------------------------------- helpers
v3 teamColor(int team) {
    if (team == 0) return v3{0.28f, 0.55f, 1.00f};
    if (team == 1) return v3{1.00f, 0.28f, 0.22f};
    return v3{0.55f, 0.60f, 0.65f};
}

m4 orientTo(v3 pos, v3 fwd, float s) {
    v3 f = normalize(fwd);
    v3 up = std::fabs(f.y) > 0.98f ? v3{1, 0, 0} : v3{0, 1, 0};
    v3 r = normalize(cross(up, f));
    v3 u = cross(f, r);
    m4 m;
    m.c[0] = v4(r * s, 0); m.c[1] = v4(u * s, 0); m.c[2] = v4(f * s, 0); m.c[3] = v4(pos, 1);
    return m;
}

// Camera pitch steepens as you zoom out, which keeps both close-in detail and
// wide tactical views readable.
float camPitch(float dist) { return lerpf(0.55f, 0.95f, saturate((dist - 28.0f) / 110.0f)); }

v3 camEye(const Camera& c, float groundY) {
    float p = camPitch(c.dist);
    v3 focus{c.focus.x, groundY, c.focus.y};
    v3 dir{std::sin(c.yaw) * std::cos(p), std::sin(p), std::cos(c.yaw) * std::cos(p)};
    return focus + dir * c.dist;
}

bool screenRay(App& a, float px, float py, v3& origin, v3& dir) {
    float ndcX = px / (float)a.pixelW * 2.0f - 1.0f;
    float ndcY = 1.0f - py / (float)a.pixelH * 2.0f;
    v4 n = a.frame.invViewProj * v4{ndcX, ndcY, 0.0f, 1.0f};
    v4 f = a.frame.invViewProj * v4{ndcX, ndcY, 1.0f, 1.0f};
    if (std::fabs(n.w) < 1e-9f || std::fabs(f.w) < 1e-9f) return false;
    origin = n.xyz() / n.w;
    dir = normalize(f.xyz() / f.w - origin);
    return true;
}

bool projectToScreen(App& a, v3 world, float& sx, float& sy) {
    v4 c = a.frame.viewProj * v4(world, 1.0f);
    if (c.w <= 0.01f) return false;
    sx = (c.x / c.w * 0.5f + 0.5f) * a.pixelW;
    sy = (1.0f - (c.y / c.w * 0.5f + 0.5f)) * a.pixelH;
    return true;
}

// Map a point in view coordinates (origin bottom-left, points) onto the
// drawable (origin top-left, pixels). Pure, so --mouse-test can check it at a
// render scale without a window or an event to hand.
struct MousePos { float nx, ny, x, y; };
MousePos mapMouse(double px, double py, double bw, double bh, int pixelW, int pixelH) {
    MousePos m;
    m.nx = (bw > 0.0) ? (float)(px / bw) : 0.5f;
    m.ny = (bh > 0.0) ? (float)(1.0 - py / bh) : 0.5f;
    m.x  = m.nx * (float)pixelW;
    m.y  = m.ny * (float)pixelH;
    return m;
}

// The adaptive-resolution controller, factored out of the frame loop so a
// sequence of measurements can be replayed through it without a GPU.
struct AdaptState {
    int   slow = 0;        // consecutive windows over budget
    int   good = 0;        // consecutive windows with room to step back up
    int   settle = 0;      // windows to ignore right after a change
    int   upFails = 0;     // step-ups that had to be undone; raises the bar each time
    float lastUp = 0.0f;   // the scale a step up reached, to recognise it being undone
};
float adaptStep(AdaptState& st, float scale, double avgMs) {
    const float kFloor = 0.7f, kStep = 0.15f;
    const double kDown = 14.0;       // missing the 16.7 ms budget
    const double kUpBudget = 12.5;   // predicted cost at the *next* scale must clear this
    // The window straddling a resize measures two different resolutions and a
    // pile of texture reallocation. It is not evidence about either one.
    if (st.settle > 0) { st.settle--; return scale; }

    if (avgMs > kDown) {
        st.good = 0;
        if (++st.slow < 2 || scale <= kFloor + 0.01f) return scale;
        // Coming back down to a scale we had just climbed up from means that
        // climb was a mistake. A machine sharing its GPU with something bursty
        // -- a chat client compositing a notification panel, say -- can pass the
        // step-up test during every quiet spell and fail it during every busy
        // one, and the resolution then changes every few seconds forever. That
        // reads as the whole HUD flickering, and it is far more distracting than
        // simply running a little softer. Each failed attempt therefore doubles
        // the clean run the next one must see -- plain exponential backoff, so
        // the loop converges in a few probes instead of ringing forever.
        if (st.lastUp > 0.0f && scale >= st.lastUp - 0.001f) st.upFails++;
        st.slow = 0; st.settle = 1;
        return std::max(kFloor, scale - kStep);
    }

    st.slow = 0;
    float next = std::min(1.0f, scale + kStep);
    if (next <= scale + 0.001f) { st.good = 0; return scale; }
    // A step up multiplies the pixel count by (new/old)^2, so the question is
    // not whether the current scale is comfortable but whether the next one
    // will be. Testing the former against a fixed threshold is what lets the
    // controller step into a budget it cannot afford and bounce straight back.
    double predicted = avgMs * (double)(next * next) / (double)(scale * scale);
    if (predicted >= kUpBudget) { st.good = 0; return scale; }
    int need = 3 << std::min(st.upFails, 3);   // 3, 6, 12, 24 windows
    if (++st.good < need) return scale;
    st.good = 0; st.settle = 1; st.lastUp = next;
    return next;
}

} // namespace

// ---------------------------------------------------------------- view
@interface GameView : NSView
@end

@implementation GameView {
    CAMetalLayer* _metal;
    NSTrackingArea* _tracking;
}

- (instancetype)initWithFrame:(NSRect)f {
    self = [super initWithFrame:f];
    if (!self) return nil;
    self.wantsLayer = YES;
    _metal = [CAMetalLayer layer];
    _metal.pixelFormat = MTLPixelFormatBGRA8Unorm;
    _metal.framebufferOnly = YES;
    self.layer = _metal;
    return self;
}
- (CAMetalLayer*)metalLayer { return _metal; }
- (BOOL)acceptsFirstResponder { return YES; }
- (BOOL)wantsUpdateLayer { return YES; }

- (void)updateTrackingAreas {
    if (_tracking) [self removeTrackingArea:_tracking];
    _tracking = [[NSTrackingArea alloc]
        initWithRect:self.bounds
             options:(NSTrackingMouseMoved | NSTrackingMouseEnteredAndExited |
                        NSTrackingActiveInKeyWindow | NSTrackingInVisibleRect)
               owner:self userInfo:nil];
    [self addTrackingArea:_tracking];
    [super updateTrackingAreas];
}

- (void)setMouseFrom:(NSEvent*)e {
    if (!gApp) return;
    NSPoint p = [self convertPoint:[e locationInWindow] fromView:nil];
    NSSize b = self.bounds.size;
    // Deliberately not backingScaleFactor: that is the size of the drawable only
    // while the renderer is running at full resolution. Under adaptive
    // resolution it is not, and scaling by it puts the cursor somewhere the
    // game is not drawing.
    MousePos m = mapMouse(p.x, p.y, b.width, b.height, gApp->pixelW, gApp->pixelH);
    gApp->in.mnx = m.nx; gApp->in.mny = m.ny;
    gApp->in.mx  = m.x;  gApp->in.my  = m.y;
    gApp->in.mouseInside = true;
}

- (void)mouseEntered:(NSEvent*)e { if (gApp) gApp->in.mouseInside = true; }
- (void)mouseExited:(NSEvent*)e  { if (gApp) gApp->in.mouseInside = false; }

- (void)mouseMoved:(NSEvent*)e      { [self setMouseFrom:e]; }
- (void)mouseDragged:(NSEvent*)e    { [self setMouseFrom:e]; if (gApp) gApp->in.dragging = true; }
- (void)rightMouseDragged:(NSEvent*)e { [self setMouseFrom:e]; }
- (void)mouseDown:(NSEvent*)e {
    [self setMouseFrom:e];
    if (!gApp) return;
    double now = CACurrentMediaTime();
    gApp->in.doubleClick = (now - gApp->in.lastClickTime) < 0.32;
    gApp->in.lastClickTime = now;
    gApp->in.lDown = true; gApp->in.lPressed = true;
    gApp->in.dragX = gApp->in.mx; gApp->in.dragY = gApp->in.my;
    gApp->in.dragging = false;
}
- (void)mouseUp:(NSEvent*)e {
    [self setMouseFrom:e];
    if (!gApp) return;
    gApp->in.lDown = false; gApp->in.lReleased = true;
}
- (void)rightMouseDown:(NSEvent*)e {
    [self setMouseFrom:e];
    if (!gApp) return;
    gApp->in.rDown = true; gApp->in.rPressed = true;
}
- (void)rightMouseUp:(NSEvent*)e { [self setMouseFrom:e]; if (gApp) gApp->in.rDown = false; }
- (void)scrollWheel:(NSEvent*)e { if (gApp) gApp->in.scroll += (float)e.scrollingDeltaY; }

- (void)keyDown:(NSEvent*)e {
    if (!gApp) return;
    NSString* s = [e charactersIgnoringModifiers];
    if (s.length) {
        unichar c = [s characterAtIndex:0];
        if (c < 256) {
            unsigned char k = (unsigned char)toupper(c);
            if (!gApp->in.keys[k]) gApp->in.keyPressed[k] = true;
            gApp->in.keys[k] = true;
        }
    }
    switch (e.keyCode) {   // arrows and Esc have no usable character code
        case 123: if (!gApp->in.keys[1]) gApp->in.keyPressed[1] = true; gApp->in.keys[1] = true; break;
        case 124: if (!gApp->in.keys[2]) gApp->in.keyPressed[2] = true; gApp->in.keys[2] = true; break;
        case 125: if (!gApp->in.keys[3]) gApp->in.keyPressed[3] = true; gApp->in.keys[3] = true; break;
        case 126: if (!gApp->in.keys[4]) gApp->in.keyPressed[4] = true; gApp->in.keys[4] = true; break;
        case 53:  gApp->in.keyPressed[27] = true; break;
        default: break;
    }
}
- (void)keyUp:(NSEvent*)e {
    if (!gApp) return;
    NSString* s = [e charactersIgnoringModifiers];
    if (s.length) {
        unichar c = [s characterAtIndex:0];
        if (c < 256) gApp->in.keys[(unsigned char)toupper(c)] = false;
    }
    switch (e.keyCode) {
        case 123: gApp->in.keys[1] = false; break;
        case 124: gApp->in.keys[2] = false; break;
        case 125: gApp->in.keys[3] = false; break;
        case 126: gApp->in.keys[4] = false; break;
        default: break;
    }
}
- (void)flagsChanged:(NSEvent*)e {
    if (!gApp) return;
    NSEventModifierFlags f = e.modifierFlags;
    gApp->in.shift = (f & NSEventModifierFlagShift) != 0;
    gApp->in.ctrl  = (f & NSEventModifierFlagControl) != 0;
    gApp->in.cmd   = (f & NSEventModifierFlagCommand) != 0;
    gApp->in.alt   = (f & NSEventModifierFlagOption) != 0;
}
@end

// ---------------------------------------------------------------- window
// Borderless full screen rather than AppKit's native one. Native full screen
// keeps the menu bar and the title bar one mouse-move away from the top of the
// screen -- and this game pans the camera when the pointer touches an edge, so
// the system chrome would drop down every time the player scrolled north.
@interface GameWindow : NSWindow
@end
@implementation GameWindow
- (BOOL)canBecomeKeyWindow  { return YES; }   // borderless windows opt out by default
- (BOOL)canBecomeMainWindow { return YES; }
@end

namespace {

bool              gFSDebug = false;   // --fs-debug reports the presentation state
GameWindow*       gWindow = nil;
bool              gIsFullscreen = false;

// Hiding the menu bar requires hiding the Dock too -- AppKit raises otherwise.
// Both revert whenever the app is not frontmost, and another app that sets its
// own presentation options while it has focus can leave ours dropped on the way
// back. So this is reasserted on every activation rather than set once: a chat
// client popping a notification panel over a full-screen game must not leave the
// menu bar sitting at the top of the screen afterwards.
void applyFullscreenPresentation() {
    NSApp.presentationOptions = NSApplicationPresentationHideDock |
                                NSApplicationPresentationHideMenuBar;
}
void reassertFullscreenPresentation() {
    if (gIsFullscreen) applyFullscreenPresentation();
}
NSRect            gSavedFrame = NSZeroRect;
NSWindowStyleMask gSavedStyle = 0;

void setFullscreen(bool on) {
    if (!gWindow || on == gIsFullscreen) return;
    if (on) {
        gSavedFrame = gWindow.frame;
        gSavedStyle = gWindow.styleMask;
        applyFullscreenPresentation();
        NSScreen* scr = gWindow.screen ?: [NSScreen mainScreen];
        gWindow.styleMask = NSWindowStyleMaskBorderless;
        [gWindow setFrame:scr.frame display:YES];
    } else {
        NSApp.presentationOptions = NSApplicationPresentationDefault;
        gWindow.styleMask = gSavedStyle;
        [gWindow setFrame:gSavedFrame display:YES];
        [gWindow center];
    }
    gIsFullscreen = on;
    // Verification hook: proves the chrome is actually gone rather than merely
    // auto-hidden. A borderless style mask means there is no title bar to slide
    // down, and HideMenuBar (as opposed to AutoHideMenuBar) means the menu bar
    // does not come back when the pointer reaches the top of the screen -- which
    // matters here because that is also how the camera pans north.
    if (gFSDebug)
        fprintf(stderr, "fullscreen=%d frame=%.0fx%.0f style=0x%lx presentation=0x%lx "
                        "menuBarVisible=%d\n", (int)on, gWindow.frame.size.width,
                gWindow.frame.size.height, (unsigned long)gWindow.styleMask,
                (unsigned long)NSApp.presentationOptions, (int)[NSMenu menuBarVisible]);
    // Changing the style mask drops the first responder on the floor.
    [gWindow makeFirstResponder:gWindow.contentView];
    [gWindow makeKeyAndOrderFront:nil];
}
void toggleFullscreen() { setFullscreen(!gIsFullscreen); }

} // namespace

// ---------------------------------------------------------------- gameplay glue
namespace {

constexpr int ACT_NONE = -1, ACT_STOP = 200, ACT_HOLD = 201, ACT_ATTACK = 202;

void resetMatch(App& a, uint32_t seed);   // defined with the frame loop below

// --- Esc menu -----------------------------------------------------------------
enum { MI_RESUME = 0, MI_RESTART, MI_FULLSCREEN, MI_HELP, MI_QUIT, MI_COUNT };
const char* kMenuLabels[MI_COUNT] = {
    "Resume", "Restart Match", "Toggle Full Screen", "Toggle Help Panel", "Quit Starforge"
};
const char* kMenuHints[MI_COUNT] = {
    "Esc", "new map, new seed", "ctrl-cmd-F", "/", "" };

// Geometry lives in one place so hit-testing and drawing cannot drift apart.
struct MenuGeom { float px, py, pw, ph, ix, iy, iw, ih, gap; };
MenuGeom menuGeom(const App& a) {
    const float S = a.scale;
    MenuGeom m;
    m.iw = 320 * S; m.ih = 42 * S; m.gap = 7 * S;
    m.pw = m.iw + 56 * S;
    m.ph = 78 * S + (float)MI_COUNT * (m.ih + m.gap) + 18 * S;
    m.px = (a.pixelW - m.pw) * 0.5f;
    m.py = (a.pixelH - m.ph) * 0.5f;
    m.ix = m.px + 28 * S;
    m.iy = m.py + 78 * S;
    return m;
}
int menuHit(const App& a, float x, float y) {
    MenuGeom m = menuGeom(a);
    if (x < m.ix || x > m.ix + m.iw) return -1;
    for (int i = 0; i < MI_COUNT; i++) {
        float iy = m.iy + i * (m.ih + m.gap);
        if (y >= iy && y <= iy + m.ih) return i;
    }
    return -1;
}

// True when the pointer is over a HUD panel you actually click. The top info bar
// is deliberately excluded: it spans the full width, so treating it as blocking
// would disable upward edge scrolling entirely.
bool overInteractiveHUD(const App& a, float x, float y) {
    const float S = a.scale;
    const float pad = 4.0f * S;
    // minimap (bottom-left)
    float mmSize = 200 * S, mmX = 14 * S, mmY = a.pixelH - mmSize - 14 * S;
    if (x >= mmX - pad && x <= mmX + mmSize + pad &&
        y >= mmY - pad && y <= mmY + mmSize + pad) return true;
    // command card (bottom-right)
    const float bw = 62 * S, bh = 40 * S, gap = 5 * S;
    float cx = a.pixelW - (bw * 3 + gap * 2) - 14 * S;
    float cy = a.pixelH - (bh * 3 + gap * 2) - 14 * S;
    if (x >= cx - pad && y >= cy - pad) return true;
    return false;
}

void updateCamera(App& a, float dt) {
    Camera& c = a.cam;
    v2 pan{0, 0};
    if (a.in.keys[1]) pan.x -= 1;      // arrow keys
    if (a.in.keys[2]) pan.x += 1;
    if (a.in.keys[4]) pan.y += 1;
    if (a.in.keys[3]) pan.y -= 1;
    // --- edge scrolling ---------------------------------------------------
    // A band wide enough to hit without aiming, with speed ramping from 35% at
    // the inner boundary to full at the screen edge, so nudging the pointer into
    // it creeps and shoving it to the edge sprints.
    if (a.in.mouseInside && !overInteractiveHUD(a, a.in.mx, a.in.my)) {
        const float EDGE = 24.0f * a.scale;
        auto ramp = [](float t) { return 0.35f + 0.65f * saturate(t); };
        if (a.in.mx >= 0.0f && a.in.mx < EDGE)
            pan.x -= ramp(1.0f - a.in.mx / EDGE);
        else if (a.in.mx > a.pixelW - EDGE && a.in.mx <= (float)a.pixelW)
            pan.x += ramp((a.in.mx - (a.pixelW - EDGE)) / EDGE);
        if (a.in.my >= 0.0f && a.in.my < EDGE)
            pan.y += ramp(1.0f - a.in.my / EDGE);
        else if (a.in.my > a.pixelH - EDGE && a.in.my <= (float)a.pixelH)
            pan.y -= ramp((a.in.my - (a.pixelH - EDGE)) / EDGE);
    }

    if (length2(pan) > 0.0f) {
        float mag = std::min(1.0f, length(pan));
        pan = normalize(pan) * mag;
        // Screen-relative panning: right is (cos,-sin), forward is (-sin,-cos).
        float s = std::sin(c.yaw), co = std::cos(c.yaw);
        v2 world{pan.x * co - pan.y * s, -pan.x * s - pan.y * co};
        c.focus += world * ((16.0f + c.dist * 0.9f) * dt);
    }
    if (a.in.keys['Q']) c.yaw -= 1.6f * dt;
    if (a.in.keys['E']) c.yaw += 1.6f * dt;
    if (a.in.scroll != 0.0f) { c.dist *= std::exp(-a.in.scroll * 0.055f); a.in.scroll = 0; }
    c.dist = clampf(c.dist, 22.0f, 155.0f);
    c.focus.x = clampf(c.focus.x, 4.0f, Terrain::SIZE - 4.0f);
    c.focus.y = clampf(c.focus.y, 4.0f, Terrain::SIZE - 4.0f);
    float gy = a.game.terrain.heightAt(c.focus.x, c.focus.y);
    c.smoothY = lerpf(c.smoothY, gy, 1.0f - std::exp(-dt * 7.0f));
}

void centreOnSelection(App& a) {
    if (a.selection.empty()) return;
    v2 sum{0, 0}; int n = 0;
    for (EntId h : a.selection) if (const Entity* e = a.game.get(h)) { sum += e->pos; n++; }
    if (n) a.cam.focus = sum / (float)n;
}

void pruneSelection(App& a) {
    a.selection.erase(std::remove_if(a.selection.begin(), a.selection.end(),
        [&](EntId h) { const Entity* e = a.game.get(h); return !e || e->deathTimer >= 0; }),
        a.selection.end());
}

// Screen-space picking is far more forgiving than a ground raycast for tall units.
EntId pickEntityScreen(App& a, float px, float py) {
    EntId best; float bestD = 34.0f * a.scale;
    for (size_t i = 0; i < a.game.ents.size(); i++) {
        const Entity& e = a.game.ents[i];
        if (!e.alive || e.deathTimer >= 0) continue;
        if (e.team == 1 && !a.game.visible(0, e.pos)) continue;
        const UnitDef& D = kDefs[e.type];
        float y = a.game.groundY(e.pos);
        float sx, sy;
        if (!projectToScreen(a, v3{e.pos.x, y + D.radius * 0.6f, e.pos.y}, sx, sy)) continue;
        float d = std::sqrt((sx - px) * (sx - px) + (sy - py) * (sy - py));
        // Bias toward units so clicking a worker on a landing pad selects the worker.
        if (D.building) d += 12.0f * a.scale;
        if (d < bestD) { bestD = d; best = a.game.handleOf((int)i); }
    }
    return best;
}

bool selectionHas(App& a, UnitType t) {
    for (EntId h : a.selection) { const Entity* e = a.game.get(h); if (e && e->type == t) return true; }
    return false;
}

void buildCommandCard(App& a) {
    a.buttons.clear();
    if (a.selection.empty()) return;
    bool owned = false;
    for (EntId h : a.selection) { const Entity* e = a.game.get(h); if (e && e->team == 0) owned = true; }
    if (!owned) return;

    const float S = a.scale;
    const float bw = 62 * S, bh = 40 * S, gap = 5 * S;
    const float x0 = a.pixelW - (bw * 3 + gap * 2) - 14 * S;
    const float y0 = a.pixelH - (bh * 3 + gap * 2) - 14 * S;
    int slot = 0;
    auto add = [&](const char* label, char key, int action, bool isBuild, bool enabled) {
        Button b{};
        b.x = x0 + (slot % 3) * (bw + gap);
        b.y = y0 + (slot / 3) * (bh + gap);
        b.w = bw; b.h = bh;
        snprintf(b.label, sizeof(b.label), "%s", label);
        b.hotkey = key; b.action = action; b.isBuild = isBuild; b.enabled = enabled;
        a.buttons.push_back(b);
        slot++;
    };
    const Faction& F = a.game.fac[0];
    bool haveWorker = selectionHas(a, UT_WORKER);
    bool haveMobile = false;
    for (EntId h : a.selection) {
        const Entity* e = a.game.get(h);
        if (e && e->team == 0 && !kDefs[e->type].building) haveMobile = true;
    }

    if (selectionHas(a, UT_FOUNDRY))  add("Digger",  'D', UT_WORKER,  false, F.ore >= 50);
    if (selectionHas(a, UT_GARRISON)) add("Trooper", 'T', UT_TROOPER, false, F.ore >= 50);
    if (selectionHas(a, UT_WORKSHOP)) add("Mauler",  'M', UT_MAULER,  false, F.ore >= 150);
    if (haveWorker) {
        add("Bunkhouse", 'B', UT_BUNKHOUSE, true, F.ore >= kDefs[UT_BUNKHOUSE].cost);
        add("Foundry",   'F', UT_FOUNDRY,   true, F.ore >= kDefs[UT_FOUNDRY].cost);
        add("Garrison",  'G', UT_GARRISON,  true, F.ore >= kDefs[UT_GARRISON].cost);
        add("Workshop",  'W', UT_WORKSHOP,  true, F.ore >= kDefs[UT_WORKSHOP].cost);
    }
    if (haveMobile) {
        add("Attack", 'A', ACT_ATTACK, false, true);
        add("Stop",   'S', ACT_STOP,   false, true);
        add("Hold",   'H', ACT_HOLD,   false, true);
    }
}

void doAction(App& a, int action, bool isBuild) {
    Game& g = a.game;
    if (action == ACT_STOP)   { g.cmdStop(a.selection); say("Stop"); return; }
    if (action == ACT_HOLD)   { g.cmdHold(a.selection); say("Holding position"); return; }
    if (action == ACT_ATTACK) { a.mode = MODE_ATTACK_TARGET; say("Attack-move: pick a target"); return; }
    if (isBuild) {
        a.mode = MODE_PLACE_BUILDING;
        a.buildType = (UnitType)action;
        say("Place the structure");
        return;
    }
    // Train at whichever selected structure produces this unit.
    UnitType want = (UnitType)action;
    UnitType from = want == UT_WORKER ? UT_FOUNDRY : (want == UT_TROOPER ? UT_GARRISON : UT_WORKSHOP);
    for (EntId h : a.selection) {
        const Entity* e = g.get(h);
        if (!e || e->type != from || e->team != 0) continue;
        if (g.cmdTrain(h, want)) { say(kDefs[want].name); return; }
    }
    if (a.game.fac[0].ore < kDefs[want].cost) say("Not enough ore");
    else say("Supply blocked - build a Bunkhouse");
}

void doMenu(App& a, int item) {
    switch (item) {
        case MI_RESUME:     a.menuOpen = false; break;
        case MI_RESTART:    a.menuOpen = false; resetMatch(a, a.seed + 1); break;
        case MI_FULLSCREEN: toggleFullscreen(); break;
        case MI_HELP:       a.showHelp = !a.showHelp; break;
        case MI_QUIT:       [NSApp terminate:nil]; break;
        default: break;
    }
}

struct Layout {
    float mmX, mmY, mmSize;
    float topH;
};
Layout layoutOf(const App& a) {
    Layout L;
    const float S = a.scale;
    L.mmSize = 200 * S;
    L.mmX = 14 * S;
    L.mmY = a.pixelH - L.mmSize - 14 * S;
    L.topH = 34 * S;
    return L;
}

void handleInput(App& a, float dt) {
    Game& g = a.game;
    Input& in = a.in;
    Layout L = layoutOf(a);
    pruneSelection(a);

    // --- Esc menu ---------------------------------------------------------
    // Esc unwinds one step at a time: first a pending placement, then the
    // selection, and only with nothing left to cancel does it open the menu.
    // Anything else would eat the cancel a player reaches for mid-order.
    if (in.keyPressed[27]) {
        if (a.menuOpen)                  a.menuOpen = false;
        else if (a.mode != MODE_NORMAL)  { a.mode = MODE_NORMAL; say("Cancelled"); }
        else if (!a.selection.empty())   a.selection.clear();
        else                             { a.menuOpen = true; a.menuIndex = 0; }
    }
    if (a.menuOpen) {
        // Modal: nothing below this point runs, so a stray click cannot order
        // units around behind the panel.
        int hit = menuHit(a, in.mx, in.my);
        if (hit >= 0) a.menuIndex = hit;
        if (in.keyPressed[3]) a.menuIndex = (a.menuIndex + 1) % MI_COUNT;
        if (in.keyPressed[4]) a.menuIndex = (a.menuIndex + MI_COUNT - 1) % MI_COUNT;
        if (in.keyPressed[13]) doMenu(a, a.menuIndex);
        if (in.lPressed && hit >= 0) doMenu(a, hit);
        a.buttons.clear();
        return;
    }

    // --- world cursor -----------------------------------------------------
    v3 ro, rd;
    a.cursorValid = false;
    if (screenRay(a, in.mx, in.my, ro, rd)) {
        v3 hit;
        if (g.terrain.raycast(ro, rd, hit)) { a.cursorWorld = hit; a.cursorValid = true; }
    }
    a.hovered = pickEntityScreen(a, in.mx, in.my);

    // --- global keys ------------------------------------------------------
    if (in.keyPressed['P']) { a.paused = !a.paused; say(a.paused ? "Paused" : "Resumed"); }
    if (in.keyPressed['I']) { a.showAI = !a.showAI; say(a.showAI ? "AI inspector on" : "AI inspector off"); }
    if (in.keyPressed[(unsigned char)'/']) a.showHelp = !a.showHelp;
    if (in.keyPressed[' ']) centreOnSelection(a);

    // Control groups: Ctrl+N assigns, N recalls.
    for (int n = 0; n <= 9; n++) {
        unsigned char k = (unsigned char)('0' + n);
        if (!in.keyPressed[k]) continue;
        if (in.ctrl || in.cmd) {
            a.groups[n] = a.selection;
            char buf[48]; snprintf(buf, sizeof(buf), "Group %d set (%zu)", n, a.selection.size());
            say(buf);
        } else if (!a.groups[n].empty()) {
            a.selection = a.groups[n];
            pruneSelection(a);
            if (in.shift) centreOnSelection(a);
        }
    }

    // Command hotkeys mirror the on-screen card.
    for (const Button& b : a.buttons) {
        if (b.hotkey && in.keyPressed[(unsigned char)b.hotkey]) {
            if (b.enabled) doAction(a, b.action, b.isBuild);
            else say("Not enough ore");
        }
    }

    // --- left click -------------------------------------------------------
    bool overMinimap = in.mx >= L.mmX && in.mx <= L.mmX + L.mmSize &&
                       in.my >= L.mmY && in.my <= L.mmY + L.mmSize;
    bool overButton = false;
    int hitAction = ACT_NONE; bool hitBuild = false, hitEnabled = false;
    for (const Button& b : a.buttons) {
        if (in.mx >= b.x && in.mx <= b.x + b.w && in.my >= b.y && in.my <= b.y + b.h) {
            overButton = true; hitAction = b.action; hitBuild = b.isBuild; hitEnabled = b.enabled;
        }
    }

    if (in.lPressed) {
        if (overButton) {
            if (hitEnabled) doAction(a, hitAction, hitBuild);
            else say("Not enough ore");
        } else if (overMinimap) {
            a.cam.focus = v2{(in.mx - L.mmX) / L.mmSize * Terrain::SIZE,
                             (in.my - L.mmY) / L.mmSize * Terrain::SIZE};
        } else if (a.mode == MODE_PLACE_BUILDING) {
            if (a.cursorValid) {
                v2 p{a.cursorWorld.x, a.cursorWorld.z};
                if (g.cmdBuild(a.selection, a.buildType, p)) {
                    say("Construction started");
                    if (!in.shift) a.mode = MODE_NORMAL;
                } else {
                    say(g.canPlace(a.buildType, p) ? "Not enough ore" : "Cannot build here");
                }
            }
        } else if (a.mode == MODE_ATTACK_TARGET) {
            if (g.get(a.hovered)) g.cmdAttack(a.selection, a.hovered);
            else if (a.cursorValid) g.cmdMove(a.selection, v2{a.cursorWorld.x, a.cursorWorld.z}, true);
            a.mode = MODE_NORMAL;
        }
    }

    if (in.lReleased && a.mode == MODE_NORMAL && !overButton && !overMinimap) {
        float dx = in.mx - in.dragX, dy = in.my - in.dragY;
        bool boxed = (std::fabs(dx) > 6 * a.scale || std::fabs(dy) > 6 * a.scale);
        if (!in.shift) a.selection.clear();
        if (boxed) {
            float x0 = std::min(in.dragX, in.mx), x1 = std::max(in.dragX, in.mx);
            float y0 = std::min(in.dragY, in.my), y1 = std::max(in.dragY, in.my);
            for (size_t i = 0; i < g.ents.size(); i++) {
                const Entity& e = g.ents[i];
                if (!e.alive || e.deathTimer >= 0 || e.team != 0) continue;
                if (kDefs[e.type].building) continue;   // marquee grabs units only
                float sx, sy;
                if (!projectToScreen(a, v3{e.pos.x, g.groundY(e.pos) + 0.6f, e.pos.y}, sx, sy)) continue;
                if (sx >= x0 && sx <= x1 && sy >= y0 && sy <= y1)
                    a.selection.push_back(g.handleOf((int)i));
            }
        } else {
            EntId h = pickEntityScreen(a, in.mx, in.my);
            if (const Entity* e = g.get(h)) {
                if (in.doubleClick && e->team == 0 && !kDefs[e->type].building) {
                    // Double click grabs every visible unit of the same type.
                    UnitType t = e->type;
                    for (size_t i = 0; i < g.ents.size(); i++) {
                        const Entity& o = g.ents[i];
                        if (!o.alive || o.deathTimer >= 0 || o.team != 0 || o.type != t) continue;
                        float sx, sy;
                        if (!projectToScreen(a, v3{o.pos.x, g.groundY(o.pos), o.pos.y}, sx, sy)) continue;
                        if (sx >= 0 && sx < a.pixelW && sy >= 0 && sy < a.pixelH)
                            a.selection.push_back(g.handleOf((int)i));
                    }
                } else {
                    a.selection.push_back(h);
                }
            }
        }
        // De-duplicate after shift-adding.
        std::sort(a.selection.begin(), a.selection.end(),
                  [](EntId x, EntId y) { return x.v < y.v; });
        a.selection.erase(std::unique(a.selection.begin(), a.selection.end(),
                          [](EntId x, EntId y) { return x.v == y.v; }), a.selection.end());
    }

    // --- right click ------------------------------------------------------
    if (in.rPressed) {
        if (a.mode != MODE_NORMAL) { a.mode = MODE_NORMAL; say("Cancelled"); }
        else if (overMinimap) {
            v2 p{(in.mx - L.mmX) / L.mmSize * Terrain::SIZE,
                 (in.my - L.mmY) / L.mmSize * Terrain::SIZE};
            if (!a.selection.empty()) g.cmdMove(a.selection, g.nav.nearestWalkable(p), false);
        } else if (!a.selection.empty()) {
            bool onlyBuildings = true;
            for (EntId h : a.selection) {
                const Entity* e = g.get(h);
                if (e && e->team == 0 && !kDefs[e->type].building) onlyBuildings = false;
            }
            if (onlyBuildings) {
                if (a.cursorValid) { g.cmdRally(a.selection, v2{a.cursorWorld.x, a.cursorWorld.z}); say("Rally point set"); }
            } else if (a.cursorValid) {
                g.cmdSmart(a.selection, v2{a.cursorWorld.x, a.cursorWorld.z}, a.hovered);
            }
        }
    }
}

} // namespace

// ---------------------------------------------------------------- scene build
namespace {

float gFogOfWar = 1.0f;   // --no-fow disables it, for screenshots and debugging

// The simulation already keeps a per-team visibility grid -- the enemy AI is
// fogged by the same data. This hands the player's copy to the renderer as a
// texture. The grid flips in discrete steps on its own timer, so each cell is
// eased toward its target instead: the fog then breathes open as a unit walks
// forward rather than popping a 4-metre square at a time.
void updateFOW(App& a, float dt) {
    const int N = Game::VIS;
    const size_t n = (size_t)N * N;
    if (a.fowVis.size() != n) { a.fowVis.assign(n, 0.0f); a.fowExp.assign(n, 0.0f); }
    a.frame.fow.resize(n * 2);
    const float k = saturate(dt * 6.0f);
    for (int z = 0; z < N; z++) {
        for (int x = 0; x < N; x++) {
            size_t i = (size_t)z * N + x;
            a.fowVis[i] += ((a.game.visibleCell(0, x, z)  ? 1.0f : 0.0f) - a.fowVis[i]) * k;
            a.fowExp[i] += ((a.game.exploredCell(0, x, z) ? 1.0f : 0.0f) - a.fowExp[i]) * k;
            a.frame.fow[i * 2 + 0] = (uint8_t)(saturate(a.fowVis[i]) * 255.0f + 0.5f);
            a.frame.fow[i * 2 + 1] = (uint8_t)(saturate(a.fowExp[i]) * 255.0f + 0.5f);
        }
    }
    a.frame.fowN = N;
    a.frame.mapSize = Terrain::SIZE;
    a.frame.fowStrength = gFogOfWar;
}

float gAnimTime = 0.0f;

void buildScene(App& a) {
    RenderFrame& F = a.frame;
    F.clear();
    Game& g = a.game;

    v3 focus{a.cam.focus.x, a.cam.smoothY, a.cam.focus.y};
    v3 eye = camEye(a.cam, a.cam.smoothY);
    float aspect = (float)a.pixelW / std::max(1, a.pixelH);
    F.view = lookAt(eye, focus, v3{0, 1, 0});
    F.viewProj = perspective(46.0f * DEG, aspect, 0.5f, 1400.0f) * F.view;
    F.invViewProj = inverse(F.viewProj);
    F.camPos = eye;
    F.time = gAnimTime;
    F.sunDir = normalize(v3{0.78f, 0.56f, -0.28f});
    F.sunIntensity = 2.15f;
    F.sunColor = v3{1.00f, 0.90f, 0.74f};
    F.ambient = 0.30f;
    F.fogColor = v3{0.60f, 0.69f, 0.80f};
    F.fogDensity = 0.00085f;
    F.waterLevel = Terrain::WATER;
    F.shadowFocus = focus;
    F.exposure = 0.78f;
    F.bloomIntensity = 0.50f;
    F.bloomThreshold = 0.85f;

    auto pushInstance = [&](int mesh, const m4& model, v3 tint, v3 team,
                            float emis, float flash, float dissolve, float cutoff) {
        InstanceData id{};
        id.model = model;
        id.tint = v4{tint.x, tint.y, tint.z, 0.0f};
        id.team = v4{team.x, team.y, team.z, 0.0f};
        id.fx = v4{emis, flash, dissolve, cutoff};
        F.instances[mesh].push_back(id);
    };

    for (size_t i = 0; i < g.ents.size(); i++) {
        const Entity& e = g.ents[i];
        if (!e.alive) continue;
        const UnitDef& D = kDefs[e.type];
        // Fog of war: enemy units are hidden unless something of ours sees them.
        if (e.team == 1 && !g.visible(0, e.pos)) continue;

        float y = g.groundY(e.pos);
        float dissolve = 1.0f, sink = 0.0f;
        if (e.deathTimer >= 0.0f) {
            float ttl = D.building ? 2.4f : 1.2f;
            dissolve = saturate(1.0f - e.deathTimer / ttl);
            sink = e.deathTimer * (D.building ? 0.35f : 0.25f);
        }
        float cutoff = 1000.0f;
        if (e.buildProgress < 1.0f) {
            float h = a.renderer.meshRange(D.meshId).height;
            cutoff = y + h * e.buildProgress + 0.05f;
        }

        // Walk cycle: a small vertical bob plus a lean into the direction of travel.
        float bobY = 0.0f, roll = 0.0f;
        float sp = length(e.vel);
        if (!D.building && !D.neutral && sp > 0.15f) {
            bobY = std::fabs(std::sin(e.bob * 2.0f)) * 0.07f;
            roll = std::sin(e.bob) * 0.045f;
        }
        v3 pos{e.pos.x, y + bobY - sink, e.pos.y};
        m4 model = translate(pos) * rotateY(e.yaw) * rotateZ(roll);
        v3 team = teamColor(e.team);
        if (e.type == UT_ORE) team = v3{0.35f, 0.85f, 1.0f};

        float emis = 0.0f;
        if (e.buildProgress < 1.0f) emis = 0.6f;
        pushInstance(D.meshId, model, v3{1, 1, 1}, team, emis, e.damageFlash, dissolve, cutoff);

        // Mauler turrets are drawn separately so they can track independently.
        if (e.type == UT_MAULER) {
            m4 tm = translate(v3{pos.x, pos.y + 1.40f, pos.z}) * rotateY(e.turretYaw);
            pushInstance(MESH_MAULER_TURRET, tm, v3{1, 1, 1}, team, 0.0f,
                         e.damageFlash, dissolve, cutoff);
        }
    }

    // Selection and hover rings.
    auto ring = [&](EntId h, v3 col, float gain) {
        const Entity* e = g.get(h);
        if (!e) return;
        const UnitDef& D = kDefs[e->type];
        float y = g.groundY(e->pos);
        // Collision radius is tighter than the visual mesh on small units, so a
        // ring sized from it alone would disappear under the chassis.
        float meshR = a.renderer.meshRange(D.meshId).radius;
        float r = std::max(D.radius * 1.15f, meshR * 1.06f);
        m4 m = translate(v3{e->pos.x, y + 0.10f, e->pos.y}) * scale(v3{r, 1.0f, r});
        pushInstance(MESH_SEL_RING, m, v3{1, 1, 1}, col, gain, 0.0f, 1.0f, 1000.0f);
    };
    for (EntId h : a.selection) ring(h, v3{0.35f, 1.0f, 0.45f}, 0.35f);
    if (a.hovered.valid()) {
        bool already = false;
        for (EntId h : a.selection) if (h.v == a.hovered.v) already = true;
        if (!already) if (const Entity* e = g.get(a.hovered)) ring(a.hovered, teamColor(e->team), 0.15f);
    }

    // Building placement ghost.
    if (a.mode == MODE_PLACE_BUILDING && a.cursorValid) {
        v2 p{a.cursorWorld.x, a.cursorWorld.z};
        bool ok = g.canPlace(a.buildType, p) && g.fac[0].ore >= kDefs[a.buildType].cost;
        v3 col = ok ? v3{0.35f, 1.0f, 0.45f} : v3{1.0f, 0.3f, 0.25f};
        float y = g.groundY(p);
        m4 m = translate(v3{p.x, y, p.y});
        pushInstance(kDefs[a.buildType].meshId, m, col, col, 1.4f, 0.0f, 0.55f, 1000.0f);
        float r = kDefs[a.buildType].radius;
        m4 rm = translate(v3{p.x, y + 0.12f, p.y}) * scale(v3{r, 1.0f, r});
        pushInstance(MESH_SEL_RING, rm, v3{1, 1, 1}, col, 2.5f, 0.0f, 1.0f, 1000.0f);
    }

    // Projectiles.
    for (const auto& p : g.projectiles) {
        if (!p.alive) continue;
        if (!g.visible(0, v2{p.pos.x, p.pos.z}) && p.team == 1) continue;
        float s = p.kind == 1 ? 1.6f : 1.0f;
        pushInstance(MESH_PROJECTILE, orientTo(p.pos, p.vel, s),
                     v3{1, 1, 1}, v3{1.0f, 0.8f, 0.4f}, 2.0f, 0.0f, 1.0f, 1000.0f);
    }

    // Particles -> billboards. Kinds 5 and 6 (scorch, smoke) are alpha-blended
    // and go to a separate list drawn ahead of the additive ones; scorch is
    // emitted first so smoke layers over the burn mark rather than under it.
    auto emit = [&](const Particle& p) {
        float t = saturate(p.life / std::max(0.001f, p.maxLife));
        Billboard b{};
        b.pos = p.pos;
        b.size = lerpf(p.size0, p.size1, t);
        b.color = lerp(p.col0, p.col1, t);
        b.rot = p.rot;
        b.kind = (float)p.kind;
        b.fade = t;              // sprite-sheet phase for kinds 4 and 6
        b.param = p.param;       // which scorch mark
        (p.kind >= 5 ? F.decals : F.billboards).push_back(b);
    };
    for (const auto& p : g.particles) if (p.alive && p.kind == 5) emit(p);
    for (const auto& p : g.particles) if (p.alive && p.kind != 5) emit(p);
}

// ---------------------------------------------------------------- HUD
// What a command card button costs and does. Single source for the caption,
// the price printed on the button and the hover tooltip, so the three cannot
// drift apart as unit stats are tuned.
struct ButtonInfo {
    const char* title;
    const char* desc;
    int   cost;        // 0 for a free action
    int   supply;      // >0 consumes supply, <0 provides it
    float buildTime;   // seconds, 0 for a free action
};
ButtonInfo buttonInfo(int action) {
    switch (action) {
        case ACT_ATTACK: return {"Attack-Move", "Advance to a point, engaging anything met on the way.", 0, 0, 0};
        case ACT_STOP:   return {"Stop",        "Cancel current orders and stand down.",                 0, 0, 0};
        case ACT_HOLD:   return {"Hold Position","Stand and fight where you are; never chase.",          0, 0, 0};
        default: break;
    }
    static const char* kDesc[UT_COUNT] = {
        "Mines crystal and raises every structure.",
        "Cheap ranged infantry. Wins on numbers, not duels.",
        "Heavy siege gun. Splash damage, slow to turn.",
        "Trains Diggers and accepts ore drop-off.",
        "Trains Troopers.",
        "Builds Maulers.",
        "Raises the supply cap.",
        "", "",
    };
    UnitType t = (UnitType)action;
    if (t < 0 || t >= UT_COUNT) return {"", "", 0, 0, 0};
    const UnitDef& D = kDefs[t];
    return {D.name, kDesc[t], D.cost,
            D.supplyGive > 0 ? -D.supplyGive : D.supplyCost, D.buildTime};
}

void buildHUD(App& a) {
    RenderFrame& F = a.frame;
    Renderer& R = a.renderer;
    Game& g = a.game;
    const float S = a.scale;
    Layout L = layoutOf(a);
    auto& U = F.uiVerts;
    char buf[160];

    const v4 panel{0.05f, 0.07f, 0.10f, 0.80f};
    const v4 edge {0.45f, 0.60f, 0.75f, 0.75f};
    const v4 white{0.92f, 0.95f, 1.00f, 1.00f};
    const v4 dim  {0.62f, 0.70f, 0.78f, 1.00f};
    const v4 gold {1.00f, 0.82f, 0.35f, 1.00f};

    // --- health bars ------------------------------------------------------
    for (size_t i = 0; i < g.ents.size(); i++) {
        const Entity& e = g.ents[i];
        if (!e.alive || e.deathTimer >= 0) continue;
        if (e.team == 2) continue;
        if (e.team == 1 && !g.visible(0, e.pos)) continue;
        const UnitDef& D = kDefs[e.type];
        bool sel = false;
        for (EntId h : a.selection) if (h.index() == (int)i) sel = true;
        float frac = saturate(e.hp / D.hp);
        if (!sel && frac > 0.995f && e.buildProgress >= 1.0f) continue;

        float top = a.renderer.meshRange(D.meshId).height;
        float sx, sy;
        if (!projectToScreen(a, v3{e.pos.x, g.groundY(e.pos) + top + 0.7f, e.pos.y}, sx, sy)) continue;
        if (sx < -100 || sy < -100 || sx > a.pixelW + 100 || sy > a.pixelH + 100) continue;
        float w = (D.building ? 54.0f : 30.0f) * S;
        float h = 5.0f * S;
        R.rect(U, sx - w * 0.5f - S, sy - S, w + 2 * S, h + 2 * S, v4{0, 0, 0, 0.65f});
        v4 hc = frac > 0.6f ? v4{0.30f, 0.90f, 0.35f, 1} :
                (frac > 0.3f ? v4{0.95f, 0.80f, 0.20f, 1} : v4{0.95f, 0.25f, 0.20f, 1});
        R.rect(U, sx - w * 0.5f, sy, w * frac, h, hc);
        if (e.buildProgress < 1.0f) {
            R.rect(U, sx - w * 0.5f, sy + h + S, w, 3 * S, v4{0, 0, 0, 0.6f});
            R.rect(U, sx - w * 0.5f, sy + h + S, w * e.buildProgress, 3 * S, v4{0.35f, 0.75f, 1.0f, 1});
        }
    }

    // --- top resource bar -------------------------------------------------
    R.rect(U, 0, 0, (float)a.pixelW, L.topH, panel);
    R.rect(U, 0, L.topH, (float)a.pixelW, 1.5f * S, edge);
    const Faction& P = g.fac[0];
    float ts = 17 * S, ty = (L.topH - R.lineHeight(ts)) * 0.5f;
    float x = 16 * S;
    snprintf(buf, sizeof(buf), "ORE %d", P.ore);
    R.text(U, buf, x, ty, ts, gold);
    x += R.textWidth(buf, ts) + 26 * S;
    bool blocked = P.supplyUsed >= P.supplyCap;
    snprintf(buf, sizeof(buf), "SUPPLY %d/%d", P.supplyUsed, P.supplyCap);
    R.text(U, buf, x, ty, ts, blocked ? v4{1.0f, 0.35f, 0.3f, 1} : white);
    x += R.textWidth(buf, ts) + 26 * S;
    snprintf(buf, sizeof(buf), "%d:%02d", (int)g.time / 60, (int)g.time % 60);
    R.text(U, buf, x, ty, ts, dim);

    RenderStats rs = R.stats();
    snprintf(buf, sizeof(buf), "%.0f fps  %.1f ms gpu  %dx%d  %d draws  %d inst", a.fps,
             rs.gpuFrameMs, a.pixelW, a.pixelH, rs.drawCalls, rs.instances);
    R.text(U, buf, a.pixelW - R.textWidth(buf, 13 * S) - 16 * S, ty + 2 * S, 13 * S, dim);

    // --- minimap ----------------------------------------------------------
    R.rect(U, L.mmX - 3 * S, L.mmY - 3 * S, L.mmSize + 6 * S, L.mmSize + 6 * S, panel);
    R.frame(U, L.mmX - 3 * S, L.mmY - 3 * S, L.mmSize + 6 * S, L.mmSize + 6 * S, 1.5f * S, edge);
    R.minimapQuad(F.minimapVerts, L.mmX, L.mmY, L.mmSize, L.mmSize, v4{1, 1, 1, 1});
    auto mmPt = [&](v2 p, float& ox, float& oy) {
        ox = L.mmX + p.x / Terrain::SIZE * L.mmSize;
        oy = L.mmY + p.y / Terrain::SIZE * L.mmSize;
    };
    // Unexplored ground is masked out.
    float cellPx = L.mmSize / Game::VIS;
    for (int z = 0; z < Game::VIS; z++)
        for (int xg = 0; xg < Game::VIS; xg++) {
            v2 p{(xg + 0.5f) * Game::VIS_CELL, (z + 0.5f) * Game::VIS_CELL};
            if (!g.explored(0, p))    R.rect(U, L.mmX + xg * cellPx, L.mmY + z * cellPx, cellPx + 1, cellPx + 1, v4{0.02f, 0.03f, 0.05f, 1.0f});
            else if (!g.visible(0, p)) R.rect(U, L.mmX + xg * cellPx, L.mmY + z * cellPx, cellPx + 1, cellPx + 1, v4{0.02f, 0.03f, 0.05f, 0.42f});
        }
    for (const auto& e : g.ents) {
        if (!e.alive || e.deathTimer >= 0) continue;
        if (e.team == 1 && !g.visible(0, e.pos)) continue;
        if (e.type == UT_BOULDER) continue;
        float ox, oy; mmPt(e.pos, ox, oy);
        v3 c = e.type == UT_ORE ? v3{0.35f, 0.85f, 1.0f} : teamColor(e.team);
        float s = kDefs[e.type].building ? 4.0f * S : 2.4f * S;
        R.rect(U, ox - s * 0.5f, oy - s * 0.5f, s, s, v4{c.x, c.y, c.z, 1.0f});
    }
    for (const auto& ping : g.pings) {
        float ox, oy; mmPt(ping.pos, ox, oy);
        float r = (6.0f + std::sin(ping.t * 9.0f) * 3.0f) * S;
        float al = saturate(1.0f - ping.t / 4.0f);
        R.frame(U, ox - r, oy - r, r * 2, r * 2, 1.5f * S, v4{1.0f, 0.3f, 0.25f, al});
    }
    // Camera footprint, from the four screen corners raycast onto the terrain.
    {
        float minx = 1e9f, maxx = -1e9f, minz = 1e9f, maxz = -1e9f;
        const float cs[4][2] = {{0, 0}, {(float)a.pixelW, 0}, {0, (float)a.pixelH}, {(float)a.pixelW, (float)a.pixelH}};
        int hits = 0;
        for (auto& c : cs) {
            v3 ro, rd, hit;
            if (screenRay(a, c[0], c[1], ro, rd) && g.terrain.raycast(ro, rd, hit)) {
                minx = std::min(minx, hit.x); maxx = std::max(maxx, hit.x);
                minz = std::min(minz, hit.z); maxz = std::max(maxz, hit.z);
                hits++;
            }
        }
        if (hits >= 3) {
            float x0, y0, x1, y1;
            mmPt(v2{clampf(minx, 0, Terrain::SIZE), clampf(minz, 0, Terrain::SIZE)}, x0, y0);
            mmPt(v2{clampf(maxx, 0, Terrain::SIZE), clampf(maxz, 0, Terrain::SIZE)}, x1, y1);
            R.frame(U, x0, y0, x1 - x0, y1 - y0, 1.2f * S, v4{1, 1, 1, 0.55f});
        }
    }

    // --- selection panel --------------------------------------------------
    if (!a.selection.empty()) {
        float pw = 330 * S, ph = 96 * S;
        float px = (a.pixelW - pw) * 0.5f, py = a.pixelH - ph - 14 * S;
        R.rect(U, px, py, pw, ph, panel);
        R.frame(U, px, py, pw, ph, 1.5f * S, edge);
        int counts[UT_COUNT] = {0};
        for (EntId h : a.selection) if (const Entity* e = g.get(h)) counts[e->type]++;
        int bestT = 0, bestC = 0;
        for (int t = 0; t < UT_COUNT; t++) if (counts[t] > bestC) { bestC = counts[t]; bestT = t; }
        if (a.selection.size() == 1) {
            if (const Entity* e = g.get(a.selection[0])) {
                snprintf(buf, sizeof(buf), "%s", kDefs[e->type].name);
                R.text(U, buf, px + 12 * S, py + 8 * S, 17 * S, white);
                snprintf(buf, sizeof(buf), "HP %d / %d", (int)std::ceil(e->hp), (int)kDefs[e->type].hp);
                R.text(U, buf, px + 12 * S, py + 32 * S, 14 * S, dim);
                if (e->type == UT_WORKER && e->carrying > 0) {
                    snprintf(buf, sizeof(buf), "Carrying %d ore", e->carrying);
                    R.text(U, buf, px + 12 * S, py + 52 * S, 13 * S, gold);
                }
                if (e->type == UT_ORE) {
                    snprintf(buf, sizeof(buf), "%d remaining", e->oreLeft);
                    R.text(U, buf, px + 12 * S, py + 52 * S, 13 * S, gold);
                }
                if (!e->queue.empty()) {
                    snprintf(buf, sizeof(buf), "Training %s  (%zu queued)",
                             kDefs[e->queue.front()].name, e->queue.size());
                    R.text(U, buf, px + 12 * S, py + 72 * S, 13 * S, v4{0.4f, 0.85f, 1.0f, 1});
                    float bw = pw - 24 * S;
                    float prog = 1.0f - saturate(e->queueTimer / std::max(0.1f, kDefs[e->queue.front()].buildTime));
                    R.rect(U, px + 12 * S, py + ph - 12 * S, bw, 4 * S, v4{0, 0, 0, 0.6f});
                    R.rect(U, px + 12 * S, py + ph - 12 * S, bw * prog, 4 * S, v4{0.35f, 0.75f, 1.0f, 1});
                }
            }
        } else {
            if (bestC == (int)a.selection.size())
                snprintf(buf, sizeof(buf), "%d x %s", bestC, kDefs[bestT].name);
            else
                snprintf(buf, sizeof(buf), "%zu units  (%d x %s)", a.selection.size(), bestC, kDefs[bestT].name);
            R.text(U, buf, px + 12 * S, py + 8 * S, 16 * S, white);
            float cx = px + 12 * S, cy = py + 34 * S;
            int shown = 0;
            for (EntId h : a.selection) {
                const Entity* e = g.get(h);
                if (!e || shown >= 24) continue;
                float w = 20 * S, hh = 24 * S;
                v3 tc = teamColor(e->team);
                R.rect(U, cx, cy, w, hh, v4{tc.x * 0.5f, tc.y * 0.5f, tc.z * 0.5f, 0.9f});
                float frac = saturate(e->hp / kDefs[e->type].hp);
                R.rect(U, cx, cy + hh - 4 * S, w * frac, 3 * S, v4{0.3f, 0.9f, 0.35f, 1});
                cx += w + 3 * S;
                if (cx > px + pw - 30 * S) { cx = px + 12 * S; cy += hh + 4 * S; }
                shown++;
            }
        }
    }

    // --- command card -----------------------------------------------------
    const Button* tipFor = nullptr;
    float cardTop = (float)a.pixelH;
    for (const Button& b : a.buttons) {
        bool hot = a.in.mx >= b.x && a.in.mx <= b.x + b.w && a.in.my >= b.y && a.in.my <= b.y + b.h;
        if (hot) tipFor = &b;
        cardTop = std::min(cardTop, b.y);
        v4 bg = b.enabled ? (hot ? v4{0.18f, 0.28f, 0.38f, 0.95f} : v4{0.09f, 0.13f, 0.18f, 0.90f})
                          : v4{0.08f, 0.08f, 0.09f, 0.85f};
        R.rect(U, b.x, b.y, b.w, b.h, bg);
        R.frame(U, b.x, b.y, b.w, b.h, 1.2f * S, b.enabled ? edge : v4{0.25f, 0.25f, 0.28f, 0.7f});
        v4 fg = b.enabled ? white : v4{0.45f, 0.45f, 0.48f, 1};
        // Shrink the label to fit rather than letting it run off the button.
        // Unit names are data, and a longer one must not silently clip -- which
        // is exactly what happened when "Depot" became "Bunkhouse".
        float ls = 12 * S;
        float avail = b.w - 10 * S;
        float lw = R.textWidth(b.label, ls);
        if (lw > avail) ls = std::max(8.0f * S, ls * avail / lw);
        R.text(U, b.label, b.x + 6 * S, b.y + 6 * S, ls, fg);
        char hk[2] = {b.hotkey, 0};
        R.text(U, hk, b.x + b.w - 12 * S, b.y + b.h - 16 * S, 12 * S, gold);
        // The price is the one number a player checks constantly, so it is on
        // the face of the button rather than only in the tooltip.
        ButtonInfo bi = buttonInfo(b.action);
        if (bi.cost > 0) {
            snprintf(buf, sizeof(buf), "%d", bi.cost);
            R.text(U, buf, b.x + 6 * S, b.y + b.h - 16 * S, 11 * S,
                   b.enabled ? gold : v4{0.55f, 0.45f, 0.30f, 1});
        }
    }

    // --- command card tooltip ---------------------------------------------
    if (tipFor) {
        ButtonInfo bi = buttonInfo(tipFor->action);
        char costLine[96] = {0};
        if (bi.cost > 0) {
            int n = snprintf(costLine, sizeof(costLine), "%d ore", bi.cost);
            if (bi.supply > 0)      n += snprintf(costLine + n, sizeof(costLine) - n, "   %d supply", bi.supply);
            else if (bi.supply < 0) n += snprintf(costLine + n, sizeof(costLine) - n, "   +%d supply", -bi.supply);
            snprintf(costLine + n, sizeof(costLine) - n, "   %.0fs", bi.buildTime);
        }
        char keyLine[48];
        snprintf(keyLine, sizeof(keyLine), "hotkey  %c%s", tipFor->hotkey,
                 tipFor->isBuild ? "   then click to place" : "");

        const float ts = 15 * S, ds = 12.5f * S;
        float w = R.textWidth(bi.title, ts);
        w = std::max(w, R.textWidth(bi.desc, ds));
        if (costLine[0]) w = std::max(w, R.textWidth(costLine, ds));
        w = std::max(w, R.textWidth(keyLine, ds));
        float lh = R.lineHeight(ds) * 1.15f;
        float h = 16 * S + R.lineHeight(ts) + lh * (costLine[0] ? 3.0f : 2.0f) + 8 * S;
        // Pinned to the screen's right edge, not to the button, so it does not
        // jump around as the pointer slides across the card.
        float tx = a.pixelW - w - 28 * S - 14 * S;
        float ty = cardTop - h - 10 * S;
        R.rect(U, tx - 14 * S, ty, w + 28 * S, h, v4{0.04f, 0.06f, 0.09f, 0.95f});
        R.frame(U, tx - 14 * S, ty, w + 28 * S, h, 1.4f * S, edge);
        float y = ty + 10 * S;
        R.text(U, bi.title, tx, y, ts, white);  y += R.lineHeight(ts) + 2 * S;
        if (costLine[0]) { R.text(U, costLine, tx, y, ds, gold); y += lh; }
        R.text(U, bi.desc, tx, y, ds, dim);     y += lh;
        R.text(U, keyLine, tx, y, ds, v4{0.45f, 0.62f, 0.78f, 1});
        if (!tipFor->enabled) {
            const char* warn = "not affordable";
            R.text(U, warn, tx + w - R.textWidth(warn, ds), ty + 10 * S, ds,
                   v4{1.0f, 0.45f, 0.40f, 1});
        }
    }

    // --- marquee ----------------------------------------------------------
    if (a.in.lDown && a.in.dragging && a.mode == MODE_NORMAL) {
        float x0 = std::min(a.in.dragX, a.in.mx), x1 = std::max(a.in.dragX, a.in.mx);
        float y0 = std::min(a.in.dragY, a.in.my), y1 = std::max(a.in.dragY, a.in.my);
        R.rect(U, x0, y0, x1 - x0, y1 - y0, v4{0.3f, 1.0f, 0.4f, 0.10f});
        R.frame(U, x0, y0, x1 - x0, y1 - y0, 1.5f * S, v4{0.35f, 1.0f, 0.45f, 0.9f});
    }

    // --- mode prompt / toast ---------------------------------------------
    if (a.mode != MODE_NORMAL) {
        const char* m = a.mode == MODE_ATTACK_TARGET ? "ATTACK-MOVE: click a target  (Esc cancels)"
                                                     : "PLACEMENT: click to build  (Esc cancels)";
        float w = R.textWidth(m, 15 * S);
        R.rect(U, (a.pixelW - w) * 0.5f - 12 * S, L.topH + 10 * S, w + 24 * S, 28 * S, panel);
        R.text(U, m, (a.pixelW - w) * 0.5f, L.topH + 16 * S, 15 * S, gold);
    }
    if (a.toastTimer > 0.0f && a.toast[0]) {
        float al = saturate(a.toastTimer / 0.7f);
        float w = R.textWidth(a.toast, 15 * S);
        R.text(U, a.toast, (a.pixelW - w) * 0.5f, L.topH + 48 * S, 15 * S, v4{1, 1, 1, al});
    }

    // --- help -------------------------------------------------------------
    if (a.showHelp) {
        const char* lines[] = {
            "STARFORGE  --  press / to toggle this help",
            "",
            "Camera    push the pointer against a screen edge to pan, or use the",
            "          arrow keys. Q/E rotate, the wheel zooms.",
            "Inspect   I shows what the AI has scouted, believes and decided",
            "Display   ctrl-cmd-F toggles full screen; Esc opens the menu",
            "Costs     hover any command card button for cost and build time",
            "Select    left-click, or drag a box; double-click picks all of a type",
            "          shift adds to selection, ctrl+0-9 sets a group, 0-9 recalls",
            "Orders    right-click moves / attacks / harvests, A attack-move,",
            "          S stop, H hold, space centres on selection",
            "Build     select a Digger then B bunkhouse, F foundry, G garrison,",
            "          W workshop, then click to place",
            "Train     select a structure then D digger, T trooper, M mauler",
            "          right-click with only structures selected sets the rally point",
            "",
            "Mine ore with Diggers, raise supply with bunkhouses, and destroy",
            "every enemy structure and Digger to win.",
        };
        int n = sizeof(lines) / sizeof(lines[0]);
        float fs = 13 * S, lh = R.lineHeight(fs) * 1.06f;
        float w = 0;
        for (int i = 0; i < n; i++) w = std::max(w, R.textWidth(lines[i], fs));
        float hx = 14 * S, hy = L.topH + 14 * S;
        R.rect(U, hx, hy, w + 28 * S, lh * n + 20 * S, panel);
        R.frame(U, hx, hy, w + 28 * S, lh * n + 20 * S, 1.5f * S, edge);
        for (int i = 0; i < n; i++)
            R.text(U, lines[i], hx + 14 * S, hy + 10 * S + lh * i, fs,
                   i == 0 ? gold : (lines[i][0] == ' ' ? dim : white));
    }

    // --- AI inspector -----------------------------------------------------
    if (a.showAI) {
        const ai::AIDebug& d = a.enemyAI.debug();
        float fs0 = 12 * S, lh0 = R.lineHeight(fs0) * 1.05f;
        float pw = 310 * S, px = a.pixelW - pw - 14 * S, py = L.topH + 14 * S;
        float ph = 18 * S + lh0 * (5.5f + (float)ai::PS_COUNT);
        R.rect(U, px, py, pw, ph, panel);
        R.frame(U, px, py, pw, ph, 1.5f * S, edge);
        float y = py + 8 * S, fs = fs0, lh = lh0;
        R.text(U, "ENEMY AI", px + 10 * S, y, 13 * S, gold); y += lh * 1.2f;

        snprintf(buf, sizeof(buf), "plan   %s", ai::strategyName(d.strategy));
        R.text(U, buf, px + 10 * S, y, fs, white); y += lh;
        snprintf(buf, sizeof(buf), "reads  %s", ai::playerStratName(d.believed));
        R.text(U, buf, px + 10 * S, y, fs, v4{0.5f, 0.9f, 1.0f, 1}); y += lh;
        snprintf(buf, sizeof(buf), "apm %.0f (peak %.0f)  scouted %.0f%%  H %.2f",
                 d.apm, d.apmPeak, d.scoutConfidence * 100.0f, d.entropy);
        R.text(U, buf, px + 10 * S, y, fs, dim); y += lh;
        snprintf(buf, sizeof(buf), "influence %s   %.2f ms/tick", d.backend, a.aiMs);
        R.text(U, buf, px + 10 * S, y, fs, dim); y += lh * 1.3f;

        // Posterior over what it thinks the player is doing.
        for (int i = 0; i < ai::PS_COUNT; i++) {
            float v = d.beliefs[i];
            R.text(U, ai::playerStratName((ai::PlayerStrat)i), px + 10 * S, y, fs,
                   i == (int)d.believed ? white : dim);
            float bx = px + 96 * S, bw = pw - 116 * S;
            R.rect(U, bx, y + 3 * S, bw, 9 * S, v4{0, 0, 0, 0.5f});
            R.rect(U, bx, y + 3 * S, bw * saturate(v), 9 * S,
                   i == (int)d.believed ? v4{0.35f, 0.85f, 1.0f, 1} : v4{0.35f, 0.45f, 0.55f, 1});
            y += lh;
        }
    }

    // --- outcome ----------------------------------------------------------
    if (g.winner >= 0) {
        const char* msg = g.winner == 0 ? "VICTORY" : "DEFEAT";
        v4 col = g.winner == 0 ? v4{0.4f, 1.0f, 0.5f, 1} : v4{1.0f, 0.35f, 0.3f, 1};
        float fs = 64 * S;
        float w = R.textWidth(msg, fs);
        R.rect(U, 0, a.pixelH * 0.5f - 60 * S, (float)a.pixelW, 120 * S, v4{0, 0, 0, 0.72f});
        R.text(U, msg, (a.pixelW - w) * 0.5f, a.pixelH * 0.5f - 40 * S, fs, col);
        const char* sub = "press R to start a new match";
        float w2 = R.textWidth(sub, 16 * S);
        R.text(U, sub, (a.pixelW - w2) * 0.5f, a.pixelH * 0.5f + 34 * S, 16 * S, white);
    }

    // --- Esc menu ---------------------------------------------------------
    if (a.menuOpen) {
        MenuGeom m = menuGeom(a);
        R.rect(U, 0, 0, (float)a.pixelW, (float)a.pixelH, v4{0.01f, 0.02f, 0.03f, 0.58f});
        R.rect(U, m.px, m.py, m.pw, m.ph, v4{0.05f, 0.07f, 0.10f, 0.97f});
        R.frame(U, m.px, m.py, m.pw, m.ph, 2.0f * S, edge);
        R.text(U, "STARFORGE", m.px + 28 * S, m.py + 22 * S, 25 * S, gold);
        R.text(U, g.winner >= 0 ? "match over" : "simulation paused",
               m.px + 28 * S, m.py + 54 * S, 12.5f * S, dim);
        for (int i = 0; i < MI_COUNT; i++) {
            float iy = m.iy + i * (m.ih + m.gap);
            bool hot = (i == a.menuIndex);
            R.rect(U, m.ix, iy, m.iw, m.ih,
                   hot ? v4{0.18f, 0.28f, 0.38f, 0.95f} : v4{0.09f, 0.13f, 0.18f, 0.90f});
            R.frame(U, m.ix, iy, m.iw, m.ih, 1.2f * S,
                    hot ? edge : v4{0.25f, 0.28f, 0.32f, 0.75f});
            R.text(U, kMenuLabels[i], m.ix + 14 * S, iy + 13 * S, 15 * S,
                   hot ? white : v4{0.78f, 0.84f, 0.90f, 1});
            if (kMenuHints[i][0]) {
                float hw = R.textWidth(kMenuHints[i], 12 * S);
                R.text(U, kMenuHints[i], m.ix + m.iw - hw - 14 * S, iy + 15 * S, 12 * S, dim);
            }
        }
    }
}

} // namespace

// ---------------------------------------------------------------- frame loop
namespace {

// Quality settings; parsed from argv before the window exists, and read by both
// tick() (for render scale) and Renderer::init.
RenderSettings gRS;
bool  gAdaptive = true;
int   gFullscreen = -1;   // -1 auto: fullscreen when interactive, windowed for bench/shot
const char* gEdgeTest = nullptr;   // pins a virtual cursor to one edge, for verification
// HUD verification hooks: select an Digger, force the Esc menu open, or park the
// pointer on a command card button so its tooltip can be captured.
bool gDecalTest = false;   // lays out the four scorch marks with nothing on top
v2   gCamAt{};             // --cam-at pins the camera somewhere specific, for shots
bool gCamAtSet = false;
bool gUIDemo = false, gUIMenu = false;
int  gHoverBtn = -1;
float gAdaptWindow = 0.0f;
AdaptState gAdapt;
double gAdaptSum = 0.0; int gAdaptN = 0;

void resetMatch(App& a, uint32_t seed) {
    a.seed = seed;
    a.game.init(seed);
    a.enemyAI.init(&a.game, 1, seed, a.gpuInfluence);
    a.renderer.rebuildTerrain(a.game.terrain);
    a.selection.clear();
    for (auto& gsel : a.groups) gsel.clear();
    a.mode = MODE_NORMAL;
    a.cam.focus = a.game.basePos[0];
    a.cam.dist = 62.0f;
    a.cam.yaw = 0.7f;
    a.cam.smoothY = a.game.terrain.heightAt(a.cam.focus.x, a.cam.focus.y);
    say("New match");
}

void tick(App& a, GameView* view) {
    // --- surface size -----------------------------------------------------
    float s = (float)view.window.backingScaleFactor;
    NSSize b = view.bounds.size;
    // Render scale shrinks the drawable and lets the display compositor upscale
    // on present -- the single biggest lever on a fragment-bound integrated GPU.
    const float rscale = clampf(gRS.renderScale, 0.4f, 1.0f);
    int pw = std::max(1, (int)(b.width * s * rscale));
    int ph = std::max(1, (int)(b.height * s * rscale));
    if (pw != a.pixelW || ph != a.pixelH || s != a.scale) {
        // Anything the input layer holds in drawable pixels has to move with the
        // drawable. The cursor is re-derived from where it sits in the view; the
        // drag anchor has no event to recompute it from, so it is rescaled.
        if (a.pixelW > 0 && a.pixelH > 0) {
            a.in.dragX *= (float)pw / (float)a.pixelW;
            a.in.dragY *= (float)ph / (float)a.pixelH;
        }
        a.pixelW = pw; a.pixelH = ph; a.scale = s;
        a.in.mx = a.in.mnx * (float)pw;
        a.in.my = a.in.mny * (float)ph;
        [view metalLayer].drawableSize = CGSizeMake(pw, ph);
        a.renderer.resize(pw, ph);
    }

    double now = CACurrentMediaTime();
    float dt = (float)(now - a.lastTime);
    a.lastTime = now;
    dt = clampf(dt, 0.0f, 0.10f);
    a.fps = lerpf(a.fps, 1.0f / std::max(dt, 1e-4f), 0.06f);
    gAnimTime += dt;

    if (a.in.keyPressed['R'] && a.game.winner >= 0) resetMatch(a, a.seed + 1);
    if (a.in.keyPressed[','] ) { a.speed = std::max(0.25f, a.speed * 0.5f); say("Slower"); }
    if (a.in.keyPressed['.'] ) { a.speed = std::min(4.0f, a.speed * 2.0f); say("Faster"); }

    // Verification hook: pin the cursor to one edge so edge scrolling can be
    // checked without a human moving a mouse.
    if (gEdgeTest) {
        a.in.mouseInside = true;
        if (!strcmp(gEdgeTest, "left"))        { a.in.mx = 1.0f;                 a.in.my = a.pixelH * 0.5f; }
        else if (!strcmp(gEdgeTest, "right"))  { a.in.mx = a.pixelW - 1.0f;      a.in.my = a.pixelH * 0.5f; }
        else if (!strcmp(gEdgeTest, "top"))    { a.in.mx = a.pixelW * 0.5f;      a.in.my = 1.0f; }
        else if (!strcmp(gEdgeTest, "bottom")) { a.in.mx = a.pixelW * 0.5f;      a.in.my = a.pixelH - 1.0f; }
        else if (!strcmp(gEdgeTest, "none"))   { a.in.mx = a.pixelW * 0.5f;      a.in.my = a.pixelH * 0.5f; }
    }

    if (!a.menuOpen) updateCamera(a, dt);
    buildCommandCard(a);          // card must reflect the selection the user sees
    handleInput(a, dt);
    if (!a.menuOpen) buildCommandCard(a);   // selection may have changed

    if (!a.paused && !a.menuOpen && a.game.winner < 0) {
        float remaining = dt * a.speed;
        int guard = 0;
        while (remaining > 1e-4f && guard++ < 8) {
            float step = std::min(remaining, 1.0f / 60.0f);
            a.game.update(step);
            double t0 = CACurrentMediaTime();
            a.enemyAI.update(step);
            a.aiMs = lerpf(a.aiMs, (float)((CACurrentMediaTime() - t0) * 1000.0), 0.05f);
            remaining -= step;
        }
    }
    if (a.toastTimer > 0.0f) a.toastTimer -= dt;

    // Adaptive resolution. A fanless-class part throttles under sustained load,
    // so rather than pick one setting up front, watch the measured GPU time and
    // trade resolution for frame rate only when it is actually needed.
    if (gAdaptive) {
        double g = a.renderer.stats().gpuFrameMs;
        if (g > 0.0) { gAdaptSum += g; gAdaptN++; }
        gAdaptWindow += dt;
        if (gAdaptWindow >= 1.0f && gAdaptN > 20) {
            double avg = gAdaptSum / gAdaptN;
            gAdaptWindow = 0.0f; gAdaptSum = 0.0; gAdaptN = 0;
            gRS.renderScale = adaptStep(gAdapt, gRS.renderScale, avg);
        }
    }

    // Verification hook: the burn marks are normally spawned under a fireball,
    // which is exactly what makes them hard to check. This lays all four out in
    // a row with nothing drawn over them.
    if (gDecalTest) {
        gDecalTest = false;
        a.cam.focus = a.cam.focus + v2{0.0f, 26.0f};   // clear ground, no base
        for (int i = 0; i < 4; i++) {
            v2 at = a.cam.focus + v2{(float)(i % 2) * 9.0f - 4.5f,
                                     (float)(i / 2) * 9.0f - 4.5f};
            Particle d{};
            d.alive = true;
            d.pos = v3{at.x, a.game.groundY(at) + 0.10f, at.y};
            d.life = 0; d.maxLife = 600.0f;
            d.size0 = d.size1 = 4.2f;
            d.col0 = v4{0.045f, 0.038f, 0.032f, 0.85f};
            d.col1 = v4{0.045f, 0.038f, 0.032f, 0.85f};
            d.param = 0.125f + 0.25f * (float)i;   // one of each mark
            d.kind = 5;
            a.game.spawnParticle(d);
        }
        for (int i = 0; i < 2; i++) {
            v2 at = a.cam.focus + v2{(float)(i * 2 - 1) * 15.0f, 0.0f};
            Particle m{};
            m.alive = true;
            m.pos = v3{at.x, a.game.groundY(at) + 1.0f, at.y};
            m.life = 0.9f + 0.7f * (float)i; m.maxLife = 9.0f;
            m.size0 = 2.0f; m.size1 = 6.0f;
            m.col0 = v4{0.50f, 0.48f, 0.46f, 0.75f};
            m.col1 = v4{0.34f, 0.34f, 0.35f, 0.0f};
            m.kind = 6;
            a.game.spawnParticle(m);
        }
    }

    if (gUIDemo && a.selection.empty()) {
        for (size_t i = 0; i < a.game.ents.size(); i++) {
            const Entity& e = a.game.ents[i];
            if (e.alive && e.deathTimer < 0 && e.team == 0 && e.type == UT_WORKER) {
                a.selection.push_back(a.game.handleOf((int)i));
                break;
            }
        }
        buildCommandCard(a);
    }
    if (gCamAtSet) a.cam.focus = gCamAt;
    if (gUIMenu) a.menuOpen = true;
    if (gHoverBtn >= 0 && gHoverBtn < (int)a.buttons.size()) {
        const Button& b = a.buttons[gHoverBtn];
        a.in.mx = b.x + b.w * 0.5f;
        a.in.my = b.y + b.h * 0.5f;
    }

    updateFOW(a, dt);
    buildScene(a);
    buildHUD(a);
    a.renderer.render(a.frame);

    // Clear edge-triggered input for the next frame.
    memset(a.in.keyPressed, 0, sizeof(a.in.keyPressed));
    a.in.lPressed = a.in.lReleased = a.in.rPressed = false;
    a.in.doubleClick = false;
}

int gBenchFrames = 0;
bool gBenchInfluence = false;
bool gBoom = false;
v2   gEdgeStart{};
bool gSelectAll = false;
int  gStress = 0;
std::string gShotPath;
int gShotFrame = 200;

} // namespace

// ---------------------------------------------------------------- delegate
@interface AppDelegate : NSObject <NSApplicationDelegate>
@end

@implementation AppDelegate {
    GameWindow*    _window;
    GameView*      _view;
    CADisplayLink* _link;
    NSTimer*       _timer;
    double         _gpuSum;
    int            _gpuN;
    double         _gpuMax;
    int            _frames;
}

- (void)applicationDidFinishLaunching:(NSNotification*)n {
    NSRect r = NSMakeRect(0, 0, 1280, 780);
    _window = [[GameWindow alloc]
        initWithContentRect:r
                  styleMask:(NSWindowStyleMaskTitled | NSWindowStyleMaskClosable |
                             NSWindowStyleMaskMiniaturizable | NSWindowStyleMaskResizable)
                    backing:NSBackingStoreBuffered
                      defer:NO];
    [_window setTitle:@"Starforge"];
    [_window center];
    // Full screen is handled by setFullscreen(), not by AppKit's native mode,
    // so the green button must not offer the native one.
    _window.collectionBehavior |= NSWindowCollectionBehaviorFullScreenNone;
    gWindow = _window;
    _view = [[GameView alloc] initWithFrame:r];
    [_window setContentView:_view];
    [_window makeFirstResponder:_view];
    [_window makeKeyAndOrderFront:nil];
    [NSApp activateIgnoringOtherApps:YES];

    // Full screen by default when someone is actually playing. Benchmarks and
    // screenshots stay windowed so their numbers remain comparable between runs;
    // --fullscreen / --windowed override either way.
    const bool headless = (gBenchFrames > 0 || !gShotPath.empty() || gBenchInfluence);
    const bool wantFS = (gFullscreen >= 0) ? (gFullscreen == 1) : !headless;
    // Deferred: the window needs to be on screen before -screen returns one.
    if (wantFS) dispatch_async(dispatch_get_main_queue(), ^{ setFullscreen(true); });

    float sc = (float)_window.backingScaleFactor;
    gApp->scale = sc;
    gApp->pixelW = (int)(r.size.width * sc);
    gApp->pixelH = (int)(r.size.height * sc);
    [_view metalLayer].drawableSize = CGSizeMake(gApp->pixelW, gApp->pixelH);
    [_view metalLayer].contentsScale = sc;

    gApp->game.init(gApp->seed);
    {
        std::string ierr;
        gApp->gpuInfluence = sf::ai::createMetalInfluenceBackend(ierr);
        if (!gApp->gpuInfluence)
            fprintf(stderr, "AI influence maps falling back to CPU: %s\n", ierr.c_str());
    }
    gApp->enemyAI.init(&gApp->game, 1, gApp->seed, gApp->gpuInfluence);

    // Measured comparison for the influence field, the one AI workload that is
    // genuinely wide parallel arithmetic.
    if (gBenchInfluence) {
        using namespace sf::ai;
        const int N = InfluenceMap::N;
        std::vector<InfluenceUnit> units;
        Rng r(4);
        for (int i = 0; i < 300; i++)
            units.push_back(InfluenceUnit{r.range(0, Terrain::SIZE), r.range(0, Terrain::SIZE),
                                          1.0f, 20.0f, (i % 2) ? 1.0f : -1.0f, 0, 0, 0});
        std::vector<float> out((size_t)N * N * 4);
        auto timeIt = [&](IInfluenceBackend* b, const char* label) {
            InfluenceMap m; m.setBackend(b);
            const int iters = 200;
            double t0 = CACurrentMediaTime();
            for (int i = 0; i < iters; i++) {
                if (b) b->compute(units.data(), (int)units.size(), N, InfluenceMap::CELL, out.data());
                else {
                    // CPU reference path, same maths as InfluenceMap::computeCPU.
                    std::fill(out.begin(), out.end(), 0.0f);
                    for (const auto& u : units) {
                        float reach = u.range * 2.0f;
                        for (int z = 0; z < N; z++) for (int x = 0; x < N; x++) {
                            float dx = (x + 0.5f) * InfluenceMap::CELL - u.x;
                            float dz = (z + 0.5f) * InfluenceMap::CELL - u.z;
                            float d2 = dx * dx + dz * dz;
                            if (d2 > reach * reach) continue;
                            out[(z * N + x) * 4 + (u.team > 0 ? 0 : 1)] +=
                                u.strength / (1.0f + d2 / (u.range * u.range));
                        }
                    }
                }
            }
            printf("influence %-10s %6.3f ms/build  (%d units, %dx%d grid, %d iters)\n",
                   label, (CACurrentMediaTime() - t0) * 1000.0 / iters,
                   (int)units.size(), N, N, iters);
        };
        timeIt(nullptr, "cpu");
        if (gApp->gpuInfluence) timeIt(gApp->gpuInfluence, "metal-gpu");
        else printf("influence metal-gpu   unavailable\n");
        [NSApp terminate:nil];
    }
    std::string err;
    if (!gApp->renderer.init((__bridge void*)[_view metalLayer], gApp->game.terrain, err, gRS)) {
        fprintf(stderr, "renderer init failed: %s\n", err.c_str());
        exit(1);
    }
    gApp->renderer.resize(gApp->pixelW, gApp->pixelH);
    // --stress spawns two armies for load testing the renderer and simulation.
    if (gStress > 0) {
        Game& g = gApp->game;
        for (int t = 0; t < 2; t++)
            for (int i = 0; i < gStress; i++) {
                float a = (float)i * 2.39996f;
                float r = 8.0f + std::sqrt((float)i) * 2.2f;
                v2 p = g.nav.nearestWalkable(g.basePos[t] + v2{std::cos(a), std::sin(a)} * r);
                g.spawn(i % 4 == 0 ? UT_MAULER : UT_TROOPER, t, p);
            }
    }
    gApp->cam.focus = gApp->game.basePos[0];
    gApp->cam.smoothY = gApp->game.terrain.heightAt(gApp->cam.focus.x, gApp->cam.focus.y);
    gApp->in.mx = gApp->pixelW * 0.5f;
    gApp->in.my = gApp->pixelH * 0.5f;
    gApp->lastTime = CACurrentMediaTime();

    // CADisplayLink is driven by display refresh, so it stops firing when the
    // screen sleeps. That is fine for interactive play (nobody is watching) but
    // it silently hangs --bench and --shot, which is exactly when no human is
    // present to notice. Headless modes therefore run off a plain timer.
    if (gBenchFrames > 0 || !gShotPath.empty() || gBenchInfluence) {
        _timer = [NSTimer scheduledTimerWithTimeInterval:1.0 / 60.0
                                                  target:self
                                                selector:@selector(onTimer:)
                                                userInfo:nil
                                                 repeats:YES];
        [[NSRunLoop currentRunLoop] addTimer:_timer forMode:NSRunLoopCommonModes];
    } else {
        _link = [_view displayLinkWithTarget:self selector:@selector(onFrame:)];
        [_link addToRunLoop:[NSRunLoop currentRunLoop] forMode:NSRunLoopCommonModes];
    }
}

- (void)onTimer:(NSTimer*)t { [self onFrame:nil]; }

- (void)onFrame:(CADisplayLink*)link {
    if (_frames == 5) gEdgeStart = gApp->cam.focus;   // after the camera settles
    tick(*gApp, _view);
    _frames++;
    // Averaged over the run: a single frame's GPU timing is far too noisy to
    // compare optimisation levers with.
    if (_frames > 40) {
        double g = gApp->renderer.stats().gpuFrameMs;
        if (g > 0.0) { _gpuSum += g; _gpuN++; _gpuMax = std::max(_gpuMax, g); }
    }
    // Deterministic effects test: stagger explosions so one capture shows the
    // sprite sheet at several different points in its animation.
    if (gBoom && !gShotPath.empty()) {
        int lead = gShotFrame - _frames;
        // The 240-frame lead is four seconds old by the time the shot is taken:
        // fire and shockwave long gone, so the burn mark it left is unobscured.
        if (lead == 240 || lead == 40 || lead == 26 || lead == 14 || lead == 5) {
            App& a = *gApp;
            float ph = (lead == 240) ? 5.9f
                     : (lead == 40) ? 0.0f : (lead == 26 ? 1.6f : (lead == 14 ? 3.2f : 4.8f));
            v2 at = a.cam.focus + v2{std::cos(ph) * 11.0f, std::sin(ph) * 11.0f};
            a.game.explosion(v3{at.x, a.game.groundY(at) + 1.0f, at.y}, 2.6f,
                             v3{1.0f, 0.62f, 0.22f});
        }
    }
    if (!gShotPath.empty() && _frames == gShotFrame) {
        if (gSelectAll) {
            gApp->selection.clear();
            for (size_t i = 0; i < gApp->game.ents.size(); i++) {
                const Entity& e = gApp->game.ents[i];
                if (e.alive && e.deathTimer < 0 && e.team == 0) gApp->selection.push_back(gApp->game.handleOf((int)i));
            }
            buildCommandCard(*gApp);
            buildHUD(*gApp);
            gApp->renderer.render(gApp->frame);
        }
        gApp->renderer.requestCapture(gShotPath);
    }
    if (!gShotPath.empty() && _frames > gShotFrame && gApp->renderer.captureComplete()) {
        printf("wrote %s\n", gShotPath.c_str());
        gShotPath.clear();
        if (gBenchFrames <= 0) [NSApp terminate:nil];
    }
    if (gBenchFrames > 0 && _frames >= gBenchFrames) {
        RenderStats rs = gApp->renderer.stats();
        printf("bench: %d frames | %.1f fps | %dx%d | %d draws | %d instances | %d tris | tex %.1f MB\n",
               _frames, gApp->fps, gApp->pixelW, gApp->pixelH, rs.drawCalls, rs.instances, rs.triangles, rs.texMB);
        printf("gpu:   render %.2f ms avg, %.2f ms peak over %d frames (shadow+scene+bloom,\n"
               "       excl. present wait)\n",
               _gpuN ? _gpuSum / _gpuN : 0.0, _gpuMax, _gpuN);
        if (gEdgeTest)
            printf("edge-test %-6s : focus moved (%+.1f, %+.1f) world units\n", gEdgeTest,
                   gApp->cam.focus.x - gEdgeStart.x, gApp->cam.focus.y - gEdgeStart.y);
        printf("sim: t=%.1fs ents=%zu particles=%zu ore=%d supply=%d/%d\n",
               gApp->game.time, gApp->game.ents.size(), gApp->game.particles.size(),
               gApp->game.fac[0].ore, gApp->game.fac[0].supplyUsed, gApp->game.fac[0].supplyCap);
        [NSApp terminate:nil];
    }
}

- (BOOL)applicationShouldTerminateAfterLastWindowClosed:(NSApplication*)s { return YES; }
- (void)applicationDidBecomeActive:(NSNotification*)n { reassertFullscreenPresentation(); }
// Target of the View menu item; reached through the responder chain.
- (void)toggleGameFullScreen:(id)sender { toggleFullscreen(); }
@end

// ---------------------------------------------------------------- entry
static void installMenu() {
    NSMenu* bar = [NSMenu new];

    NSMenuItem* appItem = [NSMenuItem new];
    [bar addItem:appItem];
    NSMenu* appMenu = [NSMenu new];
    [appMenu addItemWithTitle:@"Quit Starforge" action:@selector(terminate:) keyEquivalent:@"q"];
    [appItem setSubmenu:appMenu];

    // Gives the standard Ctrl-Cmd-F toggle. A plain letter key would collide
    // with the build hotkeys ('F' places a Workshop).
    NSMenuItem* viewItem = [NSMenuItem new];
    [bar addItem:viewItem];
    NSMenu* viewMenu = [[NSMenu alloc] initWithTitle:@"View"];
    NSMenuItem* fs = [[NSMenuItem alloc] initWithTitle:@"Toggle Full Screen"
                                               action:@selector(toggleGameFullScreen:)
                                        keyEquivalent:@"f"];
    fs.keyEquivalentModifierMask = NSEventModifierFlagCommand | NSEventModifierFlagControl;
    [viewMenu addItem:fs];
    [viewItem setSubmenu:viewMenu];

    [NSApp setMainMenu:bar];
}

int main(int argc, const char* argv[]) {
    @autoreleasepool {
        static App app;
        gApp = &app;
        app.seed = (uint32_t)time(nullptr) & 0xFFFF;
        bool mouseTest = false, adaptTest = false;
        for (int i = 1; i < argc; i++) {
            if (!strcmp(argv[i], "--seed") && i + 1 < argc) app.seed = (uint32_t)atoi(argv[++i]);
            else if (!strcmp(argv[i], "--bench") && i + 1 < argc) gBenchFrames = atoi(argv[++i]);
            else if (!strcmp(argv[i], "--no-help")) app.showHelp = false;
            else if (!strcmp(argv[i], "--shot") && i + 1 < argc) gShotPath = argv[++i];
            else if (!strcmp(argv[i], "--shot-frame") && i + 1 < argc) gShotFrame = atoi(argv[++i]);
            else if (!strcmp(argv[i], "--speed") && i + 1 < argc) app.speed = (float)atof(argv[++i]);
            else if (!strcmp(argv[i], "--cam-dist") && i + 1 < argc) app.cam.dist = (float)atof(argv[++i]);
            else if (!strcmp(argv[i], "--cam-yaw") && i + 1 < argc) app.cam.yaw = (float)atof(argv[++i]);
            else if (!strcmp(argv[i], "--select-all")) gSelectAll = true;
            else if (!strcmp(argv[i], "--ai-debug")) app.showAI = true;
            else if (!strcmp(argv[i], "--bench-influence")) gBenchInfluence = true;
            else if (!strcmp(argv[i], "--boom")) gBoom = true;
            else if (!strcmp(argv[i], "--render-scale") && i + 1 < argc) gRS.renderScale = (float)atof(argv[++i]);
            else if (!strcmp(argv[i], "--msaa") && i + 1 < argc) gRS.msaa = atoi(argv[++i]);
            else if (!strcmp(argv[i], "--shadow-res") && i + 1 < argc) gRS.shadowRes = atoi(argv[++i]);
            else if (!strcmp(argv[i], "--no-bloom")) gRS.bloom = false;
            else if (!strcmp(argv[i], "--no-adaptive")) gAdaptive = false;
            else if (!strcmp(argv[i], "--mouse-test")) mouseTest = true;
            else if (!strcmp(argv[i], "--adapt-test")) adaptTest = true;
            else if (!strcmp(argv[i], "--windowed")) gFullscreen = 0;
            else if (!strcmp(argv[i], "--fullscreen")) gFullscreen = 1;
            else if (!strcmp(argv[i], "--edge-test") && i + 1 < argc) gEdgeTest = argv[++i];
            else if (!strcmp(argv[i], "--no-fow")) gFogOfWar = 0.0f;
            else if (!strcmp(argv[i], "--decal-test")) gDecalTest = true;
            else if (!strcmp(argv[i], "--fs-debug")) gFSDebug = true;
            else if (!strcmp(argv[i], "--cam-at") && i + 2 < argc) {
                gCamAt = v2{(float)atof(argv[i + 1]), (float)atof(argv[i + 2])};
                gCamAtSet = true; i += 2;
            }
            else if (!strcmp(argv[i], "--ui-demo")) gUIDemo = true;
            else if (!strcmp(argv[i], "--ui-menu")) { gUIMenu = true; gUIDemo = true; }
            else if (!strcmp(argv[i], "--hover-button") && i + 1 < argc) { gHoverBtn = atoi(argv[++i]); gUIDemo = true; }
            else if (!strcmp(argv[i], "--stress") && i + 1 < argc) gStress = atoi(argv[++i]);
            else if (!strcmp(argv[i], "--help")) {
                printf("starforge [--seed N] [--bench FRAMES] [--shot PATH] [--shot-frame N] [--speed X] [--no-help]\n");
                return 0;
            }
        }
        // Both of these replay a bug that shipped: the cursor drifting away from
        // the game, and the HUD changing resolution every couple of seconds.
        // Neither needs a GPU or a hand on the mouse, so neither has to be
        // rediscovered by a player.
        if (mouseTest) {
            int fails = 0;
            // A 1280x780 view on a 2x display, rendering at 70% scale.
            const double bw = 1280, bh = 780;
            const int pw = (int)(bw * 2.0 * 0.7), ph = (int)(bh * 2.0 * 0.7);
            struct { const char* name; double px, py; float ex, ey; } cases[] = {
                {"top-left",     0,        bh,       0.0f,           0.0f},
                {"bottom-right", bw,       0,        (float)pw,      (float)ph},
                {"centre",       bw / 2,   bh / 2,   pw * 0.5f,      ph * 0.5f},
                {"three-qtr",    bw * 0.75, bh * 0.25, pw * 0.75f,   ph * 0.75f},
            };
            for (const auto& c : cases) {
                MousePos m = mapMouse(c.px, c.py, bw, bh, pw, ph);
                bool ok = std::fabs(m.x - c.ex) < 0.5f && std::fabs(m.y - c.ey) < 0.5f;
                if (!ok) fails++;
                printf("mouse %-13s view(%7.1f,%6.1f) -> px(%7.1f,%7.1f) expect(%7.1f,%7.1f) %s\n",
                       c.name, c.px, c.py, m.x, m.y, c.ex, c.ey, ok ? "ok" : "FAIL");
            }
            printf("mouse-test: drawable %dx%d, %d failure(s)\n", pw, ph, fails);
            return fails ? 1 : 0;
        }
        if (adaptTest) {
            // GPU cost is modelled as a fixed part (geometry, shadows, the UI)
            // plus a part proportional to pixel count, and optionally a
            // neighbouring app stealing time in bursts. The bursty case is the
            // one that shipped badly: the controller could pass its step-up test
            // during every quiet spell and fail it during every busy one, so the
            // resolution changed every few seconds for as long as you played.
            // The property that matters is convergence, not silence: finding the
            // right scale costs a couple of changes, and probing upward once is
            // correct behaviour. What is not acceptable is still changing minutes
            // later, so the assertion is that the back half of the run is quiet.
            struct Case { const char* name; double fixed, perPixel, burst; int period; };
            const Case cases[] = {
                {"steady, comfortable",  3.0,  5.0, 0.0, 0},
                {"steady, over budget",  4.0, 12.0, 0.0, 0},
                {"neighbour every 3 s",  4.0, 12.0, 6.0, 3},
                {"neighbour every 4 s",  4.0,  8.0, 7.0, 4},
                {"neighbour every 6 s",  3.5, 10.0, 8.0, 6},
                {"neighbour every 10 s", 4.0,  9.0, 9.0, 10},
            };
            int fails = 0;
            for (const auto& c : cases) {
                AdaptState st;
                float scale = 1.0f;
                const int kWindows = 160, kSettleBy = 80;
                int changes = 0, lastChange = -1;
                for (int w = 0; w < kWindows; w++) {
                    double burst = (c.period && (w / c.period) % 2 == 0) ? c.burst : 0.0;
                    double avg = c.fixed + c.perPixel * (double)(scale * scale) + burst;
                    float next = adaptStep(st, scale, avg);
                    if (std::fabs(next - scale) > 0.001f) { changes++; lastChange = w; }
                    scale = next;
                }
                bool ok = (lastChange < kSettleBy);
                if (!ok) fails++;
                printf("adapt %-20s settled %.2f  %d change(s), last at window %-3d %s\n",
                       c.name, scale, changes, lastChange, ok ? "ok" : "FAIL");
            }
            printf("adapt-test: %d failure(s) -- a run is a pass if it stops changing\n"
                   "            within %d windows (~%d s) and stays put for the rest.\n",
                   fails, 80, 80);
            return fails ? 1 : 0;
        }
        NSApplication* nsapp = [NSApplication sharedApplication];
        [nsapp setActivationPolicy:NSApplicationActivationPolicyRegular];
        installMenu();
        AppDelegate* d = [AppDelegate new];
        [nsapp setDelegate:d];
        [nsapp run];
    }
    return 0;
}
