// FramePacer.h -- when to let a frame go so that frames the host captured at
// an even cadence also come out evenly here.
//
// Every frame carries its host capture time (pts). The time it takes from
// capture to "ready here" varies frame to frame (Wi-Fi, decoder), and showing
// each frame the moment it is ready turns that variance into uneven on-screen
// durations -- on a 120 Hz panel a 60 fps stream then holds frames for 1, 2
// or 3 refreshes instead of always 2, which reads as micro-judder.
//
// The pacer releases each frame at pts + base + margin:
//   base   -- the lower envelope of (ready - pts): the quickest any recent
//             frame made it. Drops at once to a quicker frame and creeps up
//             slowly otherwise, which also follows host/client clock drift.
//   margin -- a percentile of how far above base frames land, plus 1 ms,
//             capped. A percentile, not an average that jumps on spikes: a
//             rare 30-100 ms Wi-Fi stall would otherwise hold every frame
//             ~30 ms for seconds afterwards (seen live). Frames past it just
//             go late. Which percentile and what cap is the user's
//             latency/smoothness slider (FramePacerConfigure).
// So the added latency is only what the measured spread needs. (The pacer
// this replaced anchored to one frame and pushed its schedule later on every
// late frame, ratcheting latency up toward the slowest frames, then added
// 3x the jitter on top.) A frame already past its slot goes at once and does
// not move the schedule.
//
// Header-only and side-effect free: shared by VideoDepacketizer.c (before
// decode, every platform) and client renderers that pace at the display.

#pragma once

#include <stdint.h>

#define FRAME_PACER_MARGIN_MIN_US  2000
#define FRAME_PACER_MARGIN_MAX_US  50000 // the cap of the smoothest level
// A jump in (ready - pts) this large is a new timeline (stream restart, host
// clock reset, a real stall), not jitter: start over from that frame.
#define FRAME_PACER_RESET_US       250000
// Percentile tracker: down this much on a frame below the estimate (~0.3 ms
// a second at 60 fps, every level), up enough on one above that it settles
// where (100 - percentile)% of frames are above.
#define FRAME_PACER_QUANTILE_DOWN_US 5

// The slider, level 1..FRAME_PACER_LEVELS (0 is off): the share of frames
// the margin is sized to keep on time, and its cap. Level 3 is what was
// measured live on Wi-Fi at 120 Hz: ~1% uneven frames for ~11 ms average
// hold, against ~50% uneven with no pacing.
#define FRAME_PACER_LEVELS 4
static const struct { int percentile; int maxMarginUs; } FRAME_PACER_LEVEL[FRAME_PACER_LEVELS + 1] = {
    { 0, 0 },     // off
    { 80, 10000 },
    { 90, 20000 },
    { 95, 30000 },
    { 99, 50000 },
};

typedef struct {
    int     have;
    int64_t lastPtsUs;
    int64_t baseUs;
    int64_t spreadUs; // ~percentile of (ready - pts - base)
    int     percentile;
    int64_t maxMarginUs;
} FRAME_PACER;

static inline void FramePacerReset(FRAME_PACER* p) {
    p->have = 0;
}

// level: 1..FRAME_PACER_LEVELS, clamped. Starts the pacer over.
static inline void FramePacerConfigure(FRAME_PACER* p, int level) {
    if (level < 1) level = 1;
    if (level > FRAME_PACER_LEVELS) level = FRAME_PACER_LEVELS;
    p->percentile = FRAME_PACER_LEVEL[level].percentile;
    p->maxMarginUs = FRAME_PACER_LEVEL[level].maxMarginUs;
    FramePacerReset(p);
}

static inline int64_t FramePacerMarginUs(const FRAME_PACER* p) {
    int64_t m = p->spreadUs + 1000;
    if (m > p->maxMarginUs) m = p->maxMarginUs;
    if (m < FRAME_PACER_MARGIN_MIN_US) m = FRAME_PACER_MARGIN_MIN_US;
    return m;
}

// ptsUs: the frame's host capture time; readyUs: when it became ready here
// (local clock). Returns the local time to release it at; a value <= readyUs
// means right away. Call once per frame, in pts order, after
// FramePacerConfigure.
static inline int64_t FramePacerTarget(FRAME_PACER* p, int64_t ptsUs, int64_t readyUs) {
    int64_t offset = readyUs - ptsUs;
    int64_t drift = offset - p->baseUs;
    if (!p->have || ptsUs <= p->lastPtsUs || drift > FRAME_PACER_RESET_US || drift < -FRAME_PACER_RESET_US) {
        p->have = 1;
        p->lastPtsUs = ptsUs;
        p->baseUs = offset;
        p->spreadUs = 0;
        return readyUs;
    }
    p->lastPtsUs = ptsUs;

    if (drift < 0) {
        p->baseUs = offset;
        drift = 0;
    }
    else {
        // ~1/256 of the way up per frame (and at least 2us): a lasting
        // slowdown is followed in a few seconds, one fast frame pulls it
        // right back down.
        int64_t rise = drift >> 8;
        p->baseUs += (rise > 2) ? rise : 2;
        drift = offset - p->baseUs;
        if (drift < 0) drift = 0;
    }

    if (drift > p->spreadUs) {
        // Never chase past the cap: a long bad spell would otherwise leave
        // it far above, and coming down is deliberately slow.
        if (p->spreadUs < p->maxMarginUs) {
            p->spreadUs += FRAME_PACER_QUANTILE_DOWN_US * p->percentile / (100 - p->percentile);
        }
    }
    else if (p->spreadUs > 0) {
        p->spreadUs -= FRAME_PACER_QUANTILE_DOWN_US;
    }
    return ptsUs + p->baseUs + FramePacerMarginUs(p);
}
