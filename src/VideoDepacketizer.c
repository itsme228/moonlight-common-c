#include "Limelight-internal.h"

// Uncomment to test 3 byte Annex B start sequences with GFE
//#define FORCE_3_BYTE_START_SEQUENCES

static PLENTRY nalChainHead;
static PLENTRY nalChainTail;
static int nalChainDataLength;

static unsigned int nextFrameNumber;
static unsigned int startFrameNumber;
static bool waitingForNextSuccessfulFrame;
static bool waitingForIdrFrame;
static bool waitingForRefInvalFrame;
static unsigned int lastPacketInStream;
static bool decodingFrame;
static int frameType;
static uint16_t lastPacketPayloadLength;
static bool strictIdrFrameWait;
static uint64_t syntheticPtsBaseUs;
static uint16_t frameHostProcessingLatency;
static uint64_t firstPacketReceiveTimeUs;
static uint64_t firstPacketPresentationTime;
static uint32_t firstPacketRtpTimestamp;
static bool dropStatePending;
static bool idrFrameProcessed;

// ─── Adaptive playout jitter buffer ────────────────────────────────────────
// Client-side only: absorbs transient network jitter (WiFi packet delay
// variance) between frame arrivals without adding latency on a clean link.
// NVIDIA's original GameStream protocol has no equivalent here -- it relies
// solely on per-frame Reed-Solomon FEC (RtpVideoQueue.c) to recover from
// packet loss/reorder *within* one frame's own arrival window, then submits
// every completed frame to the decoder the instant it's assembled, with no
// deliberate delay at all. That's fine on the clean, near-zero-jitter wired
// LAN it was designed for, but it means the decoder simply starves whenever
// a frame's packets arrive late as a burst on a real WiFi link -- exactly
// the "Moonlight Decoder/Network stalled" events seen in benchmarking.
//
// This estimates jitter using the RFC 3550 transit-time technique, but on
// the RTP packet header's own 90kHz timestamp rather than the higher-level
// presentationTimeUs field, because some hosts (see the `syntheticPtsBaseUs`
// fallback above) synthesize that field from local receive time when they
// don't send a real PTS extension, which would otherwise make the delta
// always ~0 and hide real jitter. The RTP header timestamp itself is a
// mandatory wire-format field for both Sunshine and this project's own
// streamer (and any other GFE-compatible host), so this works identically
// for either backend and needs no protocol changes at all -- pure local
// receive-side timing.
//
// Frames are released according to a fixed local schedule anchored to one
// reference frame: schedule(frame) = anchorLocalUs + (frame.rtp - anchorRtp)
// converted to microseconds. On a steady link this adds a constant latency
// (the buffer depth) without changing the *gap* between releases -- the
// constant offset cancels out between consecutive on-schedule frames -- so
// it smooths out arrival jitter without inflating the stall metric.
//
// If a frame's slot has already passed by the time we dequeue it, it's
// released immediately -- but the schedule is only *re-anchored* (given up
// on and restarted from here) when the lateness is large (> a few buffer
// depths' worth, see PLAYOUT_RESYNC_THRESHOLD_US), meaning a real stall or
// pause happened, not just jitter. A single late frame does NOT move the
// anchor: the next frame is still scheduled against the same reference, so
// the buffer's margin naturally refills as soon as arrivals catch back up.
//
// Design notes from earlier iterations that didn't work, kept here because
// the failure modes are non-obvious and easy to reintroduce by accident:
//   v1: gated the delay on "is the queue currently empty". Wrong -- an
//       empty queue is the *normal* steady-state (the decoder consumes
//       frames about as fast as they arrive), so that added the delay to
//       nearly every frame's own submit gap, directly inflating the stall
//       metric it was meant to reduce (93 vs. 79 stalls/90s on a live
//       WiFi retest).
//   v2: anchored schedule as described above, but re-anchored on *every*
//       single late frame, not just large/sustained lateness. That's
//       exactly backwards: frequent small lateness is precisely what
//       happens when the network is jittery -- i.e. exactly when the
//       buffer is needed -- and resyncing on every one of those misses
//       collapses the schedule back to "release immediately" right when
//       smoothing matters most, making it a no-op under load (measured no
//       better than baseline, and worse in one run).
//
// v7: this now also applies on the CAPABILITY_DIRECT_SUBMIT path (see the
// call site in reassembleFrame() below), not just the pull/queued decode
// thread. Two earlier full architectures were tried and abandoned first:
//   - Removing CAPABILITY_DIRECT_SUBMIT to activate moonlight-common-c's
//     own queue+decoder-thread machinery (LiWaitForNextVideoFrame) worked
//     for the buffer itself (confirmed: applied delay tracked jitter
//     correctly, stalls dropped sharply), but running decode/render on a
//     separate thread from the network receive thread caused a *worse*,
//     unrelated regression on this project's macOS Metal/CVDisplayLink
//     path: real render throughput to the screen collapsed to ~10-15fps
//     while decode itself kept running at the full ~60fps (confirmed live:
//     CVDisplayLink's own fire rate halved from ~60Hz to ~30Hz with the
//     extra thread running, reproduced twice back-to-back by toggling the
//     capability bit with nothing else changed). Not something this file
//     can fix -- it's downstream Metal/AppKit thread-scheduling behavior.
//
// Applying the delay directly on the DIRECT_SUBMIT path avoids that
// regression entirely (same one thread as before, no new thread), but
// changes the risk profile: this thread is also the one reading the video
// UDP socket, so sleeping here delays draining it, not just delaying
// presentation. The video socket's receive buffer is sized generously
// (RTP_RECV_PACKETS_BUFFERED = 2048 packets, VideoStream.c) -- around 2MB,
// which even at a high 80 Mbps stream bitrate absorbs on the order of 200ms
// of buffered traffic -- so a bounded sleep here has real margin before
// risking a kernel-buffer overflow, but PLAYOUT_DELAY_MAX_US below is kept
// well under that margin rather than pushed to it. See
// PLAYOUT_POST_RESYNC_GRACE_US for the other DIRECT_SUBMIT-specific risk
// (a backlog of already-arrived frames draining right after a stall) and
// why it matters more here than on the queued path.
#define PLAYOUT_JITTER_SHIFT_GROW    2      // fast growth on a real jitter spike (~1/4 weight)
#define PLAYOUT_JITTER_SHIFT_DECAY   6      // slow decay back down (~1/64 weight) once it's calm again
#define PLAYOUT_DELAY_MULTIPLIER     3      // target delay = jitter estimate * this
#define PLAYOUT_DELAY_MAX_US         50000  // hard cap -- never add more than 50ms. Kept well under the
                                             // ~200ms the video socket's receive buffer can absorb (see
                                             // above) since this now runs on the same thread that drains
                                             // that socket, unlike the abandoned queued-decode-thread
                                             // design which could afford a 100ms cap.
#define PLAYOUT_RESYNC_THRESHOLD_US  150000 // give up on the schedule only past this much lateness
#define PLAYOUT_POST_RESYNC_GRACE_US 200000 // no delay at all for this long after any resync (including
                                             // the very first frame of a connection). A resync means we
                                             // just fell badly behind (a real stall/pause) -- on the
                                             // DIRECT_SUBMIT path, the frames right after very often
                                             // arrive as a back-to-back burst already sitting in the
                                             // socket buffer, and individually re-pacing each one back to
                                             // nominal cadence would mean this thread -- which also reads
                                             // that socket -- spends the whole burst duration asleep
                                             // instead of draining it. Skipping delay for a while after
                                             // any resync lets the burst drain immediately instead.
#define PLAYOUT_STATUS_LOG_FRAMES    120    // ~every 2s at 60fps -- tuning visibility, not spam
#define PLAYOUT_DELAY_MAX_STEP_US    1500   // v4 bug: the raw jitter estimate can jump by tens of ms in
                                             // one step (that's the point of its fast-growth EWMA), and
                                             // v4 fed that directly into the per-frame schedule -- so the
                                             // *delay itself* changing between two consecutive frames
                                             // perturbed their release gap by that same amount, recreating
                                             // the stall pattern the buffer exists to remove (confirmed:
                                             // 155 stalls even after jitter calmed to 2-8ms late in a
                                             // test run). This caps how fast the *applied* delay can move
                                             // per frame, decoupled from how fast the estimate itself
                                             // reacts.
#define PLAYOUT_RAW_LEAD_ALLOWANCE_US 3000  // how far the RAW schedule (before this frame's own
                                             // adaptive margin) may sit ahead of "now" before it's
                                             // treated as a stale anchor artifact instead of normal
                                             // jitter -- see the v9 fix at the "now < idealReleaseUs"
                                             // branch below. Set well above real measured transit-time
                                             // variance on a clean link (recvToEnqueue ~15-30us) but
                                             // well below a single frame interval, so it never fights
                                             // the deliberate jitter margin on a genuinely noisy link.
#define RTP_CLOCK_RATE_HZ            90000  // fixed by the Moonlight/GameStream RTP profile
static bool havePrevFrameTiming;
static uint64_t prevFrameReceiveTimeUs;
static uint32_t prevFrameRtpTimestamp;
static int64_t networkJitterUs;
static uint64_t appliedPlayoutDelayUs;
static unsigned int playoutStatusLogCounter;

static bool havePlayoutAnchor;
static uint64_t playoutAnchorLocalUs;
static uint32_t playoutAnchorRtpTimestamp;
static uint64_t lastResyncLocalUs;

// Converts a duration expressed in 90kHz RTP clock ticks to microseconds.
static uint64_t rtpTicksToUs(uint32_t deltaTicks) {
    return ((uint64_t)deltaTicks * 1000ULL) / (RTP_CLOCK_RATE_HZ / 1000);
}

// Returns the raw, instantaneous playout delay target derived from the
// current jitter estimate -- NOT what should be fed directly into the
// per-frame schedule (see advancePlayoutDelayUs()).
static uint64_t targetPlayoutDelayUs(void) {
    int64_t delay = networkJitterUs * PLAYOUT_DELAY_MULTIPLIER;
    if (delay < 0) {
        delay = 0;
    }
    else if (delay > PLAYOUT_DELAY_MAX_US) {
        delay = PLAYOUT_DELAY_MAX_US;
    }
    return (uint64_t)delay;
}

// Moves the actually-applied delay one step closer to targetPlayoutDelayUs(),
// by at most PLAYOUT_DELAY_MAX_STEP_US, and returns the new applied value.
// Must be called exactly once per frame (from playoutDelayForFrame) --
// calling it more or less often changes the effective ramp rate.
static uint64_t advancePlayoutDelayUs(void) {
    uint64_t target = targetPlayoutDelayUs();
    if (target > appliedPlayoutDelayUs) {
        uint64_t step = target - appliedPlayoutDelayUs;
        appliedPlayoutDelayUs += (step > PLAYOUT_DELAY_MAX_STEP_US) ? PLAYOUT_DELAY_MAX_STEP_US : step;
    }
    else if (target < appliedPlayoutDelayUs) {
        uint64_t step = appliedPlayoutDelayUs - target;
        appliedPlayoutDelayUs -= (step > PLAYOUT_DELAY_MAX_STEP_US) ? PLAYOUT_DELAY_MAX_STEP_US : step;
    }
    return appliedPlayoutDelayUs;
}

// Updates the jitter estimate using this newly-completed frame's RTP
// timestamp and local receive time versus the previous frame's.
static void updatePlayoutJitterEstimate(uint64_t receiveTimeUs, uint32_t rtpTimestamp) {
    if (havePrevFrameTiming) {
        // Signed 32-bit subtraction correctly handles the 90kHz RTP clock's own
        // wraparound (~13.25h) via modular arithmetic. A non-positive result means
        // this frame's RTP timestamp did not advance versus the last one we
        // measured -- observed after RFI/IDR recovery reorders which frame's
        // timing this function sees next -- and isn't a valid transit-time
        // sample. Previously the delta was cast straight to uint32_t, so a
        // negative delta wrapped to ~4 billion ticks and briefly sent
        // networkJitterUs into the billions of "microseconds" (seen live as
        // e.g. "jitter=17123437318us" in PlayoutBuffer's own status log).
        // Harmless today only because targetPlayoutDelayUs() hard-clamps to
        // PLAYOUT_DELAY_MAX_US, but it made the jitter metric meaningless for
        // diagnosing real network conditions during exactly the periods that
        // matter most.
        int32_t rtpDeltaTicks = (int32_t)(rtpTimestamp - prevFrameRtpTimestamp);
        if (rtpDeltaTicks > 0) {
            int64_t rtpDeltaUs = (int64_t)rtpTicksToUs((uint32_t)rtpDeltaTicks);
            int64_t receiveDeltaUs = (int64_t)(receiveTimeUs - prevFrameReceiveTimeUs);
            int64_t transitDeltaUs = receiveDeltaUs - rtpDeltaUs;
            if (transitDeltaUs < 0) {
                transitDeltaUs = -transitDeltaUs;
            }

            int shift = (transitDeltaUs > networkJitterUs) ? PLAYOUT_JITTER_SHIFT_GROW : PLAYOUT_JITTER_SHIFT_DECAY;
            networkJitterUs += (transitDeltaUs - networkJitterUs) >> shift;
        }
    }

    prevFrameReceiveTimeUs = receiveTimeUs;
    prevFrameRtpTimestamp = rtpTimestamp;
    havePrevFrameTiming = true;

    if (++playoutStatusLogCounter >= PLAYOUT_STATUS_LOG_FRAMES) {
        playoutStatusLogCounter = 0;
        Limelog("PlayoutBuffer: jitter=%lldus target=%lluus applied=%lluus\n",
                (long long)networkJitterUs, (unsigned long long)targetPlayoutDelayUs(),
                (unsigned long long)appliedPlayoutDelayUs);
    }
}

// Returns how long (in microseconds, possibly 0) the caller should wait
// before releasing this already-dequeued frame to the decoder, and advances
// the playout schedule. See the design comment above for the anchoring and
// re-sync-on-miss behavior.
//
// v3 bug (found via the debug logging below): the anchor's offset from raw
// arrival time was baked in once, from currentPlayoutDelayUs() as read at
// the very first frame -- when the jitter estimate is still 0, since there's
// no history yet. Because resyncs are rare by design (see above), that
// zero offset then never got refreshed for the rest of the session: the
// buffer was measuring jitter correctly (confirmed in the logs) but never
// actually acting on it, a pure no-op indistinguishable from no buffer at
// all. Fixed in v4 by keeping the anchor itself purely about raw arrival
// timing (no offset) and adding the *current* delay on top fresh for every
// frame, so the buffer depth actually tracks live jitter instead of a
// stale reading from connection start.
static uint64_t playoutDelayForFrame(uint64_t receiveTimeUs, uint32_t rtpTimestamp) {
    uint64_t now = PltGetMicroseconds();

    if (!havePlayoutAnchor) {
        playoutAnchorRtpTimestamp = rtpTimestamp;
        playoutAnchorLocalUs = receiveTimeUs;
        havePlayoutAnchor = true;
        lastResyncLocalUs = now;
    }

    // See PLAYOUT_POST_RESYNC_GRACE_US's doc comment above: skip delay
    // entirely for a while after any resync, so a burst of already-arrived
    // frames right after a stall drains immediately instead of each being
    // individually re-paced back to nominal cadence.
    if (now - lastResyncLocalUs < PLAYOUT_POST_RESYNC_GRACE_US) {
        return 0;
    }

    // Signed 32-bit subtraction correctly handles the RTP clock's own
    // wraparound; a non-positive result means this frame's RTP timestamp is
    // at or behind the anchor's -- observed after RFI/IDR recovery reorders
    // which frame this function sees next -- so the anchor is stale. Re-anchor
    // immediately instead of feeding it through rtpTicksToUs(): the previous
    // unsigned cast turned a negative delta into ~4 billion ticks (~47721s)
    // and idealReleaseUs landed that far in the future, so `now < idealReleaseUs`
    // below was true and this returned that many *microseconds* as the delay
    // before releasing the frame -- a real PltSleepMs() call on the thread that
    // also drains the video socket, i.e. video freezing for hours, not the
    // brief stall this buffer is meant to add. This can't self-heal via the
    // lateByUs resync check further down because that check only runs when
    // idealReleaseUs has already passed, which a schedule tens of thousands of
    // seconds in the future never does.
    int32_t rtpDeltaTicks = (int32_t)(rtpTimestamp - playoutAnchorRtpTimestamp);
    if (rtpDeltaTicks < 0) {
        playoutAnchorRtpTimestamp = rtpTimestamp;
        playoutAnchorLocalUs = receiveTimeUs;
        lastResyncLocalUs = now;
        return 0;
    }

    uint64_t rawScheduleUs = playoutAnchorLocalUs + rtpTicksToUs((uint32_t)rtpDeltaTicks);
    uint64_t idealReleaseUs = rawScheduleUs + advancePlayoutDelayUs();

    if (now < idealReleaseUs) {
        // v9 bug (found live on a real point-to-point cable link after the
        // late-side fix above landed): the anchor's own one-time reference
        // transit delay -- sampled from whichever single frame happened to
        // set it (connection start, or the frame right after a resync) --
        // can itself be anomalously large (e.g. RTSP/session-setup overhead
        // racing the first video packet). Unlike genuine per-frame jitter,
        // that one-off excess gets baked into rawScheduleUs for every
        // subsequent frame with NO existing correction path: the resync/
        // nudge logic further below only ever fires when we're running LATE
        // against the schedule, never when we're persistently AHEAD of it.
        // Confirmed live: jitter/applied stayed small (~1-3ms, no resyncs)
        // yet every single frame still measured a consistent ~13-17ms
        // mandatory sleep -- far more than the measured real transit-time
        // variance (recvToEnqueue ~15-30us) could ever justify, i.e. this
        // was schedule lead baked in at the anchor, not real buffering.
        //
        // If the RAW schedule (before this frame's own small adaptive
        // margin) is already ahead of "now" by more than a small allowance
        // for genuine jitter, gradually pull the anchor's local-time
        // reference earlier so future frames' schedule catches back down to
        // real transit time -- same bounded per-frame step as the late-side
        // fix below, so it can't itself perturb frame-to-frame release gaps.
        if (rawScheduleUs > now + PLAYOUT_RAW_LEAD_ALLOWANCE_US) {
            uint64_t excessLeadUs = rawScheduleUs - now - PLAYOUT_RAW_LEAD_ALLOWANCE_US;
            uint64_t nudge = (excessLeadUs > PLAYOUT_DELAY_MAX_STEP_US) ? PLAYOUT_DELAY_MAX_STEP_US : excessLeadUs;
            playoutAnchorLocalUs -= nudge;
        }
        return idealReleaseUs - now;
    }

    // This frame's slot already passed. A small miss is normal jitter --
    // release immediately but leave the anchor alone, so the schedule (and
    // its buffering margin) survives for the next frame. Only give up and
    // fully re-anchor here if the miss is large enough to mean a real stall
    // or pause, not just jitter -- otherwise every jittery period would keep
    // resetting the schedule right when it's needed most (see the v2 design
    // note above).
    uint64_t lateByUs = now - idealReleaseUs;
    if (lateByUs > PLAYOUT_RESYNC_THRESHOLD_US) {
        Limelog("PlayoutBuffer: resync after %llums of drift\n", (unsigned long long)(lateByUs / 1000));
        playoutAnchorRtpTimestamp = rtpTimestamp;
        playoutAnchorLocalUs = receiveTimeUs;
        lastResyncLocalUs = now;
    }
    else if (lateByUs > 0) {
        // v8 bug (found live on a real point-to-point cable link with
        // near-zero actual network jitter, confirmed absent on WiFi where
        // this buffer was originally tuned): any lateness under the resync
        // threshold above was previously accepted and never corrected -- the
        // anchor only moves on a >150ms drift, so a moderate offset that
        // crept in just once (e.g. during connection startup, before
        // anything had a chance to trigger a real resync) stayed baked into
        // every subsequent frame's schedule for the rest of the session: a
        // fixed, smooth-but-permanent extra delay, not jitter -- reported
        // live as a stable ~200-300ms perceived lag that vanished entirely
        // once this buffer was removed outright. On WiFi this was masked
        // because real jitter crosses the 150ms resync threshold often
        // enough on its own to keep re-correcting the schedule; a clean
        // cable link rarely does, so a bad offset can persist indefinitely.
        //
        // A full resync here (like v2's mistake above) would itself perturb
        // release timing every time normal jitter causes a small miss, so
        // instead nudge just the anchor's local-time reference forward by a
        // small slew-limited step -- same technique and step size already
        // proven safe in production for appliedPlayoutDelayUs's own ramp
        // (PLAYOUT_DELAY_MAX_STEP_US) -- so persistent lateness drains away
        // over roughly a second instead of being locked in until a
        // catastrophic stall happens to trigger a full resync.
        uint64_t nudge = (lateByUs > PLAYOUT_DELAY_MAX_STEP_US) ? PLAYOUT_DELAY_MAX_STEP_US : lateByUs;
        playoutAnchorLocalUs += nudge;
    }
    return 0;
}

#define DR_CLEANUP -1000

#define CONSECUTIVE_DROP_LIMIT 120
static unsigned int consecutiveFrameDrops;

static LINKED_BLOCKING_QUEUE decodeUnitQueue;

typedef struct _BUFFER_DESC {
    char* data;
    unsigned int offset;
    unsigned int length;
} BUFFER_DESC, *PBUFFER_DESC;

typedef struct _LENTRY_INTERNAL {
    LENTRY entry;
    void* allocPtr;
} LENTRY_INTERNAL, *PLENTRY_INTERNAL;

#define H264_NAL_TYPE(x) ((x) & 0x1F)
#define HEVC_NAL_TYPE(x) (((x) & 0x7E) >> 1)

#define H264_NAL_TYPE_SEI 6
#define H264_NAL_TYPE_SPS 7
#define H264_NAL_TYPE_PPS 8
#define H264_NAL_TYPE_AUD 9
#define H264_NAL_TYPE_FILLER 12
#define HEVC_NAL_TYPE_VPS 32
#define HEVC_NAL_TYPE_SPS 33
#define HEVC_NAL_TYPE_PPS 34
#define HEVC_NAL_TYPE_AUD 35
#define HEVC_NAL_TYPE_FILLER 38
#define HEVC_NAL_TYPE_SEI 39

// Init
void initializeVideoDepacketizer(int pktSize) {
    LbqInitializeLinkedBlockingQueue(&decodeUnitQueue, 15);

    nextFrameNumber = 1;
    startFrameNumber = 0;
    waitingForNextSuccessfulFrame = false;
    waitingForIdrFrame = true;
    waitingForRefInvalFrame = false;
    lastPacketInStream = UINT32_MAX;
    decodingFrame = false;
    syntheticPtsBaseUs = 0;
    frameHostProcessingLatency = 0;
    firstPacketReceiveTimeUs = 0;
    firstPacketPresentationTime = 0;
    firstPacketRtpTimestamp = 0;
    lastPacketPayloadLength = 0;
    dropStatePending = false;
    idrFrameProcessed = false;
    strictIdrFrameWait = !isReferenceFrameInvalidationEnabled();

    havePrevFrameTiming = false;
    prevFrameReceiveTimeUs = 0;
    prevFrameRtpTimestamp = 0;
    networkJitterUs = 0;
    appliedPlayoutDelayUs = 0;
    playoutStatusLogCounter = 0;

    havePlayoutAnchor = false;
    playoutAnchorLocalUs = 0;
    playoutAnchorRtpTimestamp = 0;
    lastResyncLocalUs = 0;
}

// Free the NAL chain
static void cleanupFrameState(void) {
    PLENTRY_INTERNAL lastEntry;

    while (nalChainHead != NULL) {
        lastEntry = (PLENTRY_INTERNAL)nalChainHead;
        nalChainHead = lastEntry->entry.next;
        free(lastEntry->allocPtr);
    }

    nalChainTail = NULL;

    nalChainDataLength = 0;
}

// Cleanup frame state and set that we're waiting for an IDR Frame
static void dropFrameState(void) {
    // This may only be called at frame boundaries
    LC_ASSERT(!decodingFrame);

    // We're dropping frame state now
    dropStatePending = false;

    if (strictIdrFrameWait || !idrFrameProcessed || waitingForIdrFrame || (nalChainHead && frameType == FRAME_TYPE_IDR)) {
        // We'll need an IDR frame now if we're in non-RFI mode, if we've never
        // received an IDR frame, if we explicitly need an IDR frame, or if we
        // just dropped a partially processed IDR frame.
        waitingForIdrFrame = true;
    }
    else {
        waitingForRefInvalFrame = true;
    }

    // Count the number of consecutive frames dropped
    consecutiveFrameDrops++;

    // If we reach our limit, immediately request an IDR frame and reset
    if (consecutiveFrameDrops == CONSECUTIVE_DROP_LIMIT) {
        Limelog("Reached consecutive drop limit\n");

        // Restart the count
        consecutiveFrameDrops = 0;

        // Request an IDR frame
        waitingForIdrFrame = true;
        LiRequestIdrFrame();
    }

    cleanupFrameState();
}

// Cleanup the list of decode units
static void freeDecodeUnitList(PLINKED_BLOCKING_QUEUE_ENTRY entry) {
    PLINKED_BLOCKING_QUEUE_ENTRY nextEntry;

    while (entry != NULL) {
        nextEntry = entry->flink;

        // Complete this with a failure status
        LiCompleteVideoFrame(entry->data, DR_CLEANUP);

        entry = nextEntry;
    }
}

void stopVideoDepacketizer(void) {
    LbqSignalQueueShutdown(&decodeUnitQueue);
}

// Cleanup video depacketizer and free malloced memory
void destroyVideoDepacketizer(void) {
    freeDecodeUnitList(LbqDestroyLinkedBlockingQueue(&decodeUnitQueue));
    cleanupFrameState();
}

// NB: This function also ensures an additional byte for the NALU type exists after the start sequence
static bool getAnnexBStartSequence(PBUFFER_DESC current, PBUFFER_DESC startSeq) {
    // We must not get called for other codecs
    LC_ASSERT(NegotiatedVideoFormat & (VIDEO_FORMAT_MASK_H264 | VIDEO_FORMAT_MASK_H265));

    if (current->length <= 3) {
        return false;
    }

    if (current->data[current->offset] == 0 &&
        current->data[current->offset + 1] == 0) {
        if (current->data[current->offset + 2] == 0) {
            if (current->length > 4 && current->data[current->offset + 3] == 1) {
                // Frame start
                if (startSeq != NULL) {
                    startSeq->data = current->data;
                    startSeq->offset = current->offset;
                    startSeq->length = 4;
                }
                return true;
            }
        }
        else if (current->data[current->offset + 2] == 1) {
            // NAL start
            if (startSeq != NULL) {
                startSeq->data = current->data;
                startSeq->offset = current->offset;
                startSeq->length = 3;
            }
            return true;
        }
    }

    return false;
}

void validateDecodeUnitForPlayback(PDECODE_UNIT decodeUnit) {
    // Frames must always have at least one buffer
    LC_ASSERT(decodeUnit->bufferList != NULL);
    LC_ASSERT(decodeUnit->fullLength != 0);

    // Validate the buffers in the frame
    if (decodeUnit->frameType == FRAME_TYPE_IDR) {
        // IDR frames always start with codec configuration data
        if (NegotiatedVideoFormat & VIDEO_FORMAT_MASK_H264) {
            // H.264 IDR frames should have an SPS, PPS, then picture data
            LC_ASSERT_VT(decodeUnit->bufferList->bufferType == BUFFER_TYPE_SPS);
            LC_ASSERT_VT(decodeUnit->bufferList->next != NULL);
            LC_ASSERT_VT(decodeUnit->bufferList->next->bufferType == BUFFER_TYPE_PPS);
            LC_ASSERT_VT(decodeUnit->bufferList->next->next != NULL);
        }
        else if (NegotiatedVideoFormat & VIDEO_FORMAT_MASK_H265) {
            // HEVC IDR frames should have an VPS, SPS, PPS, then picture data
            LC_ASSERT_VT(decodeUnit->bufferList->bufferType == BUFFER_TYPE_VPS);
            LC_ASSERT_VT(decodeUnit->bufferList->next != NULL);
            LC_ASSERT_VT(decodeUnit->bufferList->next->bufferType == BUFFER_TYPE_SPS);
            LC_ASSERT_VT(decodeUnit->bufferList->next->next != NULL);
            LC_ASSERT_VT(decodeUnit->bufferList->next->next->bufferType == BUFFER_TYPE_PPS);
            LC_ASSERT_VT(decodeUnit->bufferList->next->next->next != NULL);
        }
        else if (NegotiatedVideoFormat & VIDEO_FORMAT_MASK_AV1) {
            // We don't parse the AV1 bitstream
            LC_ASSERT_VT(decodeUnit->bufferList->bufferType == BUFFER_TYPE_PICDATA);
        }
        else {
            LC_ASSERT(false);
        }
    }
    else {
        LC_ASSERT(decodeUnit->frameType == FRAME_TYPE_PFRAME);

        // P frames always start with picture data
        LC_ASSERT(decodeUnit->bufferList->bufferType == BUFFER_TYPE_PICDATA);

        // We must not dequeue a P frame before an IDR frame has been successfully processed
        LC_ASSERT(idrFrameProcessed);
    }
}

bool LiWaitForNextVideoFrame(VIDEO_FRAME_HANDLE* frameHandle, PDECODE_UNIT* decodeUnit) {
    PQUEUED_DECODE_UNIT qdu;

    int err = LbqWaitForQueueElement(&decodeUnitQueue, (void**)&qdu);
    if (err != LBQ_SUCCESS) {
        return false;
    }

    // Adaptive playout delay -- see the design comment above playoutDelayForFrame().
    // Skipped entirely whenever another frame is already queued up behind this
    // one: a non-empty queue means the decoder has fallen behind arrivals (e.g.
    // a network stall just cleared and several frames arrived in a burst), and
    // continuing to pace individual frames at cadence in that state starves
    // this same queue from the *enqueue* side -- the RTP receive thread keeps
    // depositing new frames while this thread sleeps -- which can overflow the
    // queue's bound and force a hard IDR-recovery reset.
    //
    // Confirmed live: shipping without this check caused a genuine playback
    // outage (stuck at ~0.6fps, an endless "Video decode unit queue overflow"
    // -> IDR-request -> overflow-again loop, since even the recovery IDR frame
    // got the same treatment). Draining backlog immediately here is always
    // safe: it's exactly what would happen anyway with no buffer at all, so
    // this can never be worse than the pre-buffer baseline -- the buffer only
    // ever adds delay once the decoder has genuinely caught back up.
    if (LbqGetItemCount(&decodeUnitQueue) == 0) {
        uint64_t delayUs = playoutDelayForFrame(qdu->decodeUnit.receiveTimeUs, qdu->decodeUnit.rtpTimestamp);
        if (delayUs > 0) {
            PltSleepMs((int)(delayUs / 1000));
        }
    }

    validateDecodeUnitForPlayback(&qdu->decodeUnit);

    *frameHandle = qdu;
    *decodeUnit = &qdu->decodeUnit;
    return true;
}

bool LiPollNextVideoFrame(VIDEO_FRAME_HANDLE* frameHandle, PDECODE_UNIT* decodeUnit) {
    PQUEUED_DECODE_UNIT qdu;

    int err = LbqPollQueueElement(&decodeUnitQueue, (void**)&qdu);
    if (err != LBQ_SUCCESS) {
        return false;
    }

    validateDecodeUnitForPlayback(&qdu->decodeUnit);

    *frameHandle = qdu;
    *decodeUnit = &qdu->decodeUnit;
    return true;
}

bool LiPeekNextVideoFrame(PDECODE_UNIT* decodeUnit) {
    PQUEUED_DECODE_UNIT qdu;

    int err = LbqPeekQueueElement(&decodeUnitQueue, (void**)&qdu);
    if (err != LBQ_SUCCESS) {
        return false;
    }

    validateDecodeUnitForPlayback(&qdu->decodeUnit);

    *decodeUnit = &qdu->decodeUnit;
    return true;
}

void LiWakeWaitForVideoFrame(void) {
    LbqSignalQueueUserWake(&decodeUnitQueue);
}

// Cleanup a decode unit by freeing the buffer chain and the holder
void LiCompleteVideoFrame(VIDEO_FRAME_HANDLE handle, int drStatus) {
    PQUEUED_DECODE_UNIT qdu = handle;
    PLENTRY_INTERNAL lastEntry;

    if (drStatus == DR_NEED_IDR) {
        Limelog("Requesting IDR frame on behalf of DR\n");
        requestDecoderRefresh();
    }
    else if (drStatus == DR_OK && qdu->decodeUnit.frameType == FRAME_TYPE_IDR) {
        // Remember that the IDR frame was processed. We can now use
        // reference frame invalidation.
        idrFrameProcessed = true;
    }

    while (qdu->decodeUnit.bufferList != NULL) {
        lastEntry = (PLENTRY_INTERNAL)qdu->decodeUnit.bufferList;
        qdu->decodeUnit.bufferList = lastEntry->entry.next;
        free(lastEntry->allocPtr);
    }

    // We will have stack-allocated entries iff we have a direct-submit decoder
    if ((VideoCallbacks.capabilities & CAPABILITY_DIRECT_SUBMIT) == 0) {
        free(qdu);
    }
}

static bool isSeqReferenceFrameStart(PBUFFER_DESC buffer) {
    BUFFER_DESC startSeq;

    if (!getAnnexBStartSequence(buffer, &startSeq)) {
        return false;
    }

    if (NegotiatedVideoFormat & VIDEO_FORMAT_MASK_H264) {
        return H264_NAL_TYPE(startSeq.data[startSeq.offset + startSeq.length]) == 5;
    }
    else if (NegotiatedVideoFormat & VIDEO_FORMAT_MASK_H265) {
        switch (HEVC_NAL_TYPE(startSeq.data[startSeq.offset + startSeq.length])) {
            case 16:
            case 17:
            case 18:
            case 19:
            case 20:
            case 21:
                return true;

            default:
                return false;
        }
    }
    else {
        LC_ASSERT(false);
        return false;
    }
}

static bool isAccessUnitDelimiter(PBUFFER_DESC buffer) {
    BUFFER_DESC startSeq;

    if (!getAnnexBStartSequence(buffer, &startSeq)) {
        return false;
    }

    if (NegotiatedVideoFormat & VIDEO_FORMAT_MASK_H264) {
        return H264_NAL_TYPE(startSeq.data[startSeq.offset + startSeq.length]) == H264_NAL_TYPE_AUD;
    }
    else if (NegotiatedVideoFormat & VIDEO_FORMAT_MASK_H265) {
        return HEVC_NAL_TYPE(startSeq.data[startSeq.offset + startSeq.length]) == HEVC_NAL_TYPE_AUD;
    }
    else {
        LC_ASSERT(false);
        return false;
    }
}

static bool isSeiNal(PBUFFER_DESC buffer) {
    BUFFER_DESC startSeq;

    if (!getAnnexBStartSequence(buffer, &startSeq)) {
        return false;
    }

    if (NegotiatedVideoFormat & VIDEO_FORMAT_MASK_H264) {
        return H264_NAL_TYPE(startSeq.data[startSeq.offset + startSeq.length]) == H264_NAL_TYPE_SEI;
    }
    else if (NegotiatedVideoFormat & VIDEO_FORMAT_MASK_H265) {
        return HEVC_NAL_TYPE(startSeq.data[startSeq.offset + startSeq.length]) == HEVC_NAL_TYPE_SEI;
    }
    else {
        LC_ASSERT(false);
        return false;
    }
}

#ifdef LC_DEBUG
static bool isFillerDataNal(PBUFFER_DESC buffer) {
    BUFFER_DESC startSeq;

    if (!getAnnexBStartSequence(buffer, &startSeq)) {
        return false;
    }

    if (NegotiatedVideoFormat & VIDEO_FORMAT_MASK_H264) {
        return H264_NAL_TYPE(startSeq.data[startSeq.offset + startSeq.length]) == H264_NAL_TYPE_FILLER;
    }
    else if (NegotiatedVideoFormat & VIDEO_FORMAT_MASK_H265) {
        return HEVC_NAL_TYPE(startSeq.data[startSeq.offset + startSeq.length]) == HEVC_NAL_TYPE_FILLER;
    }
    else {
        LC_ASSERT(false);
        return false;
    }
}
#endif

static bool isPictureParameterSetNal(PBUFFER_DESC buffer) {
    BUFFER_DESC startSeq;

    if (!getAnnexBStartSequence(buffer, &startSeq)) {
        return false;
    }

    if (NegotiatedVideoFormat & VIDEO_FORMAT_MASK_H264) {
        return H264_NAL_TYPE(startSeq.data[startSeq.offset + startSeq.length]) == H264_NAL_TYPE_PPS;
    }
    else if (NegotiatedVideoFormat & VIDEO_FORMAT_MASK_H265) {
        return HEVC_NAL_TYPE(startSeq.data[startSeq.offset + startSeq.length]) == HEVC_NAL_TYPE_PPS;
    }
    else {
        LC_ASSERT(false);
        return false;
    }
}

// Advance the buffer descriptor to the start of the next NAL or end of buffer
static void skipToNextNalOrEnd(PBUFFER_DESC buffer) {
    BUFFER_DESC startSeq;

    // If we're starting on a NAL boundary, skip to the next one
    if (getAnnexBStartSequence(buffer, &startSeq)) {
        buffer->offset += startSeq.length;
        buffer->length -= startSeq.length;
    }

    // Loop until we find an Annex B start sequence (3 or 4 byte)
    while (!getAnnexBStartSequence(buffer, NULL)) {
        if (buffer->length == 0) {
            // Reached the end of the buffer
            return;
        }

        buffer->offset++;
        buffer->length--;
    }
}

// Advance the buffer descriptor to the start of the next NAL
static void skipToNextNal(PBUFFER_DESC buffer) {
    skipToNextNalOrEnd(buffer);

    // If we skipped all the data, something has gone horribly wrong
    LC_ASSERT(buffer->length > 0);
}

static bool isIdrFrameStart(PBUFFER_DESC buffer) {
    BUFFER_DESC startSeq;

    if (!getAnnexBStartSequence(buffer, &startSeq)) {
        return false;
    }

    if (NegotiatedVideoFormat & VIDEO_FORMAT_MASK_H264) {
        return H264_NAL_TYPE(startSeq.data[startSeq.offset + startSeq.length]) == H264_NAL_TYPE_SPS;
    }
    else if (NegotiatedVideoFormat & VIDEO_FORMAT_MASK_H265) {
        return HEVC_NAL_TYPE(startSeq.data[startSeq.offset + startSeq.length]) == HEVC_NAL_TYPE_VPS;
    }
    else {
        LC_ASSERT(false);
        return false;
    }
}

// Reassemble the frame with the given frame number
static void reassembleFrame(int frameNumber, bool frameIsLTR) {
    if (nalChainHead != NULL) {
        QUEUED_DECODE_UNIT qduDS;
        PQUEUED_DECODE_UNIT qdu;

        // Use a stack allocation if we won't be queuing this
        if ((VideoCallbacks.capabilities & CAPABILITY_DIRECT_SUBMIT) == 0) {
            qdu = (PQUEUED_DECODE_UNIT)malloc(sizeof(*qdu));
        }
        else {
            qdu = &qduDS;
        }

        if (qdu != NULL) {
            qdu->decodeUnit.bufferList = nalChainHead;
            qdu->decodeUnit.fullLength = nalChainDataLength;
            qdu->decodeUnit.frameType = frameType;
            qdu->decodeUnit.frameNumber = frameNumber;
            qdu->decodeUnit.frameHostProcessingLatency = frameHostProcessingLatency;
            qdu->decodeUnit.receiveTimeUs = firstPacketReceiveTimeUs;
            qdu->decodeUnit.presentationTimeUs = firstPacketPresentationTime;
            qdu->decodeUnit.rtpTimestamp = firstPacketRtpTimestamp;
            qdu->decodeUnit.enqueueTimeUs = PltGetMicroseconds();

            updatePlayoutJitterEstimate(qdu->decodeUnit.receiveTimeUs, qdu->decodeUnit.rtpTimestamp);

            // These might be wrong for a few frames during a transition between SDR and HDR,
            // but the effects shouldn't very noticable since that's an infrequent operation.
            //
            // If we start sending this state in the frame header, we can make it 100% accurate.
            qdu->decodeUnit.hdrActive = LiGetCurrentHostDisplayHdrMode();
            qdu->decodeUnit.colorspace = (uint8_t)(qdu->decodeUnit.hdrActive ? COLORSPACE_REC_2020 : StreamConfig.colorSpace);

            // Invoke the key frame callback if needed
            if (nalChainHead->bufferType != BUFFER_TYPE_PICDATA || qdu->decodeUnit.frameType == FRAME_TYPE_IDR) {
                qdu->decodeUnit.frameType = FRAME_TYPE_IDR;
                notifyKeyFrameReceived();
            }
            else {
                qdu->decodeUnit.frameType = FRAME_TYPE_PFRAME;
            }

            nalChainHead = nalChainTail = NULL;
            nalChainDataLength = 0;

            if ((VideoCallbacks.capabilities & CAPABILITY_DIRECT_SUBMIT) == 0) {
                if (LbqOfferQueueItem(&decodeUnitQueue, qdu, &qdu->entry) == LBQ_BOUND_EXCEEDED) {
                    Limelog("Video decode unit queue overflow\n");

                    // RFI recovery is not supported here
                    waitingForIdrFrame = true;

                    // Clear NAL state for the frame that we failed to enqueue
                    nalChainHead = qdu->decodeUnit.bufferList;
                    nalChainDataLength = qdu->decodeUnit.fullLength;
                    dropFrameState();

                    // Free the DU we were going to queue
                    free(qdu);

                    // Free all frames in the decode unit queue
                    freeDecodeUnitList(LbqFlushQueueItems(&decodeUnitQueue));

                    // Request an IDR frame to recover
                    LiRequestIdrFrame();
                    return;
                }
            }
            else {
                // Adaptive playout delay -- see the design comment above
                // playoutDelayForFrame(). This runs on the RTP receive thread
                // itself here (DIRECT_SUBMIT), so the sleep also delays
                // draining the video socket, not just presentation -- see
                // that same comment for why PLAYOUT_DELAY_MAX_US and
                // PLAYOUT_POST_RESYNC_GRACE_US are sized the way they are.
                uint64_t tPlayout0 = PltGetMicroseconds();
                uint64_t delayUs = playoutDelayForFrame(qdu->decodeUnit.receiveTimeUs, qdu->decodeUnit.rtpTimestamp);
                uint64_t tPlayout1 = PltGetMicroseconds();
                if (delayUs > 0) {
                    PltSleepMs((int)(delayUs / 1000));
                }
                uint64_t tSleepEnd = PltGetMicroseconds();

                // Submit the frame to the decoder
                validateDecodeUnitForPlayback(&qdu->decodeUnit);
                uint64_t tSubmitStart = PltGetMicroseconds();
                LiCompleteVideoFrame(qdu, VideoCallbacks.submitDecodeUnit(&qdu->decodeUnit));
                uint64_t tSubmitEnd = PltGetMicroseconds();

                // Brackets exactly where VideoStream.c's "SLOW video receive-loop
                // iteration" time (addPacket, when it completes a frame) actually
                // goes: computing the schedule, the sleep that computation asked
                // for, or the platform submitDecodeUnit callback itself (decode +
                // readback + render submit on Windows). Without this split, a
                // multi-second addPacket() reading could be any of the three --
                // this pins it down instead of requiring cross-referencing
                // separate log lines from separate threads/files by timestamp.
                if (tSubmitEnd - tPlayout0 > 5000) {
                    Limelog("PlayoutBuffer: SLOW direct-submit frame %u: %lluus total (playoutCalc=%lluus sleep=%lluus[requested=%lluus] submitDecodeUnit=%lluus)\n",
                            frameNumber,
                            (unsigned long long)(tSubmitEnd - tPlayout0),
                            (unsigned long long)(tPlayout1 - tPlayout0),
                            (unsigned long long)(tSleepEnd - tPlayout1),
                            (unsigned long long)delayUs,
                            (unsigned long long)(tSubmitEnd - tSubmitStart));
                }
            }

            // Notify the control connection
            connectionReceivedCompleteFrame(frameNumber, frameIsLTR);

            // Clear frame drops
            consecutiveFrameDrops = 0;

            // Move the start of our (potential) RFI window to the next frame
            startFrameNumber = nextFrameNumber;
        }
    }
}

static int getBufferFlags(char* data, int length) {
    BUFFER_DESC buffer;
    BUFFER_DESC candidate;

    // We only parse H.264 and HEVC bitstreams
    if (!(NegotiatedVideoFormat & (VIDEO_FORMAT_MASK_H264 | VIDEO_FORMAT_MASK_H265))) {
        return BUFFER_TYPE_PICDATA;
    }

    buffer.data = data;
    buffer.length = (unsigned int)length;
    buffer.offset = 0;

    if (!getAnnexBStartSequence(&buffer, &candidate)) {
        return BUFFER_TYPE_PICDATA;
    }

    if (NegotiatedVideoFormat & VIDEO_FORMAT_MASK_H264) {
        switch (H264_NAL_TYPE(candidate.data[candidate.offset + candidate.length])) {
        case H264_NAL_TYPE_SPS:
            return BUFFER_TYPE_SPS;

        case H264_NAL_TYPE_PPS:
            return BUFFER_TYPE_PPS;

        default:
            return BUFFER_TYPE_PICDATA;
        }
    }
    else if (NegotiatedVideoFormat & VIDEO_FORMAT_MASK_H265) {
        switch (HEVC_NAL_TYPE(candidate.data[candidate.offset + candidate.length])) {
            case HEVC_NAL_TYPE_SPS:
                return BUFFER_TYPE_SPS;

            case HEVC_NAL_TYPE_PPS:
                return BUFFER_TYPE_PPS;

            case HEVC_NAL_TYPE_VPS:
                return BUFFER_TYPE_VPS;

            default:
                return BUFFER_TYPE_PICDATA;
        }
    }
    else {
        LC_ASSERT(false);
        return BUFFER_TYPE_PICDATA;
    }
}

// As an optimization, we can cast the existing packet buffer to a PLENTRY and avoid
// a malloc() and a memcpy() of the packet data.
static void queueFragment(PLENTRY_INTERNAL* existingEntry, char* data, int offset, int length) {
    PLENTRY_INTERNAL entry;

    if (existingEntry == NULL || *existingEntry == NULL) {
        entry = (PLENTRY_INTERNAL)malloc(sizeof(*entry) + length);
    }
    else {
        entry = *existingEntry;
    }

    if (entry != NULL) {
        entry->entry.next = NULL;
        entry->entry.length = length;

        // If we had to allocate a new entry, we must copy the data. If not,
        // the data already resides within the LENTRY allocation.
        if (existingEntry == NULL || *existingEntry == NULL) {
            entry->allocPtr = entry;

            entry->entry.data = (char*)(entry + 1);
            memcpy(entry->entry.data, &data[offset], entry->entry.length);
        }
        else {
            entry->entry.data = &data[offset];

            // The caller should have already set this up for us
            LC_ASSERT(entry->allocPtr != NULL);

            // We now own the packet buffer and will manage freeing it
            *existingEntry = NULL;
        }

        entry->entry.bufferType = getBufferFlags(entry->entry.data, entry->entry.length);

        nalChainDataLength += entry->entry.length;

        if (nalChainTail == NULL) {
            LC_ASSERT(nalChainHead == NULL);
            nalChainHead = nalChainTail = (PLENTRY)entry;
        }
        else {
            LC_ASSERT(nalChainHead != NULL);
            nalChainTail->next = (PLENTRY)entry;
            nalChainTail = nalChainTail->next;
        }
    }
}

// Process an RTP Payload using the slow path that handles multiple NALUs per packet
static void processAvcHevcRtpPayloadSlow(PBUFFER_DESC currentPos, PLENTRY_INTERNAL* existingEntry) {
    // We should not have any NALUs when processing the first packet in an IDR frame
    LC_ASSERT(nalChainHead == NULL);
    LC_ASSERT(nalChainTail == NULL);

    while (currentPos->length != 0) {
        // Skip through any padding bytes
        if (!getAnnexBStartSequence(currentPos, NULL)) {
            skipToNextNal(currentPos);
        }

        // Skip any prepended AUD or SEI NALUs. We may have padding between
        // these on IDR frames, so the check in processRtpPayload() is not
        // completely sufficient to handle that case.
        while (isAccessUnitDelimiter(currentPos) || isSeiNal(currentPos)) {
            skipToNextNal(currentPos);
        }

        int start = currentPos->offset;
        bool containsPicData = false;

#ifdef FORCE_3_BYTE_START_SEQUENCES
        start++;
#endif

        if (isSeqReferenceFrameStart(currentPos)) {
            // No longer waiting for an IDR frame
            waitingForIdrFrame = false;
            waitingForRefInvalFrame = false;

            // Cancel any pending IDR frame request
            waitingForNextSuccessfulFrame = false;

            // Use the cached LENTRY for this NALU since it will be
            // the bulk of the data in this packet.
            containsPicData = true;

            // This is an IDR frame
            frameType = FRAME_TYPE_IDR;
        }

        // Move to the next NALU
        skipToNextNalOrEnd(currentPos);

        // If this is the picture data, we expect it to extend to the end of the packet
        if (containsPicData) {
            while (currentPos->length != 0) {
                // Any NALUs we encounter on the way to the end of the packet must be
                // reference frame slices or filler data.
                LC_ASSERT_VT(isSeqReferenceFrameStart(currentPos) || isFillerDataNal(currentPos));
                skipToNextNalOrEnd(currentPos);
            }
        }

        // To minimize copies, we'll allocate for SPS, PPS, and VPS to allow
        // us to reuse the packet buffer for the picture data in the I-frame.
        queueFragment(containsPicData ? existingEntry : NULL,
                      currentPos->data, start, currentPos->offset - start);
    }
}

// Dumps the decode unit queue and ensures the next frame submitted to the decoder will be
// an IDR frame
void requestDecoderRefresh(void) {
    // Wait for the next IDR frame
    waitingForIdrFrame = true;

    // Flush the decode unit queue
    freeDecodeUnitList(LbqFlushQueueItems(&decodeUnitQueue));

    // Request the receive thread drop its state
    // on the next call. We can't do it here because
    // it may be trying to queue DUs and we'll nuke
    // the state out from under it.
    dropStatePending = true;

    // Request the IDR frame
    LiRequestIdrFrame();
}

// Return 1 if packet is the first one in the frame
static bool isFirstPacket(uint8_t flags, uint8_t fecBlockNumber) {
    // Clear the picture data flag
    flags &= ~FLAG_CONTAINS_PIC_DATA;

    // Check if it's just the start or both start and end of a frame
    return (flags == (FLAG_SOF | FLAG_EOF) || flags == FLAG_SOF) && fecBlockNumber == 0;
}

// Process an RTP Payload
// The caller will free *existingEntry unless we NULL it
static void processRtpPayload(PNV_VIDEO_PACKET videoPacket, int length,
                       uint64_t receiveTimeUs, uint64_t presentationTimeUs, uint32_t rtpTimestamp,
                       PLENTRY_INTERNAL* existingEntry) {
    BUFFER_DESC currentPos;
    uint32_t frameIndex;
    uint8_t flags;
    uint8_t extraFlags;
    bool firstPacket, lastPacket;
    uint32_t streamPacketIndex;
    uint8_t fecCurrentBlockNumber;
    uint8_t fecLastBlockNumber;

    // Mask the top 8 bits from the SPI
    videoPacket->streamPacketIndex >>= 8;
    videoPacket->streamPacketIndex &= 0xFFFFFF;

    currentPos.data = (char*)(videoPacket + 1);
    currentPos.offset = 0;
    currentPos.length = length - sizeof(*videoPacket);

    fecCurrentBlockNumber = (videoPacket->multiFecBlocks >> 4) & 0x3;
    fecLastBlockNumber = (videoPacket->multiFecBlocks >> 6) & 0x3;
    frameIndex = videoPacket->frameIndex;
    flags = videoPacket->flags;
    extraFlags = videoPacket->extraFlags;
    firstPacket = isFirstPacket(flags, fecCurrentBlockNumber);
    lastPacket = (flags & FLAG_EOF) && fecCurrentBlockNumber == fecLastBlockNumber;

    LC_ASSERT_VT((flags & ~(FLAG_SOF | FLAG_EOF | FLAG_CONTAINS_PIC_DATA)) == 0);

    streamPacketIndex = videoPacket->streamPacketIndex;

    // Drop packets from a previously corrupt frame
    if (isBefore32(frameIndex, nextFrameNumber)) {
        return;
    }

    // The FEC queue can sometimes recover corrupt frames (see comments in RtpFecQueue).
    // It almost always detects them before they get to us, but in case it doesn't
    // the streamPacketIndex not matching correctly should find nearly all of the rest.
    if (isBefore24(streamPacketIndex, U24(lastPacketInStream + 1)) ||
            (!(flags & FLAG_SOF) && streamPacketIndex != U24(lastPacketInStream + 1))) {
        Limelog("Depacketizer detected corrupt frame: %d", frameIndex);
        decodingFrame = false;
        nextFrameNumber = frameIndex + 1;
        dropFrameState();
        if (waitingForIdrFrame) {
            LiRequestIdrFrame();
        }
        else {
            connectionDetectedFrameLoss(startFrameNumber, frameIndex);
        }
        return;
    }

    // Verify that we didn't receive an incomplete frame
    LC_ASSERT(firstPacket ^ decodingFrame);

    // Check sequencing of this frame to ensure we didn't
    // miss one in between
    if (firstPacket) {
        // Make sure this is the next consecutive frame
        if (isBefore32(nextFrameNumber, frameIndex)) {
            if (nextFrameNumber + 1 == frameIndex) {
                Limelog("Network dropped 1 frame (frame %d)\n", frameIndex - 1);
            }
            else {
                Limelog("Network dropped %d frames (frames %d to %d)\n",
                        frameIndex - nextFrameNumber,
                        nextFrameNumber,
                        frameIndex - 1);
            }

            nextFrameNumber = frameIndex;

            // Wait until next complete frame
            waitingForNextSuccessfulFrame = true;
            dropFrameState();
        }
        else {
            LC_ASSERT(nextFrameNumber == frameIndex);
        }

        // We're now decoding a frame
        decodingFrame = true;
        frameType = FRAME_TYPE_PFRAME;
        firstPacketReceiveTimeUs = receiveTimeUs;

        // Some versions of Sunshine don't send a valid PTS, so we will
        // synthesize one using the receive time as the time base.
        if (!syntheticPtsBaseUs) {
            syntheticPtsBaseUs = receiveTimeUs;
        }

        if (!presentationTimeUs && frameIndex > 0) {
            firstPacketPresentationTime = receiveTimeUs - syntheticPtsBaseUs;
        }
        else {
            firstPacketPresentationTime = presentationTimeUs;
        }

        firstPacketRtpTimestamp = rtpTimestamp;
    }

    lastPacketInStream = streamPacketIndex;

    // If this is the first packet, skip the frame header (if one exists)
    uint32_t frameHeaderSize;
    LC_ASSERT_VT(currentPos.length > 0);
    if (firstPacket && currentPos.length > 0) {
        // Parse the frame type from the header
        LC_ASSERT_VT(currentPos.length >= 4);
        if (APP_VERSION_AT_LEAST(7, 1, 350) && currentPos.length >= 4) {
            switch (currentPos.data[currentPos.offset + 3]) {
            case 1: // Normal P-frame
                break;
            case 2: // IDR frame
                // For other codecs, we trust the frame header rather than parsing the bitstream
                // to determine if a given frame is an IDR frame.
                if (!(NegotiatedVideoFormat & (VIDEO_FORMAT_MASK_H264 | VIDEO_FORMAT_MASK_H265))) {
                    waitingForIdrFrame = false;
                    waitingForNextSuccessfulFrame = false;
                    frameType = FRAME_TYPE_IDR;
                }
                // Fall-through
            case 4: // Intra-refresh
            case 5: // P-frame with reference frames invalidated
                if (waitingForRefInvalFrame) {
                    Limelog("Next post-invalidation frame is: %d (%s-frame)\n",
                            frameIndex,
                            currentPos.data[currentPos.offset + 3] == 5 ? "P" : "I");
                    waitingForRefInvalFrame = false;
                    waitingForNextSuccessfulFrame = false;
                }
                break;
            case 104: // Sunshine hardcoded header
                break;
            default:
                Limelog("Unrecognized frame type: %d", currentPos.data[currentPos.offset + 3]);
                LC_ASSERT_VT(false);
                break;
            }
        }
        else {
            // Hope for the best with older servers
            if (waitingForRefInvalFrame) {
                connectionDetectedFrameLoss(startFrameNumber, frameIndex - 1);
                waitingForRefInvalFrame = false;
                waitingForNextSuccessfulFrame = false;
            }
        }

        // Sunshine can provide host processing latency of the frame
        LC_ASSERT_VT(currentPos.length >= 3);
        if (IS_SUNSHINE() && currentPos.length >= 3) {
            BYTE_BUFFER bb;
            BbInitializeWrappedBuffer(&bb, currentPos.data, currentPos.offset + 1, 2, BYTE_ORDER_LITTLE);
            BbGet16(&bb, &frameHostProcessingLatency);
        }

        // Codecs like H.264 and HEVC handle the FEC trailing zero padding just fine, but other
        // codecs need the exact length encoded separately.
        LC_ASSERT_VT(currentPos.length >= 6);
        if (!(NegotiatedVideoFormat & (VIDEO_FORMAT_MASK_H264 | VIDEO_FORMAT_MASK_H265)) && currentPos.length >= 6) {
            BYTE_BUFFER bb;
            BbInitializeWrappedBuffer(&bb, currentPos.data, currentPos.offset + 4, 2, BYTE_ORDER_LITTLE);
            BbGet16(&bb, &lastPacketPayloadLength);
        }

        if (APP_VERSION_AT_LEAST(7, 1, 450)) {
            // >= 7.1.450 uses 2 different header lengths based on the first byte:
            // 0x01 indicates an 8 byte header
            // 0x81 indicates a 44 byte header
            if (currentPos.data[0] == 0x01) {
                frameHeaderSize = 8;
            }
            else {
                LC_ASSERT_VT(currentPos.data[0] == (char)0x81);
                frameHeaderSize = 44;
            }
        }
        else if (APP_VERSION_AT_LEAST(7, 1, 446)) {
            // [7.1.446, 7.1.450) uses 2 different header lengths based on the first byte:
            // 0x01 indicates an 8 byte header
            // 0x81 indicates a 41 byte header
            if (currentPos.data[0] == 0x01) {
                frameHeaderSize = 8;
            }
            else {
                LC_ASSERT_VT(currentPos.data[0] == (char)0x81);
                frameHeaderSize = 41;
            }
        }
        else if (APP_VERSION_AT_LEAST(7, 1, 415)) {
            // [7.1.415, 7.1.446) uses 2 different header lengths based on the first byte:
            // 0x01 indicates an 8 byte header
            // 0x81 indicates a 24 byte header
            if (currentPos.data[0] == 0x01) {
                frameHeaderSize = 8;
            }
            else {
                LC_ASSERT_VT(currentPos.data[0] == (char)0x81);
                frameHeaderSize = 24;
            }
        }
        else if (APP_VERSION_AT_LEAST(7, 1, 350)) {
            // [7.1.350, 7.1.415) should use the 8 byte header again
            frameHeaderSize = 8;
        }
        else if (APP_VERSION_AT_LEAST(7, 1, 320)) {
            // [7.1.320, 7.1.350) should use the 12 byte frame header
            frameHeaderSize = 12;
        }
        else if (APP_VERSION_AT_LEAST(5, 0, 0)) {
            // [5.x, 7.1.320) should use the 8 byte header
            frameHeaderSize = 8;
        }
        else {
            // Other versions don't have a frame header at all
            frameHeaderSize = 0;
        }

        LC_ASSERT_VT(currentPos.length >= frameHeaderSize);
        if (currentPos.length >= frameHeaderSize) {
            // Skip past the frame header
            currentPos.offset += frameHeaderSize;
            currentPos.length -= frameHeaderSize;
        }

        // We only parse H.264 and HEVC at the NALU level
        if (NegotiatedVideoFormat & (VIDEO_FORMAT_MASK_H264 | VIDEO_FORMAT_MASK_H265)) {
            // The Annex B NALU start prefix must be next
            if (!getAnnexBStartSequence(&currentPos, NULL)) {
                // If we aren't starting on a start prefix, something went wrong.
                LC_ASSERT_VT(false);

                // For release builds, we will try to recover by searching for one.
                // This mimics the way most decoders handle this situation.
                skipToNextNal(&currentPos);
            }

            // If an AUD NAL is prepended to this frame data, remove it.
            // Other parts of this code are not prepared to deal with a
            // NAL of that type, so stripping it is the easiest option.
            if (isAccessUnitDelimiter(&currentPos)) {
                skipToNextNal(&currentPos);
            }

            // There may be one or more SEI NAL units prepended to the
            // frame data *after* the (optional) AUD.
            while (isSeiNal(&currentPos)) {
                skipToNextNal(&currentPos);
            }
        }
    }
    else {
        // There is no frame header on later packets
        frameHeaderSize = 0;
    }

    if (NegotiatedVideoFormat & (VIDEO_FORMAT_MASK_H264 | VIDEO_FORMAT_MASK_H265)) {
        if (firstPacket && isIdrFrameStart(&currentPos)) {
            // SPS and PPS prefix is padded between NALs, so we must decode it with the slow path
            processAvcHevcRtpPayloadSlow(&currentPos, existingEntry);
        }
        else {
            // Intel's H.264 Media Foundation encoder prepends a PPS to each P-frame.
            // Skip it to avoid confusing clients.
            if (firstPacket && isPictureParameterSetNal(&currentPos)) {
                skipToNextNal(&currentPos);
            }

#ifdef FORCE_3_BYTE_START_SEQUENCES
            if (firstPacket) {
                currentPos.offset++;
                currentPos.length--;
            }
#endif

            queueFragment(existingEntry, currentPos.data, currentPos.offset, currentPos.length);
        }
    }
    else {
        // We fixup the length of the last packet for other codecs since they may not be tolerant
        // of trailing zero padding like H.264/HEVC Annex B bitstream parsers are.
        if (lastPacket) {
            // The payload length includes the frame header, so it cannot be smaller than that
            LC_ASSERT_VT(lastPacketPayloadLength > frameHeaderSize);

            // The payload length cannot be smaller than the actual received payload
            // NB: currentPos.length is already adjusted to exclude the frameHeaderSize from above
            LC_ASSERT_VT(lastPacketPayloadLength - frameHeaderSize <= currentPos.length);

            // If the payload length is valid, truncate the packet. If not, discard this frame.
            if (lastPacketPayloadLength > frameHeaderSize && lastPacketPayloadLength - frameHeaderSize <= currentPos.length) {
                currentPos.length = lastPacketPayloadLength - frameHeaderSize;
            }
            else {
                if (lastPacketPayloadLength <= frameHeaderSize) {
                    Limelog("Invalid last payload length for header on frame %u: %u <= %u",
                            frameIndex, lastPacketPayloadLength, frameHeaderSize);
                }
                else {
                    Limelog("Invalid last payload length for packet size on frame %u: %u > %u",
                            frameIndex, lastPacketPayloadLength - frameHeaderSize, currentPos.length);
                }

                // Skip to the next frame and tell the host we lost this one
                decodingFrame = false;
                nextFrameNumber = frameIndex + 1;
                dropFrameState();
                if (waitingForIdrFrame) {
                    LiRequestIdrFrame();
                }
                else {
                    connectionDetectedFrameLoss(startFrameNumber, frameIndex);
                }

                return;
            }
        }

        // Other codecs are just passed through as is.
        queueFragment(existingEntry, currentPos.data, currentPos.offset, currentPos.length);
    }

    if (lastPacket) {
        // Move on to the next frame
        decodingFrame = false;
        nextFrameNumber = frameIndex + 1;

        // If we can't submit this frame due to a discontinuity in the bitstream,
        // inform the host (if needed) and drop the data.
        if (waitingForIdrFrame || waitingForRefInvalFrame) {
            // IDR wait takes priority over RFI wait (and an IDR frame will satisfy both)
            if (waitingForIdrFrame) {
                Limelog("Waiting for IDR frame\n");

                // We wait for the first fully received frame after a loss to approximate
                // detection of the recovery of the network. Requesting an IDR frame while
                // the network is unstable will just contribute to congestion collapse.
                if (waitingForNextSuccessfulFrame) {
                    LiRequestIdrFrame();
                }
            }
            else {
                // If we need an RFI frame first, then drop this frame
                // and update the reference frame invalidation window.
                Limelog("Waiting for RFI frame\n");
                connectionDetectedFrameLoss(startFrameNumber, frameIndex);
            }

            waitingForNextSuccessfulFrame = false;
            dropFrameState();
            return;
        }

        LC_ASSERT(!waitingForNextSuccessfulFrame);

        // Carry out any pending state drops. We can't just do this
        // arbitrarily in the middle of processing a frame because
        // may cause the depacketizer state to become corrupted. For
        // example, if we drop state after the first packet, the
        // depacketizer will next try to process a non-SOF packet,
        // and cause it to assert.
        if (dropStatePending) {
            if (nalChainHead && frameType == FRAME_TYPE_IDR) {
                // Don't drop the frame state if this frame is an IDR frame itself,
                // otherwise we'll lose this IDR frame without another in flight
                // and have to wait until we hit our consecutive drop limit to
                // request a new one (potentially several seconds).
                dropStatePending = false;
            }
            else {
                dropFrameState();
                return;
            }
        }

        reassembleFrame(frameIndex, extraFlags & NV_VIDEO_PACKET_EXTRA_FLAG_LTR_FRAME);
    }
}

// Called by the video RTP FEC queue to notify us of a lost frame
// if it determines the frame to be unrecoverable. This lets us
// avoid having to wait until the next received frame to determine
// that we lost a frame and submit an RFI request.
void notifyFrameLost(unsigned int frameNumber, bool speculative) {
    // We may not invalidate frames that we've already received
    LC_ASSERT(frameNumber >= startFrameNumber);

    // Drop state and determine if we need an IDR frame or if RFI is okay
    dropFrameState();

    // If dropFrameState() determined that RFI was usable, issue it now
    if (!waitingForIdrFrame) {
        LC_ASSERT(waitingForRefInvalFrame);

        if (speculative) {
            Limelog("Sending speculative RFI request for predicted loss of frame %d\n", frameNumber);
        }
        else {
            Limelog("Sending RFI request for unrecoverable frame %d\n", frameNumber);
        }

        // Advance the frame number since we won't be expecting this one anymore
        nextFrameNumber = frameNumber + 1;

        // Notify the host that we lost this one
        connectionDetectedFrameLoss(startFrameNumber, frameNumber);
    }
}

// Add an RTP Packet to the queue
void queueRtpPacket(PRTPV_QUEUE_ENTRY queueEntryPtr) {
    int dataOffset;
    RTPV_QUEUE_ENTRY queueEntry = *queueEntryPtr;

    LC_ASSERT(!queueEntry.isParity);
    LC_ASSERT(queueEntry.receiveTimeUs != 0);

    dataOffset = sizeof(*queueEntry.packet);
    if (queueEntry.packet->header & FLAG_EXTENSION) {
        dataOffset += 4; // 2 additional fields
    }

    // The packet length was validated by the RtpVideoQueue
    LC_ASSERT(queueEntry.length >= dataOffset + (int)sizeof(NV_VIDEO_PACKET));

    // Reuse the memory reserved for the RTPFEC_QUEUE_ENTRY to store the LENTRY_INTERNAL
    // now that we're in the depacketizer. We saved a copy of the real FEC queue entry
    // on the stack here so we can safely modify this memory in place.
    LC_ASSERT(sizeof(LENTRY_INTERNAL) <= sizeof(RTPV_QUEUE_ENTRY));
    PLENTRY_INTERNAL existingEntry = (PLENTRY_INTERNAL)queueEntryPtr;
    existingEntry->allocPtr = queueEntry.packet;

    processRtpPayload((PNV_VIDEO_PACKET)(((char*)queueEntry.packet) + dataOffset),
                      queueEntry.length - dataOffset,
                      queueEntry.receiveTimeUs,
                      queueEntry.presentationTimeUs,
                      queueEntry.rtpTimestamp,
                      &existingEntry);

    if (existingEntry != NULL) {
        // processRtpPayload didn't want this packet, so just free it
        free(existingEntry->allocPtr);
    }
}

int LiGetPendingVideoFrames(void) {
    return LbqGetItemCount(&decodeUnitQueue);
}
