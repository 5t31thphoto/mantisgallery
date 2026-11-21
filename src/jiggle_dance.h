/**
 * Dance mode — finds a steady rhythm in the IMU and fires beat events in sync with it.
 * No Arduino dependencies (unit-tested on a PC).
 *
 *  1. Gravity is removed upstream; we get the "dynamic" acceleration vector each loop.
 *  2. Resampled to 50 Hz into a 4 s ring buffer, projected onto the main direction of motion
 *     (so bobbing, swaying or walking all work, whichever way the Core2 is held).
 *  3. Every 100 ms an autocorrelation over 60–200 BPM gives the tempo and how rhythmic it is.
 *  4. Peaks in the motion (the "hits") steer an internal metronome — a phase-locked loop — so beats
 *     land on the hits rather than trailing behind them.
 *  5. Steady rhythm → lock, dance strength fades in. Hits stop → unlock, fade out within ~1 s.
 *
 * AudioOnset turns microphone audio into the same kind of signal: one "hit strength" value per 20 ms
 * chunk (rises in bass and overall loudness — kicks, snares, claps), fed to a second Rhythm.
 */
#pragma once
#include <math.h>
#include <stdint.h>
#include <string.h>

#ifndef JIG_MUSIC_RUNS
#define JIG_MUSIC_RUNS 6
#endif

namespace jig {

class Rhythm {
public:
    static constexpr int   FS = 50;                 // analysis sample rate (Hz)
    static constexpr int   N = 200;                 // 4 s window
    static constexpr int   LAG_MIN = 15;            // 0.30 s → 200 BPM
    static constexpr int   LAG_MAX = 50;            // 1.00 s →  60 BPM
    static constexpr float MIN_RMS = 0.018f;        // g: quieter than this isn't dancing
    static constexpr float HIT_FLOOR = 0.035f;      // g: a hit must be at least this strong
    static constexpr float LOCK_CONF = 0.5f;        // autocorrelation needed to lock
    static constexpr float KEEP_CONF = 0.25f;       // ...and to stay locked
    static constexpr float LEAD = 0.04f;            // s: default lead (spring swing time); see set_lead
    static constexpr int   HITS = 6;                // recent hit times kept for the evenness check

    // dt: seconds since last call. d*: dynamic acceleration (g, gravity removed).
    // up*: unit vector pointing "up" (from the gravity baseline). Returns true if a beat fired.
    bool update(float dt, float dx, float dy, float dz, float upx, float upy, float upz) {
        bool beat = false;
        t_ += dt;
        acc_[0] += dx * dt; acc_[1] += dy * dt; acc_[2] += dz * dt; acc_w_ += dt;
        up_[0] = upx; up_[1] = upy; up_[2] = upz;
        while (t_ - last_sample_t_ >= 1.0f / FS) {
            last_sample_t_ += 1.0f / FS;
            float w = acc_w_ > 1e-6f ? 1.0f / acc_w_ : 0.0f;
            push_sample(acc_[0] * w, acc_[1] * w, acc_[2] * w);
            acc_[0] = acc_[1] = acc_[2] = 0; acc_w_ = 0;
        }

        // Metronome
        if (locked_ || amp_ > 0.0f) {
            phase_ += dt / period_;
            while (phase_ >= 1.0f) { phase_ -= 1.0f; fired_ = false; }
            if (!fired_ && phase_ >= 1.0f - lead_ / period_) {
                fired_ = true;
                beat = true;
                sign_ = -sign_;
                ++beats_;
            }
        }
        if (locked_) {
            float rise = dt / (1.5f * period_);           // full strength after ~1.5 beats
            amp_ = amp_ + rise > 1.0f ? 1.0f : amp_ + rise;
        } else {
            amp_ = amp_ - dt / coast_ > 0.0f ? amp_ - dt / coast_ : 0.0f;   // fade over the coast time
            if (amp_ <= 0.0f) phase_ = 0.0f;
        }
        return beat && amp_ > 0.0f;
    }

    // Fire beats this many seconds before the predicted hit (spring swing time + input latency).
    void  set_lead(float s) { lead_ = s; }
    // After the rhythm stops, the metronome keeps running while strength fades over this time (s).
    void  set_coast(float s) { coast_ = s; }
    // Music has off-beats (hi-hats, bass on the "and"): count hits on half-beats as on the grid.
    // Audio beats land between 20 ms chunks, which smears the autocorrelation, so music locks at a
    // lower score — the hit-spacing check still has to pass.
    void  set_music(bool on) { music_ = on; lock_conf_ = on ? 0.35f : LOCK_CONF; lock_runs_ = on ? JIG_MUSIC_RUNS : 4; }
    bool  locked() const { return locked_; }
    float amp() const { return amp_; }                     // 0..1 dance strength
    float bpm() const { return 60.0f / period_; }
    int   beat_sign() const { return sign_; }              // alternates ±1 every beat
    float confidence() const { return conf_; }
    float phase() const { return phase_; }
    uint32_t beats() const { return beats_; }

private:
    float buf_[N][3] = {};      // dynamic accel, resampled
    float sig_[N] = {};         // projected onto motion axis
    int   head_ = 0;            // next write index
    int   filled_ = 0;
    float t_ = 0, last_sample_t_ = 0;
    float acc_[3] = {0, 0, 0}, acc_w_ = 0;
    float up_[3] = {0, 0, 1};
    float axis_[3] = {0, 0, 1};
    int   since_analysis_ = 0;

    float thr_ = HIT_FLOOR;
    float last_hit_t_ = -10.0f;
    float conf_ = 0, rms_ = 0, period_ac_ = 0.5f;
    int   good_runs_ = 0;
    float bad_time_ = 0;

    float hit_t_[HITS] = {};
    int   nhits_ = 0;

    bool  locked_ = false, fired_ = true;
    float lead_ = LEAD;
    float coast_ = 1.0f;
    bool  music_ = false;
    float hit_max_ = 0;
    float lock_conf_ = LOCK_CONF;
    int   lock_runs_ = 4;                           // consecutive good analyses (100 ms apart) to lock
    float period_ = 0.5f, phase_ = 0.0f, amp_ = 0.0f;
    int   sign_ = 1;
    uint32_t beats_ = 0;

    const float& sig_at(int ago) const { return sig_[(head_ - 1 - ago + 2 * N) % N]; }

    void push_sample(float x, float y, float z) {
        buf_[head_][0] = x; buf_[head_][1] = y; buf_[head_][2] = z;
        sig_[head_] = x * axis_[0] + y * axis_[1] + z * axis_[2];
        head_ = (head_ + 1) % N;
        if (filled_ < N) ++filled_;
        const float now = last_sample_t_;

        // Hit = local maximum above threshold (detected one sample late).
        if (filled_ >= 3) {
            float a = sig_at(2), b = sig_at(1), c = sig_at(0);
            float min_gap = locked_ ? 0.5f * period_ : 0.25f;
            float when = now - 1.5f / FS;      // peak was 1 sample ago; averaging adds half a sample
            // Music: only the strongest hits steer (kicks), so snares and talking don't drag the tempo.
            hit_max_ *= expf(-1.0f / (FS * 4.0f));
            if (b > hit_max_) hit_max_ = b;
            bool strong = !music_ || b >= 0.5f * hit_max_;
            if (b > a && b >= c && b > thr_ && strong && when - last_hit_t_ > min_gap) on_hit(when);
        }
        if (++since_analysis_ >= FS / 10) { since_analysis_ = 0; analyse(); }

        // Hits stopped → the rhythm is over.
        if (locked_ && now - last_hit_t_ > 2.2f * period_) unlock();
    }

    void on_hit(float when) {
        last_hit_t_ = when;
        for (int i = HITS - 1; i > 0; --i) hit_t_[i] = hit_t_[i - 1];
        hit_t_[0] = when;
        if (nhits_ < HITS) ++nhits_;
        if (!locked_) return;
        // Phase error in cycles at the moment of the hit, wrapped to [-0.5, 0.5).
        float ph = phase_ - (t_ - when) / period_;
        ph -= floorf(ph + 0.5f);
        phase_ -= 0.35f * ph;                 // pull the metronome toward the hit
        period_ *= 1.0f - 0.05f * ph;         // and nudge the tempo
        if (ph < -0.1f && phase_ >= 1.0f - lead_ / period_) fired_ = true;   // late hit: don't double-fire
        if (phase_ < 0) phase_ = 0.0f;        // never wrap backwards: that would fire the beat twice
    }

    // Are the recent hits evenly spaced at period p (allowing a missed hit = 2p)?
    bool hits_even(float p) const {
        if (nhits_ < 4) return false;
        int good = 0, n = nhits_ - 1;
        // Music: half-beat gaps count too (off-beats), but tighter, and most gaps must be whole beats
        // — kicks give that, syllables of speech almost never do.
        const float unit = music_ ? 0.5f * p : p;
        const float tol = music_ ? 0.12f * p : 0.2f * p;
        int whole = 0;
        for (int i = 0; i < n; ++i) {
            float iv = hit_t_[i] - hit_t_[i + 1];
            float k = roundf(iv / unit);
            if (k >= 1 && k * unit <= 2 * p && fabsf(iv - k * unit) < tol) {
                ++good;
                if (!music_ || ((int)k % 2) == 0) ++whole;
            }
        }
        return good >= n - 1 && 2 * whole >= n;
    }

    void unlock() {
        locked_ = false;
        good_runs_ = 0;
    }

    void analyse() {
        if (filled_ < N / 2) return;
        const int n = filled_;
        int start = (head_ - n + N) % N;

        // Main motion axis: power iteration on the covariance of the window.
        float m[3] = {0, 0, 0};
        for (int i = 0; i < n; ++i) for (int k = 0; k < 3; ++k) m[k] += buf_[(start + i) % N][k];
        for (int k = 0; k < 3; ++k) m[k] /= n;
        float C[3][3] = {};
        for (int i = 0; i < n; ++i) {
            const float* v = buf_[(start + i) % N];
            float d[3] = {v[0] - m[0], v[1] - m[1], v[2] - m[2]};
            for (int r = 0; r < 3; ++r) for (int c = 0; c < 3; ++c) C[r][c] += d[r] * d[c];
        }
        float ax[3] = {axis_[0], axis_[1], axis_[2]};
        {   // If the old axis is blind to the motion (e.g. it's all sideways), restart from the
            // direction with the most variance so the iteration can't get stuck at zero.
            float t0 = C[0][0] * ax[0] + C[0][1] * ax[1] + C[0][2] * ax[2];
            float t1 = C[1][0] * ax[0] + C[1][1] * ax[1] + C[1][2] * ax[2];
            float t2 = C[2][0] * ax[0] + C[2][1] * ax[1] + C[2][2] * ax[2];
            float tr = C[0][0] + C[1][1] + C[2][2];
            if (t0 * t0 + t1 * t1 + t2 * t2 < 1e-4f * tr * tr) {
                int k = C[1][1] > C[0][0] ? 1 : 0;
                if (C[2][2] > C[k][k]) k = 2;
                ax[0] = ax[1] = ax[2] = 0;
                ax[k] = 1;
            }
        }
        for (int it = 0; it < 8; ++it) {
            float nx[3];
            for (int r = 0; r < 3; ++r) nx[r] = C[r][0] * ax[0] + C[r][1] * ax[1] + C[r][2] * ax[2];
            float len = sqrtf(nx[0] * nx[0] + nx[1] * nx[1] + nx[2] * nx[2]);
            if (len < 1e-9f) break;
            for (int r = 0; r < 3; ++r) ax[r] = nx[r] / len;
        }
        // Project the window on the new axis.
        float x[N];
        for (int i = 0; i < n; ++i) {
            const float* v = buf_[(start + i) % N];
            x[i] = v[0] * ax[0] + v[1] * ax[1] + v[2] * ax[2];
        }
        float mean = 0;
        for (int i = 0; i < n; ++i) mean += x[i];
        mean /= n;
        float var = 0, m3 = 0;
        for (int i = 0; i < n; ++i) { float d = x[i] - mean; var += d * d; m3 += d * d * d; }
        var /= n; m3 /= n;
        float sd = sqrtf(var);
        // Orient so a "hit" points positive. A hit is the sharp, spiky side of the motion (skewness);
        // if the motion is too symmetric to tell, use "up" (the push at the bottom of a bob), and
        // failing that keep the previous direction.
        float skew = sd > 1e-6f ? m3 / (var * sd) : 0.0f;
        float upd = ax[0] * up_[0] + ax[1] * up_[1] + ax[2] * up_[2];
        float keep = ax[0] * axis_[0] + ax[1] * axis_[1] + ax[2] * axis_[2];
        bool flip = fabsf(skew) > 0.25f ? skew < 0 : (fabsf(upd) > 0.3f ? upd < 0 : keep < 0);
        if (flip) {
            for (int r = 0; r < 3; ++r) ax[r] = -ax[r];
            for (int i = 0; i < n; ++i) x[i] = -x[i];
            mean = -mean;
        }
        bool changed = (ax[0] * axis_[0] + ax[1] * axis_[1] + ax[2] * axis_[2]) < 0.995f;
        for (int r = 0; r < 3; ++r) axis_[r] = ax[r];
        if (changed) for (int i = 0; i < n; ++i) sig_[(start + i) % N] = x[i];
        for (int i = 0; i < n; ++i) x[i] -= mean;
        // RMS of the last second decides "still moving?"
        float recent = 0;
        for (int i = n - FS; i < n; ++i) recent += x[i] * x[i];
        rms_ = sqrtf(recent / FS);
        thr_ = mean + 0.8f * sd;
        if (thr_ < HIT_FLOOR) thr_ = HIT_FLOOR;

        // Normalised autocorrelation over the tempo range.
        float r[LAG_MAX + 2];
        for (int L = LAG_MIN - 1; L <= LAG_MAX + 1; ++L) {
            float sxy = 0, sxx = 0, syy = 0;
            for (int i = 0; i + L < n; ++i) {
                sxy += x[i] * x[i + L]; sxx += x[i] * x[i]; syy += x[i + L] * x[i + L];
            }
            r[L] = (sxx > 0 && syy > 0) ? sxy / sqrtf(sxx * syy) : 0;
        }
        int lo = LAG_MIN, hi = LAG_MAX;
        if (locked_) {                                   // follow the tempo we have; no octave jumps
            int c = (int)lroundf(period_ * FS);
            lo = (int)(c * 0.75f); hi = (int)ceilf(c * 1.25f);
            if (lo < LAG_MIN) lo = LAG_MIN;
            if (hi > LAG_MAX) hi = LAG_MAX;
        }
        // Mild preference for dance tempos (~120 BPM) so double-time hats or half-time steps don't win.
        // Music repeats at the bar too (kick-snare every 2 beats), so it gets a firmer pull to ~120.
        const float width = music_ ? 0.6f : 1.1f;
        auto prior = [width](int L) {
            float o = log2f((60.0f * FS / L) / 120.0f) / width;
            return expf(-0.5f * o * o);
        };
        int best = lo;
        for (int L = lo; L <= hi; ++L) if (r[L] * prior(L) > r[best] * prior(best)) best = L;
        // Prefer the faster tempo when half the lag is nearly as strong (avoids locking to every 2nd
        // beat) — unless that would be above 160 BPM, which is almost always double-time.
        int half = (best + 1) / 2;
        if (!locked_ && half >= LAG_MIN && r[half] > 0.8f * r[best] && 60.0f * FS / half <= 160.0f) best = half;
        // Parabolic refinement for sub-sample lag.
        float y0 = r[best - 1], y1 = r[best], y2 = r[best + 1];
        float den = y0 - 2 * y1 + y2;
        float off = fabsf(den) > 1e-6f ? 0.5f * (y0 - y2) / den : 0.0f;
        if (off > 0.5f) off = 0.5f;
        if (off < -0.5f) off = -0.5f;
        conf_ = y1;
        period_ac_ = (best + off) / FS;

        const float dt = 0.1f;
        bool moving = rms_ >= MIN_RMS && sd >= MIN_RMS * 0.8f;
        if (!locked_) {
            good_runs_ = (moving && conf_ >= lock_conf_) ? good_runs_ + 1 : 0;
            float bpm_ac = 60.0f / period_ac_;
            bool tempo_ok = !music_ || (bpm_ac >= 70.0f && bpm_ac <= 160.0f);
            if (good_runs_ >= lock_runs_ && tempo_ok && t_ - last_hit_t_ < period_ac_ && hits_even(period_ac_)) {
                locked_ = true;
                period_ = period_ac_;
                phase_ = (t_ - last_hit_t_) / period_;      // last hit was "beat 0"
                bad_time_ = 0;
                fired_ = true;                               // beat 0 already happened (the last hit)
            }
        } else {
            period_ += 0.25f * (period_ac_ - period_);     // follow tempo changes
            bad_time_ = (moving && conf_ >= KEEP_CONF) ? 0 : bad_time_ + dt;
            if (bad_time_ > 1.0f) unlock();
        }
    }
};

// Motion echo, like a bobblehead: sway the Core2 back and forth a couple of times and stop, and the
// forms keep swaying on their own springs for a while, dying out slowly. Nothing is replayed — this
// only spots "that was back-and-forth" (jolts, gravity swinging while you rock or tip it, twisting)
// and, once you stop, reports a sustain level (1 → 0 over fade_s) that lowers the springs' damping.
class MotionEcho {
public:
    static constexpr int   FS = 50;                 // samples per second
    static constexpr int   N = 300;                 // 6 s of history
    static constexpr int   LAG_MIN = 10;            // 0.2 s per back-and-forth (quick shake)
    static constexpr int   LAG_MAX = 125;           // 2.5 s per back-and-forth (slow rocking)
    static constexpr float MOVE = 0.035f;           // g (rms): you are moving
    static constexpr float CONF = 0.5f;             // how back-and-forth it must be
    static constexpr float SWAY_HOLD = 0.6f;        // damping ease-off while you're still swaying

    float fade_s = 9.0f;                            // the sway dies out over this long

    // dyn: jolts (g, gravity removed) · gdev: real down minus its slow average · w: twist (rad/s)
    void update(float dt, const float dyn[3], const float gdev[3], float w) {
        t_ += dt;
        float v2 = 0;                               // moving = jolts, gravity turning, or twisting
        for (int k = 0; k < 3; ++k) {
            const float rate = dt > 1e-6f ? (gdev[k] - gprev_[k]) / dt : 0.0f;
            gprev_[k] = gdev[k];
            v2 += dyn[k] * dyn[k] + (0.05f * rate) * (0.05f * rate);
        }
        v2 += (w * 0.05f) * (w * 0.05f);
        e_ += (v2 - e_) * (1.0f - expf(-dt / 0.15f));
        const bool moving = sqrtf(e_) > MOVE;
        // you've stopped swaying as soon as your motion clearly drops off (not when it reaches zero),
        // so the forms' own sway isn't damped away before the echo takes over
        peak_ = e_ > peak_ ? e_ : peak_ * expf(-dt / 2.0f);
        const bool swaying = sqrtf(e_) > (MOVE > 0.3f * sqrtf(peak_) ? MOVE : 0.3f * sqrtf(peak_));
        for (int k = 0; k < 3; ++k) acc_[k] += (dyn[k] + gdev[k]) * dt;
        acc_[3] += w * dt; acc_w_ += dt;
        while (t_ - last_t_ >= 1.0f / FS) {
            last_t_ += 1.0f / FS;
            const float k = acc_w_ > 1e-6f ? 1.0f / acc_w_ : 0.0f;
            for (int c = 0; c < 4; ++c) { h_[head_][c] = acc_[c] * k; acc_[c] = 0; }
            acc_w_ = 0;
            head_ = (head_ + 1) % N;
            if (filled_ < N) ++filled_;
            if (++since_ >= FS / 10) { since_ = 0; analyse(moving); }
        }
        if (moving) moving_t_ += dt;
        if (swaying) {
            // once it's clearly back-and-forth, start easing the damping off already (part way), so
            // the forms still have their full sway when you stop
            still_t_ = 0; fading_ = false;
            const float target = swayed_ ? SWAY_HOLD : 0.0f;
            sustain_ += (target - sustain_) * (1.0f - expf(-dt / 0.3f));
        } else {
            still_t_ += dt;
            if (swayed_ && !fading_) { fading_ = true; fade_t_ = 0; }
            if (!swayed_ && !moving && still_t_ > 0.6f) moving_t_ = 0;
            if (fading_) {
                fade_t_ += dt;
                const float u = fade_t_ / fade_s;            // stays near full, eases off at the end
                sustain_ = u < 1.0f ? 1.0f - u * u * u : 0.0f;
                if (u >= 1.0f) { fading_ = false; swayed_ = false; }
            }
        }
    }

    float strength() const { return sustain_; }     // 1 just after you stop → 0 when it has died out
    bool  rhythmic() const { return swayed_; }
    float period() const { return period_; }

private:
    float h_[N][4] = {};
    int   head_ = 0, filled_ = 0, since_ = 0, good_ = 0;
    float t_ = 0, last_t_ = 0, acc_[4] = {}, acc_w_ = 0, e_ = 0, gprev_[3] = {0, 0, 0};
    float period_ = 1.0f, moving_t_ = 0, still_t_ = 0, fade_t_ = 0, sustain_ = 0, peak_ = 0;
    bool  swayed_ = false, fading_ = false;

    const float* at(int ago) const { return h_[(head_ - 1 - ago + 2 * N) % N]; }

    // How back-and-forth x is: its self-similarity must dip below zero half a cycle in and peak again
    // one cycle in (a one-off tilt or a slow turn never does). Returns the peak; pick = cycle length.
    static float periodicity(const float* x0, int n, int& pick) {
        static float x[N], r[LAG_MAX + 2];
        pick = 0;
        float mean = 0;
        for (int i = 0; i < n; ++i) mean += x0[i];
        mean /= n;
        for (int i = 0; i < n; ++i) x[i] = x0[i] - mean;
        const int lmax = (n / 2 < LAG_MAX) ? n / 2 : LAG_MAX;
        if (lmax <= LAG_MIN) return 0.0f;
        int dip = -1;
        for (int L = 1; L <= lmax + 1; ++L) {
            float sxy = 0, sxx = 0, syy = 0;
            for (int i = 0; i + L < n; ++i) { sxy += x[i] * x[i + L]; sxx += x[i] * x[i]; syy += x[i + L] * x[i + L]; }
            r[L] = (sxx > 0 && syy > 0) ? sxy / sqrtf(sxx * syy) : 0.0f;
            if (dip < 0 && r[L] < -0.1f) dip = L;
        }
        if (dip < 0) return 0.0f;
        const int from = dip > LAG_MIN ? dip : LAG_MIN;
        float best = 0;
        for (int L = from; L <= lmax; ++L)
            if (r[L] >= r[L - 1] && r[L] >= r[L + 1] && r[L] > best) { best = r[L]; pick = L; }
        return pick ? best : 0.0f;
    }

    void analyse(bool moving) {
        const int n = filled_ < N ? filled_ : N;
        if (n < 2 * LAG_MIN + 10 || !moving) return;
        // 1-D signal: the main direction of the motion, plus twist
        static float vv[N][3], x[N];
        float m[3] = {0, 0, 0};
        for (int i = 0; i < n; ++i) { const float* s = at(n - 1 - i); for (int k = 0; k < 3; ++k) { vv[i][k] = s[k]; m[k] += s[k]; } }
        for (int k = 0; k < 3; ++k) m[k] /= n;
        float C[3][3] = {};
        for (int i = 0; i < n; ++i) {
            float d[3] = {vv[i][0] - m[0], vv[i][1] - m[1], vv[i][2] - m[2]};
            for (int r = 0; r < 3; ++r) for (int c = 0; c < 3; ++c) C[r][c] += d[r] * d[c];
        }
        int kmax = C[1][1] > C[0][0] ? 1 : 0; if (C[2][2] > C[kmax][kmax]) kmax = 2;
        float ax[3] = {0, 0, 0}; ax[kmax] = 1;
        for (int it = 0; it < 10; ++it) {
            float nx[3];
            for (int r = 0; r < 3; ++r) nx[r] = C[r][0] * ax[0] + C[r][1] * ax[1] + C[r][2] * ax[2];
            float l = sqrtf(nx[0] * nx[0] + nx[1] * nx[1] + nx[2] * nx[2]);
            if (l < 1e-12f) break;
            for (int r = 0; r < 3; ++r) ax[r] = nx[r] / l;
        }
        for (int i = 0; i < n; ++i)
            x[i] = (vv[i][0] - m[0]) * ax[0] + (vv[i][1] - m[1]) * ax[1] + (vv[i][2] - m[2]) * ax[2] + 0.05f * at(n - 1 - i)[3];
        // the last 3 s (quicker swaying) or the whole 6 s (slow rocking), whichever is clearer
        const int nw = n < 3 * FS ? n : 3 * FS;
        int p1 = 0, p2 = 0;
        const float c1 = periodicity(x + (n - nw), nw, p1), c2 = periodicity(x, n, p2);
        const float conf = c1 >= c2 ? c1 : c2;
        const int pick = c1 >= c2 ? p1 : p2;
        // back and forth a couple of times → arm the sway-on
        if (pick && conf >= CONF && moving_t_ > 1.5f * pick / FS) {
            if (++good_ >= 2) { swayed_ = true; period_ = (float)pick / FS; }
        } else {
            good_ = 0;
            if (conf < 0.3f) swayed_ = false;
        }
    }
};

// Microphone → hit strength + "is music playing?". Feed each recorded chunk (mono int16).
// The room-noise floor is learned only while no music is playing — it freezes during music, so the
// music itself can never become the floor. Music = clearly above the floor AND hitting (kicks, snares,
// claps); steady noise like a fan has no hits and doesn't count.
#ifndef JIG_KICK_HZ
#define JIG_KICK_HZ 120.0f
#endif
class AudioOnset {
public:
    static constexpr float KICK_HZ = JIG_KICK_HZ;   // low-pass for the kick/bass band
    static constexpr float OPEN_DB = 6.0f;          // music must be this far above the room floor…
    static constexpr float CLOSE_DB = 3.0f;         // …and counts as stopped below this (for 2 s)
    static constexpr float SCALE = 0.05f;           // dB of rise → hit strength
    static constexpr float HIT = 0.06f;             // a "hit" for the music detector
    static constexpr float FLOOR_START = -60.0f;    // dBFS: assume a quiet room until we learn otherwise

    float process(const int16_t* s, size_t n, uint32_t rate) {
        if (!n) return 0.0f;
        const float a = 1.0f - expf(-6.2831853f * KICK_HZ / rate);
        const float hp = 1.0f - expf(-6.2831853f * 20.0f / rate);   // remove DC / rumble
        float el = 0, ea = 0;
        for (size_t i = 0; i < n; ++i) {
            float x = s[i] * (1.0f / 32768.0f);
            dc_ += (x - dc_) * hp;
            x -= dc_;
            lp1_ += a * (x - lp1_);
            lp2_ += a * (lp1_ - lp2_);
            el += lp2_ * lp2_;
            ea += x * x;
        }
        float low = 10.0f * log10f(el / n + 1e-10f), all = 10.0f * log10f(ea / n + 1e-10f);
        level_db_ = all;
        const float dt = (float)n / rate;
        // Floor: falls quickly to quieter levels; rises slowly only while there's no music.
        if (all < floor_db_) floor_db_ += (all - floor_db_) * (1.0f - expf(-dt / 3.0f));
        else if (!music_)    floor_db_ += (all - floor_db_) * (1.0f - expf(-dt / 15.0f));
        float rise = (low - prev_low_ > 0 ? low - prev_low_ : 0) + 0.3f * (all - prev_all_ > 0 ? all - prev_all_ : 0);
        prev_low_ = low;
        prev_all_ = all;
        float hit = rise * SCALE;
        // Music detector: loud enough + hitting, with hysteresis so gaps between beats don't stop it.
        activity_ = activity_ * expf(-dt / 1.5f) + (hit > HIT && all > floor_db_ + CLOSE_DB ? 1.0f : 0.0f);
        bool loud = all > floor_db_ + OPEN_DB;
        if (!music_) {
            on_t_ = (loud && activity_ > 1.5f) ? on_t_ + dt : 0.0f;
            if (on_t_ >= 0.4f) { music_ = true; off_t_ = 0; }
        } else {
            off_t_ = (all < floor_db_ + CLOSE_DB || activity_ < 0.2f) ? off_t_ + dt : 0.0f;
            if (off_t_ >= 2.0f) { music_ = false; on_t_ = 0; }
        }
        last_hit_ = all > floor_db_ + CLOSE_DB ? hit : 0.0f;
        return last_hit_;
    }
    float level_db() const { return level_db_; }
    float floor_db() const { return floor_db_; }
    bool  music() const { return music_; }
    float last_hit() const { return last_hit_; }

private:
    float dc_ = 0, lp1_ = 0, lp2_ = 0;
    float prev_low_ = -100.0f, prev_all_ = -100.0f, level_db_ = -100.0f, floor_db_ = FLOOR_START;
    float activity_ = 0, on_t_ = 0, off_t_ = 0, last_hit_ = 0;
    bool  music_ = false;
};

}  // namespace jig

