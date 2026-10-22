# APM v2 / NERVA autonomy plan

Status: active development on `codex/nerva-v2-autonomy` and `codex/apm-v2-autonomy`.  The v1.0 snapshots remain
unchanged.  This is an engineering plan, not a claim that the proposed behaviours have been demonstrated.

## Audit and reuse

NERVA already has a local MuJoCo closed loop with a rendered robot-eye RGB-D camera, colour/depth detections,
multi-object tracks, ego-motion-compensated approach speed, short track retention, entity/episodic/place memory,
an explicit world model, grounded outcome learning, and safe orient/approach/inspect/watch/retreat primitives.
Its current v2 profile uses grounded memory and appraisal frames.  The existing `VisionPerception` path uses image
pixels and depth for perception; its ground-truth detector remains a named baseline only.  Track continuity and
persistent identity stay separate: appearance matching is uncertain and short-lived track IDs are never treated as
proof of re-identification.

APM v1.0 contributes a robot-independent, revisable expectation architecture and an independent safety supervisor.
It is deliberately not copied into NERVA.  The first v2 increment extends NERVA's embodiment-specific attention
and behaviour layer, then defines a small adapter at the APM boundary only after the loop has local evidence.

## First increment

1. Add a deterministic, auditable attention manager over existing camera tracks.  It ranks novelty, measured
   relative motion, unexpected detector events, track uncertainty and active goals; it includes fixation dwell and
   a switch margin.
2. Feed the selected target into the existing utility selector, retaining its independent safety pre-emption.
   Positive approach/greeting preference is permitted only after an attributed `benign_contact` outcome; calm
   observation by itself cannot make a person socially positive.
3. Run local unit tests and a short MuJoCo vision scenario.  Record target acquisition, fixation continuity,
   inspection/approach timing and any safety interventions.
4. Produce two short local videos only after the physical loop is stable: novel-ball first versus repeat encounter,
   and matched social histories.  They will identify scripted world events separately from decisions made by NERVA.

## Acceptance criteria before a cloud run

- robot-eye RGB-D, not simulated entity coordinates, creates the target tracks used by attention;
- novel-object acquisition and later habituation are visible in fresh local seeds;
- attention persists through ordinary detector jitter but switches after a measured sudden event;
- all locomotion, head and safety limits remain the existing validated limits;
- no falls, collisions or new safety interventions in the short local pilot.

Cloud rendering or Isaac work needs a local runtime estimate, a smallest-machine cost estimate and new spending
approval.  Any approved v2 spend remains within the existing US$500 total cap; this local phase uses no cloud spend.

## First local pilot (development evidence only)

Seed 62, the existing B2 neutral controller, rendered RGB-D, no GPU or cloud work:

- The ball was inspected for 24.78 s on first encounter and 4.98 s after its return; both encounters were selected
  from camera tracks by the attention manager.
- Two lifetimes had identical calm test meetings after 38 s.  A history with measured gentle contact produced a
  1.0 s stationary greeting; a history with an observed near miss produced no greeting.
- Peak torso tilt was 14.85° (curiosity), 11.98° (positive history) and 14.54° (cautious history).  The deterministic
  safety supervisor did not intervene.

The review bundle in `experiments/v2_autonomy/results_review_seed62_close/` is a direct local simulation render:
observer view plus robot-eye feed, no retiming, and no history labels in the social comparison.  It includes full
20 ms JSONL traces, settings, seed, scenario schedules and SHA-256 manifest; labelled copies live in `debug/`.
It is engineering review material, not a human-perception study or a confirmatory evaluation.  The social greeting
must still be judged visually before treating it as a compelling demo.
