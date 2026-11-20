use std::collections::{BTreeSet, HashMap, HashSet};
// Shared frozen core for the EXP150 capacity and EXP151 runtime studies.
use std::env;
use std::fs::File;
use std::hash::{Hash, Hasher};
use std::io::{BufReader, BufWriter, Read, Write};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{mpsc, Arc, OnceLock, RwLock};
use std::time::Instant;


static PROFILE_ENABLED: OnceLock<bool> = OnceLock::new();

static TRAIN_STEP_OFFSET: AtomicU64 = AtomicU64::new(0);

fn training_step(local: usize) -> u64 {
    TRAIN_STEP_OFFSET
        .load(Ordering::Relaxed)
        .saturating_add(local as u64)
}

fn model_last_step(model: &Model) -> u64 {
    model
        .gcp
        .values()
        .chain(model.gm1.values())
        .chain(model.glp.values())
        .flat_map(|node| {
            std::iter::once(node.last_seen)
                .chain(node.edges.iter().map(|edge| edge.last_seen))
        })
        .max()
        .unwrap_or(0)
}

fn negative_feedback_enabled() -> bool {
    std::env::var("NGU_NEGATIVE_FEEDBACK")
        .map(|value| value != "0" && !value.eq_ignore_ascii_case("false"))
        .unwrap_or(true)
}

static PROFILE_DISTRIBUTION_NS: AtomicU64 = AtomicU64::new(0);
static PROFILE_DISTRIBUTION_CALLS: AtomicU64 = AtomicU64::new(0);
static PROFILE_ROUTED_GET_NS: AtomicU64 = AtomicU64::new(0);
static PROFILE_ROUTED_GET_CALLS: AtomicU64 = AtomicU64::new(0);
static PROFILE_MERGE_NS: AtomicU64 = AtomicU64::new(0);
static PROFILE_MERGE_CALLS: AtomicU64 = AtomicU64::new(0);
static PROFILE_PRUNE_NS: AtomicU64 = AtomicU64::new(0);
static PROFILE_PRUNE_CALLS: AtomicU64 = AtomicU64::new(0);
static PROFILE_ROUTE_REFRESH_NS: AtomicU64 = AtomicU64::new(0);
static PROFILE_ROUTE_REFRESH_CALLS: AtomicU64 = AtomicU64::new(0);
static PROFILE_SLEEP_NS: AtomicU64 = AtomicU64::new(0);
static PROFILE_SLEEP_CALLS: AtomicU64 = AtomicU64::new(0);
static GROUPED_SPEC_WINDOWS: AtomicU64 = AtomicU64::new(0);
static GROUPED_SPEC_TASKS: AtomicU64 = AtomicU64::new(0);

fn profile_enabled() -> bool {
    *PROFILE_ENABLED.get_or_init(|| {
        std::env::var("NGU_PROFILE")
            .map(|value| value == "1" || value.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
    })
}

fn profile_timer() -> Option<Instant> {
    profile_enabled().then(Instant::now)
}

fn profile_finish(timer: Option<Instant>, total: &AtomicU64, calls: &AtomicU64) {
    if let Some(start) = timer {
        total.fetch_add(start.elapsed().as_nanos() as u64, Ordering::Relaxed);
        calls.fetch_add(1, Ordering::Relaxed);
    }
}

fn profile_snapshot() -> [u64; 14] {
    [
        PROFILE_DISTRIBUTION_NS.load(Ordering::Relaxed),
        PROFILE_DISTRIBUTION_CALLS.load(Ordering::Relaxed),
        PROFILE_ROUTED_GET_NS.load(Ordering::Relaxed),
        PROFILE_ROUTED_GET_CALLS.load(Ordering::Relaxed),
        PROFILE_MERGE_NS.load(Ordering::Relaxed),
        PROFILE_MERGE_CALLS.load(Ordering::Relaxed),
        PROFILE_PRUNE_NS.load(Ordering::Relaxed),
        PROFILE_PRUNE_CALLS.load(Ordering::Relaxed),
        PROFILE_ROUTE_REFRESH_NS.load(Ordering::Relaxed),
        PROFILE_ROUTE_REFRESH_CALLS.load(Ordering::Relaxed),
        PROFILE_SLEEP_NS.load(Ordering::Relaxed),
        PROFILE_SLEEP_CALLS.load(Ordering::Relaxed),
        GROUPED_SPEC_WINDOWS.load(Ordering::Relaxed),
        GROUPED_SPEC_TASKS.load(Ordering::Relaxed),
    ]
}


static POOL_WINDOWS: AtomicU64 = AtomicU64::new(0);
static POOL_TASKS: AtomicU64 = AtomicU64::new(0);
static POOL_POSITIONS: AtomicU64 = AtomicU64::new(0);
static POOL_PREP_NS: AtomicU64 = AtomicU64::new(0);
static POOL_PREP_CALLS: AtomicU64 = AtomicU64::new(0);
static POOL_DISPATCH_NS: AtomicU64 = AtomicU64::new(0);
static POOL_DISPATCH_CALLS: AtomicU64 = AtomicU64::new(0);
static POOL_WAIT_NS: AtomicU64 = AtomicU64::new(0);
static POOL_WAIT_CALLS: AtomicU64 = AtomicU64::new(0);
static POOL_REDUCE_NS: AtomicU64 = AtomicU64::new(0);
static POOL_REDUCE_CALLS: AtomicU64 = AtomicU64::new(0);
static POOL_WORK_NS: AtomicU64 = AtomicU64::new(0);
static POOL_WORK_CALLS: AtomicU64 = AtomicU64::new(0);

fn exp174_pool_mode() -> u8 {
    std::env::var("NGU_POOL_MODE")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(0)
}

fn pool_snapshot() -> [u64; 13] {
    [
        POOL_WINDOWS.load(Ordering::Relaxed),
        POOL_TASKS.load(Ordering::Relaxed),
        POOL_POSITIONS.load(Ordering::Relaxed),
        POOL_PREP_NS.load(Ordering::Relaxed),
        POOL_PREP_CALLS.load(Ordering::Relaxed),
        POOL_DISPATCH_NS.load(Ordering::Relaxed),
        POOL_DISPATCH_CALLS.load(Ordering::Relaxed),
        POOL_WAIT_NS.load(Ordering::Relaxed),
        POOL_WAIT_CALLS.load(Ordering::Relaxed),
        POOL_REDUCE_NS.load(Ordering::Relaxed),
        POOL_REDUCE_CALLS.load(Ordering::Relaxed),
        POOL_WORK_NS.load(Ordering::Relaxed),
        POOL_WORK_CALLS.load(Ordering::Relaxed),
    ]
}


static PHASE_WINDOWS: AtomicU64 = AtomicU64::new(0);
static PHASE_WAVES: AtomicU64 = AtomicU64::new(0);
static PHASE_MAX_WAVES: AtomicU64 = AtomicU64::new(0);
static PHASE_POSITIONS: AtomicU64 = AtomicU64::new(0);
static PHASE_PLAN_NS: AtomicU64 = AtomicU64::new(0);
static PHASE_PLAN_CALLS: AtomicU64 = AtomicU64::new(0);
static PHASE_COMMIT_NS: AtomicU64 = AtomicU64::new(0);
static PHASE_COMMIT_CALLS: AtomicU64 = AtomicU64::new(0);
static PHASE_DISTRIBUTION_NS: AtomicU64 = AtomicU64::new(0);
static PHASE_DISTRIBUTION_CALLS: AtomicU64 = AtomicU64::new(0);

fn exp175_phase_mode() -> u8 {
    std::env::var("NGU_PHASE_MODE")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(0)
}

fn phase_snapshot() -> [u64; 10] {
    [
        PHASE_WINDOWS.load(Ordering::Relaxed),
        PHASE_WAVES.load(Ordering::Relaxed),
        PHASE_MAX_WAVES.load(Ordering::Relaxed),
        PHASE_POSITIONS.load(Ordering::Relaxed),
        PHASE_PLAN_NS.load(Ordering::Relaxed),
        PHASE_PLAN_CALLS.load(Ordering::Relaxed),
        PHASE_COMMIT_NS.load(Ordering::Relaxed),
        PHASE_COMMIT_CALLS.load(Ordering::Relaxed),
        PHASE_DISTRIBUTION_NS.load(Ordering::Relaxed),
        PHASE_DISTRIBUTION_CALLS.load(Ordering::Relaxed),
    ]
}


static LAYOUT_LOOKUP_NS: AtomicU64 = AtomicU64::new(0);
static LAYOUT_LOOKUP_CALLS: AtomicU64 = AtomicU64::new(0);
static LAYOUT_LOOKUP_HITS: AtomicU64 = AtomicU64::new(0);
static LAYOUT_REBUILD_NS: AtomicU64 = AtomicU64::new(0);
static LAYOUT_REBUILD_CALLS: AtomicU64 = AtomicU64::new(0);

fn exp176_layout_mode() -> u8 {
    std::env::var("NGU_LAYOUT_MODE")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(1)
}

fn layout_snapshot() -> [u64; 5] {
    [
        LAYOUT_LOOKUP_NS.load(Ordering::Relaxed),
        LAYOUT_LOOKUP_CALLS.load(Ordering::Relaxed),
        LAYOUT_LOOKUP_HITS.load(Ordering::Relaxed),
        LAYOUT_REBUILD_NS.load(Ordering::Relaxed),
        LAYOUT_REBUILD_CALLS.load(Ordering::Relaxed),
    ]
}

static GM1_GCP_READS: AtomicU64 = AtomicU64::new(0);
static GM1_MEDIAL_READS: AtomicU64 = AtomicU64::new(0);
static GM1_GLP_READS: AtomicU64 = AtomicU64::new(0);
static GM1_GCP_STOPS: AtomicU64 = AtomicU64::new(0);
static GM1_MEDIAL_STOPS: AtomicU64 = AtomicU64::new(0);
static SHARD_QUERIES: AtomicU64 = AtomicU64::new(0);
static SHARD_PROBES: AtomicU64 = AtomicU64::new(0);
static SHARD_ROUTE_PREDICTIONS: AtomicU64 = AtomicU64::new(0);
static SHARD_ROUTE_CORRECT: AtomicU64 = AtomicU64::new(0);
static SHARD_ROUTE_WRONG: AtomicU64 = AtomicU64::new(0);
static SHARD_FALLBACKS: AtomicU64 = AtomicU64::new(0);
static SHARD_LOST_CONTEXTS: AtomicU64 = AtomicU64::new(0);
static SHARD_FOUND_CONTEXTS: AtomicU64 = AtomicU64::new(0);

fn reset_layer_access() {
    GM1_GCP_READS.store(0, Ordering::Relaxed);
    GM1_MEDIAL_READS.store(0, Ordering::Relaxed);
    GM1_GLP_READS.store(0, Ordering::Relaxed);
    GM1_GCP_STOPS.store(0, Ordering::Relaxed);
    GM1_MEDIAL_STOPS.store(0, Ordering::Relaxed);
    SHARD_QUERIES.store(0, Ordering::Relaxed);
    SHARD_PROBES.store(0, Ordering::Relaxed);
    SHARD_ROUTE_PREDICTIONS.store(0, Ordering::Relaxed);
    SHARD_ROUTE_CORRECT.store(0, Ordering::Relaxed);
    SHARD_ROUTE_WRONG.store(0, Ordering::Relaxed);
    SHARD_FALLBACKS.store(0, Ordering::Relaxed);
    SHARD_LOST_CONTEXTS.store(0, Ordering::Relaxed);
    SHARD_FOUND_CONTEXTS.store(0, Ordering::Relaxed);
    for counter in [
        &PROFILE_DISTRIBUTION_NS, &PROFILE_DISTRIBUTION_CALLS,
        &PROFILE_ROUTED_GET_NS, &PROFILE_ROUTED_GET_CALLS,
        &PROFILE_MERGE_NS, &PROFILE_MERGE_CALLS,
        &PROFILE_PRUNE_NS, &PROFILE_PRUNE_CALLS,
        &PROFILE_ROUTE_REFRESH_NS, &PROFILE_ROUTE_REFRESH_CALLS,
        &PROFILE_SLEEP_NS, &PROFILE_SLEEP_CALLS,
        &GROUPED_SPEC_WINDOWS, &GROUPED_SPEC_TASKS,
        &POOL_WINDOWS, &POOL_TASKS, &POOL_POSITIONS,
        &POOL_PREP_NS, &POOL_PREP_CALLS,
        &POOL_DISPATCH_NS, &POOL_DISPATCH_CALLS,
        &POOL_WAIT_NS, &POOL_WAIT_CALLS,
        &POOL_REDUCE_NS, &POOL_REDUCE_CALLS,
        &POOL_WORK_NS, &POOL_WORK_CALLS,
        &PHASE_WINDOWS, &PHASE_WAVES, &PHASE_MAX_WAVES, &PHASE_POSITIONS,
        &PHASE_PLAN_NS, &PHASE_PLAN_CALLS,
        &PHASE_COMMIT_NS, &PHASE_COMMIT_CALLS,
        &PHASE_DISTRIBUTION_NS, &PHASE_DISTRIBUTION_CALLS,
        &LAYOUT_LOOKUP_NS, &LAYOUT_LOOKUP_CALLS, &LAYOUT_LOOKUP_HITS,
        &LAYOUT_REBUILD_NS, &LAYOUT_REBUILD_CALLS,
    ] {
        counter.store(0, Ordering::Relaxed);
    }
}

fn layer_access() -> [u64; 5] {
    [
        GM1_GCP_READS.load(Ordering::Relaxed),
        GM1_MEDIAL_READS.load(Ordering::Relaxed),
        GM1_GLP_READS.load(Ordering::Relaxed),
        GM1_GCP_STOPS.load(Ordering::Relaxed),
        GM1_MEDIAL_STOPS.load(Ordering::Relaxed),
    ]
}

fn route_access() -> [u64; 8] {
    [
        SHARD_QUERIES.load(Ordering::Relaxed),
        SHARD_PROBES.load(Ordering::Relaxed),
        SHARD_ROUTE_PREDICTIONS.load(Ordering::Relaxed),
        SHARD_ROUTE_CORRECT.load(Ordering::Relaxed),
        SHARD_ROUTE_WRONG.load(Ordering::Relaxed),
        SHARD_FALLBACKS.load(Ordering::Relaxed),
        SHARD_LOST_CONTEXTS.load(Ordering::Relaxed),
        SHARD_FOUND_CONTEXTS.load(Ordering::Relaxed),
    ]
}

// EXP96: matched Dolma pretraining of the post-EXP95 graph language core.
// The same raw OLMo-token stream is consumed in its original document order.
// Prospective candidate-GR assembly and wake/sleep reflection increase the
// opportunity to consolidate constructions while every graph remains
// generic, ordered, non-neural and strictly unidirectional.
const MAGIC: &[u8; 8] = b"NGU168V1";
const ALPHA: f64 = 0.1;
const DEFAULT_MAX_SUCCESSORS: usize = 32;
const WAVE_WINDOW: usize = 4;
const WAVE_DEPTH: usize = 4;
const WAVE_BRANCH: usize = 4;
const WAVE_BEAM: usize = 8;
const WAVE_MAX_GRS: usize = 16;
const MEETING_TOLERANCE: f64 = 1.25;
const FULL_TOLERANCE: f64 = 0.75;
const PROSPECTIVE_MARGIN_FLOOR: f64 = -0.05;
const PROSPECTIVE_MAX_WRONG_PROMOTION: f64 = 0.01;
const ROUTE_SET_MAX_ROUTES: usize = 8;
const ROUTE_SET_BEAM_WIDTH: usize = 16;
const DEFERRED_FUTURE_SAMPLE: usize = 256;
const DEFERRED_RELEVANT_LIMIT: usize = 32;
const DEFERRED_REQUIRED_WINDOWS: u8 = 2;
const DEFERRED_MAX_AGE: u8 = 4;
const DEFERRED_LIMIT: usize = 4096;
const METAGRAPH_MAX_PAIRS: usize = 16;
const METAGRAPH_PAIR_EXAMPLES: usize = 16;
const CONSTRUCTION_INTERFACE: usize = 4;
const CONSTRUCTION_SLEEP_EXAMPLES: usize = 8192;
const CONSTRUCTION_NEW_LIMIT: usize = 512;
const CONSTRUCTION_LIMIT: usize = 8192;
const CONSTRUCTION_FUTURE_LIMIT: usize = 64;
const CONSTRUCTION_REQUIRED_WINDOWS: u8 = 2;
const CONSTRUCTION_MAX_AGE: u8 = 4;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Mode {
    Base,
    Unit,
    Frequency,
    Gradient,
    FrequencyGradient,
    Full,
    Shuffle,
}

impl Mode {
    fn name(self) -> &'static str {
        match self {
            Self::Base => "base",
            Self::Unit => "w_unit",
            Self::Frequency => "w_freq",
            Self::Gradient => "w_grad",
            Self::FrequencyGradient => "w_fg",
            Self::Full => "w_fgd",
            Self::Shuffle => "shuffle",
        }
    }
    fn waves(self) -> bool {
        self != Self::Base
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SleepPolicy {
    Cycle,
    Uniform,
    Contrast,
    Bridge,
    BridgeContrast,
    Shuffle,
    ValidatedBridge,
    ValidatedBridgeContrast,
    ValidatedShuffle,
    MultiDoseContrast,
    CumulativeContrast,
    CumulativeShuffle,
    PhaseContrast,
    CumulativePhase,
    CumulativePhaseShuffle,
    SequentialPhase,
    OriginSequential,
    OriginSequentialShuffle,
    OriginBankNoPhase,
    OriginBankPhase,
    OriginBankShuffle,
    RouteSetGreedy,
    RouteSetBeam,
    RouteSetNoInterference,
    RouteSetShuffle,
    DeferredRoute,
    DeferredEdge,
    DeferredMetagraph,
    DeferredShuffle,
    ConstructionFixed,
    ConstructionSlotsImmediate,
    ConstructionDeferred,
    ConstructionBoth,
    ConstructionShuffle,
}

impl SleepPolicy {
    fn parse(value: &str) -> Self {
        match value {
            "cycle" => Self::Cycle,
            "uniform" => Self::Uniform,
            "contrast" => Self::Contrast,
            "bridge" => Self::Bridge,
            "bridge_contrast" => Self::BridgeContrast,
            "shuffle" => Self::Shuffle,
            "validated_bridge" => Self::ValidatedBridge,
            "validated_bridge_contrast" => Self::ValidatedBridgeContrast,
            "validated_shuffle" => Self::ValidatedShuffle,
            "multidose_contrast" => Self::MultiDoseContrast,
            "cumulative_contrast" => Self::CumulativeContrast,
            "cumulative_shuffle" => Self::CumulativeShuffle,
            "phase_contrast" => Self::PhaseContrast,
            "cumulative_phase" => Self::CumulativePhase,
            "cumulative_phase_shuffle" => Self::CumulativePhaseShuffle,
            "sequential_phase" => Self::SequentialPhase,
            "origin_sequential" => Self::OriginSequential,
            "origin_sequential_shuffle" => Self::OriginSequentialShuffle,
            "origin_bank_no_phase" => Self::OriginBankNoPhase,
            "origin_bank_phase" => Self::OriginBankPhase,
            "origin_bank_shuffle" => Self::OriginBankShuffle,
            "route_set_greedy" => Self::RouteSetGreedy,
            "route_set_beam" => Self::RouteSetBeam,
            "route_set_no_interference" => Self::RouteSetNoInterference,
            "route_set_shuffle" => Self::RouteSetShuffle,
            "deferred_route" => Self::DeferredRoute,
            "deferred_edge" => Self::DeferredEdge,
            "deferred_metagraph" => Self::DeferredMetagraph,
            "deferred_shuffle" => Self::DeferredShuffle,
            "gr_none" => Self::ConstructionFixed,
            "gr_role" => Self::ConstructionSlotsImmediate,
            "gr_prediction" => Self::ConstructionDeferred,
            "gr_both" => Self::ConstructionBoth,
            "gr_shuffle" => Self::ConstructionShuffle,
            _ => panic!("unknown sleep policy: {}", value),
        }
    }
    fn name(self) -> &'static str {
        match self {
            Self::Cycle => "cycle",
            Self::Uniform => "uniform",
            Self::Contrast => "contrast",
            Self::Bridge => "bridge",
            Self::BridgeContrast => "bridge_contrast",
            Self::Shuffle => "shuffle",
            Self::ValidatedBridge => "validated_bridge",
            Self::ValidatedBridgeContrast => "validated_bridge_contrast",
            Self::ValidatedShuffle => "validated_shuffle",
            Self::MultiDoseContrast => "multidose_contrast",
            Self::CumulativeContrast => "cumulative_contrast",
            Self::CumulativeShuffle => "cumulative_shuffle",
            Self::PhaseContrast => "phase_contrast",
            Self::CumulativePhase => "cumulative_phase",
            Self::CumulativePhaseShuffle => "cumulative_phase_shuffle",
            Self::SequentialPhase => "sequential_phase",
            Self::OriginSequential => "origin_sequential",
            Self::OriginSequentialShuffle => "origin_sequential_shuffle",
            Self::OriginBankNoPhase => "origin_bank_no_phase",
            Self::OriginBankPhase => "origin_bank_phase",
            Self::OriginBankShuffle => "origin_bank_shuffle",
            Self::RouteSetGreedy => "route_set_greedy",
            Self::RouteSetBeam => "route_set_beam",
            Self::RouteSetNoInterference => "route_set_no_interference",
            Self::RouteSetShuffle => "route_set_shuffle",
            Self::DeferredRoute => "deferred_route",
            Self::DeferredEdge => "deferred_edge",
            Self::DeferredMetagraph => "deferred_metagraph",
            Self::DeferredShuffle => "deferred_shuffle",
            Self::ConstructionFixed => "gr_none",
            Self::ConstructionSlotsImmediate => "gr_role",
            Self::ConstructionDeferred => "gr_prediction",
            Self::ConstructionBoth => "gr_both",
            Self::ConstructionShuffle => "gr_shuffle",
        }
    }
    fn interval(self, options: &Options) -> usize {
        options.wake_interval
    }
    fn active(self) -> bool {
        self != Self::Cycle
    }
    fn bridge(self) -> bool {
        matches!(
            self,
            Self::Bridge
                | Self::BridgeContrast
                | Self::Shuffle
                | Self::ValidatedBridge
                | Self::ValidatedBridgeContrast
                | Self::ValidatedShuffle
                | Self::MultiDoseContrast
                | Self::CumulativeContrast
                | Self::CumulativeShuffle
                | Self::CumulativePhase
                | Self::CumulativePhaseShuffle
                | Self::SequentialPhase
                | Self::OriginSequential
                | Self::OriginSequentialShuffle
                | Self::OriginBankNoPhase
                | Self::OriginBankPhase
                | Self::OriginBankShuffle
                | Self::RouteSetGreedy
                | Self::RouteSetBeam
                | Self::RouteSetNoInterference
                | Self::RouteSetShuffle
                | Self::DeferredRoute
                | Self::DeferredEdge
                | Self::DeferredMetagraph
                | Self::DeferredShuffle
        )
    }
    fn contrast(self) -> bool {
        matches!(
            self,
            Self::Contrast
                | Self::BridgeContrast
                | Self::Shuffle
                | Self::ValidatedBridgeContrast
                | Self::ValidatedShuffle
                | Self::MultiDoseContrast
                | Self::CumulativeContrast
                | Self::CumulativeShuffle
                | Self::PhaseContrast
                | Self::CumulativePhase
                | Self::CumulativePhaseShuffle
                | Self::SequentialPhase
                | Self::OriginSequential
                | Self::OriginSequentialShuffle
                | Self::OriginBankNoPhase
                | Self::OriginBankPhase
                | Self::OriginBankShuffle
                | Self::RouteSetGreedy
                | Self::RouteSetBeam
                | Self::RouteSetNoInterference
                | Self::RouteSetShuffle
                | Self::DeferredRoute
                | Self::DeferredEdge
                | Self::DeferredMetagraph
                | Self::DeferredShuffle
                | Self::ConstructionFixed
                | Self::ConstructionSlotsImmediate
                | Self::ConstructionDeferred
                | Self::ConstructionBoth
                | Self::ConstructionShuffle
        )
    }
    fn validated(self) -> bool {
        matches!(
            self,
            Self::ValidatedBridge | Self::ValidatedBridgeContrast | Self::ValidatedShuffle
        )
    }
    fn multidose(self) -> bool {
        matches!(
            self,
            Self::MultiDoseContrast
                | Self::CumulativeContrast
                | Self::CumulativeShuffle
                | Self::CumulativePhase
                | Self::CumulativePhaseShuffle
                | Self::SequentialPhase
                | Self::OriginSequential
                | Self::OriginSequentialShuffle
                | Self::OriginBankNoPhase
                | Self::OriginBankPhase
                | Self::OriginBankShuffle
                | Self::RouteSetGreedy
                | Self::RouteSetBeam
                | Self::RouteSetNoInterference
                | Self::RouteSetShuffle
                | Self::DeferredRoute
                | Self::DeferredEdge
                | Self::DeferredMetagraph
                | Self::DeferredShuffle
                | Self::ConstructionFixed
                | Self::ConstructionSlotsImmediate
                | Self::ConstructionDeferred
                | Self::ConstructionBoth
                | Self::ConstructionShuffle
        )
    }
    fn cumulative(self) -> bool {
        matches!(
            self,
            Self::CumulativeContrast
                | Self::CumulativeShuffle
                | Self::CumulativePhase
                | Self::CumulativePhaseShuffle
                | Self::SequentialPhase
                | Self::OriginSequential
                | Self::OriginSequentialShuffle
                | Self::OriginBankNoPhase
                | Self::OriginBankPhase
                | Self::OriginBankShuffle
                | Self::RouteSetGreedy
                | Self::RouteSetBeam
                | Self::RouteSetNoInterference
                | Self::RouteSetShuffle
                | Self::DeferredRoute
                | Self::DeferredEdge
                | Self::DeferredMetagraph
                | Self::DeferredShuffle
                | Self::ConstructionFixed
                | Self::ConstructionSlotsImmediate
                | Self::ConstructionDeferred
                | Self::ConstructionBoth
                | Self::ConstructionShuffle
        )
    }
    fn sequential(self) -> bool {
        matches!(
            self,
            Self::SequentialPhase
                | Self::OriginSequential
                | Self::OriginSequentialShuffle
                | Self::OriginBankNoPhase
                | Self::OriginBankPhase
                | Self::OriginBankShuffle
                | Self::RouteSetGreedy
                | Self::RouteSetBeam
                | Self::RouteSetNoInterference
                | Self::RouteSetShuffle
                | Self::DeferredRoute
                | Self::DeferredEdge
                | Self::DeferredMetagraph
                | Self::DeferredShuffle
                | Self::ConstructionFixed
                | Self::ConstructionSlotsImmediate
                | Self::ConstructionDeferred
                | Self::ConstructionBoth
                | Self::ConstructionShuffle
        )
    }
    fn origin(self) -> bool {
        matches!(
            self,
            Self::OriginSequential
                | Self::OriginSequentialShuffle
                | Self::OriginBankNoPhase
                | Self::OriginBankPhase
                | Self::OriginBankShuffle
                | Self::RouteSetGreedy
                | Self::RouteSetBeam
                | Self::RouteSetNoInterference
                | Self::RouteSetShuffle
                | Self::DeferredRoute
                | Self::DeferredEdge
                | Self::DeferredMetagraph
                | Self::DeferredShuffle
        )
    }
    fn origin_bank(self) -> bool {
        matches!(
            self,
            Self::OriginBankNoPhase
                | Self::OriginBankPhase
                | Self::OriginBankShuffle
                | Self::RouteSetGreedy
                | Self::RouteSetBeam
                | Self::RouteSetNoInterference
                | Self::RouteSetShuffle
                | Self::DeferredRoute
                | Self::DeferredEdge
                | Self::DeferredMetagraph
                | Self::DeferredShuffle
        )
    }
    fn route_set(self) -> bool {
        matches!(
            self,
            Self::RouteSetGreedy
                | Self::RouteSetBeam
                | Self::RouteSetNoInterference
                | Self::RouteSetShuffle
        )
    }
    fn deferred(self) -> bool {
        matches!(
            self,
            Self::DeferredRoute
                | Self::DeferredEdge
                | Self::DeferredMetagraph
                | Self::DeferredShuffle
        )
    }
    fn construction(self) -> bool {
        matches!(
            self,
            Self::ConstructionFixed
                | Self::ConstructionSlotsImmediate
                | Self::ConstructionDeferred
                | Self::ConstructionBoth
                | Self::ConstructionShuffle
        )
    }
    fn construction_deferred(self) -> bool {
        self.construction()
    }
    fn construction_dag(self) -> bool {
        self.construction()
    }
    fn construction_edge_role_credit(self) -> bool {
        self.construction()
    }
    fn construction_edge_prediction_credit(self) -> bool {
        self.construction()
    }
    fn edge_application(self) -> u8 {
        match self {
            Self::ConstructionFixed => 0b0010, // route, no GR term
            Self::ConstructionSlotsImmediate => 0b0110, // route + role term
            Self::ConstructionDeferred => 0b1010, // route + prediction term
            Self::ConstructionBoth | Self::ConstructionShuffle => 0b1110,
            _ => 0,
        }
    }
    fn separate_phase(self) -> bool {
        matches!(
            self,
            Self::SequentialPhase
                | Self::OriginSequential
                | Self::OriginSequentialShuffle
                | Self::OriginBankPhase
                | Self::OriginBankShuffle
        )
    }
    fn phase(self) -> bool {
        matches!(
            self,
            Self::PhaseContrast | Self::CumulativePhase | Self::CumulativePhaseShuffle
        )
    }
    fn shuffled(self) -> bool {
        matches!(
            self,
            Self::Shuffle
                | Self::ValidatedShuffle
                | Self::CumulativeShuffle
                | Self::CumulativePhaseShuffle
                | Self::OriginSequentialShuffle
                | Self::OriginBankShuffle
                | Self::RouteSetShuffle
                | Self::DeferredShuffle
                | Self::ConstructionShuffle
        )
    }
}

#[derive(Clone, Copy, Default, Eq, PartialEq)]
struct ContextKey(u64);

impl ContextKey {
    const TOKEN_MASK: u64 = 0x0fff;

    fn new(len: u8, tokens: [u16; 4]) -> Self {
        assert!(len <= 4);
        let mut packed = len as u64;
        for (index, token) in tokens.iter().enumerate() {
            assert!((*token as u64) <= Self::TOKEN_MASK);
            packed |= (*token as u64) << (4 + 12 * index);
        }
        Self(packed)
    }
    fn len(self) -> u8 {
        (self.0 & 0x0f) as u8
    }
    fn token(self, index: usize) -> u16 {
        ((self.0 >> (4 + 12 * index)) & Self::TOKEN_MASK) as u16
    }
    fn tokens(self) -> [u16; 4] {
        [self.token(0), self.token(1), self.token(2), self.token(3)]
    }
    fn packed(self) -> u64 {
        self.0
    }
}

impl Ord for ContextKey {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.len()
            .cmp(&other.len())
            .then_with(|| self.tokens().cmp(&other.tokens()))
    }
}

impl PartialOrd for ContextKey {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Hash for ContextKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_u64(self.0);
    }
}

#[derive(Default)]
struct ContextHasher {
    state: u64,
}

impl Hasher for ContextHasher {
    fn finish(&self) -> u64 {
        self.state
    }
    fn write(&mut self, bytes: &[u8]) {
        let mut value = 0x517cc1b727220a95u64;
        for &byte in bytes {
            value = value.rotate_left(5) ^ byte as u64;
            value = value.wrapping_mul(0x9e3779b185ebca87);
        }
        self.state = value;
    }
    fn write_u64(&mut self, value: u64) {
        let mut x = value.wrapping_add(0x9e3779b97f4a7c15);
        x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
        self.state = x ^ (x >> 31);
    }
    fn write_u16(&mut self, value: u16) {
        self.state ^= (value as u64).wrapping_mul(0x9e3779b185ebca87);
        self.state = self.state.rotate_left(23).wrapping_mul(0x165667b19e3779f9);
    }
}

type ContextMap<V> = HashMap<
    ContextKey,
    V,
    std::hash::BuildHasherDefault<ContextHasher>,
>;

type TokenMap<V> = HashMap<
    u16,
    V,
    std::hash::BuildHasherDefault<ContextHasher>,
>;

#[derive(Default)]
struct WaveSuccessorCache {
    enabled: bool,
    rows: TokenMap<Vec<(u16, f64, f64, f64)>>,
}

thread_local! {
    static WAVE_SUCCESSOR_CACHE: std::cell::RefCell<WaveSuccessorCache> =
        std::cell::RefCell::new(WaveSuccessorCache::default());
}

fn wave_successor_cache_begin() {
    WAVE_SUCCESSOR_CACHE.with(|cell| {
        let mut cache = cell.borrow_mut();
        cache.rows.clear();
        cache.enabled = true;
    });
}

fn cached_frequency_gradient_successors(
    model: &Model,
    source: u16,
    base: &[f64],
    base_total: f64,
) -> Option<Vec<(u16, f64, f64, f64)>> {
    WAVE_SUCCESSOR_CACHE.with(|cell| {
        let mut cache = cell.borrow_mut();
        if !cache.enabled {
            return None;
        }
        if let Some(rows) = cache.rows.get(&source) {
            return Some(rows.clone());
        }
        let rows = successors_modified(
            model,
            source,
            base,
            base_total,
            Mode::FrequencyGradient,
            None,
            None,
        );
        cache.rows.insert(source, rows.clone());
        Some(rows)
    })
}

thread_local! {
    static FEATURE_BUFFER: std::cell::RefCell<Vec<Feature>> =
        std::cell::RefCell::new(Vec::with_capacity(256));
    static RANKED_BUFFER: std::cell::RefCell<Vec<(u16, f64)>> =
        std::cell::RefCell::new(Vec::with_capacity(2));
}

fn take_feature_buffer() -> Vec<Feature> {
    FEATURE_BUFFER.with(|cell| std::mem::take(&mut *cell.borrow_mut()))
}

fn recycle_feature_buffer(mut rows: Vec<Feature>) {
    rows.clear();
    FEATURE_BUFFER.with(|cell| *cell.borrow_mut() = rows);
}

fn take_ranked_buffer() -> Vec<(u16, f64)> {
    RANKED_BUFFER.with(|cell| std::mem::take(&mut *cell.borrow_mut()))
}

fn recycle_ranked_buffer(mut rows: Vec<(u16, f64)>) {
    rows.clear();
    RANKED_BUFFER.with(|cell| *cell.borrow_mut() = rows);
}


#[derive(Clone, Copy, Default)]
struct TokenVertex {
    count: u32,
    potential: f32,
}

#[derive(Clone, Copy, Default)]
struct Edge {
    token: u16,
    count: u32,
    weight: f32,
    gradient_sum: f32,
    gradient_sq: f32,
    conflicts: u32,
    last_seen: u64,
    sleep_bias: f32,
}

#[derive(Clone, Copy, Default)]
struct EdgeCredit {
    weight_delta: f32,
    gradient_sum: f32,
    gradient_sq: f32,
    conflicts: u32,
    last_seen: u64,
}

#[derive(Clone, Default)]
struct Node {
    total: u32,
    last_seen: u64,
    potential: f32,
    edges: Vec<Edge>,
    // Runtime-only deterministic accelerator. Keeping the learned edges in
    // insertion order preserves the exact checkpoint and inference semantics.
    edge_index: Option<Vec<(u16, usize)>>,
}

impl Node {
    fn new(order: usize) -> Self {
        Self {
            potential: match order {
                1 => -1.5,
                2 => -1.0,
                3 => -0.5,
                _ => 0.0,
            },
            ..Self::default()
        }
    }
    fn ensure_edge_index(&mut self, enabled: bool) {
        const INDEX_THRESHOLD: usize = 16;
        if !enabled || self.edges.len() < INDEX_THRESHOLD || self.edge_index.is_some() {
            return;
        }
        let mut index: Vec<_> = self
            .edges
            .iter()
            .enumerate()
            .map(|(position, edge)| (edge.token, position))
            .collect();
        index.sort_unstable_by_key(|row| row.0);
        self.edge_index = Some(index);
    }
    fn edge_position(&self, token: u16, use_index: bool) -> Option<usize> {
        if use_index {
            if let Some(index) = &self.edge_index {
                return index
                    .binary_search_by_key(&token, |row| row.0)
                    .ok()
                    .map(|position| index[position].1);
            }
        }
        self.edges.iter().position(|edge| edge.token == token)
    }
    fn index_inserted_edge(&mut self, token: u16, position: usize) {
        if let Some(index) = &mut self.edge_index {
            let insertion = index
                .binary_search_by_key(&token, |row| row.0)
                .unwrap_or_else(|position| position);
            index.insert(insertion, (token, position));
        }
    }
    fn observe(
        &mut self,
        token: u16,
        step: u64,
        max_successors: usize,
        capacity_policy: CapacityPolicy,
        use_index: bool,
    ) -> ObserveOutcome {
        self.total = self.total.saturating_add(1);
        self.last_seen = step;
        self.ensure_edge_index(use_index);
        if let Some(position) = self.edge_position(token, use_index) {
            let edge = &mut self.edges[position];
            edge.count = edge.count.saturating_add(1);
            edge.last_seen = step;
            return ObserveOutcome::Updated;
        }
        if self.edges.len() < capacity_policy.limit(self, max_successors) {
            let position = self.edges.len();
            self.edges.push(Edge {
                token,
                count: 1,
                last_seen: step,
                ..Edge::default()
            });
            self.index_inserted_edge(token, position);
            self.ensure_edge_index(use_index);
            return ObserveOutcome::Created;
        }
        ObserveOutcome::RejectedAtCapacity
    }
}


#[derive(Clone, Copy, Default)]
struct FlatNode {
    total: u32,
    last_seen: u64,
    potential: f32,
    edge_start: usize,
    edge_len: usize,
}

#[derive(Clone, Default)]
struct FlatGraph {
    slots: Vec<u32>,
    keys: Vec<ContextKey>,
    nodes: Vec<Node>,
    flat_nodes: Vec<FlatNode>,
    edges: Vec<Edge>,
    sorted_edges: bool,
}

#[derive(Clone, Copy)]
struct NodeView<'a> {
    total: u32,
    potential: f32,
    edges: &'a [Edge],
    edge_index: Option<&'a [(u16, usize)]>,
    sorted_edges: bool,
}

impl<'a> NodeView<'a> {
    fn canonical(node: &'a Node) -> Self {
        Self {
            total: node.total,
            potential: node.potential,
            edges: &node.edges,
            edge_index: node.edge_index.as_deref(),
            sorted_edges: false,
        }
    }
}

impl FlatGraph {
    fn hash(key: ContextKey) -> usize {
        let mut value = key.packed().wrapping_add(0x9e3779b97f4a7c15);
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d049bb133111eb);
        (value ^ (value >> 31)) as usize
    }

    fn rebuild(&mut self, graph: &ContextMap<Node>, mode: u8) {
        self.slots.clear();
        self.keys.clear();
        self.nodes.clear();
        self.flat_nodes.clear();
        self.edges.clear();
        self.sorted_edges = mode >= 4;
        if graph.is_empty() {
            return;
        }
        let mut rows: Vec<_> = graph.iter().collect();
        rows.sort_unstable_by_key(|row| *row.0);
        let capacity = (rows.len() * 2).next_power_of_two().max(8);
        self.slots.resize(capacity, u32::MAX);
        self.keys.reserve(rows.len());
        if mode == 2 {
            self.nodes.reserve(rows.len());
        } else {
            self.flat_nodes.reserve(rows.len());
            self.edges
                .reserve(rows.iter().map(|row| row.1.edges.len()).sum());
        }
        for (dense_id, (&key, node)) in rows.into_iter().enumerate() {
            self.keys.push(key);
            if mode == 2 {
                self.nodes.push(node.clone());
            } else {
                let edge_start = self.edges.len();
                self.edges.extend_from_slice(&node.edges);
                if mode >= 4 {
                    self.edges[edge_start..].sort_unstable_by_key(|edge| edge.token);
                }
                self.flat_nodes.push(FlatNode {
                    total: node.total,
                    last_seen: node.last_seen,
                    potential: node.potential,
                    edge_start,
                    edge_len: node.edges.len(),
                });
            }
            let mut slot = Self::hash(key) & (capacity - 1);
            while self.slots[slot] != u32::MAX {
                slot = (slot + 1) & (capacity - 1);
            }
            self.slots[slot] = dense_id as u32;
        }
    }

    fn dense_id(&self, key: &ContextKey) -> Option<usize> {
        if self.slots.is_empty() {
            return None;
        }
        let mask = self.slots.len() - 1;
        let mut slot = Self::hash(*key) & mask;
        loop {
            let dense = self.slots[slot];
            if dense == u32::MAX {
                return None;
            }
            let dense = dense as usize;
            if self.keys[dense] == *key {
                return Some(dense);
            }
            slot = (slot + 1) & mask;
        }
    }

    fn view(&self, key: &ContextKey, mode: u8) -> Option<NodeView<'_>> {
        let dense = self.dense_id(key)?;
        if mode == 2 {
            return Some(NodeView::canonical(&self.nodes[dense]));
        }
        let node = self.flat_nodes[dense];
        let edges = &self.edges[node.edge_start..node.edge_start + node.edge_len];
        Some(NodeView {
            total: node.total,
            potential: node.potential,
            edges,
            edge_index: None,
            sorted_edges: self.sorted_edges,
        })
    }

    fn bytes(&self) -> usize {
        self.slots.capacity() * std::mem::size_of::<u32>()
            + self.keys.capacity() * std::mem::size_of::<ContextKey>()
            + self.nodes.capacity() * std::mem::size_of::<Node>()
            + self.flat_nodes.capacity() * std::mem::size_of::<FlatNode>()
            + self.edges.capacity() * std::mem::size_of::<Edge>()
    }
}


fn active_successor_limit() -> usize {
    std::env::var("NGU_ACTIVE_SUCCESSORS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(0)
}

fn select_active_edges(node: &Node, limit: usize) -> Vec<Edge> {
    if limit == 0 {
        return Vec::new();
    }
    let mut rows = node.edges.clone();
    rows.sort_unstable_by(|left, right| {
        right
            .count
            .cmp(&left.count)
            .then_with(|| right.last_seen.cmp(&left.last_seen))
            .then_with(|| left.token.cmp(&right.token))
    });
    rows.truncate(limit);
    rows.sort_unstable_by_key(|edge| edge.token);
    rows
}

fn refresh_active_glp_incremental(
    model: &mut Model,
    touched: &ContextMap<Vec<u16>>,
    removed: &[ContextKey],
) {
    if model.active_successor_limit == 0 {
        model.active_glp.clear();
        return;
    }
    for key in removed {
        model.active_glp.remove(key);
    }
    for (key, tokens) in touched {
        let Some(node) = model.glp.get(key) else {
            model.active_glp.remove(key);
            continue;
        };
        let mut candidates = model.active_glp.remove(key).unwrap_or_default();
        for token in tokens {
            let Some(position) = node.edge_position(*token, true) else {
                continue;
            };
            let current = node.edges[position];
            if let Some(active) = candidates
                .iter_mut()
                .find(|edge| edge.token == *token)
            {
                *active = current;
            } else {
                candidates.push(current);
            }
        }
        candidates.sort_unstable_by(|left, right| {
            right
                .count
                .cmp(&left.count)
                .then_with(|| right.last_seen.cmp(&left.last_seen))
                .then_with(|| left.token.cmp(&right.token))
        });
        candidates.truncate(model.active_successor_limit);
        candidates.sort_unstable_by_key(|edge| edge.token);
        model.active_glp.insert(*key, candidates);
    }
}

fn refresh_active_glp_all(model: &mut Model) {
    model.active_glp.clear();
    if model.active_successor_limit == 0 {
        return;
    }
    model.active_glp.reserve(model.glp.len());
    for (key, node) in &model.glp {
        model.active_glp.insert(
            *key,
            select_active_edges(node, model.active_successor_limit),
        );
    }
}

fn refresh_flat_layout(model: &mut Model) {
    if model.layout_mode < 2 {
        model.flat_glp = FlatGraph::default();
        return;
    }
    let timer = Instant::now();
    model.flat_glp.rebuild(&model.glp, model.layout_mode);
    LAYOUT_REBUILD_NS.fetch_add(timer.elapsed().as_nanos() as u64, Ordering::Relaxed);
    LAYOUT_REBUILD_CALLS.fetch_add(1, Ordering::Relaxed);
}

fn model_glp_view<'a>(model: &'a Model, key: &ContextKey) -> Option<NodeView<'a>> {
    if model.layout_mode < 2 {
        let node = model.glp.get(key)?;
        if model.active_successor_limit > 0 {
            if let Some(edges) = model.active_glp.get(key) {
                return Some(NodeView {
                    total: node.total,
                    potential: node.potential,
                    edges,
                    edge_index: None,
                    sorted_edges: true,
                });
            }
        }
        return Some(NodeView::canonical(node));
    }
    let timer = profile_timer();
    let result = model.flat_glp.view(key, model.layout_mode);
    if result.is_some() {
        LAYOUT_LOOKUP_HITS.fetch_add(1, Ordering::Relaxed);
    }
    LAYOUT_LOOKUP_CALLS.fetch_add(1, Ordering::Relaxed);
    if let Some(start) = timer {
        LAYOUT_LOOKUP_NS.fetch_add(start.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }
    result
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CapacityPolicy {
    Fixed,
    Adaptive,
}

impl CapacityPolicy {
    fn parse(value: &str) -> Self {
        match value {
            "fixed" => Self::Fixed,
            "adaptive" => Self::Adaptive,
            _ => panic!("unknown capacity policy {}", value),
        }
    }
    fn name(self) -> &'static str {
        match self {
            Self::Fixed => "fixed",
            Self::Adaptive => "adaptive",
        }
    }
    fn limit(self, node: &Node, maximum: usize) -> usize {
        if self == Self::Fixed {
            return maximum;
        }
        let diversity = node.edges.len();
        let support = node.total;
        let adaptive = if support >= 4096 && diversity >= 256 {
            512
        } else if support >= 1024 && diversity >= 128 {
            256
        } else if support >= 256 && diversity >= 64 {
            128
        } else if support >= 64 && diversity >= 32 {
            64
        } else {
            32
        };
        adaptive.min(maximum)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ObserveOutcome {
    Updated,
    Created,
    RejectedAtCapacity,
}

// A construction is a graph motif with an ordered boundary. `slot_mask` marks
// variable role vertices; symbols at those positions are deliberately zero.
// This representation is modality-agnostic: only the boundary ordering is
// specific to the current 1-D text instantiation.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
struct ConstructionKey {
    len: u8,
    slot_mask: u8,
    symbols: [u16; CONSTRUCTION_INTERFACE],
    target: u16,
}

const DAG_CONSTRUCTION_VERTEX: u8 = CONSTRUCTION_INTERFACE as u8;
const DAG_TARGET_VERTEX: u8 = CONSTRUCTION_INTERFACE as u8 + 1;
const DAG_ROLE_EDGE: u8 = 1;
const DAG_PREDICTION_EDGE: u8 = 2;
const DAG_EDGE_COUNT: usize = CONSTRUCTION_INTERFACE + 1;

#[derive(Clone, Copy, Debug, Default)]
struct DirectedConstructionEdge {
    source: u8,
    target: u8,
    relation: u8,
    weight: f32,
    gradient_sum: f32,
    gradient_sq: f32,
    credit_count: u32,
}

#[derive(Clone, Copy, Debug)]
struct Construction {
    key: ConstructionKey,
    support: u32,
    confirmations: u16,
    value: f32,
    dag_edge_count: u8,
    dag_edges: [DirectedConstructionEdge; DAG_EDGE_COUNT],
}

impl Construction {
    fn new(key: ConstructionKey, support: u32, dag: bool) -> Self {
        let mut row = Self {
            key,
            support,
            confirmations: 0,
            value: 0.0,
            dag_edge_count: 0,
            dag_edges: [DirectedConstructionEdge::default(); DAG_EDGE_COUNT],
        };
        if dag {
            for position in 0..CONSTRUCTION_INTERFACE {
                row.dag_edges[position] = DirectedConstructionEdge {
                    source: position as u8,
                    target: DAG_CONSTRUCTION_VERTEX,
                    relation: DAG_ROLE_EDGE,
                    weight: 1.0,
                    gradient_sum: 0.0,
                    gradient_sq: 0.0,
                    credit_count: 0,
                };
            }
            row.dag_edges[CONSTRUCTION_INTERFACE] = DirectedConstructionEdge {
                source: DAG_CONSTRUCTION_VERTEX,
                target: DAG_TARGET_VERTEX,
                relation: DAG_PREDICTION_EDGE,
                weight: 1.0,
                gradient_sum: 0.0,
                gradient_sq: 0.0,
                credit_count: 0,
            };
            row.dag_edge_count = DAG_EDGE_COUNT as u8;
        }
        row
    }

    fn dag_is_forward_only(&self) -> bool {
        if self.dag_edge_count == 0 {
            return true;
        }
        if self.dag_edge_count as usize != DAG_EDGE_COUNT {
            return false;
        }
        for position in 0..CONSTRUCTION_INTERFACE {
            let edge = self.dag_edges[position];
            if edge.source != position as u8
                || edge.target != DAG_CONSTRUCTION_VERTEX
                || edge.relation != DAG_ROLE_EDGE
                || edge.source == edge.target
            {
                return false;
            }
        }
        let prediction = self.dag_edges[CONSTRUCTION_INTERFACE];
        prediction.source == DAG_CONSTRUCTION_VERTEX
            && prediction.target == DAG_TARGET_VERTEX
            && prediction.relation == DAG_PREDICTION_EDGE
            && prediction.source != prediction.target
    }

    fn forward_gain(&self) -> f64 {
        if self.dag_edge_count == 0 {
            return 1.0;
        }
        if !self.dag_is_forward_only() {
            return 0.0;
        }
        let role = self.dag_edges[..CONSTRUCTION_INTERFACE]
            .iter()
            .map(|edge| edge.weight as f64)
            .sum::<f64>()
            / CONSTRUCTION_INTERFACE as f64;
        let prediction = self.dag_edges[CONSTRUCTION_INTERFACE].weight as f64;
        (role * prediction).clamp(0.0, 1.50)
    }
}

#[derive(Clone, Default)]
struct CreditNode {
    rows: Vec<(u16, EdgeCredit)>,
}

impl CreditNode {
    fn get(&self, token: &u16) -> Option<&EdgeCredit> {
        self.rows
            .binary_search_by_key(token, |row| row.0)
            .ok()
            .map(|position| &self.rows[position].1)
    }
    fn get_or_insert_default(&mut self, token: u16) -> &mut EdgeCredit {
        let position = self
            .rows
            .binary_search_by_key(&token, |row| row.0)
            .unwrap_or_else(|position| {
                self.rows.insert(position, (token, EdgeCredit::default()));
                position
            });
        &mut self.rows[position].1
    }
    fn get_or_insert_ordered(
        &mut self,
        token: u16,
        cursor: &mut usize,
    ) -> &mut EdgeCredit {
        while *cursor < self.rows.len() && self.rows[*cursor].0 < token {
            *cursor += 1;
        }
        if *cursor == self.rows.len() || self.rows[*cursor].0 != token {
            self.rows.insert(*cursor, (token, EdgeCredit::default()));
        }
        let position = *cursor;
        *cursor += 1;
        &mut self.rows[position].1
    }
}


#[derive(Clone, Copy, Eq, Hash, PartialEq, Ord, PartialOrd)]
struct MedialDreamKey {
    context: ContextKey,
    target: u16,
}

#[derive(Clone)]
struct MedialDreamCandidate {
    key: MedialDreamKey,
    recent_top: u16,
    medial_top: u16,
    long_top: u16,
    age: u8,
    supported_windows: u8,
    adverse_windows: u8,
    evaluated: u32,
    credit_sum: f64,
}


const EXP170_SHARDS: usize = 8;
const EXP170_DIRECT_ROUTES: usize = 200_000;

#[derive(Clone, Copy, Eq, Hash, PartialEq, Ord, PartialOrd)]
struct RouteSignature {
    len: u8,
    themes: [u8; 4],
}

#[derive(Clone)]
struct RouteMediator {
    signature: RouteSignature,
    votes: [u32; EXP170_SHARDS],
    top: u8,
    confirmations: u8,
}

type RouteMap<V> = HashMap<
    RouteSignature,
    V,
    std::hash::BuildHasherDefault<ContextHasher>,
>;

#[derive(Clone)]
struct Model {
    vocab_size: usize,
    max_order: usize,
    gcp_tokens: Vec<TokenVertex>,
    gm1_tokens: Vec<TokenVertex>,
    glp_tokens: Vec<TokenVertex>,
    gcp: ContextMap<Node>,
    gm1: ContextMap<Node>,
    // Composite subgraphs refreshed only during sleep. They contain the GM1
    // relation plus its matching GLP evidence, avoiding an online GLP read.
    gm1_cache: ContextMap<Node>,
    glp: ContextMap<Node>,
    layout_mode: u8,
    flat_glp: FlatGraph,
    // Runtime-only sparse activation view. The authoritative GLP and its
    // checkpoint retain every learned successor.
    active_successor_limit: usize,
    active_glp: ContextMap<Vec<Edge>>,
    gcp_credit: ContextMap<CreditNode>,
    glp_constructions: Vec<Construction>,
    // Runtime-only inference policy. Checkpoints contain learned structure,
    // while the command line chooses where those weights participate.
    edge_application: u8,
    // Runtime-only generation of the authoritative GLP. It replaces full
    // wake-boundary scans without entering the learned state or checkpoint.
    glp_version: u64,
    // 0 baseline, 1 observer, 2 active controlled, 3 active additive,
    // 4 active additive with tri-hierarchical dream.
    gm1_mode: u8,
    // 0 composed control, 1 semantic-only, 2 immediate dream,
    // 3 causal dream, 4 causal dream with composed fallback.
    medial_mode: u8,
    medial_pending: Vec<MedialDreamCandidate>,
    // 0 monolithic; 1 GCP local; 2 GCP reconstructed; 3 GM1 direct;
    // 4 GM1 abstract top-1; 5 GM1 abstract top-2 plus fallback.
    shard_mode: u8,
    // Physical shards and the GCP reconstruction are runtime-only.
    glp_shards: Vec<ContextMap<Node>>,
    token_themes: Vec<u8>,
    gcp_routes: ContextMap<u8>,
    gcp_route_mediators: RouteMap<RouteMediator>,
    // GM1 routing edges are persistent learned graph state.
    gm1_routes: ContextMap<u8>,
    gm1_route_mediators: RouteMap<RouteMediator>,
    // EXP171 runtime index. Themes are serialized as compact shard
    // descriptors; physical shard clones and GCP routes are reconstructible.
    shard_assignments: ContextMap<u8>,
    route_support: ContextMap<u32>,
    route_order: BTreeSet<(std::cmp::Reverse<u32>, ContextKey)>,
    compact_routes: Vec<(ContextKey, u8)>,
    // EXP172: views contain membership only; the authoritative Node remains
    // in GLP. The eviction index reproduces the exact global pruning order.
    glp_shard_views: Vec<ContextMap<()>>,
    eviction_rank: ContextMap<(u32, u64)>,
    eviction_order: BTreeSet<(u32, u64, ContextKey)>,
}

#[derive(Clone, Copy)]
struct Options {
    vocab_size: usize,
    max_order: usize,
    train_limit: usize,
    gcp_max: usize,
    gm1_max: usize,
    glp_max: usize,
    max_successors: usize,
    capacity_policy: CapacityPolicy,
    successor_index: bool,
    light_interval: usize,
    deep_interval: usize,
    wake_interval: usize,
    lr: f64,
    sleep_policy: SleepPolicy,
}

#[derive(Default)]
struct TrainStats {
    tokens: usize,
    predictions: usize,
    nll: f64,
    created_edges: u64,
    edge_updates: u64,
    conflicts: u64,
    light_reflections: usize,
    deep_reflections: usize,
    pruned_gcp: u64,
    pruned_gm1: u64,
    pruned_glp: u64,
    gcp_to_gm1_contexts: u64,
    gm1_to_glp_contexts: u64,
    gm1_dream_candidates: u64,
    gm1_dream_reinforced: u64,
    gm1_dream_inhibited: u64,
    gm1_peak: usize,
    gm1_cache_refreshes: u64,
    gm1_cache_contexts: u64,
    gm1_cache_peak: usize,
    medial_dream_proposed: u64,
    medial_dream_immediate: u64,
    medial_dream_future_windows: u64,
    medial_dream_supported_windows: u64,
    medial_dream_adverse_windows: u64,
    medial_dream_materialized: u64,
    medial_dream_created_edges: u64,
    medial_dream_reinforced_edges: u64,
    medial_dream_expired: u64,
    medial_dream_pending_peak: usize,
    medial_dream_credit_sum: f64,
    shard_rebuilds: u64,
    shard_contexts_materialized: u64,
    route_mediators_proposed: u64,
    route_mediators_confirmed: u64,
    route_mediators_reset: u64,
    route_direct_entries: u64,
    incremental_initial_splits: u64,
    incremental_refreshes: u64,
    incremental_journal_keys: u64,
    incremental_contexts_examined: u64,
    incremental_migrations: u64,
    incremental_removals: u64,
    incremental_full_scanned_contexts: u64,
    compact_route_rebuilds: u64,
    eviction_index_builds: u64,
    eviction_rank_updates: u64,
    indexed_prune_calls: u64,
    indexed_evictions: u64,
    global_prune_calls: u64,
    global_prune_contexts_scanned: u64,
    victim_events: u64,
    victim_fingerprint: u64,
    view_memberships_written: u64,
    successor_capacity_rejections: u64,
    successor_capacity_rejections_by_order: [u64; 4],
    sleep_successor_capacity_rejections: u64,
    max_gcp: usize,
    gradient_audit_max_abs: f64,
    sleeps_attempted: usize,
    sleeps_accepted: usize,
    sleeps_rolled_back: usize,
    sleep_passes: usize,
    replay_updates: u64,
    wake_integrity_checks: usize,
    wake_glp_mutation_violations: usize,
    sleep_examples: u64,
    sleep_gold_candidates_before_feedback: u64,
    sleep_full_gold_candidates_before_feedback: u64,
    sleep_transaction_edges: u64,
    sleep_candidate_traces_before_feedback: u64,
    counterfactual_interventions: u64,
    cf_gold_edges_tested: u64,
    cf_gold_edges_causal: u64,
    cf_wrong_edges_tested: u64,
    cf_wrong_edges_causal: u64,
    bootstrap_updates: u64,
    contrast_edges_tested: u64,
    contrast_gold_favored: u64,
    contrast_wrong_favored: u64,
    contrast_neutral: u64,
    contrast_gold_necessity_sum: f64,
    contrast_wrong_necessity_sum: f64,
    contrast_abs_difference_sum: f64,
    bridge_attempts: u64,
    bridge_found: u64,
    bridge_full_coverage: u64,
    bridge_edges_credited: u64,
    bridge_coverage_sum: f64,
    bridge_dispersion_sum: f64,
    bridge_cost_sum: f64,
    bridge_validations: u64,
    bridge_validated: u64,
    bridge_rejected_no_gold: u64,
    bridge_rejected_margin: u64,
    bridge_rejected_wrong_promotion: u64,
    prospective_gold_bonus_sum: f64,
    prospective_margin_sum: f64,
    prospective_wrong_promotion_sum: f64,
    multidose_virtual_traces: u64,
    multidose_dose1_pass: u64,
    multidose_dose2_pass: u64,
    multidose_dose4_pass: u64,
    multidose_rejected: u64,
    multidose_monotonic_rejected: u64,
    pending_enqueued: u64,
    pending_revalidated: u64,
    pending_recredited: u64,
    pending_resolved: u64,
    pending_expired: u64,
    pending_peak: usize,
    phase_attempts: u64,
    phase_full_coverage_attempts: u64,
    phase_missing_coverage_attempts: u64,
    phase_vector_edges: u64,
    phase_accepted: u64,
    phase_full_created: u64,
    phase_rejected_gold_lost: u64,
    phase_rejected_coverage: u64,
    phase_rejected_no_improvement: u64,
    phase_rejected_bonus: u64,
    phase_rejected_wrong_promotion: u64,
    phase_dispersion_before_sum: f64,
    phase_dispersion_after_sum: f64,
    phase_abs_delta_sum: f64,
    origin_attempts: u64,
    origin_paths_found: u64,
    origin_validated: u64,
    origin_full_created: u64,
    origin_dose1_pass: u64,
    origin_dose2_pass: u64,
    origin_dose4_pass: u64,
    origin_rejected_no_gain: u64,
    origin_edges_credited: u64,
    sequential_semantic_edges: u64,
    sequential_origin_edges: u64,
    sequential_phase_edges: u64,
    origin_bank_enqueued: u64,
    origin_bank_revalidated: u64,
    origin_bank_recredited: u64,
    origin_bank_incorporated: u64,
    origin_bank_expired: u64,
    origin_bank_peak: usize,
    route_set_groups: u64,
    route_set_candidates: u64,
    route_set_states_evaluated: u64,
    route_set_selected_groups: u64,
    route_set_selected_routes: u64,
    route_set_selected_edges: u64,
    route_set_full_created: u64,
    route_set_revalidated_groups: u64,
    route_set_rejected_no_joint_gain: u64,
    route_set_negative_interactions: u64,
    route_set_nonnegative_interactions: u64,
    route_set_joint_gain_sum: f64,
    route_set_individual_gain_sum: f64,
    route_set_interference_sum: f64,
    route_set_selected_bonus_sum: f64,
    route_set_selected_margin_sum: f64,
    route_set_wrong_promotion_sum: f64,
    deferred_proposed: u64,
    deferred_evidence_windows: u64,
    deferred_future_examples_screened: u64,
    deferred_future_examples_relevant: u64,
    deferred_overlay_evaluations: u64,
    deferred_supported_windows: u64,
    deferred_adverse_windows: u64,
    deferred_neutral_windows: u64,
    deferred_matured: u64,
    deferred_committed: u64,
    deferred_expired: u64,
    deferred_route_commits: u64,
    deferred_edge_commits: u64,
    deferred_quarantine_peak: usize,
    deferred_gold_delta_sum: f64,
    deferred_full_delta_sum: f64,
    deferred_coverage_delta_sum: f64,
    deferred_margin_delta_sum: f64,
    deferred_wrong_promotion_sum: f64,
    deferred_value_sum: f64,
    deferred_journal_edges: u64,
    metagraph_pairs_evaluated: u64,
    metagraph_synergy: u64,
    metagraph_conflict: u64,
    metagraph_neutral: u64,
    metagraph_interaction_sum: f64,
    construction_candidates: u64,
    construction_proposed: u64,
    construction_fixed_proposed: u64,
    construction_slot_proposed: u64,
    construction_slot_fillings: u64,
    construction_future_matches: u64,
    construction_evidence_windows: u64,
    construction_supported_windows: u64,
    construction_adverse_windows: u64,
    construction_neutral_windows: u64,
    construction_matured: u64,
    construction_committed: u64,
    construction_immediate_committed: u64,
    construction_expired: u64,
    construction_quarantine_peak: usize,
    construction_glp_peak: usize,
    construction_gold_delta_sum: f64,
    construction_full_delta_sum: f64,
    construction_coverage_delta_sum: f64,
    construction_margin_delta_sum: f64,
    construction_wrong_promotion_sum: f64,
    construction_value_sum: f64,
    directional_edge_tests: u64,
    directional_role_edge_tests: u64,
    directional_prediction_edge_tests: u64,
    directional_edge_positive: u64,
    directional_edge_negative: u64,
    directional_edge_neutral: u64,
    directional_edge_value_sum: f64,
}

#[derive(Clone, Copy)]
struct EdgeAggregate {
    token: u16,
    count: u32,
    weight: f64,
    gradient_sum: f64,
    gradient_sq: f64,
    conflicts: u32,
    last_seen: u64,
    sleep_bias: f64,
}

impl EdgeAggregate {
    fn merge(&mut self, edge: &Edge) {
        let old = self.count as f64;
        let add = edge.count as f64;
        let denom = (old + add).max(1.0);
        self.weight = (self.weight * old + edge.weight as f64 * add) / denom;
        self.sleep_bias = (self.sleep_bias * old + edge.sleep_bias as f64 * add) / denom;
        self.count = self.count.saturating_add(edge.count);
        self.gradient_sum += edge.gradient_sum as f64;
        self.gradient_sq += edge.gradient_sq as f64;
        self.conflicts = self.conflicts.saturating_add(edge.conflicts);
        self.last_seen = self.last_seen.max(edge.last_seen);
    }
}

#[derive(Clone, Copy)]
struct Feature {
    key: ContextKey,
    token: u16,
    relation: f64,
    gate: f64,
    probability: f64,
}

#[derive(Clone)]
struct WaveState {
    token: u16,
    signal: f64,
    arrival: f64,
    cost: f64,
    conflicts: f64,
    path: [u16; WAVE_DEPTH + 1],
    path_len: u8,
}

#[derive(Clone, Default)]
struct CandidateGr {
    token: u16,
    coverage: usize,
    full: bool,
    dispersion: f64,
    mean_cost: f64,
    mean_signal: f64,
    conflict: f64,
    energy: f64,
    bonus: f64,
    support_edges: Vec<(u16, u16)>,
    phase_deltas: Vec<((u16, u16), f64)>,
    origin_present: [bool; WAVE_WINDOW],
}

#[derive(Clone, Default)]
struct WaveTrace {
    candidates: Vec<CandidateGr>,
    full: bool,
    max_coverage: f64,
    early_active_constructions: usize,
    early_modified_transitions: usize,
    early_route_reorders: usize,
    early_gr_adjustments: usize,
}

#[derive(Default)]
struct EvalStats {
    predictions: usize,
    nll: f64,
    unigram_nll: f64,
    top1: usize,
    top5: usize,
    wave_predictions: usize,
    wave_candidate_sum: u64,
    wave_full: usize,
    wave_coverage_sum: f64,
    gold_candidate: usize,
    gold_full_candidate: usize,
    dispersion_sum: f64,
    cost_sum: f64,
    signal_sum: f64,
    conflict_sum: f64,
    energy_sum: f64,
    candidate_count: u64,
    early_active_constructions: u64,
    early_modified_transitions: u64,
    early_route_reorders: u64,
    early_gr_adjustments: u64,
    paired_predictions: u64,
    paired_top1_changed: u64,
    paired_gold_rank_improved: u64,
    paired_gold_rank_worsened: u64,
    paired_gold_rank_equal: u64,
    paired_gold_margin_delta_sum: f64,
    paired_gold_probability_delta_sum: f64,
    paired_gr_set_changed: u64,
    context_queries_by_order: [u64; 4],
    context_hits_by_order: [u64; 4],
    frequency_bin_predictions: [u64; 3],
    frequency_bin_top1: [u64; 3],
    frequency_bin_nll: [f64; 3],
    correct_gr_margin_sum: f64,
    correct_gr_margin_count: u64,
}

#[derive(Default)]
struct GenerationStats {
    continuations: Vec<Vec<u16>>,
    baseline_peak: u64,
    gr_peak: u64,
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn make_key(history: &[u16], order: usize) -> ContextKey {
    let mut tokens = [0u16; 4];
    for (i, &token) in history[history.len() - order..].iter().enumerate() {
        tokens[i] = token;
    }
    ContextKey::new(order as u8, tokens)
}

fn combined_token(model: &Model, token: usize) -> TokenVertex {
    TokenVertex {
        count: model.gcp_tokens[token]
            .count
            .saturating_add(if model.gm1_mode >= 2 {
                model.gm1_tokens[token].count
            } else {
                0
            })
            .saturating_add(model.glp_tokens[token].count),
        potential: if model.gcp_tokens[token].count > 0 {
            model.gcp_tokens[token].potential
        } else if model.gm1_mode >= 2 && model.gm1_tokens[token].count > 0 {
            model.gm1_tokens[token].potential
        } else {
            model.glp_tokens[token].potential
        },
    }
}

fn base_weights(model: &Model) -> (Vec<f64>, f64, Vec<u16>) {
    let mut values = Vec::with_capacity(model.vocab_size);
    let mut total = 0.0;
    for token in 0..model.vocab_size {
        let v = combined_token(model, token);
        let w = (v.count as f64 + ALPHA) * (v.potential as f64).clamp(-4.0, 4.0).exp();
        values.push(w);
        total += w;
    }
    let mut ids: Vec<usize> = (0..model.vocab_size).collect();
    ids.sort_unstable_by(|&a, &b| {
        values[b]
            .partial_cmp(&values[a])
            .unwrap()
            .then_with(|| a.cmp(&b))
    });
    (
        values,
        total,
        ids.into_iter().take(64).map(|x| x as u16).collect(),
    )
}

struct EdgeOrder<'a> {
    view: NodeView<'a>,
    owned: Option<Vec<(u16, usize)>>,
}

impl<'a> EdgeOrder<'a> {
    fn new(view: NodeView<'a>) -> Self {
        let owned = if view.sorted_edges || view.edge_index.is_some() {
            None
        } else {
            let mut rows: Vec<_> = view
                .edges
                .iter()
                .enumerate()
                .map(|(position, edge)| (edge.token, position))
                .collect();
            rows.sort_unstable_by_key(|row| row.0);
            Some(rows)
        };
        Self { view, owned }
    }
    fn len(&self) -> usize {
        self.view.edges.len()
    }
    fn get(&self, position: usize) -> Option<&'a Edge> {
        if position >= self.len() {
            return None;
        }
        if self.view.sorted_edges {
            return self.view.edges.get(position);
        }
        let index = self
            .view
            .edge_index
            .map(|rows| rows[position].1)
            .or_else(|| self.owned.as_ref().map(|rows| rows[position].1))?;
        self.view.edges.get(index)
    }
}



fn edge_aggregate(edge: &Edge) -> EdgeAggregate {
    EdgeAggregate {
        token: edge.token,
        count: edge.count,
        weight: edge.weight as f64,
        gradient_sum: edge.gradient_sum as f64,
        gradient_sq: edge.gradient_sq as f64,
        conflicts: edge.conflicts,
        last_seen: edge.last_seen,
        sleep_bias: edge.sleep_bias as f64,
    }
}


fn recent_resolves(node: &Node) -> bool {
    if node.total < 8 {
        return false;
    }
    let dominant = node.edges.iter().map(|edge| edge.count).max().unwrap_or(0);
    dominant as f64 / node.total.max(1) as f64 >= 0.55
}

fn medial_resolves(key: &ContextKey, node: &Node) -> bool {
    let dominant = node.edges.iter().map(|edge| edge.count).max().unwrap_or(0);
    let ratio = dominant as f64 / node.total.max(1) as f64;
    (key.len() >= 3 && ratio >= 0.50) || (node.total >= 4 && ratio >= 0.40)
}


fn top_shard(votes: &[u32; EXP170_SHARDS]) -> (u8, u32, u32) {
    let mut rows: Vec<(u32, u8)> = votes
        .iter()
        .enumerate()
        .map(|(shard, &count)| (count, shard as u8))
        .collect();
    rows.sort_unstable_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
    (rows[0].1, rows[0].0, rows[1].0)
}

fn learn_token_themes(model: &Model) -> Vec<u8> {
    let mut frequency = vec![0u64; model.vocab_size];
    for node in model.glp.values() {
        for edge in &node.edges {
            if (edge.token as usize) < frequency.len() {
                frequency[edge.token as usize] =
                    frequency[edge.token as usize].saturating_add(edge.count as u64);
            }
        }
    }
    let mut hubs: Vec<usize> = (0..model.vocab_size).collect();
    hubs.sort_unstable_by(|&a, &b| {
        frequency[b].cmp(&frequency[a]).then_with(|| a.cmp(&b))
    });
    hubs.truncate(EXP170_SHARDS.min(hubs.len()));
    let mut labels = vec![u8::MAX; model.vocab_size];
    for (theme, token) in hubs.into_iter().enumerate() {
        labels[token] = theme as u8;
    }
    for _ in 0..4 {
        let old = labels.clone();
        for (key, node) in &model.glp {
            if key.len() != 1 {
                continue;
            }
            let source = key.tokens()[0] as usize;
            if source >= labels.len() || labels[source] < EXP170_SHARDS as u8 {
                continue;
            }
            let mut votes = [0u64; EXP170_SHARDS];
            for edge in &node.edges {
                let target = edge.token as usize;
                if target < old.len() && old[target] < EXP170_SHARDS as u8 {
                    let theme = old[target] as usize;
                    votes[theme] = votes[theme].saturating_add(edge.count as u64);
                }
            }
            if let Some((theme, &weight)) = votes
                .iter()
                .enumerate()
                .max_by_key(|(theme, weight)| (**weight, std::cmp::Reverse(*theme)))
            {
                if weight > 0 {
                    labels[source] = theme as u8;
                }
            }
        }
    }
    for (token, label) in labels.iter_mut().enumerate() {
        if *label >= EXP170_SHARDS as u8 {
            *label = (token % EXP170_SHARDS) as u8;
        }
    }
    labels
}

fn node_shard(node: &Node, themes: &[u8]) -> u8 {
    node.edges
        .iter()
        .max_by_key(|edge| (edge.count, std::cmp::Reverse(edge.token)))
        .map(|edge| {
            themes
                .get(edge.token as usize)
                .copied()
                .unwrap_or((edge.token as usize % EXP170_SHARDS) as u8)
        })
        .unwrap_or(0)
}

fn route_signature(key: &ContextKey, themes: &[u8]) -> RouteSignature {
    let mut signature = RouteSignature {
        len: key.len(),
        themes: [0; 4],
    };
    for index in 0..key.len() as usize {
        let token = key.tokens()[index] as usize;
        signature.themes[index] = themes
            .get(token)
            .copied()
            .unwrap_or((token % EXP170_SHARDS) as u8);
    }
    signature
}

fn build_partitions(
    model: &mut Model,
) -> (RouteMap<[u32; EXP170_SHARDS]>, Vec<(ContextKey, u32, u8)>) {
    model.token_themes = learn_token_themes(model);
    model.glp_shards = (0..EXP170_SHARDS)
        .map(|_| ContextMap::default())
        .collect();
    let mut signatures: RouteMap<[u32; EXP170_SHARDS]> = RouteMap::default();
    let mut direct = Vec::with_capacity(model.glp.len());
    for (key, node) in &model.glp {
        let shard = node_shard(node, &model.token_themes);
        model.glp_shards[shard as usize].insert(*key, node.clone());
        let signature = route_signature(key, &model.token_themes);
        signatures.entry(signature).or_insert([0; EXP170_SHARDS])
            [shard as usize] = signatures
            .get(&signature)
            .map(|row| row[shard as usize])
            .unwrap_or(0)
            .saturating_add(node.total.max(1));
        direct.push((*key, node.total, shard));
    }
    direct.sort_unstable_by(|a, b| {
        b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0))
    });
    (signatures, direct)
}

fn current_mediators(
    signatures: &RouteMap<[u32; EXP170_SHARDS]>,
) -> RouteMap<RouteMediator> {
    let mut rows = RouteMap::default();
    for (&signature, &votes) in signatures {
        let (top, first, second) = top_shard(&votes);
        let total: u32 = votes.iter().copied().sum();
        if total >= 4 && first > second {
            rows.insert(
                signature,
                RouteMediator {
                    signature,
                    votes,
                    top,
                    confirmations: 1,
                },
            );
        }
    }
    rows
}




fn exp173_full_directory(mode: u8) -> bool {
    matches!(mode, 13 | 14 | 15)
}

fn exp173_owner_storage(mode: u8) -> bool {
    matches!(mode, 14 | 15)
}

fn exp172_indexed_forgetting(mode: u8) -> bool {
    matches!(mode, 10 | 12)
}

fn exp172_canonical_views(mode: u8) -> bool {
    matches!(mode, 11 | 12 | 13)
}

fn update_victim_fingerprint(stats: &mut TrainStats, removed: &[ContextKey]) {
    let mut keys = removed.to_vec();
    keys.sort_unstable();
    let mut hash = if stats.victim_fingerprint == 0 {
        0xcbf29ce484222325u64
    } else {
        stats.victim_fingerprint
    };
    hash ^= keys.len() as u64;
    hash = hash.wrapping_mul(0x100000001b3);
    for key in keys {
        hash ^= key.len() as u64;
        hash = hash.wrapping_mul(0x100000001b3);
        for token in key.tokens() {
            hash ^= token as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        }
    }
    stats.victim_fingerprint = hash;
    stats.victim_events += 1;
}

fn rebuild_eviction_index(model: &mut Model, stats: &mut TrainStats) {
    model.eviction_rank.clear();
    model.eviction_order.clear();
    for (key, node) in &model.glp {
        model.eviction_rank.insert(*key, (node.total, node.last_seen));
        model
            .eviction_order
            .insert((node.total, node.last_seen, *key));
    }
    stats.eviction_index_builds += 1;
}

fn update_eviction_index(
    model: &mut Model,
    changed: &[ContextKey],
    stats: &mut TrainStats,
) {
    if model.eviction_rank.is_empty() && !model.glp.is_empty() {
        rebuild_eviction_index(model, stats);
        return;
    }
    let mut journal = BTreeSet::new();
    journal.extend(changed.iter().copied());
    for key in journal {
        if let Some((total, last_seen)) = model.eviction_rank.remove(&key) {
            model.eviction_order.remove(&(total, last_seen, key));
        }
        if let Some(node) = model.glp.get(&key) {
            model.eviction_rank.insert(key, (node.total, node.last_seen));
            model
                .eviction_order
                .insert((node.total, node.last_seen, key));
        }
        stats.eviction_rank_updates += 1;
    }
}

fn prune_graph_indexed(
    model: &mut Model,
    maximum: usize,
    stats: &mut TrainStats,
) -> Vec<ContextKey> {
    stats.indexed_prune_calls += 1;
    if model.glp.len() <= maximum {
        return Vec::new();
    }
    let remove = model.glp.len() - maximum;
    let victims: Vec<_> = model
        .eviction_order
        .iter()
        .take(remove)
        .map(|row| row.2)
        .collect();
    for key in &victims {
        let rank = model
            .eviction_rank
            .remove(key)
            .expect("indexed victim must have a rank");
        model.eviction_order.remove(&(rank.0, rank.1, *key));
        model.glp.remove(key).expect("indexed victim must exist in GLP");
    }
    stats.indexed_evictions += victims.len() as u64;
    victims
}

const EXP171_SPLIT_CONTEXTS: usize = 250_000;

fn sync_exp171_shard_node(model: &mut Model, key: ContextKey) {
    if !matches!(model.shard_mode, 6 | 7 | 8 | 10)
        || model.glp_shards.len() != EXP170_SHARDS
    {
        return;
    }
    let shard = model.shard_assignments.get(&key).copied();
    let node = model.glp.get(&key).cloned();
    if let (Some(shard), Some(node)) = (shard, node) {
        model.glp_shards[shard as usize].insert(key, node);
    }
}

fn install_exp171_routes(model: &mut Model, stats: &mut TrainStats) {
    let route_limit = if exp173_full_directory(model.shard_mode) {
        usize::MAX
    } else {
        EXP170_DIRECT_ROUTES
    };
    let selected: Vec<_> = model
        .route_order
        .iter()
        .take(route_limit)
        .map(|(_, key)| {
            let shard = *model
                .shard_assignments
                .get(key)
                .expect("ranked EXP171 route assignment");
            (*key, shard)
        })
        .collect();
    match model.shard_mode {
        6 | 9 | 10 | 11 | 12 | 13 | 14 | 15 => {
            model.gcp_routes.clear();
            model.gcp_routes.reserve(selected.len());
            model.gcp_routes.extend(selected);
            stats.route_direct_entries += model.gcp_routes.len() as u64;
        }
        7 => {
            model.gm1_routes.clear();
            model.gm1_routes.reserve(selected.len());
            model.gm1_routes.extend(selected);
            stats.route_direct_entries += model.gm1_routes.len() as u64;
        }
        8 => {
            model.compact_routes = selected;
            model.compact_routes.sort_unstable_by_key(|row| row.0);
            stats.compact_route_rebuilds += 1;
            stats.route_direct_entries += model.compact_routes.len() as u64;
        }
        _ => {}
    }
}

fn full_exp171_partition(model: &mut Model, stats: &mut TrainStats) {
    if model.token_themes.len() != model.vocab_size {
        if model.glp.len() < EXP171_SPLIT_CONTEXTS {
            model.glp_shards.clear();
            model.glp_shard_views.clear();
            return;
        }
        model.token_themes = learn_token_themes(model);
        stats.incremental_initial_splits += 1;
    }
    model.glp_shards = (0..EXP170_SHARDS)
        .map(|_| ContextMap::default())
        .collect();
    model.glp_shard_views = if exp172_canonical_views(model.shard_mode) {
        (0..EXP170_SHARDS)
            .map(|_| ContextMap::default())
            .collect()
    } else {
        Vec::new()
    };
    model.shard_assignments.clear();
    model.route_support.clear();
    model.route_order.clear();
    for (key, node) in &model.glp {
        let shard = node_shard(node, &model.token_themes);
        if exp172_canonical_views(model.shard_mode) {
            model.glp_shard_views[shard as usize].insert(*key, ());
            stats.view_memberships_written += 1;
        } else if !exp173_owner_storage(model.shard_mode) {
            model.glp_shards[shard as usize].insert(*key, node.clone());
        }
        model.shard_assignments.insert(*key, shard);
        model.route_support.insert(*key, node.total);
        model
            .route_order
            .insert((std::cmp::Reverse(node.total), *key));
    }
    stats.shard_rebuilds += 1;
    stats.shard_contexts_materialized += model.glp.len() as u64;
    stats.incremental_full_scanned_contexts += model.glp.len() as u64;
    install_exp171_routes(model, stats);
}

fn incremental_exp171_partition(
    model: &mut Model,
    changed: &[ContextKey],
    removed: &[ContextKey],
    stats: &mut TrainStats,
) {
    if model.glp_shards.len() != EXP170_SHARDS {
        full_exp171_partition(model, stats);
        return;
    }
    stats.incremental_refreshes += 1;
    let mut journal = BTreeSet::new();
    journal.extend(changed.iter().copied());
    journal.extend(removed.iter().copied());
    stats.incremental_journal_keys += journal.len() as u64;
    for key in journal {
        stats.incremental_contexts_examined += 1;
        let old_shard = model.shard_assignments.remove(&key);
        if let Some(shard) = old_shard {
            if exp172_canonical_views(model.shard_mode) {
                model.glp_shard_views[shard as usize].remove(&key);
            } else if !exp173_owner_storage(model.shard_mode) {
                model.glp_shards[shard as usize].remove(&key);
            }
        }
        if let Some(old_support) = model.route_support.remove(&key) {
            model
                .route_order
                .remove(&(std::cmp::Reverse(old_support), key));
        }
        if let Some(node) = model.glp.get(&key) {
            let shard = node_shard(node, &model.token_themes);
            if old_shard.map(|old| old != shard).unwrap_or(false) {
                stats.incremental_migrations += 1;
            }
            if exp172_canonical_views(model.shard_mode) {
                model.glp_shard_views[shard as usize].insert(key, ());
                stats.view_memberships_written += 1;
            } else if !exp173_owner_storage(model.shard_mode) {
                model.glp_shards[shard as usize].insert(key, node.clone());
            }
            model.shard_assignments.insert(key, shard);
            model.route_support.insert(key, node.total);
            model
                .route_order
                .insert((std::cmp::Reverse(node.total), key));
        } else if old_shard.is_some() {
            stats.incremental_removals += 1;
        }
    }
    install_exp171_routes(model, stats);
}

fn refresh_exp171_routing_inner(
    model: &mut Model,
    changed: &[ContextKey],
    removed: &[ContextKey],
    stats: &mut TrainStats,
) {
    match model.shard_mode {
        9 => full_exp171_partition(model, stats),
        6 | 7 | 8 | 10 | 11 | 12 | 13 | 14 | 15 => {
            incremental_exp171_partition(model, changed, removed, stats)
        }
        _ => refresh_exp170_routing(model, stats),
    }
}


fn refresh_exp171_routing(
    model: &mut Model,
    changed: &[ContextKey],
    removed: &[ContextKey],
    stats: &mut TrainStats,
) {
    let timer = profile_timer();
    refresh_exp171_routing_inner(model, changed, removed, stats);
    profile_finish(timer, &PROFILE_ROUTE_REFRESH_NS, &PROFILE_ROUTE_REFRESH_CALLS);
}

fn initialize_exp171_after_load(model: &mut Model) {
    if !matches!(model.shard_mode, 6..=15) {
        initialize_exp170_after_load(model);
        return;
    }
    let saved_routes = std::mem::take(&mut model.gm1_routes);
    let mut ignored = TrainStats::default();
    full_exp171_partition(model, &mut ignored);
    if model.shard_mode == 7 && !saved_routes.is_empty() {
        model.gm1_routes = saved_routes;
    }
}

fn compact_route(model: &Model, key: &ContextKey) -> Option<u8> {
    model
        .compact_routes
        .binary_search_by_key(key, |row| row.0)
        .ok()
        .map(|position| model.compact_routes[position].1)
}

fn refresh_exp170_routing(model: &mut Model, stats: &mut TrainStats) {
    if model.shard_mode == 0 {
        return;
    }
    let (signatures, direct) = build_partitions(model);
    stats.shard_rebuilds += 1;
    stats.shard_contexts_materialized += model.glp.len() as u64;
    match model.shard_mode {
        2 => {
            model.gcp_routes.clear();
            for (key, _, shard) in direct.into_iter().take(EXP170_DIRECT_ROUTES) {
                model.gcp_routes.insert(key, shard);
            }
            stats.route_direct_entries += model.gcp_routes.len() as u64;
        }
        3 => {
            model.gm1_routes.clear();
            for (key, _, shard) in direct.into_iter().take(EXP170_DIRECT_ROUTES) {
                model.gm1_routes.insert(key, shard);
            }
            stats.route_direct_entries += model.gm1_routes.len() as u64;
        }
        4 | 5 => {
            let observed = current_mediators(&signatures);
            stats.route_mediators_proposed += observed.len() as u64;
            let mut next = RouteMap::default();
            for (signature, mut current) in observed {
                if let Some(old) = model.gm1_route_mediators.get(&signature) {
                    if old.top == current.top {
                        current.confirmations = old.confirmations.saturating_add(1);
                    } else {
                        stats.route_mediators_reset += 1;
                    }
                }
                if current.confirmations >= 2 {
                    stats.route_mediators_confirmed += 1;
                }
                next.insert(signature, current);
            }
            model.gm1_route_mediators = next;
        }
        _ => {}
    }
}

fn initialize_exp170_after_load(model: &mut Model) {
    if model.shard_mode == 0 {
        return;
    }
    let saved_routes = std::mem::take(&mut model.gm1_routes);
    let saved_mediators = std::mem::take(&mut model.gm1_route_mediators);
    let mut ignored = TrainStats::default();
    refresh_exp170_routing(model, &mut ignored);
    if model.shard_mode == 3 && !saved_routes.is_empty() {
        model.gm1_routes = saved_routes;
    }
    if model.shard_mode >= 4 && !saved_mediators.is_empty() {
        model.gm1_route_mediators = saved_mediators;
    }
}

fn probe_shard<'a>(
    model: &'a Model,
    shard: u8,
    key: &ContextKey,
) -> Option<NodeView<'a>> {
    SHARD_PROBES.fetch_add(1, Ordering::Relaxed);
    if exp173_owner_storage(model.shard_mode) {
        return (model.shard_assignments.get(key).copied() == Some(shard))
            .then(|| model_glp_view(model, key))
            .flatten();
    }
    if exp172_canonical_views(model.shard_mode) {
        let present = model
            .glp_shard_views
            .get(shard as usize)
            .map(|view| view.contains_key(key))
            .unwrap_or(false);
        return present.then(|| model_glp_view(model, key)).flatten();
    }
    model
        .glp_shards
        .get(shard as usize)
        .and_then(|graph| graph.get(key))
        .map(NodeView::canonical)
}

fn fallback_shards<'a>(
    model: &'a Model,
    key: &ContextKey,
    skip: &[u8],
) -> Option<NodeView<'a>> {
    SHARD_FALLBACKS.fetch_add(1, Ordering::Relaxed);
    for shard in 0..EXP170_SHARDS as u8 {
        if skip.contains(&shard) {
            continue;
        }
        if let Some(node) = probe_shard(model, shard, key) {
            return Some(node);
        }
    }
    None
}

fn routed_glp_get_inner<'a>(
    model: &'a Model,
    key: &ContextKey,
    gcp: Option<&Node>,
) -> Option<NodeView<'a>> {
    SHARD_QUERIES.fetch_add(1, Ordering::Relaxed);
    if model.glp_shards.len() != EXP170_SHARDS {
        return model_glp_view(model, key);
    }
    let mut candidates = Vec::new();
    match model.shard_mode {
        1 => {
            if let Some(node) = gcp {
                candidates.push(node_shard(node, &model.token_themes));
            }
        }
        2 | 6 | 9 | 10 | 11 | 12 | 13 | 14 | 15 => {
            if let Some(&shard) = model.gcp_routes.get(key) {
                candidates.push(shard);
            }
        }
        7 => {
            if let Some(&shard) = model.gm1_routes.get(key) {
                candidates.push(shard);
            }
        }
        8 => {
            if let Some(shard) = compact_route(model, key) {
                candidates.push(shard);
            }
        }
        3 => {
            if let Some(&shard) = model.gm1_routes.get(key) {
                candidates.push(shard);
            }
        }
        4 | 5 => {
            let signature = route_signature(key, &model.token_themes);
            if let Some(mediator) = model.gm1_route_mediators.get(&signature) {
                if mediator.confirmations >= 2 {
                    let mut rows: Vec<(u32, u8)> = mediator
                        .votes
                        .iter()
                        .enumerate()
                        .map(|(shard, &count)| (count, shard as u8))
                        .collect();
                    rows.sort_unstable_by(|a, b| {
                        b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1))
                    });
                    candidates.push(rows[0].1);
                    if model.shard_mode == 5 && rows[1].0 > 0 {
                        candidates.push(rows[1].1);
                    }
                }
            }
        }
        _ => {}
    }
    candidates.sort_unstable();
    candidates.dedup();
    if exp173_full_directory(model.shard_mode) && candidates.is_empty() {
        return None;
    }
    for &shard in &candidates {
        SHARD_ROUTE_PREDICTIONS.fetch_add(1, Ordering::Relaxed);
        if let Some(node) = probe_shard(model, shard, key) {
            SHARD_ROUTE_CORRECT.fetch_add(1, Ordering::Relaxed);
            SHARD_FOUND_CONTEXTS.fetch_add(1, Ordering::Relaxed);
            return Some(node);
        }
        SHARD_ROUTE_WRONG.fetch_add(1, Ordering::Relaxed);
    }
    let found = if candidates.is_empty() || model.shard_mode == 5 {
        fallback_shards(model, key, &candidates)
    } else {
        None
    };
    if found.is_some() {
        SHARD_FOUND_CONTEXTS.fetch_add(1, Ordering::Relaxed);
    } else if model.glp.contains_key(key) {
        SHARD_LOST_CONTEXTS.fetch_add(1, Ordering::Relaxed);
    }
    found
}

fn routed_glp_get<'a>(
    model: &'a Model,
    key: &ContextKey,
    gcp: Option<&Node>,
) -> Option<NodeView<'a>> {
    let timer = profile_timer();
    let result = routed_glp_get_inner(model, key, gcp);
    profile_finish(timer, &PROFILE_ROUTED_GET_NS, &PROFILE_ROUTED_GET_CALLS);
    result
}

fn shard_sizes(model: &Model) -> [usize; EXP170_SHARDS] {
    let mut sizes = [0usize; EXP170_SHARDS];
    if exp173_owner_storage(model.shard_mode) {
        for shard in model.shard_assignments.values() {
            sizes[*shard as usize] += 1;
        }
    } else if exp172_canonical_views(model.shard_mode) {
        for (index, view) in model
            .glp_shard_views
            .iter()
            .enumerate()
            .take(EXP170_SHARDS)
        {
            sizes[index] = view.len();
        }
    } else {
        for (index, graph) in model.glp_shards.iter().enumerate().take(EXP170_SHARDS) {
            sizes[index] = graph.len();
        }
    }
    sizes
}

fn mediated_long_view<'a>(
    model: &'a Model,
    key: &ContextKey,
    gcp: Option<&Node>,
) -> Option<NodeView<'a>> {
    GM1_GCP_READS.fetch_add(1, Ordering::Relaxed);
    if matches!(
        model.shard_mode,
        1 | 2 | 6 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15
    ) {
        if gcp.map(recent_resolves).unwrap_or(false) {
            GM1_GCP_STOPS.fetch_add(1, Ordering::Relaxed);
            return None;
        }
        GM1_GLP_READS.fetch_add(1, Ordering::Relaxed);
        return routed_glp_get(model, key, gcp);
    }
    if model.shard_mode == 7 {
        if gcp.map(recent_resolves).unwrap_or(false) {
            GM1_GCP_STOPS.fetch_add(1, Ordering::Relaxed);
            return None;
        }
        GM1_MEDIAL_READS.fetch_add(1, Ordering::Relaxed);
        GM1_GLP_READS.fetch_add(1, Ordering::Relaxed);
        return routed_glp_get(model, key, gcp);
    }
    if (3..=5).contains(&model.shard_mode) {
        if gcp.map(recent_resolves).unwrap_or(false) {
            GM1_GCP_STOPS.fetch_add(1, Ordering::Relaxed);
            return None;
        }
        GM1_MEDIAL_READS.fetch_add(1, Ordering::Relaxed);
        if let Some(composite) = model.gm1_cache.get(key) {
            GM1_MEDIAL_STOPS.fetch_add(1, Ordering::Relaxed);
            return Some(NodeView::canonical(composite));
        }
        let medial = model.gm1.get(key);
        if medial.map(|node| medial_resolves(key, node)).unwrap_or(false) {
            GM1_MEDIAL_STOPS.fetch_add(1, Ordering::Relaxed);
            return medial.map(NodeView::canonical);
        }
        GM1_GLP_READS.fetch_add(1, Ordering::Relaxed);
        return routed_glp_get(model, key, gcp)
            .or_else(|| medial.map(NodeView::canonical));
    }
    if model.gm1_mode < 2 {
        GM1_GLP_READS.fetch_add(1, Ordering::Relaxed);
        return model_glp_view(model, key);
    }
    if gcp.map(recent_resolves).unwrap_or(false) {
        GM1_GCP_STOPS.fetch_add(1, Ordering::Relaxed);
        return None;
    }
    GM1_MEDIAL_READS.fetch_add(1, Ordering::Relaxed);
    if let Some(composite) = model.gm1_cache.get(key) {
        GM1_MEDIAL_STOPS.fetch_add(1, Ordering::Relaxed);
        return Some(NodeView::canonical(composite));
    }
    let medial = model.gm1.get(key);
    if medial.map(|node| medial_resolves(key, node)).unwrap_or(false) {
        GM1_MEDIAL_STOPS.fetch_add(1, Ordering::Relaxed);
        return medial.map(NodeView::canonical);
    }
    GM1_GLP_READS.fetch_add(1, Ordering::Relaxed);
    model_glp_view(model, key).or_else(|| medial.map(NodeView::canonical))
}

fn gm1_support_profile(model: &Model) -> [usize; 6] {
    let mut rows = [0usize; 6];
    for node in model.gm1.values() {
        rows[0] += 1;
        rows[1] += (node.total >= 2) as usize;
        rows[2] += (node.total >= 4) as usize;
        rows[3] += (node.total >= 8) as usize;
        let dominant = node.edges.iter().map(|edge| edge.count).max().unwrap_or(0);
        let ratio = dominant as f64 / node.total.max(1) as f64;
        rows[4] += (node.total >= 2 && ratio >= 0.40) as usize;
        rows[5] += (node.total >= 2 && ratio >= 0.25) as usize;
    }
    rows
}

fn aggregate_node(model: &Model, key: &ContextKey) -> Option<(u32, f64, Vec<EdgeAggregate>)> {
    let gcp_node = model.gcp.get(key);
    let gcp = gcp_node.map(NodeView::canonical);
    let glp = mediated_long_view(model, key, gcp_node);
    if gcp.is_none() && glp.is_none() {
        return None;
    }
    let mut total = 0u32;
    let mut potential = 0.0;
    let mut weight = 0.0;
    for node in [glp, gcp].into_iter().flatten() {
        total = total.saturating_add(node.total);
        let local_weight = node.total.max(1) as f64;
        potential += node.potential as f64 * local_weight;
        weight += local_weight;
    }
    let left = glp.map(EdgeOrder::new);
    let right = gcp.map(EdgeOrder::new);
    let left_len = left.as_ref().map(EdgeOrder::len).unwrap_or(0);
    let right_len = right.as_ref().map(EdgeOrder::len).unwrap_or(0);
    let mut edges = Vec::with_capacity(left_len + right_len);
    let (mut i, mut j) = (0usize, 0usize);
    while i < left_len || j < right_len {
        let left_edge = left.as_ref().and_then(|rows| rows.get(i));
        let right_edge = right.as_ref().and_then(|rows| rows.get(j));
        let mut row;
        if right_edge.is_none()
            || left_edge
                .zip(right_edge)
                .map(|(left, right)| left.token < right.token)
                .unwrap_or(false)
        {
            row = edge_aggregate(left_edge.expect("left edge"));
            i += 1;
        } else if left_edge.is_none()
            || right_edge
                .zip(left_edge)
                .map(|(right, left)| right.token < left.token)
                .unwrap_or(false)
        {
            row = edge_aggregate(right_edge.expect("right edge"));
            j += 1;
        } else {
            // Reference order is GLP first and GCP second.
            row = edge_aggregate(left_edge.expect("left edge"));
            row.merge(right_edge.expect("right edge"));
            i += 1;
            j += 1;
        }
        if let Some(credit) = model
            .gcp_credit
            .get(key)
            .and_then(|credits| credits.get(&row.token))
        {
            row.weight = (row.weight + credit.weight_delta as f64).clamp(-4.0, 4.0);
            row.gradient_sum += credit.gradient_sum as f64;
            row.gradient_sq += credit.gradient_sq as f64;
            row.conflicts = row.conflicts.saturating_add(credit.conflicts);
            row.last_seen = row.last_seen.max(credit.last_seen);
        }
        edges.push(row);
    }
    Some((total, potential / weight.max(1.0), edges))
}

fn collect_features(
    model: &Model,
    history: &[u16],
    base: &[f64],
    base_total: f64,
) -> Vec<Feature> {
    let mut features = take_feature_buffer();
    for order in 1..=model.max_order.min(history.len()) {
        let key = make_key(history, order);
        let gcp_node = model.gcp.get(&key);
        let gcp = gcp_node.map(NodeView::canonical);
        let glp = mediated_long_view(model, &key, gcp_node);
        if gcp.is_none() && glp.is_none() {
            continue;
        }
        let mut total = 0u32;
        let mut potential = 0.0;
        let mut potential_weight = 0.0;
        for node in [glp, gcp].into_iter().flatten() {
            total = total.saturating_add(node.total);
            let weight = node.total.max(1) as f64;
            potential += node.potential as f64 * weight;
            potential_weight += weight;
        }
        let left = glp.map(EdgeOrder::new);
        let right = gcp.map(EdgeOrder::new);
        let left_len = left.as_ref().map(EdgeOrder::len).unwrap_or(0);
        let right_len = right.as_ref().map(EdgeOrder::len).unwrap_or(0);

        let (mut li, mut ri, mut edge_count) = (0usize, 0usize, 0usize);
        while li < left_len || ri < right_len {
            let l = left.as_ref().and_then(|rows| rows.get(li));
            let r = right.as_ref().and_then(|rows| rows.get(ri));
            if r.is_none()
                || l.zip(r).map(|(a, b)| a.token < b.token).unwrap_or(false)
            {
                li += 1;
            } else if l.is_none()
                || r.zip(l).map(|(a, b)| a.token < b.token).unwrap_or(false)
            {
                ri += 1;
            } else {
                li += 1;
                ri += 1;
            }
            edge_count += 1;
        }
        let denom = total as f64 + 0.5 * edge_count.max(1) as f64;
        let gate = sigmoid(potential / potential_weight.max(1.0));
        li = 0;
        ri = 0;
        while li < left_len || ri < right_len {
            let l = left.as_ref().and_then(|rows| rows.get(li));
            let r = right.as_ref().and_then(|rows| rows.get(ri));
            let mut edge;
            if r.is_none()
                || l.zip(r).map(|(a, b)| a.token < b.token).unwrap_or(false)
            {
                edge = edge_aggregate(l.expect("left feature edge"));
                li += 1;
            } else if l.is_none()
                || r.zip(l).map(|(a, b)| a.token < b.token).unwrap_or(false)
            {
                edge = edge_aggregate(r.expect("right feature edge"));
                ri += 1;
            } else {
                edge = edge_aggregate(l.expect("left feature edge"));
                edge.merge(r.expect("right feature edge"));
                li += 1;
                ri += 1;
            }
            if let Some(credit) = model
                .gcp_credit
                .get(&key)
                .and_then(|credits| credits.get(&edge.token))
            {
                edge.weight =
                    (edge.weight + credit.weight_delta as f64).clamp(-4.0, 4.0);
                edge.gradient_sum += credit.gradient_sum as f64;
                edge.gradient_sq += credit.gradient_sq as f64;
                edge.conflicts = edge.conflicts.saturating_add(credit.conflicts);
                edge.last_seen = edge.last_seen.max(credit.last_seen);
            }
            let local = (edge.count as f64 + 0.5) / denom.max(0.5);
            let global = (base[edge.token as usize] / base_total).max(1e-15);
            features.push(Feature {
                key,
                token: edge.token,
                relation: (local / global).ln().clamp(-6.0, 6.0) + edge.weight,
                gate,
                probability: 0.0,
            });
        }
    }
    features
}

fn utility(edge: &EdgeAggregate) -> f64 {
    (-edge.gradient_sum / (edge.gradient_sq + 1e-8).sqrt()).clamp(-3.0, 3.0)
}

fn raw_length(edge: &EdgeAggregate, mode: Mode) -> f64 {
    if mode == Mode::Unit {
        return 1.0;
    }
    let frequency = 1.0 / (1.0 + 0.08 * (1.0 + edge.count as f64).ln());
    let gradient = (-0.08 * utility(edge)).exp();
    let conflict = (0.80 * edge.conflicts as f64 / edge.count.max(1) as f64).exp();
    let value = match mode {
        Mode::Frequency => frequency,
        Mode::Gradient => gradient,
        Mode::FrequencyGradient => frequency * gradient,
        Mode::Full | Mode::Shuffle => frequency * gradient * conflict,
        Mode::Base | Mode::Unit => 1.0,
    };
    (value * edge.sleep_bias.clamp(-0.60, 0.60).exp()).clamp(0.35, 2.50)
}

fn assigned_length(edges: &[EdgeAggregate], index: usize, source: u16, mode: Mode) -> f64 {
    if mode != Mode::Shuffle {
        return raw_length(&edges[index], mode);
    }
    if edges.len() < 2 {
        return raw_length(&edges[index], Mode::Full);
    }
    let shift = 1 + (source as usize % (edges.len() - 1));
    raw_length(&edges[(index + shift) % edges.len()], Mode::Full)
}

fn successors_modified(
    model: &Model,
    source: u16,
    base: &[f64],
    base_total: f64,
    mode: Mode,
    masked: Option<(u16, u16)>,
    overlay: Option<&HashMap<(u16, u16), f64>>,
) -> Vec<(u16, f64, f64, f64)> {
    let key = ContextKey::new(1, [source, 0, 0, 0]);
    let Some((total, _, mut edges)) = aggregate_node(model, &key) else {
        return Vec::new();
    };
    let removed = edges
        .iter()
        .filter(|edge| masked == Some((source, edge.token)))
        .map(|edge| edge.count)
        .sum::<u32>();
    edges.retain(|edge| masked != Some((source, edge.token)));
    if edges.is_empty() {
        return Vec::new();
    }
    edges.sort_unstable_by_key(|edge| edge.token);
    let denom = total.saturating_sub(removed) as f64 + 0.5 * edges.len().max(1) as f64;
    let mut rows: Vec<_> = (0..edges.len())
        .map(|index| {
            let edge = &edges[index];
            let local = (edge.count as f64 + 0.5) / denom.max(0.5);
            let global = (base[edge.token as usize] / base_total).max(1e-15);
            let relation = (local / global).ln().clamp(-3.0, 4.0) + edge.weight;
            let mut length = assigned_length(&edges, index, source, mode);
            if let Some(delta) = overlay.and_then(|rows| rows.get(&(source, edge.token))) {
                // The overlay is an uncommitted sleep_bias delta. Applying it to the
                // already aggregated W-FG length is exact away from the clamp bounds.
                length = (length * delta.clamp(-0.60, 0.60).exp()).clamp(0.35, 2.50);
            }
            let conflict = edge.conflicts as f64 / edge.count.max(1) as f64;
            (edge.token, relation, length, conflict)
        })
        .collect();
    rows.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap().then_with(|| a.0.cmp(&b.0)));
    rows.truncate(WAVE_BRANCH);
    rows
}

fn successors(
    model: &Model,
    source: u16,
    base: &[f64],
    base_total: f64,
    mode: Mode,
    masked: Option<(u16, u16)>,
) -> Vec<(u16, f64, f64, f64)> {
    successors_modified(model, source, base, base_total, mode, masked, None)
}

#[derive(Default)]
struct ActiveConstructionOverlay {
    role_weights: HashMap<(u16, u16), f64>,
    prediction_weights: HashMap<u16, f64>,
    active_constructions: usize,
}

fn active_construction_overlay(model: &Model, history: &[u16]) -> ActiveConstructionOverlay {
    if model.edge_application == 0 {
        return ActiveConstructionOverlay::default();
    }
    let mut best: HashMap<u16, Construction> = HashMap::new();
    for construction in &model.glp_constructions {
        if !construction.dag_is_forward_only()
            || !construction_matches(&construction.key, history)
        {
            continue;
        }
        best.entry(construction.key.target)
            .and_modify(|old| {
                let new_score = construction_bonus(construction);
                let old_score = construction_bonus(old);
                if new_score > old_score
                    || (new_score == old_score && construction.key < old.key)
                {
                    *old = *construction;
                }
            })
            .or_insert(*construction);
    }
    let mut overlay = ActiveConstructionOverlay {
        active_constructions: best.len(),
        ..ActiveConstructionOverlay::default()
    };
    let mut selected: Vec<_> = best.into_values().collect();
    selected.sort_unstable_by_key(|construction| construction.key);
    for construction in selected {
        let len = construction.key.len as usize;
        let input = &history[history.len() - len..];
        let mut local: HashMap<u16, (f64, u32)> = HashMap::new();
        for (position, &source) in input.iter().enumerate() {
            let row = local.entry(source).or_insert((0.0, 0));
            row.0 += construction.dag_edges[position].weight as f64;
            row.1 += 1;
        }
        for (source, (sum, count)) in local {
            overlay.role_weights.insert(
                (source, construction.key.target),
                sum / count.max(1) as f64,
            );
        }
        overlay.prediction_weights.insert(
            construction.key.target,
            construction.dag_edges[CONSTRUCTION_INTERFACE].weight as f64,
        );
    }
    overlay
}

fn apply_role_weight(row: &mut (u16, f64, f64, f64), weight: f64) {
    let weight = weight.clamp(0.65, 1.35);
    row.1 += weight.ln();
    row.2 = (row.2 / weight).clamp(0.35, 2.50);
}

fn successors_early(
    model: &Model,
    source: u16,
    base: &[f64],
    base_total: f64,
    mode: Mode,
    masked: Option<(u16, u16)>,
    semantic_overlay: Option<&HashMap<(u16, u16), f64>>,
    construction_overlay: &ActiveConstructionOverlay,
) -> (Vec<(u16, f64, f64, f64)>, usize, bool) {
    if model.edge_application == 0
        && mode == Mode::FrequencyGradient
        && masked.is_none()
        && semantic_overlay.is_none()
    {
        if let Some(rows) =
            cached_frequency_gradient_successors(model, source, base, base_total)
        {
            return (rows, 0, false);
        }
    }
    if model.edge_application == 0 || construction_overlay.role_weights.is_empty() {
        return (
            successors_modified(model, source, base, base_total, mode, masked, semantic_overlay),
            0,
            false,
        );
    }
    let key = ContextKey::new(1, [source, 0, 0, 0]);
    let Some((total, _, mut edges)) = aggregate_node(model, &key) else {
        return (Vec::new(), 0, false);
    };
    let removed = edges
        .iter()
        .filter(|edge| masked == Some((source, edge.token)))
        .map(|edge| edge.count)
        .sum::<u32>();
    edges.retain(|edge| masked != Some((source, edge.token)));
    if edges.is_empty() {
        return (Vec::new(), 0, false);
    }
    edges.sort_unstable_by_key(|edge| edge.token);
    let denom = total.saturating_sub(removed) as f64 + 0.5 * edges.len().max(1) as f64;
    let mut rows: Vec<_> = (0..edges.len())
        .map(|index| {
            let edge = &edges[index];
            let local = (edge.count as f64 + 0.5) / denom.max(0.5);
            let global = (base[edge.token as usize] / base_total).max(1e-15);
            let relation = (local / global).ln().clamp(-3.0, 4.0) + edge.weight;
            let mut length = assigned_length(&edges, index, source, mode);
            if let Some(delta) =
                semantic_overlay.and_then(|values| values.get(&(source, edge.token)))
            {
                length = (length * delta.clamp(-0.60, 0.60).exp()).clamp(0.35, 2.50);
            }
            let conflict = edge.conflicts as f64 / edge.count.max(1) as f64;
            (edge.token, relation, length, conflict)
        })
        .collect();
    rows.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap().then_with(|| a.0.cmp(&b.0)));
    let baseline_top: Vec<u16> = rows.iter().take(WAVE_BRANCH).map(|row| row.0).collect();
    let mut modified = 0;
    if model.edge_application == 1 {
        rows.truncate(WAVE_BRANCH);
        for row in &mut rows {
            if let Some(&weight) = construction_overlay.role_weights.get(&(source, row.0)) {
                apply_role_weight(row, weight);
                modified += 1;
            }
        }
        rows.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap().then_with(|| a.0.cmp(&b.0)));
        return (rows, modified, false);
    }
    for row in &mut rows {
        if let Some(&weight) = construction_overlay.role_weights.get(&(source, row.0)) {
            apply_role_weight(row, weight);
            modified += 1;
        }
    }
    rows.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap().then_with(|| a.0.cmp(&b.0)));
    rows.truncate(WAVE_BRANCH);
    let weighted_top: Vec<u16> = rows.iter().map(|row| row.0).collect();
    (rows, modified, weighted_top != baseline_top)
}

fn construction_matches(key: &ConstructionKey, history: &[u16]) -> bool {
    let len = key.len as usize;
    if len == 0 || len > CONSTRUCTION_INTERFACE || history.len() < len {
        return false;
    }
    let input = &history[history.len() - len..];
    input.iter().enumerate().all(|(index, token)| {
        key.slot_mask & (1 << index) != 0 || key.symbols[index] == *token
    })
}

fn construction_bonus(construction: &Construction) -> f64 {
    let support = (construction.support.max(1) as f64).ln_1p();
    let confirmations = construction.confirmations as f64;
    ((0.025
        + 0.008 * support
        + 0.006 * confirmations
        + 0.012 * (construction.value as f64).tanh())
        .clamp(0.02, 0.10)
        * construction.forward_gain())
        .clamp(0.0, 0.10)
}

fn construction_bonus_applied(construction: &Construction, edge_application: u8) -> f64 {
    let late = construction_bonus(construction);
    if edge_application & 0b1100 == 0 || !construction.dag_is_forward_only() {
        return late;
    }
    let role_mean = construction.dag_edges[..CONSTRUCTION_INTERFACE]
        .iter()
        .map(|edge| edge.weight as f64)
        .sum::<f64>()
        / CONSTRUCTION_INTERFACE as f64;
    let prediction = construction.dag_edges[CONSTRUCTION_INTERFACE].weight as f64;
    let role_term = if edge_application & 0b0100 != 0 {
        0.10 * role_mean.clamp(0.65, 1.35).ln()
    } else {
        0.0
    };
    let prediction_term = if edge_application & 0b1000 != 0 {
        0.10 * prediction.clamp(0.65, 1.35).ln()
    } else {
        0.0
    };
    (late + role_term + prediction_term).clamp(-0.15, 0.35)
}

fn inject_construction_candidates(
    history: &[u16],
    constructions: &[Construction],
    candidates: &mut Vec<CandidateGr>,
    edge_application: u8,
) -> (usize, usize) {
    let window = history.len().min(WAVE_WINDOW);
    let mut best: HashMap<u16, Construction> = HashMap::new();
    for construction in constructions {
        // Activation follows role -> construction -> target only. A malformed
        // or reverse DAG is inert instead of being silently traversed.
        if !construction.dag_is_forward_only()
            || !construction_matches(&construction.key, history)
        {
            continue;
        }
        best.entry(construction.key.target)
            .and_modify(|old| {
                if construction_bonus_applied(construction, edge_application)
                    > construction_bonus_applied(old, edge_application)
                    || (construction_bonus_applied(construction, edge_application)
                        == construction_bonus_applied(old, edge_application)
                        && construction.key < old.key)
                {
                    *old = *construction;
                }
            })
            .or_insert(*construction);
    }
    let active = best.len();
    let mut adjustments = 0;
    let mut selected: Vec<_> = best.into_values().collect();
    selected.sort_unstable_by_key(|construction| construction.key);
    for construction in selected {
        let len = construction.key.len as usize;
        let coverage = len.min(window);
        let full = coverage == window;
        let bonus = construction_bonus_applied(&construction, edge_application);
        if edge_application & 0b1100 != 0
            && (bonus - construction_bonus(&construction)).abs() > 1e-12
        {
            adjustments += 1;
        }
        if bonus <= 0.0 {
            continue;
        }
        let input = &history[history.len() - len..];
        let mut origin_present = [false; WAVE_WINDOW];
        let offset = window.saturating_sub(len);
        for index in 0..coverage {
            origin_present[offset + index] = true;
        }
        let mut support_edges: Vec<_> = input
            .iter()
            .copied()
            .map(|source| (source, construction.key.target))
            .collect();
        support_edges.sort_unstable();
        support_edges.dedup();
        if let Some(candidate) = candidates
            .iter_mut()
            .find(|candidate| candidate.token == construction.key.target)
        {
            candidate.coverage = candidate.coverage.max(coverage);
            candidate.full |= full;
            candidate.bonus = (candidate.bonus + bonus).clamp(-0.15, 0.35);
            candidate.mean_signal += bonus;
            for (index, present) in origin_present.iter().enumerate() {
                candidate.origin_present[index] |= *present;
            }
            for edge in support_edges {
                if !candidate.support_edges.contains(&edge) {
                    candidate.support_edges.push(edge);
                }
            }
        } else {
            candidates.push(CandidateGr {
                token: construction.key.target,
                coverage,
                full,
                dispersion: 0.0,
                mean_cost: 1.0,
                mean_signal: bonus,
                conflict: 0.0,
                energy: 1.0 - bonus,
                bonus,
                support_edges,
                phase_deltas: Vec::new(),
                origin_present,
            });
        }
    }
    (active, adjustments)
}

fn wave_trace_modified(
    model: &Model,
    history: &[u16],
    base: &[f64],
    base_total: f64,
    mode: Mode,
    masked: Option<(u16, u16)>,
    overlay: Option<&HashMap<(u16, u16), f64>>,
) -> WaveTrace {
    if !mode.waves() || history.len() < 2 {
        return WaveTrace::default();
    }
    let window = history.len().min(WAVE_WINDOW);
    let input = &history[history.len() - window..];
    let construction_overlay = active_construction_overlay(model, history);
    let mut early_modified_transitions = 0usize;
    let mut early_route_reorders = 0usize;
    let mut evidence: HashMap<u16, Vec<Option<WaveState>>> = HashMap::new();
    // EXP151 arena stage: reuse the sparse successor table across every
    // origin/depth in this reflection. `drain` retains its allocation.
    let mut best_arena: HashMap<u16, WaveState> = HashMap::new();
    for origin in 0..window {
        let mut states = vec![WaveState {
            token: input[origin],
            signal: 0.0,
            arrival: origin as f64,
            cost: 0.0,
            conflicts: 0.0,
            path: std::array::from_fn(|i| if i == 0 { input[origin] } else { 0 }),
            path_len: 1,
        }];
        for _depth in 1..=WAVE_DEPTH {
            debug_assert!(best_arena.is_empty());
            for state in &states {
                let (successors, modified, reordered) = successors_early(
                    model,
                    state.token,
                    base,
                    base_total,
                    mode,
                    masked,
                    overlay,
                    &construction_overlay,
                );
                early_modified_transitions += modified;
                early_route_reorders += reordered as usize;
                for (next, relation, length, conflict) in successors {
                    if state.path[..state.path_len as usize].contains(&next) {
                        continue;
                    }
                    let mut candidate = state.clone();
                    candidate.token = next;
                    candidate.arrival += length;
                    candidate.cost += length;
                    candidate.signal += relation * (-0.25 * candidate.cost).exp();
                    candidate.conflicts += conflict;
                    if (candidate.path_len as usize) < candidate.path.len() {
                        candidate.path[candidate.path_len as usize] = next;
                        candidate.path_len += 1;
                    }
                    let quality =
                        candidate.signal - 0.10 * candidate.cost - 0.25 * candidate.conflicts;
                    let replace = best_arena
                        .get(&next)
                        .map(|old| quality > old.signal - 0.10 * old.cost - 0.25 * old.conflicts)
                        .unwrap_or(true);
                    if replace {
                        best_arena.insert(next, candidate);
                    }
                }
            }
            states = best_arena.drain().map(|(_, state)| state).collect();
            states.sort_unstable_by(|a, b| {
                (b.signal - 0.1 * b.cost)
                    .partial_cmp(&(a.signal - 0.1 * a.cost))
                    .unwrap()
                    .then_with(|| a.token.cmp(&b.token))
                    .then_with(|| a.path_len.cmp(&b.path_len))
                    .then_with(|| a.path.cmp(&b.path))
            });
            states.truncate(WAVE_BEAM);
            for state in &states {
                let slots = evidence
                    .entry(state.token)
                    .or_insert_with(|| vec![None; window]);
                let replace = slots[origin]
                    .as_ref()
                    .map(|old| state.signal - 0.1 * state.cost > old.signal - 0.1 * old.cost)
                    .unwrap_or(true);
                if replace {
                    slots[origin] = Some(state.clone());
                }
            }
            if states.is_empty() {
                break;
            }
        }
    }
    let mut candidates = Vec::new();
    for (token, slots) in evidence {
        if slots[window - 1].is_none() {
            continue;
        }
        let selected: Vec<&WaveState> = slots.iter().filter_map(|row| row.as_ref()).collect();
        let mut origin_present = [false; WAVE_WINDOW];
        for (index, state) in slots.iter().enumerate() {
            origin_present[index] = state.is_some();
        }
        let coverage = selected.len();
        if coverage < 2 {
            continue;
        }
        let min = selected
            .iter()
            .map(|s| s.arrival)
            .fold(f64::INFINITY, f64::min);
        let max = selected
            .iter()
            .map(|s| s.arrival)
            .fold(f64::NEG_INFINITY, f64::max);
        let dispersion = max - min;
        if dispersion > MEETING_TOLERANCE {
            continue;
        }
        let mean_cost = selected.iter().map(|s| s.cost).sum::<f64>() / coverage as f64;
        let mean_signal = selected.iter().map(|s| s.signal).sum::<f64>() / coverage as f64;
        let conflict = selected.iter().map(|s| s.conflicts).sum::<f64>() / coverage as f64;
        let mut support: HashSet<(u16, u16)> = HashSet::new();
        for state in &selected {
            for pair in state.path[..state.path_len as usize].windows(2) {
                support.insert((pair[0], pair[1]));
            }
        }
        let mut support_edges: Vec<_> = support.into_iter().collect();
        support_edges.sort_unstable();
        let mean_arrival =
            selected.iter().map(|state| state.arrival).sum::<f64>() / selected.len() as f64;
        let mut phase_rows: HashMap<(u16, u16), (f64, u32)> = HashMap::new();
        for state in &selected {
            let path_edges = (state.path_len as usize).saturating_sub(1).max(1);
            let residual = state.arrival - mean_arrival;
            // Late paths receive negative bias (shorter); early paths receive
            // positive bias (longer). Division by path length distributes the
            // phase correction jointly instead of blaming one arbitrary edge.
            let delta =
                (-0.03 * residual / MEETING_TOLERANCE / path_edges as f64).clamp(-0.03, 0.03);
            for pair in state.path[..state.path_len as usize].windows(2) {
                let row = phase_rows.entry((pair[0], pair[1])).or_insert((0.0, 0));
                row.0 += delta;
                row.1 += 1;
            }
        }
        let mut phase_deltas: Vec<_> = phase_rows
            .into_iter()
            .map(|(edge, (sum, count))| (edge, sum / count.max(1) as f64))
            .filter(|(_, delta)| delta.abs() >= 1e-6)
            .collect();
        phase_deltas.sort_unstable_by_key(|row| row.0);
        let full = coverage == window && dispersion <= FULL_TOLERANCE;
        let coverage_ratio = coverage as f64 / window as f64;
        let energy = mean_cost + dispersion + conflict - mean_signal;
        let raw = 1.10 * coverage_ratio + 0.60 * (-dispersion).exp() + 0.15 * mean_signal
            - 0.10 * mean_cost
            - 0.50 * conflict
            + if full { 0.25 } else { 0.0 };
        candidates.push(CandidateGr {
            token,
            coverage,
            full,
            dispersion,
            mean_cost,
            mean_signal,
            conflict,
            energy,
            bonus: (0.10 * raw).clamp(-0.15, 0.35),
            support_edges,
            phase_deltas,
            origin_present,
        });
    }
    let (active_injected, early_gr_adjustments) = inject_construction_candidates(
        history,
        &model.glp_constructions,
        &mut candidates,
        model.edge_application,
    );
    candidates.sort_unstable_by(|a, b| {
        b.bonus
            .partial_cmp(&a.bonus)
            .unwrap()
            .then_with(|| a.token.cmp(&b.token))
    });
    candidates.truncate(WAVE_MAX_GRS);
    WaveTrace {
        full: candidates.iter().any(|gr| gr.full),
        max_coverage: candidates
            .iter()
            .map(|gr| gr.coverage as f64 / window as f64)
            .fold(0.0, f64::max),
        candidates,
        early_active_constructions: construction_overlay
            .active_constructions
            .max(active_injected),
        early_modified_transitions,
        early_route_reorders,
        early_gr_adjustments,
    }
}

fn wave_trace_masked(
    model: &Model,
    history: &[u16],
    base: &[f64],
    base_total: f64,
    mode: Mode,
    masked: Option<(u16, u16)>,
) -> WaveTrace {
    wave_trace_modified(model, history, base, base_total, mode, masked, None)
}

fn wave_trace(
    model: &Model,
    history: &[u16],
    base: &[f64],
    base_total: f64,
    mode: Mode,
) -> WaveTrace {
    wave_trace_modified(model, history, base, base_total, mode, None, None)
}

struct DenseDeltaScratch {
    values: Vec<f64>,
    factors: Vec<f64>,
    seen: Vec<bool>,
    touched: Vec<u16>,
    id_seen: Vec<bool>,
    ids: Vec<u16>,
}

impl DenseDeltaScratch {
    fn new() -> Self {
        Self {
            values: Vec::new(),
            factors: Vec::new(),
            seen: Vec::new(),
            touched: Vec::new(),
            id_seen: Vec::new(),
            ids: Vec::new(),
        }
    }
    fn begin(&mut self, vocab_size: usize) {
        for &token in &self.touched {
            self.values[token as usize] = 0.0;
            self.seen[token as usize] = false;
        }
        for &token in &self.ids {
            self.id_seen[token as usize] = false;
        }
        self.touched.clear();
        self.ids.clear();
        if self.values.len() != vocab_size {
            self.values = vec![0.0; vocab_size];
            self.factors = vec![1.0; vocab_size];
            self.seen = vec![false; vocab_size];
            self.id_seen = vec![false; vocab_size];
        }
    }
    fn add(&mut self, token: u16, value: f64) {
        let index = token as usize;
        if !self.seen[index] {
            self.seen[index] = true;
            self.touched.push(token);
        }
        self.values[index] += value;
    }
}

thread_local! {
    static DENSE_DELTA: std::cell::RefCell<DenseDeltaScratch> =
        std::cell::RefCell::new(DenseDeltaScratch::new());
}

fn distribution(
    model: &Model,
    history: &[u16],
    gold: u16,
    base: &[f64],
    base_total: f64,
    base_top: &[u16],
    mode: Mode,
) -> (f64, Vec<(u16, f64)>, Vec<Feature>, WaveTrace) {
    let mut features = collect_features(model, history, base, base_total);
    let trace = wave_trace(model, history, base, base_total, mode);
    DENSE_DELTA.with(|cell| {
        let mut scratch = cell.borrow_mut();
        scratch.begin(model.vocab_size);
        for feature in &features {
            scratch.add(feature.token, feature.gate * feature.relation);
        }
        for gr in &trace.candidates {
            scratch.add(gr.token, gr.bonus);
        }
        scratch.touched.sort_unstable();
        let mut z = base_total;
        for position in 0..scratch.touched.len() {
            let token = scratch.touched[position];
            let delta = scratch.values[token as usize];
            let factor = delta.clamp(-12.0, 12.0).exp();
            scratch.factors[token as usize] = factor;
            z += base[token as usize] * (factor - 1.0);
        }
        for feature in &mut features {
            let token = feature.token as usize;
            let factor = if scratch.seen[token] {
                scratch.factors[token]
            } else {
                1.0
            };
            feature.probability = base[token] * factor / z;
        }
        let p_gold = (base[gold as usize]
            * if scratch.seen[gold as usize] {
                scratch.factors[gold as usize]
            } else {
                1.0
            }
            / z)
            .max(1e-30);
        let mut ids = base_top.to_vec();
        ids.extend(scratch.touched.iter().copied());
        ids.sort_unstable();
        ids.dedup();
        let mut ranked: Vec<_> = ids
            .into_iter()
            .map(|token| {
                let factor = if scratch.seen[token as usize] {
                    scratch.factors[token as usize]
                } else {
                    1.0
                };
                (token, base[token as usize] * factor / z)
            })
            .collect();
        ranked.sort_unstable_by(|a, b| {
            b.1.partial_cmp(&a.1).unwrap().then_with(|| a.0.cmp(&b.0))
        });
        (p_gold, ranked, features, trace)
    })
}

fn training_distribution_inner(
    model: &Model,
    history: &[u16],
    gold: u16,
    base: &[f64],
    base_total: f64,
    base_top: &[u16],
) -> (f64, Vec<(u16, f64)>, Vec<Feature>, WaveTrace) {
    let mut features = collect_features(model, history, base, base_total);
    DENSE_DELTA.with(|cell| {
        let mut scratch = cell.borrow_mut();
        scratch.begin(model.vocab_size);
        for feature in &features {
            scratch.add(feature.token, feature.gate * feature.relation);
        }
        scratch.touched.sort_unstable();
        let mut z = base_total;
        for position in 0..scratch.touched.len() {
            let token = scratch.touched[position];
            let delta = scratch.values[token as usize];
            let factor = delta.clamp(-12.0, 12.0).exp();
            scratch.factors[token as usize] = factor;
            z += base[token as usize] * (factor - 1.0);
        }
        for feature in &mut features {
            let token = feature.token as usize;
            let factor = if scratch.seen[token] {
                scratch.factors[token]
            } else {
                1.0
            };
            feature.probability = base[token] * factor / z;
        }
        let p_gold = (base[gold as usize]
            * if scratch.seen[gold as usize] {
                scratch.factors[gold as usize]
            } else {
                1.0
            }
            / z)
            .max(1e-30);
        for &token in base_top {
            if !scratch.id_seen[token as usize] {
                scratch.id_seen[token as usize] = true;
                scratch.ids.push(token);
            }
        }
        for position in 0..scratch.touched.len() {
            let token = scratch.touched[position];
            if !scratch.id_seen[token as usize] {
                scratch.id_seen[token as usize] = true;
                scratch.ids.push(token);
            }
        }
        let mut top: Option<(u16, f64)> = None;
        let mut top_wrong: Option<(u16, f64)> = None;
        for &token in &scratch.ids {
            let factor = if scratch.seen[token as usize] {
                scratch.factors[token as usize]
            } else {
                1.0
            };
            let row = (token, base[token as usize] * factor / z);
            let better = |candidate: (u16, f64), old: Option<(u16, f64)>| {
                old.map(|value| {
                    candidate.1 > value.1
                        || (candidate.1 == value.1 && candidate.0 < value.0)
                })
                .unwrap_or(true)
            };
            if better(row, top) {
                top = Some(row);
            }
            if token != gold && better(row, top_wrong) {
                top_wrong = Some(row);
            }
        }
        let mut rows = take_ranked_buffer();
        if let Some(row) = top {
            rows.push(row);
        }
        if top.map(|row| row.0) == Some(gold) {
            if let Some(row) = top_wrong {
                rows.push(row);
            }
        }
        (p_gold, rows, features, WaveTrace::default())
    })
}


fn training_distribution(
    model: &Model,
    history: &[u16],
    gold: u16,
    base: &[f64],
    base_total: f64,
    base_top: &[u16],
) -> (f64, Vec<(u16, f64)>, Vec<Feature>, WaveTrace) {
    let timer = profile_timer();
    let result = training_distribution_inner(
        model, history, gold, base, base_total, base_top,
    );
    profile_finish(
        timer,
        &PROFILE_DISTRIBUTION_NS,
        &PROFILE_DISTRIBUTION_CALLS,
    );
    result
}

struct DenseProbabilityScratch {
    values: Vec<f64>,
    touched: Vec<u16>,
}

impl DenseProbabilityScratch {
    fn begin(&mut self, vocab_size: usize) {
        for &token in &self.touched {
            self.values[token as usize] = 0.0;
        }
        self.touched.clear();
        if self.values.len() != vocab_size {
            self.values = vec![0.0; vocab_size];
        }
    }
}

thread_local! {
    static DENSE_PROBABILITY: std::cell::RefCell<DenseProbabilityScratch> =
        std::cell::RefCell::new(DenseProbabilityScratch {
            values: Vec::new(),
            touched: Vec::new(),
        });
}

fn update(
    model: &mut Model,
    history: &[u16],
    gold: u16,
    p_gold: f64,
    ranked: &[(u16, f64)],
    features: &[Feature],
    options: &Options,
    stats: &mut TrainStats,
) {
    model.gcp_tokens[gold as usize].count = model.gcp_tokens[gold as usize].count.saturating_add(1);
    let mut top: Option<(u16, f64)> = None;
    let mut top_wrong: Option<(u16, f64)> = None;
    for &row in ranked {
        let better = |candidate: (u16, f64), old: Option<(u16, f64)>| {
            old.map(|value| {
                candidate.1 > value.1
                    || (candidate.1 == value.1 && candidate.0 < value.0)
            })
            .unwrap_or(true)
        };
        if better(row, top) {
            top = Some(row);
        }
        if row.0 != gold && better(row, top_wrong) {
            top_wrong = Some(row);
        }
    }
    for order in 1..=model.max_order.min(history.len()) {
        let key = make_key(history, order);
        let node = model.gcp.entry(key).or_insert_with(|| Node::new(order));
        match node.observe(
            gold,
            training_step(stats.tokens),
            options.max_successors,
            options.capacity_policy,
            options.successor_index,
        ) {
            ObserveOutcome::Created => stats.created_edges += 1,
            ObserveOutcome::RejectedAtCapacity => {
                stats.successor_capacity_rejections += 1;
                stats.successor_capacity_rejections_by_order[order - 1] += 1;
            }
            ObserveOutcome::Updated => {}
        }
    }
    {
        let mut offset = 0usize;
        while offset < features.len() {
            let key = features[offset].key;
            let mut end = offset + 1;
            while end < features.len() && features[end].key == key {
                end += 1;
            }
            // Features are emitted contiguously by context/order. Resolve the
            // outer context map once instead of once per successor edge.
            let credits = model.gcp_credit.entry(key).or_default();
            let mut credit_cursor = 0usize;
            for feature in &features[offset..end] {
                if !negative_feedback_enabled() && feature.token != gold {
                    continue;
                }
                let p = feature.probability;
                let target = if feature.token == gold { 1.0 } else { 0.0 };
                let gradient = feature.gate * (p - target);
                let credit =
                    credits.get_or_insert_ordered(feature.token, &mut credit_cursor);
                credit.gradient_sum += gradient as f32;
                credit.gradient_sq += (gradient * gradient) as f32;
                credit.weight_delta =
                    (credit.weight_delta as f64 - options.lr * gradient).clamp(-4.0, 4.0) as f32;
                credit.last_seen = training_step(stats.tokens);
                stats.edge_updates += 1;
            }
            offset = end;
        }
    }
    if let Some((wrong, _)) = top_wrong {
        if negative_feedback_enabled() && top.map(|row| row.0) != Some(gold) {
            let mut associated = false;
            for order in 1..=model.max_order.min(history.len()) {
                let key = make_key(history, order);
                let exists = model
                    .gcp
                    .get(&key)
                    .or_else(|| model.glp.get(&key))
                    .map(|node| node.edges.iter().any(|edge| edge.token == wrong))
                    .unwrap_or(false);
                if exists {
                    let credit = model
                        .gcp_credit
                        .entry(key)
                        .or_default()
                        .get_or_insert_default(wrong);
                    credit.conflicts = credit.conflicts.saturating_add(1);
                    credit.last_seen = training_step(stats.tokens);
                    associated = true;
                }
            }
            stats.conflicts += associated as u64;
        }
    }
    let _ = p_gold;
}


#[derive(Clone)]
struct PreparedUpdate {
    keys: [ContextKey; 4],
    key_count: usize,
    top_token: Option<u16>,
    top_wrong: Option<u16>,
}

fn prepare_update(
    history: &[u16],
    gold: u16,
    ranked: &[(u16, f64)],
    max_order: usize,
) -> PreparedUpdate {
    let mut top: Option<(u16, f64)> = None;
    let mut top_wrong: Option<(u16, f64)> = None;
    for &row in ranked {
        let better = |candidate: (u16, f64), old: Option<(u16, f64)>| {
            old.map(|value| {
                candidate.1 > value.1
                    || (candidate.1 == value.1 && candidate.0 < value.0)
            })
            .unwrap_or(true)
        };
        if better(row, top) {
            top = Some(row);
        }
        if row.0 != gold && better(row, top_wrong) {
            top_wrong = Some(row);
        }
    }
    let key_count = max_order.min(history.len());
    let mut keys = [ContextKey::new(0, [0; 4]); 4];
    for order in 1..=key_count {
        keys[order - 1] = make_key(history, order);
    }
    PreparedUpdate {
        keys,
        key_count,
        top_token: top.map(|row| row.0),
        top_wrong: top_wrong.map(|row| row.0),
    }
}

fn apply_prepared_update(
    model: &mut Model,
    gold: u16,
    p_gold: f64,
    features: &[Feature],
    plan: &PreparedUpdate,
    options: &Options,
    stats: &mut TrainStats,
) {
    model.gcp_tokens[gold as usize].count =
        model.gcp_tokens[gold as usize].count.saturating_add(1);
    for order in 1..=plan.key_count {
        let key = plan.keys[order - 1];
        let node = model.gcp.entry(key).or_insert_with(|| Node::new(order));
        match node.observe(
            gold,
            training_step(stats.tokens),
            options.max_successors,
            options.capacity_policy,
            options.successor_index,
        ) {
            ObserveOutcome::Created => stats.created_edges += 1,
            ObserveOutcome::RejectedAtCapacity => {
                stats.successor_capacity_rejections += 1;
                stats.successor_capacity_rejections_by_order[order - 1] += 1;
            }
            ObserveOutcome::Updated => {}
        }
    }
    let mut offset = 0usize;
    while offset < features.len() {
        let key = features[offset].key;
        let mut end = offset + 1;
        while end < features.len() && features[end].key == key {
            end += 1;
        }
        let credits = model.gcp_credit.entry(key).or_default();
        let mut credit_cursor = 0usize;
        for feature in &features[offset..end] {
            if !negative_feedback_enabled() && feature.token != gold {
                continue;
            }
            let p = feature.probability;
            let target = if feature.token == gold { 1.0 } else { 0.0 };
            let gradient = feature.gate * (p - target);
            let credit =
                credits.get_or_insert_ordered(feature.token, &mut credit_cursor);
            credit.gradient_sum += gradient as f32;
            credit.gradient_sq += (gradient * gradient) as f32;
            credit.weight_delta =
                (credit.weight_delta as f64 - options.lr * gradient)
                    .clamp(-4.0, 4.0) as f32;
            credit.last_seen = training_step(stats.tokens);
            stats.edge_updates += 1;
        }
        offset = end;
    }
    if let Some(wrong) = plan.top_wrong {
        if negative_feedback_enabled() && plan.top_token != Some(gold) {
            let mut associated = false;
            for &key in &plan.keys[..plan.key_count] {
                let exists = model
                    .gcp
                    .get(&key)
                    .or_else(|| model.glp.get(&key))
                    .map(|node| node.edges.iter().any(|edge| edge.token == wrong))
                    .unwrap_or(false);
                if exists {
                    let credit = model
                        .gcp_credit
                        .entry(key)
                        .or_default()
                        .get_or_insert_default(wrong);
                    credit.conflicts = credit.conflicts.saturating_add(1);
                    credit.last_seen = training_step(stats.tokens);
                    associated = true;
                }
            }
            stats.conflicts += associated as u64;
        }
    }
    let _ = p_gold;
}

fn merge_node(
    destination: &mut Node,
    source: &Node,
    max_successors: usize,
    capacity_policy: CapacityPolicy,
    use_index: bool,
) -> u64 {
    let mut rejected = 0;
    let old = destination.total as f64;
    let add = source.total as f64;
    let denom = (old + add).max(1.0);
    destination.potential =
        ((destination.potential as f64 * old + source.potential as f64 * add) / denom) as f32;
    destination.total = destination.total.saturating_add(source.total);
    destination.last_seen = destination.last_seen.max(source.last_seen);
    for source_edge in &source.edges {
        destination.ensure_edge_index(use_index);
        if let Some(position) = destination.edge_position(source_edge.token, use_index) {
            let edge = &mut destination.edges[position];
            let old_count = edge.count as f64;
            let add_count = source_edge.count as f64;
            let edge_denom = (old_count + add_count).max(1.0);
            edge.weight = ((edge.weight as f64 * old_count + source_edge.weight as f64 * add_count)
                / edge_denom) as f32;
            edge.sleep_bias = ((edge.sleep_bias as f64 * old_count
                + source_edge.sleep_bias as f64 * add_count)
                / edge_denom) as f32;
            edge.count = edge.count.saturating_add(source_edge.count);
            edge.gradient_sum += source_edge.gradient_sum;
            edge.gradient_sq += source_edge.gradient_sq;
            edge.conflicts = edge.conflicts.saturating_add(source_edge.conflicts);
            edge.last_seen = edge.last_seen.max(source_edge.last_seen);
        } else if destination.edges.len() < capacity_policy.limit(destination, max_successors) {
            let position = destination.edges.len();
            destination.edges.push(*source_edge);
            destination.index_inserted_edge(source_edge.token, position);
            destination.ensure_edge_index(use_index);
        } else {
            rejected += 1;
        }
    }
    rejected
}

fn prune_graph<S: std::hash::BuildHasher>(
    graph: &mut HashMap<ContextKey, Node, S>,
    maximum: usize,
) -> usize {
    if graph.len() <= maximum {
        return 0;
    }
    let remove = graph.len() - maximum;
    let mut rows: Vec<_> = graph
        .iter()
        .map(|(key, node)| (*key, node.total, node.last_seen))
        .collect();
    // Context keys close the otherwise ambiguous boundary when several nodes
    // have identical support and recency. This makes pruning independent of
    // HashMap iteration/randomization and therefore exactly reproducible.
    // EXP152: the full ordering is not observable. The context key makes
    // every pruning key unique, so deterministic nth-selection removes
    // exactly the same set as a complete sort in expected linear time.
    if remove < rows.len() {
        rows.select_nth_unstable_by_key(remove, |row| (row.1, row.2, row.0));
    }
    for &(key, _, _) in &rows[..remove] {
        graph.remove(&key);
    }
    remove
}


fn prune_graph_collect_inner<S: std::hash::BuildHasher>(
    graph: &mut HashMap<ContextKey, Node, S>,
    maximum: usize,
) -> Vec<ContextKey> {
    if graph.len() <= maximum {
        return Vec::new();
    }
    let remove = graph.len() - maximum;
    let mut rows: Vec<_> = graph
        .iter()
        .map(|(key, node)| (*key, node.total, node.last_seen))
        .collect();
    if remove < rows.len() {
        rows.select_nth_unstable_by_key(remove, |row| (row.1, row.2, row.0));
    }
    let removed: Vec<_> = rows[..remove].iter().map(|row| row.0).collect();
    for key in &removed {
        graph.remove(key);
    }
    removed
}


fn prune_graph_collect<S: std::hash::BuildHasher>(
    graph: &mut HashMap<ContextKey, Node, S>,
    maximum: usize,
) -> Vec<ContextKey> {
    let timer = profile_timer();
    let result = prune_graph_collect_inner(graph, maximum);
    profile_finish(timer, &PROFILE_PRUNE_NS, &PROFILE_PRUNE_CALLS);
    result
}

fn light_reflection(model: &mut Model, options: &Options, stats: &mut TrainStats) {
    stats.light_reflections += 1;
    stats.pruned_gcp += prune_graph(&mut model.gcp, options.gcp_max) as u64;
}

fn parallel_merge_gcp_snapshot(
    glp: &ContextMap<Node>,
    graph: ContextMap<Node>,
    options: &Options,
    workers: usize,
) -> Vec<(ContextKey, Node, u64)> {
    let sources: Vec<_> = graph.into_iter().collect();
    if sources.is_empty() {
        return Vec::new();
    }
    let workers = workers.min(sources.len());
    let chunk_size = (sources.len() + workers - 1) / workers;
    let groups: Vec<Vec<(ContextKey, Node, u64)>> = std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for chunk in sources.chunks(chunk_size) {
            handles.push(scope.spawn(move || {
                let mut journal = Vec::with_capacity(chunk.len());
                for (key, source) in chunk {
                    let mut destination = glp
                        .get(key)
                        .cloned()
                        .unwrap_or_else(|| Node::new(key.len() as usize));
                    let rejected = merge_node(
                        &mut destination,
                        source,
                        options.max_successors,
                        options.capacity_policy,
                        options.successor_index,
                    );
                    journal.push((*key, destination, rejected));
                }
                journal
            }));
        }
        handles
            .into_iter()
            .map(|handle| handle.join().expect("EXP152 deep-merge worker panicked"))
            .collect()
    });
    groups.into_iter().flatten().collect()
}

fn merge_graph_canonically_inner(
    target: &mut ContextMap<Node>,
    graph: ContextMap<Node>,
    options: &Options,
    stats: &mut TrainStats,
) {
    let workers = parallel_train_workers();
    if workers > 1 && graph.len() >= 1024 {
        let journal = parallel_merge_gcp_snapshot(target, graph, options, workers);
        for (key, node, rejected) in journal {
            target.insert(key, node);
            stats.sleep_successor_capacity_rejections += rejected;
        }
    } else {
        let mut rows: Vec<_> = graph.into_iter().collect();
        rows.sort_unstable_by_key(|row| row.0);
        for (key, node) in rows {
            stats.sleep_successor_capacity_rejections += merge_node(
                target.entry(key).or_insert_with(|| Node::new(key.len() as usize)),
                &node,
                options.max_successors,
                options.capacity_policy,
                options.successor_index,
            );
        }
    }
}


fn merge_graph_canonically(
    target: &mut ContextMap<Node>,
    graph: ContextMap<Node>,
    options: &Options,
    stats: &mut TrainStats,
) {
    let timer = profile_timer();
    merge_graph_canonically_inner(target, graph, options, stats);
    profile_finish(timer, &PROFILE_MERGE_NS, &PROFILE_MERGE_CALLS);
}

fn apply_credit_to_graph(
    graph: &mut ContextMap<Node>,
    credits: ContextMap<CreditNode>,
    options: &Options,
) {
    for (key, rows) in credits {
        for (token, credit) in rows.rows {
            if let Some(node) = graph.get_mut(&key) {
                node.ensure_edge_index(options.successor_index);
                let Some(position) = node.edge_position(token, options.successor_index) else {
                    continue;
                };
                let edge = &mut node.edges[position];
                edge.weight =
                    (edge.weight as f64 + credit.weight_delta as f64).clamp(-4.0, 4.0) as f32;
                edge.gradient_sum += credit.gradient_sum;
                edge.gradient_sq += credit.gradient_sq;
                edge.conflicts = edge.conflicts.saturating_add(credit.conflicts);
                edge.last_seen = edge.last_seen.max(credit.last_seen);
            }
        }
    }
}

fn tri_hierarchical_dream(model: &mut Model, stats: &mut TrainStats) {
    let mut keys: Vec<_> = model.gcp.keys().copied().collect();
    keys.sort_unstable();
    keys.truncate(8192);
    for key in keys {
        let Some(recent) = model.gcp.get(&key) else { continue };
        let Some(long) = model.glp.get(&key) else { continue };
        let Some(recent_top) = recent.edges.iter().max_by_key(|edge| (edge.count, std::cmp::Reverse(edge.token))).map(|edge| edge.token) else { continue };
        let Some(long_top) = long.edges.iter().max_by_key(|edge| (edge.count, std::cmp::Reverse(edge.token))).map(|edge| edge.token) else { continue };
        let Some(medial) = model.gm1.get_mut(&key) else { continue };
        stats.gm1_dream_candidates += 1;
        let desired = if recent_top == long_top { recent_top } else { long_top };
        if let Some(edge) = medial.edges.iter_mut().find(|edge| edge.token == desired) {
            if recent_top == long_top {
                edge.sleep_bias = (edge.sleep_bias + 0.02).clamp(-0.60, 0.60);
                edge.weight = (edge.weight + 0.005).clamp(-4.0, 4.0);
                stats.gm1_dream_reinforced += 1;
            } else {
                edge.sleep_bias = (edge.sleep_bias - 0.01).clamp(-0.60, 0.60);
                edge.weight = (edge.weight - 0.0025).clamp(-4.0, 4.0);
                edge.conflicts = edge.conflicts.saturating_add(1);
                stats.gm1_dream_inhibited += 1;
            }
        }
    }
}

fn refresh_gm1_cache(model: &mut Model, options: &Options, stats: &mut TrainStats) {
    model.gm1_cache.clear();
    let mut keys: Vec<_> = model.gm1.keys().copied().collect();
    keys.sort_unstable();
    model.gm1_cache.reserve(keys.len());
    for key in keys {
        let medial = model.gm1.get(&key).expect("GM1 key").clone();
        let mut composite = model
            .glp
            .get(&key)
            .cloned()
            .unwrap_or_else(|| Node::new(key.len() as usize));
        stats.sleep_successor_capacity_rejections += merge_node(
            &mut composite,
            &medial,
            options.max_successors,
            options.capacity_policy,
            options.successor_index,
        );
        model.gm1_cache.insert(key, composite);
    }
    stats.gm1_cache_refreshes += 1;
    stats.gm1_cache_contexts += model.gm1_cache.len() as u64;
    stats.gm1_cache_peak = stats.gm1_cache_peak.max(model.gm1_cache.len());
}


fn lazy_admission_quota() -> Option<usize> {
    std::env::var("NGU_ADMISSION_QUOTA")
        .ok()
        .and_then(|value| value.parse().ok())
}

fn bounded_glp_admission(
    model: &mut Model,
    graph: ContextMap<Node>,
    maximum: usize,
) -> (ContextMap<Node>, Vec<ContextKey>) {
    let Some(quota) = lazy_admission_quota() else {
        return (graph, Vec::new());
    };
    let freeze_identities = std::env::var("NGU_FREEZE_IDENTITIES")
        .map(|value| value == "1" || value.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    if model.glp.len() < maximum && !freeze_identities {
        return (graph, Vec::new());
    }

    let mut existing = ContextMap::default();
    let mut novel = Vec::new();
    for (key, node) in graph {
        if model.glp.contains_key(&key) {
            existing.insert(key, node);
        } else {
            novel.push((key, node));
        }
    }
    if novel.is_empty() || quota == 0 {
        return (existing, Vec::new());
    }

    let protected: BTreeSet<_> = existing.keys().copied().collect();
    let mut victims = Vec::with_capacity(quota);
    for (_, key) in model.route_order.iter().rev() {
        if protected.contains(key) {
            continue;
        }
        victims.push(*key);
        if victims.len() == quota || victims.len() == novel.len() {
            break;
        }
    }
    if victims.is_empty() {
        return (existing, Vec::new());
    }

    novel.sort_unstable_by(|left, right| {
        right
            .1
            .total
            .cmp(&left.1.total)
            .then_with(|| right.1.last_seen.cmp(&left.1.last_seen))
            .then_with(|| left.0.cmp(&right.0))
    });
    novel.truncate(victims.len());

    for key in &victims {
        model.glp.remove(key);
    }
    existing.reserve(novel.len());
    existing.extend(novel);
    (existing, victims)
}

fn deep_reflection(model: &mut Model, options: &Options, stats: &mut TrainStats) {
    stats.deep_reflections += 1;
    if model.gm1_mode == 4 {
        tri_hierarchical_dream(model, stats);
    }

    if model.gm1_mode < 2 {
        // M0 is the exact legacy path. M1 additionally creates a shadow GM1,
        // which is never read by inference and therefore cannot change it.
        let shadow = (model.gm1_mode == 1).then(|| model.gcp.clone());
        for token in 0..model.vocab_size {
            let source = model.gcp_tokens[token];
            model.glp_tokens[token].count =
                model.glp_tokens[token].count.saturating_add(source.count);
            if source.count > 0 {
                model.glp_tokens[token].potential = source.potential;
            }
            if model.gm1_mode == 1 {
                model.gm1_tokens[token].count =
                    model.gm1_tokens[token].count.saturating_add(source.count);
                if source.count > 0 {
                    model.gm1_tokens[token].potential = source.potential;
                }
            }
            model.gcp_tokens[token] = TokenVertex::default();
        }
        let graph = std::mem::take(&mut model.gcp);
        let credits = std::mem::take(&mut model.gcp_credit);
        let (graph, mut admission_removed) =
            bounded_glp_admission(model, graph, options.glp_max);
        let mut active_touched: ContextMap<Vec<u16>> = ContextMap::default();
        for (key, node) in &graph {
            active_touched
                .entry(*key)
                .or_default()
                .extend(node.edges.iter().map(|edge| edge.token));
        }
        for (key, rows) in &credits {
            if model.glp.contains_key(key) || graph.contains_key(key) {
                active_touched
                    .entry(*key)
                    .or_default()
                    .extend(rows.rows.iter().map(|row| row.0));
            }
        }
        let mut route_changed: Vec<_> = graph.keys().copied().collect();
        route_changed.extend(
            credits
                .keys()
                .filter(|key| model.glp.contains_key(key))
                .copied(),
        );
        merge_graph_canonically(&mut model.glp, graph, options, stats);
        apply_credit_to_graph(&mut model.glp, credits, options);
        if let Some(graph) = shadow {
            stats.gcp_to_gm1_contexts += graph.len() as u64;
            merge_graph_canonically(&mut model.gm1, graph, options, stats);
            stats.pruned_gm1 += prune_graph(&mut model.gm1, options.gm1_max) as u64;
            stats.gm1_peak = stats.gm1_peak.max(model.gm1.len());
        }
        let mut route_removed = if exp172_indexed_forgetting(model.shard_mode) {
            update_eviction_index(model, &route_changed, stats);
            prune_graph_indexed(model, options.glp_max, stats)
        } else if matches!(
            model.shard_mode,
            6 | 7 | 8 | 9 | 11 | 13 | 14 | 15
        ) {
            if model.glp.len() > options.glp_max {
                stats.global_prune_calls += 1;
                stats.global_prune_contexts_scanned += model.glp.len() as u64;
            }
            prune_graph_collect(&mut model.glp, options.glp_max)
        } else {
            let count = prune_graph(&mut model.glp, options.glp_max);
            stats.pruned_glp += count as u64;
            Vec::new()
        };
        route_removed.append(&mut admission_removed);
        if matches!(model.shard_mode, 6..=15) {
            stats.pruned_glp += route_removed.len() as u64;
            if !route_removed.is_empty() {
                update_victim_fingerprint(stats, &route_removed);
            }
        }
        refresh_active_glp_incremental(model, &active_touched, &route_removed);
        model.glp_version = model.glp_version.wrapping_add(1);
        refresh_exp171_routing(model, &route_changed, &route_removed, stats);
        refresh_flat_layout(model);
        return;
    }

    // Promote only sufficiently supported medial contexts that survived a
    // complete sleep interval. New GCP evidence enters GM1 after promotion,
    // so it cannot jump directly to GLP in the same cycle.
    let now = model.gcp.values().map(|node| node.last_seen).max().unwrap_or(0);
    let old_medial = std::mem::take(&mut model.gm1);
    let mut keep = ContextMap::default();
    let mut promote = ContextMap::default();
    let age = options.deep_interval as u64;
    let mut rows: Vec<_> = old_medial.into_iter().collect();
    rows.sort_unstable_by_key(|row| row.0);
    for (key, node) in rows {
        if (node.total >= 2 || key.len() >= 3)
            && node.last_seen.saturating_add(age) <= now
        {
            promote.insert(key, node);
        } else {
            keep.insert(key, node);
        }
    }
    stats.gm1_to_glp_contexts += promote.len() as u64;
    model.gm1 = keep;
    merge_graph_canonically(&mut model.glp, promote, options, stats);

    for token in 0..model.vocab_size {
        let promote_count = model.gm1_tokens[token].count / 4;
        model.gm1_tokens[token].count -= promote_count;
        model.glp_tokens[token].count =
            model.glp_tokens[token].count.saturating_add(promote_count);
        if promote_count > 0 {
            model.glp_tokens[token].potential = model.gm1_tokens[token].potential;
        }
        let source = model.gcp_tokens[token];
        model.gm1_tokens[token].count =
            model.gm1_tokens[token].count.saturating_add(source.count);
        if source.count > 0 {
            model.gm1_tokens[token].potential = source.potential;
        }
        model.gcp_tokens[token] = TokenVertex::default();
    }

    let graph = std::mem::take(&mut model.gcp);
    stats.gcp_to_gm1_contexts += graph.len() as u64;
    merge_graph_canonically(&mut model.gm1, graph, options, stats);
    apply_credit_to_graph(
        &mut model.gm1,
        std::mem::take(&mut model.gcp_credit),
        options,
    );
    stats.pruned_gm1 += prune_graph(&mut model.gm1, options.gm1_max) as u64;
    stats.pruned_glp += prune_graph(&mut model.glp, options.glp_max) as u64;
    stats.gm1_peak = stats.gm1_peak.max(model.gm1.len());
    // This composite is a deep medial reflection, not a wake cache.
    if (model.medial_mode == 0 || model.medial_mode == 4)
        && (stats.deep_reflections % 10 == 0
            || options.train_limit <= options.deep_interval)
    {
        refresh_gm1_cache(model, options, stats);
    } else if model.medial_mode != 0 && model.medial_mode != 4 {
        model.gm1_cache.clear();
    }
    model.glp_version = model.glp_version.wrapping_add(1);
    refresh_exp170_routing(model, stats);
    refresh_flat_layout(model);
}

fn glp_fingerprint(model: &Model) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    let mut mix = |value: u64| {
        hash ^= value;
        hash = hash.wrapping_mul(0x100000001b3);
    };
    for vertex in &model.glp_tokens {
        mix(vertex.count as u64);
        mix(vertex.potential.to_bits() as u64);
    }
    for (key, node) in &model.glp {
        mix(key.len() as u64);
        for token in key.tokens() {
            mix(token as u64);
        }
        mix(node.total as u64);
        mix(node.last_seen);
        mix(node.potential.to_bits() as u64);
        for edge in &node.edges {
            mix(edge.token as u64);
            mix(edge.count as u64);
            mix(edge.weight.to_bits() as u64);
            mix(edge.gradient_sum.to_bits() as u64);
            mix(edge.gradient_sq.to_bits() as u64);
            mix(edge.conflicts as u64);
            mix(edge.last_seen);
            mix(edge.sleep_bias.to_bits() as u64);
        }
    }
    let mut constructions = model.glp_constructions.clone();
    constructions.sort_unstable_by_key(|row| row.key);
    for construction in constructions {
        mix(construction.key.len as u64);
        mix(construction.key.slot_mask as u64);
        for symbol in construction.key.symbols {
            mix(symbol as u64);
        }
        mix(construction.key.target as u64);
        mix(construction.support as u64);
        mix(construction.confirmations as u64);
        mix(construction.value.to_bits() as u64);
        mix(construction.dag_edge_count as u64);
        for edge in &construction.dag_edges[..construction.dag_edge_count as usize] {
            mix(edge.source as u64);
            mix(edge.target as u64);
            mix(edge.relation as u64);
            mix(edge.weight.to_bits() as u64);
            mix(edge.gradient_sum.to_bits() as u64);
            mix(edge.gradient_sq.to_bits() as u64);
            mix(edge.credit_count as u64);
        }
    }
    hash
}

#[derive(Clone)]
struct SleepExample {
    history: Vec<u16>,
    gold: u16,
}

fn sleep_examples(tokens: &[u16], max_examples: usize, max_order: usize) -> Vec<SleepExample> {
    let usable = tokens.len() * 9 / 10;
    if usable < 2 {
        return Vec::new();
    }
    let stride = (usable / max_examples.max(1)).max(1);
    let mut history = Vec::with_capacity(max_order);
    let mut rows = Vec::new();
    for (index, &gold) in tokens[..usable].iter().enumerate() {
        if index % 512 == 0 || gold == 0 {
            history.clear();
        }
        if !history.is_empty() && index % stride == 0 {
            rows.push(SleepExample {
                history: history.clone(),
                gold,
            });
            if rows.len() == max_examples {
                break;
            }
        }
        history.push(gold);
        if history.len() > max_order {
            history.remove(0);
        }
    }
    rows
}

#[derive(Clone, Copy)]
struct BiasChange {
    gcp: bool,
    key: ContextKey,
    token: u16,
    old: f32,
    new: f32,
}

fn add_path_credit(credit: &mut HashMap<(u16, u16), (f64, u32)>, edges: &[(u16, u16)], value: f64) {
    for &edge in edges {
        let row = credit.entry(edge).or_insert((0.0, 0));
        row.0 += value;
        row.1 += 1;
    }
}

fn direct_edge_exists(model: &Model, source: u16, target: u16) -> bool {
    let key = ContextKey::new(1, [source, 0, 0, 0]);
    model
        .gcp
        .get(&key)
        .into_iter()
        .chain(model.gm1.get(&key))
        .chain(model.glp.get(&key))
        .any(|node| node.edges.iter().any(|edge| edge.token == target))
}

fn necessity_from_trace(target: &CandidateGr, intervened: &WaveTrace) -> f64 {
    let Some(after) = intervened
        .candidates
        .iter()
        .find(|candidate| candidate.token == target.token)
    else {
        return 1.0;
    };
    let drop = target.bonus - after.bonus;
    if drop < 0.01 {
        0.0
    } else {
        (drop / 0.05).clamp(0.0, 2.0)
    }
}

#[derive(Clone)]
struct BridgeState {
    node: u16,
    cost: f64,
    edges: Vec<(u16, u16)>,
    nodes: Vec<u16>,
}

#[derive(Clone)]
struct Bridge {
    edges: Vec<(u16, u16)>,
    coverage: f64,
    dispersion: f64,
    cost: f64,
    full: bool,
}

#[derive(Clone, Copy, Debug)]
struct BridgeValidation {
    accepted: bool,
    gold_found: bool,
    margin: f64,
    wrong_promotion: f64,
    gold_bonus: f64,
}

#[derive(Clone)]
struct PendingBridge {
    history: Vec<u16>,
    gold: u16,
    bridge: Bridge,
    magnitude: f64,
    age: u8,
}

#[derive(Clone)]
struct PendingOrigin {
    history: Vec<u16>,
    gold: u16,
    path: Vec<(u16, u16)>,
    baseline_coverage: usize,
    age: u8,
}

#[derive(Clone)]
struct DeferredProposal {
    origin_history: Vec<u16>,
    origin_gold: u16,
    routes: Vec<Vec<(u16, u16)>>,
    edges: Vec<(u16, u16)>,
    dose: u8,
    age: u8,
    evidence_windows: u8,
    supported_windows: u8,
    adverse_windows: u8,
    evaluated_examples: u32,
    gold_delta: f64,
    full_delta: f64,
    coverage_delta: f64,
    margin_delta: f64,
    wrong_promotion: f64,
    value: f64,
    last_value: f64,
    edge_value: HashMap<(u16, u16), f64>,
    synergy: u16,
    conflict: u16,
}

#[derive(Clone, Copy, Debug, Default)]
struct DeferredEvidence {
    evaluated: usize,
    gold_delta: f64,
    full_delta: f64,
    coverage_delta: f64,
    margin_delta: f64,
    wrong_promotion: f64,
    value: f64,
    supported: bool,
    adverse: bool,
}

#[derive(Clone, Debug)]
struct PendingConstruction {
    construction: Construction,
    age: u8,
    evidence_windows: u8,
    supported_windows: u8,
    adverse_windows: u8,
    evaluated_examples: u32,
    gold_delta: f64,
    full_delta: f64,
    coverage_delta: f64,
    margin_delta: f64,
    wrong_promotion: f64,
    value: f64,
    edge_value: [f64; DAG_EDGE_COUNT],
    edge_value_sq: [f64; DAG_EDGE_COUNT],
    edge_tests: [u32; DAG_EDGE_COUNT],
}

#[derive(Clone, Copy, Debug, Default)]
struct ConstructionEvidence {
    matched: usize,
    gold_delta: f64,
    full_delta: f64,
    coverage_delta: f64,
    margin_delta: f64,
    wrong_promotion: f64,
    value: f64,
    edge_value: [f64; DAG_EDGE_COUNT],
    edge_value_sq: [f64; DAG_EDGE_COUNT],
    edge_tests: [u32; DAG_EDGE_COUNT],
    supported: bool,
    adverse: bool,
}

#[derive(Clone, Debug)]
struct RouteSetChoice {
    indices: Vec<usize>,
    dose: u8,
    accepted: bool,
    full: bool,
    coverage: usize,
    bonus: f64,
    margin: f64,
    wrong_promotion: f64,
    edge_count: usize,
    score: f64,
}

#[derive(Clone, Copy, Debug)]
struct MultiDoseValidation {
    accepted: bool,
    dose: u8,
    last: BridgeValidation,
    monotonic: bool,
}

#[derive(Clone, Copy, Debug)]
struct PhaseValidation {
    accepted: bool,
    gold_found: bool,
    coverage_preserved: bool,
    improved: bool,
    bonus_preserved: bool,
    wrong_ok: bool,
    full_created: bool,
    after_dispersion: f64,
}

fn validate_phase_vector(
    model: &Model,
    history: &[u16],
    gold: u16,
    baseline: &WaveTrace,
    good: &CandidateGr,
    base: &[f64],
    total: f64,
) -> PhaseValidation {
    let overlay: HashMap<_, _> = good.phase_deltas.iter().copied().collect();
    let prospective = wave_trace_modified(
        model,
        history,
        base,
        total,
        Mode::FrequencyGradient,
        None,
        Some(&overlay),
    );
    let baseline_wrong = baseline
        .candidates
        .iter()
        .filter(|candidate| candidate.token != gold)
        .map(|candidate| candidate.bonus)
        .fold(-0.15, f64::max);
    let prospective_wrong = prospective
        .candidates
        .iter()
        .filter(|candidate| candidate.token != gold)
        .map(|candidate| candidate.bonus)
        .fold(-0.15, f64::max);
    let after = prospective
        .candidates
        .iter()
        .find(|candidate| candidate.token == gold);
    let gold_found = after.is_some();
    let coverage_preserved = after
        .map(|candidate| candidate.coverage >= good.coverage)
        .unwrap_or(false);
    let full_created = after.map(|candidate| candidate.full).unwrap_or(false) && !good.full;
    let after_dispersion = after
        .map(|candidate| candidate.dispersion)
        .unwrap_or(f64::INFINITY);
    let improved = full_created || after_dispersion <= good.dispersion - 0.02;
    let bonus_preserved = after
        .map(|candidate| candidate.bonus + 0.01 >= good.bonus)
        .unwrap_or(false);
    let wrong_ok = prospective_wrong - baseline_wrong <= PROSPECTIVE_MAX_WRONG_PROMOTION;
    PhaseValidation {
        accepted: !good.phase_deltas.is_empty()
            && gold_found
            && coverage_preserved
            && improved
            && bonus_preserved
            && wrong_ok,
        gold_found,
        coverage_preserved,
        improved,
        bonus_preserved,
        wrong_ok,
        full_created,
        after_dispersion,
    }
}

fn validate_bridge_at_dose(
    model: &Model,
    history: &[u16],
    gold: u16,
    baseline: &WaveTrace,
    bridge: &Bridge,
    magnitude: f64,
    base: &[f64],
    total: f64,
    dose: u8,
) -> BridgeValidation {
    // This is the exact per-example delta proposed by the sleep rule, before
    // cycle-level averaging. The real journal remains untouched at this point.
    let single = (-0.02 * magnitude).clamp(-0.03, 0.03);
    let delta = (single * dose as f64).clamp(-0.12, 0.12);
    let overlay: HashMap<_, _> = bridge.edges.iter().map(|&edge| (edge, delta)).collect();
    let prospective = wave_trace_modified(
        model,
        history,
        base,
        total,
        Mode::FrequencyGradient,
        None,
        Some(&overlay),
    );
    let baseline_wrong = baseline
        .candidates
        .iter()
        .filter(|gr| gr.token != gold)
        .map(|gr| gr.bonus)
        .fold(-0.15, f64::max);
    let prospective_wrong = prospective
        .candidates
        .iter()
        .filter(|gr| gr.token != gold)
        .map(|gr| gr.bonus)
        .fold(-0.15, f64::max);
    let gold_bonus = prospective
        .candidates
        .iter()
        .find(|gr| gr.token == gold)
        .map(|gr| gr.bonus);
    let wrong_promotion = prospective_wrong - baseline_wrong;
    let margin = gold_bonus
        .map(|bonus| bonus - prospective_wrong)
        .unwrap_or(f64::NEG_INFINITY);
    let gold_found = gold_bonus.is_some();
    BridgeValidation {
        accepted: gold_found
            && margin >= PROSPECTIVE_MARGIN_FLOOR
            && wrong_promotion <= PROSPECTIVE_MAX_WRONG_PROMOTION,
        gold_found,
        margin,
        wrong_promotion,
        gold_bonus: gold_bonus.unwrap_or(-0.15),
    }
}

fn validate_bridge_prospectively(
    model: &Model,
    history: &[u16],
    gold: u16,
    baseline: &WaveTrace,
    bridge: &Bridge,
    magnitude: f64,
    base: &[f64],
    total: f64,
) -> BridgeValidation {
    validate_bridge_at_dose(
        model, history, gold, baseline, bridge, magnitude, base, total, 1,
    )
}

fn validate_bridge_multidose(
    model: &Model,
    history: &[u16],
    gold: u16,
    baseline: &WaveTrace,
    bridge: &Bridge,
    magnitude: f64,
    base: &[f64],
    total: f64,
    stats: &mut TrainStats,
) -> MultiDoseValidation {
    let mut previous: Option<BridgeValidation> = None;
    let mut monotonic = true;
    let mut last = BridgeValidation {
        accepted: false,
        gold_found: false,
        margin: f64::NEG_INFINITY,
        wrong_promotion: 0.0,
        gold_bonus: -0.15,
    };
    for dose in [1u8, 2, 4] {
        stats.multidose_virtual_traces += 1;
        let current = validate_bridge_at_dose(
            model, history, gold, baseline, bridge, magnitude, base, total, dose,
        );
        if let Some(old) = previous {
            if old.gold_found
                && current.gold_found
                && (current.gold_bonus + 1e-12 < old.gold_bonus
                    || current.margin + 1e-12 < old.margin)
            {
                monotonic = false;
            }
        }
        last = current;
        if current.accepted && monotonic {
            match dose {
                1 => stats.multidose_dose1_pass += 1,
                2 => stats.multidose_dose2_pass += 1,
                4 => stats.multidose_dose4_pass += 1,
                _ => unreachable!(),
            }
            return MultiDoseValidation {
                accepted: true,
                dose,
                last: current,
                monotonic,
            };
        }
        previous = Some(current);
    }
    stats.multidose_rejected += 1;
    if !monotonic {
        stats.multidose_monotonic_rejected += 1;
    }
    MultiDoseValidation {
        accepted: false,
        dose: 0,
        last,
        monotonic,
    }
}

fn build_reverse_index(model: &Model, base: &[f64], total: f64) -> Vec<Vec<(u16, f64)>> {
    let mut reverse = vec![Vec::new(); model.vocab_size];
    for source in 0..model.vocab_size {
        for (target, _, length, _) in successors(
            model,
            source as u16,
            base,
            total,
            Mode::FrequencyGradient,
            None,
        ) {
            reverse[target as usize].push((source as u16, length));
        }
    }
    for rows in &mut reverse {
        rows.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap().then_with(|| a.0.cmp(&b.0)));
        rows.dedup_by_key(|row| row.0);
        rows.truncate(4);
    }
    reverse
}

fn bidirectional_bridge(
    model: &Model,
    history: &[u16],
    gold: u16,
    base: &[f64],
    total: f64,
    reverse: &[Vec<(u16, f64)>],
) -> Option<Bridge> {
    if history.len() < 2 {
        return None;
    }
    let mut backward: HashMap<u16, BridgeState> = HashMap::new();
    let start = BridgeState {
        node: gold,
        cost: 0.0,
        edges: Vec::new(),
        nodes: vec![gold],
    };
    backward.insert(gold, start.clone());
    let mut frontier = vec![start];
    for _ in 0..3 {
        let mut next: HashMap<u16, BridgeState> = HashMap::new();
        for state in &frontier {
            for &(pred, length) in &reverse[state.node as usize] {
                if state.nodes.contains(&pred) {
                    continue;
                }
                let mut candidate = state.clone();
                candidate.node = pred;
                candidate.cost += length;
                candidate.edges.insert(0, (pred, state.node));
                candidate.nodes.insert(0, pred);
                if next
                    .get(&pred)
                    .map(|old| candidate.cost < old.cost)
                    .unwrap_or(true)
                {
                    next.insert(pred, candidate);
                }
            }
        }
        let mut rows: Vec<_> = next.into_values().collect();
        rows.sort_unstable_by(|a, b| {
            a.cost
                .partial_cmp(&b.cost)
                .unwrap()
                .then_with(|| a.node.cmp(&b.node))
        });
        rows.truncate(8);
        for state in &rows {
            if backward
                .get(&state.node)
                .map(|old| state.cost < old.cost)
                .unwrap_or(true)
            {
                backward.insert(state.node, state.clone());
            }
        }
        frontier = rows;
        if frontier.is_empty() {
            break;
        }
    }
    let window = history.len().min(4);
    let input = &history[history.len() - window..];
    let mut meetings: HashMap<u16, Vec<(usize, BridgeState)>> = HashMap::new();
    for (origin, &token) in input.iter().enumerate() {
        let initial = BridgeState {
            node: token,
            cost: 0.0,
            edges: Vec::new(),
            nodes: vec![token],
        };
        let mut reached = HashMap::new();
        reached.insert(token, initial.clone());
        let mut forward = vec![initial];
        for _ in 0..3 {
            let mut next: HashMap<u16, BridgeState> = HashMap::new();
            for state in &forward {
                for (target, _, length, _) in successors(
                    model,
                    state.node,
                    base,
                    total,
                    Mode::FrequencyGradient,
                    None,
                ) {
                    if state.nodes.contains(&target) {
                        continue;
                    }
                    let mut candidate = state.clone();
                    candidate.node = target;
                    candidate.cost += length;
                    candidate.edges.push((state.node, target));
                    candidate.nodes.push(target);
                    if next
                        .get(&target)
                        .map(|old| candidate.cost < old.cost)
                        .unwrap_or(true)
                    {
                        next.insert(target, candidate);
                    }
                }
            }
            let mut rows: Vec<_> = next.into_values().collect();
            rows.sort_unstable_by(|a, b| {
                a.cost
                    .partial_cmp(&b.cost)
                    .unwrap()
                    .then_with(|| a.node.cmp(&b.node))
            });
            rows.truncate(8);
            for state in &rows {
                if reached
                    .get(&state.node)
                    .map(|old| state.cost < old.cost)
                    .unwrap_or(true)
                {
                    reached.insert(state.node, state.clone());
                }
            }
            forward = rows;
            if forward.is_empty() {
                break;
            }
        }
        for (node, state) in reached {
            if backward.contains_key(&node) {
                meetings.entry(node).or_default().push((origin, state));
            }
        }
    }
    let mut best: Option<(f64, u16, Bridge)> = None;
    for (meeting, states) in meetings {
        if states.len() < 2 {
            continue;
        }
        let back = &backward[&meeting];
        let arrivals: Vec<_> = states
            .iter()
            .map(|(origin, state)| *origin as f64 + state.cost + back.cost)
            .collect();
        let min = arrivals.iter().copied().fold(f64::INFINITY, f64::min);
        let max = arrivals.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let dispersion = max - min;
        if dispersion > 2.5 {
            continue;
        }
        let coverage = states.len() as f64 / window as f64;
        let cost = states
            .iter()
            .map(|(_, state)| state.cost + back.cost)
            .sum::<f64>()
            / states.len() as f64;
        let mut edges: Vec<_> = states
            .iter()
            .flat_map(|(_, state)| state.edges.iter().copied())
            .chain(back.edges.iter().copied())
            .collect();
        edges.sort_unstable();
        edges.dedup();
        if edges.is_empty() {
            continue;
        }
        let bridge = Bridge {
            edges,
            coverage,
            dispersion,
            cost,
            full: states.len() == window,
        };
        let score = 2.0 * coverage - 0.5 * dispersion - 0.05 * cost;
        if best
            .as_ref()
            .map(|(old, node, _)| score > *old || (score == *old && meeting < *node))
            .unwrap_or(true)
        {
            best = Some((score, meeting, bridge));
        }
    }
    best.map(|row| row.2)
}

fn find_origin_path(
    model: &Model,
    source: u16,
    gold: u16,
    base: &[f64],
    total: f64,
) -> Option<Vec<(u16, u16)>> {
    let mut frontier = vec![BridgeState {
        node: source,
        cost: 0.0,
        edges: Vec::new(),
        nodes: vec![source],
    }];
    for _ in 0..WAVE_DEPTH {
        let mut best: HashMap<u16, BridgeState> = HashMap::new();
        for state in &frontier {
            for (target, _, length, _) in successors(
                model,
                state.node,
                base,
                total,
                Mode::FrequencyGradient,
                None,
            ) {
                if state.nodes.contains(&target) {
                    continue;
                }
                let mut candidate = state.clone();
                candidate.node = target;
                candidate.cost += length;
                candidate.edges.push((state.node, target));
                candidate.nodes.push(target);
                if target == gold {
                    return Some(candidate.edges);
                }
                if best
                    .get(&target)
                    .map(|old| candidate.cost < old.cost)
                    .unwrap_or(true)
                {
                    best.insert(target, candidate);
                }
            }
        }
        frontier = best.into_values().collect();
        frontier.sort_unstable_by(|a, b| a.cost.partial_cmp(&b.cost).unwrap());
        frontier.truncate(16);
        if frontier.is_empty() {
            break;
        }
    }
    None
}

fn journal_from_credit(model: &Model, credit: HashMap<(u16, u16), (f64, u32)>) -> Vec<BiasChange> {
    let deltas: HashMap<_, _> = credit
        .into_iter()
        .map(|(edge, (sum, count))| (edge, (0.02 * sum / count.max(1) as f64).clamp(-0.03, 0.03)))
        .collect();
    let mut journal = Vec::new();
    for (gcp, graph) in [(false, &model.glp), (true, &model.gcp)] {
        for (key, node) in graph {
            if key.len() != 1 {
                continue;
            }
            let source = key.tokens()[0];
            for edge in &node.edges {
                if let Some(delta) = deltas.get(&(source, edge.token)) {
                    journal.push(BiasChange {
                        gcp,
                        key: *key,
                        token: edge.token,
                        old: edge.sleep_bias,
                        new: (edge.sleep_bias as f64 + delta).clamp(-0.60, 0.60) as f32,
                    });
                }
            }
        }
    }
    journal
}

fn route_set_overlay(
    routes: &[Vec<(u16, u16)>],
    indices: &[usize],
    dose: u8,
) -> HashMap<(u16, u16), f64> {
    let delta = -0.03 * dose as f64;
    let mut overlay = HashMap::new();
    for &index in indices {
        for &edge in &routes[index] {
            overlay.entry(edge).or_insert(delta);
        }
    }
    overlay
}

fn evaluate_route_set(
    model: &Model,
    history: &[u16],
    gold: u16,
    baseline: &WaveTrace,
    routes: &[Vec<(u16, u16)>],
    indices: &[usize],
    dose: u8,
    base: &[f64],
    total: f64,
) -> RouteSetChoice {
    let baseline_gold = baseline.candidates.iter().find(|candidate| candidate.token == gold);
    let baseline_coverage = baseline_gold.map(|candidate| candidate.coverage).unwrap_or(0);
    let baseline_bonus = baseline_gold.map(|candidate| candidate.bonus).unwrap_or(-0.15);
    let baseline_full = baseline_gold.map(|candidate| candidate.full).unwrap_or(false);
    let baseline_wrong = baseline
        .candidates
        .iter()
        .filter(|candidate| candidate.token != gold)
        .map(|candidate| candidate.bonus)
        .fold(-0.15, f64::max);
    let overlay = route_set_overlay(routes, indices, dose);
    let prospective = wave_trace_modified(
        model,
        history,
        base,
        total,
        Mode::FrequencyGradient,
        None,
        Some(&overlay),
    );
    let after = prospective.candidates.iter().find(|candidate| candidate.token == gold);
    let wrong = prospective
        .candidates
        .iter()
        .filter(|candidate| candidate.token != gold)
        .map(|candidate| candidate.bonus)
        .fold(-0.15, f64::max);
    let coverage = after.map(|candidate| candidate.coverage).unwrap_or(0);
    let full = after.map(|candidate| candidate.full).unwrap_or(false);
    let bonus = after.map(|candidate| candidate.bonus).unwrap_or(-0.15);
    let margin = bonus - wrong;
    let wrong_promotion = wrong - baseline_wrong;
    let structural_gain = coverage > baseline_coverage || (full && !baseline_full);
    let accepted = !indices.is_empty()
        && after.is_some()
        && structural_gain
        && bonus + 0.01 >= baseline_bonus
        && wrong_promotion <= PROSPECTIVE_MAX_WRONG_PROMOTION;
    let score = if after.is_some() {
        (full as u8 as f64) * 1_000_000.0
            + coverage as f64 * 10_000.0
            + margin * 100.0
            + bonus
            - overlay.len() as f64 * 0.001
    } else {
        f64::NEG_INFINITY
    };
    RouteSetChoice {
        indices: indices.to_vec(),
        dose,
        accepted,
        full,
        coverage,
        bonus,
        margin,
        wrong_promotion,
        edge_count: overlay.len(),
        score,
    }
}

fn better_route_choice(a: &RouteSetChoice, b: &RouteSetChoice) -> bool {
    a.accepted && (!b.accepted || a.score > b.score + 1e-12)
}

fn best_for_indices(
    model: &Model,
    history: &[u16],
    gold: u16,
    baseline: &WaveTrace,
    routes: &[Vec<(u16, u16)>],
    indices: &[usize],
    base: &[f64],
    total: f64,
    stats: &mut TrainStats,
) -> RouteSetChoice {
    let mut best = RouteSetChoice {
        indices: indices.to_vec(),
        dose: 1,
        accepted: false,
        full: false,
        coverage: 0,
        bonus: -0.15,
        margin: -1.0,
        wrong_promotion: 0.0,
        edge_count: 0,
        score: f64::NEG_INFINITY,
    };
    for dose in [1u8, 2, 4] {
        stats.route_set_states_evaluated += 1;
        let candidate = evaluate_route_set(
            model, history, gold, baseline, routes, indices, dose, base, total,
        );
        if (candidate.accepted && !best.accepted)
            || (candidate.accepted == best.accepted && candidate.score > best.score + 1e-12)
        {
            best = candidate;
        }
    }
    best
}

fn select_route_set(
    model: &Model,
    history: &[u16],
    gold: u16,
    baseline: &WaveTrace,
    routes: &[Vec<(u16, u16)>],
    policy: SleepPolicy,
    base: &[f64],
    total: f64,
    stats: &mut TrainStats,
) -> Option<RouteSetChoice> {
    if routes.is_empty() {
        return None;
    }
    if policy == SleepPolicy::RouteSetNoInterference {
        let mut selected = Vec::new();
        for index in 0..routes.len() {
            let single = best_for_indices(
                model, history, gold, baseline, routes, &[index], base, total, stats,
            );
            if single.accepted {
                selected.push(index);
            }
        }
        if selected.is_empty() {
            return None;
        }
        let joint = best_for_indices(
            model, history, gold, baseline, routes, &selected, base, total, stats,
        );
        return joint.accepted.then_some(joint);
    }
    if policy == SleepPolicy::RouteSetGreedy {
        let mut selected = Vec::new();
        let mut best: Option<RouteSetChoice> = None;
        loop {
            let mut next: Option<RouteSetChoice> = None;
            for index in 0..routes.len() {
                if selected.contains(&index) {
                    continue;
                }
                let mut trial = selected.clone();
                trial.push(index);
                trial.sort_unstable();
                let candidate = best_for_indices(
                    model, history, gold, baseline, routes, &trial, base, total, stats,
                );
                if candidate.accepted
                    && next
                        .as_ref()
                        .map(|current| better_route_choice(&candidate, current))
                        .unwrap_or(true)
                {
                    next = Some(candidate);
                }
            }
            let Some(candidate) = next else { break };
            if best
                .as_ref()
                .map(|current| candidate.score <= current.score + 1e-12)
                .unwrap_or(false)
            {
                break;
            }
            selected = candidate.indices.clone();
            best = Some(candidate);
            if selected.len() >= ROUTE_SET_MAX_ROUTES {
                break;
            }
        }
        return best;
    }
    #[derive(Clone)]
    struct BeamState {
        indices: Vec<usize>,
        choice: Option<RouteSetChoice>,
        score: f64,
    }
    let mut beam = vec![BeamState {
        indices: Vec::new(),
        choice: None,
        score: f64::NEG_INFINITY,
    }];
    for index in 0..routes.len().min(ROUTE_SET_MAX_ROUTES) {
        let mut expanded = beam.clone();
        for state in &beam {
            let mut indices = state.indices.clone();
            indices.push(index);
            let choice = best_for_indices(
                model, history, gold, baseline, routes, &indices, base, total, stats,
            );
            expanded.push(BeamState {
                indices,
                score: choice.score,
                choice: Some(choice),
            });
        }
        expanded.sort_unstable_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        expanded.dedup_by(|a, b| a.indices == b.indices);
        expanded.truncate(ROUTE_SET_BEAM_WIDTH);
        beam = expanded;
    }
    beam.into_iter()
        .filter_map(|state| state.choice)
        .filter(|choice| choice.accepted)
        .max_by(|a, b| a.score.partial_cmp(&b.score).unwrap())
}

fn record_route_set_choice(
    model: &Model,
    history: &[u16],
    gold: u16,
    baseline: &WaveTrace,
    routes: &[Vec<(u16, u16)>],
    choice: &RouteSetChoice,
    base: &[f64],
    total: f64,
    stats: &mut TrainStats,
) {
    let baseline_gold = baseline.candidates.iter().find(|candidate| candidate.token == gold);
    let baseline_coverage = baseline_gold.map(|candidate| candidate.coverage).unwrap_or(0);
    let baseline_full = baseline_gold.map(|candidate| candidate.full).unwrap_or(false);
    let joint_gain = choice.coverage.saturating_sub(baseline_coverage) as f64
        + ((choice.full && !baseline_full) as u8 as f64);
    let mut individual_gain = 0.0;
    for &index in &choice.indices {
        stats.route_set_states_evaluated += 1;
        let single = evaluate_route_set(
            model,
            history,
            gold,
            baseline,
            routes,
            &[index],
            choice.dose,
            base,
            total,
        );
        individual_gain += single.coverage.saturating_sub(baseline_coverage) as f64
            + ((single.full && !baseline_full) as u8 as f64);
    }
    let interference = joint_gain - individual_gain;
    stats.route_set_joint_gain_sum += joint_gain;
    stats.route_set_individual_gain_sum += individual_gain;
    stats.route_set_interference_sum += interference;
    if choice.indices.len() >= 2 {
        if interference < -1e-12 {
            stats.route_set_negative_interactions += 1;
        } else {
            stats.route_set_nonnegative_interactions += 1;
        }
    }
    stats.route_set_selected_groups += 1;
    stats.route_set_selected_routes += choice.indices.len() as u64;
    stats.route_set_selected_edges += choice.edge_count as u64;
    stats.route_set_full_created += (choice.full && !baseline_full) as u64;
    stats.route_set_selected_bonus_sum += choice.bonus;
    stats.route_set_selected_margin_sum += choice.margin;
    stats.route_set_wrong_promotion_sum += choice.wrong_promotion;
}

fn route_set_origin_journal(
    model: &Model,
    examples: &[SleepExample],
    policy: SleepPolicy,
    stats: &mut TrainStats,
    pending_origin: &mut Vec<PendingOrigin>,
) -> Vec<BiasChange> {
    let (base, total, _) = base_weights(model);
    let mut credit = HashMap::new();
    let mut next_origin = Vec::new();
    let mut groups: HashMap<(Vec<u16>, u16), Vec<PendingOrigin>> = HashMap::new();
    for hypothesis in pending_origin.drain(..) {
        groups
            .entry((hypothesis.history.clone(), hypothesis.gold))
            .or_default()
            .push(hypothesis);
    }
    for ((history, gold), hypotheses) in groups {
        stats.route_set_revalidated_groups += 1;
        stats.origin_bank_revalidated += hypotheses.len() as u64;
        let baseline = wave_trace(model, &history, &base, total, Mode::FrequencyGradient);
        let baseline_gold = baseline.candidates.iter().find(|candidate| candidate.token == gold);
        let old_coverage = hypotheses
            .iter()
            .map(|hypothesis| hypothesis.baseline_coverage)
            .max()
            .unwrap_or(0);
        if baseline_gold
            .map(|candidate| candidate.coverage > old_coverage || candidate.full)
            .unwrap_or(false)
        {
            stats.origin_bank_incorporated += hypotheses.len() as u64;
            continue;
        }
        let routes: Vec<_> = hypotheses
            .iter()
            .take(ROUTE_SET_MAX_ROUTES)
            .map(|hypothesis| hypothesis.path.clone())
            .collect();
        stats.route_set_groups += 1;
        stats.route_set_candidates += routes.len() as u64;
        let choice = select_route_set(
            model, &history, gold, &baseline, &routes, policy, &base, total, stats,
        );
        if let Some(choice) = choice {
            record_route_set_choice(
                model, &history, gold, &baseline, &routes, &choice, &base, total, stats,
            );
            let selected: HashSet<_> = choice.indices.iter().copied().collect();
            for (index, mut hypothesis) in hypotheses.into_iter().enumerate() {
                if selected.contains(&index) {
                    add_path_credit(&mut credit, &hypothesis.path, -1.5);
                    stats.origin_bank_recredited += 1;
                    hypothesis.age += 1;
                    if hypothesis.age < 4 && next_origin.len() < 4096 {
                        next_origin.push(hypothesis);
                    } else {
                        stats.origin_bank_expired += 1;
                    }
                } else {
                    stats.origin_bank_expired += 1;
                }
            }
        } else {
            stats.route_set_rejected_no_joint_gain += 1;
            stats.origin_bank_expired += hypotheses.len() as u64;
        }
    }
    let shuffled: Vec<u16> = (0..examples.len())
        .map(|i| examples[(i * 17 + 7) % examples.len()].gold)
        .collect();
    for (index, example) in examples.iter().enumerate() {
        let gold = if policy.shuffled() { shuffled[index] } else { example.gold };
        let baseline = wave_trace(model, &example.history, &base, total, Mode::FrequencyGradient);
        let Some(good) = baseline
            .candidates
            .iter()
            .find(|candidate| candidate.token == gold && !candidate.full)
        else {
            continue;
        };
        let window = WAVE_WINDOW.min(example.history.len());
        let input = &example.history[example.history.len() - window..];
        let mut routes = Vec::new();
        for origin in 0..window {
            if good.origin_present[origin] {
                continue;
            }
            stats.origin_attempts += 1;
            let Some(path) = find_origin_path(model, input[origin], gold, &base, total) else {
                continue;
            };
            stats.origin_paths_found += 1;
            if !routes.contains(&path) && routes.len() < ROUTE_SET_MAX_ROUTES {
                routes.push(path);
            }
        }
        if routes.is_empty() {
            continue;
        }
        stats.route_set_groups += 1;
        stats.route_set_candidates += routes.len() as u64;
        let choice = select_route_set(
            model,
            &example.history,
            gold,
            &baseline,
            &routes,
            policy,
            &base,
            total,
            stats,
        );
        if let Some(choice) = choice {
            record_route_set_choice(
                model,
                &example.history,
                gold,
                &baseline,
                &routes,
                &choice,
                &base,
                total,
                stats,
            );
            stats.origin_validated += choice.indices.len() as u64;
            stats.origin_full_created += choice.full as u64;
            match choice.dose {
                1 => stats.origin_dose1_pass += choice.indices.len() as u64,
                2 => stats.origin_dose2_pass += choice.indices.len() as u64,
                4 => stats.origin_dose4_pass += choice.indices.len() as u64,
                _ => unreachable!(),
            }
            for &route_index in &choice.indices {
                let path = &routes[route_index];
                add_path_credit(&mut credit, path, -1.5);
                stats.origin_edges_credited += path.len() as u64;
                if next_origin.len() < 4096 {
                    next_origin.push(PendingOrigin {
                        history: example.history.clone(),
                        gold,
                        path: path.clone(),
                        baseline_coverage: good.coverage,
                        age: 1,
                    });
                    stats.origin_bank_enqueued += 1;
                }
            }
        } else {
            stats.route_set_rejected_no_joint_gain += 1;
            stats.origin_rejected_no_gain += 1;
        }
    }
    *pending_origin = next_origin;
    stats.origin_bank_peak = stats.origin_bank_peak.max(pending_origin.len());
    journal_from_credit(model, credit)
}

fn trace_outcome(trace: &WaveTrace, gold: u16) -> (f64, f64, f64, f64, f64) {
    let candidate = trace.candidates.iter().find(|candidate| candidate.token == gold);
    let wrong = trace
        .candidates
        .iter()
        .filter(|candidate| candidate.token != gold)
        .map(|candidate| candidate.bonus)
        .fold(-0.15, f64::max);
    let bonus = candidate.map(|candidate| candidate.bonus).unwrap_or(-0.15);
    (
        candidate.is_some() as u8 as f64,
        candidate.map(|candidate| candidate.full).unwrap_or(false) as u8 as f64,
        candidate.map(|candidate| candidate.coverage).unwrap_or(0) as f64,
        bonus - wrong,
        wrong,
    )
}

fn unique_edges(routes: &[Vec<(u16, u16)>]) -> Vec<(u16, u16)> {
    let mut edges = Vec::new();
    for route in routes {
        for &edge in route {
            if !edges.contains(&edge) {
                edges.push(edge);
            }
        }
    }
    edges
}

fn edges_overlay(edges: &[(u16, u16)], dose: u8) -> HashMap<(u16, u16), f64> {
    let delta = -0.03 * dose as f64;
    edges.iter().copied().map(|edge| (edge, delta)).collect()
}

fn proposal_relevant(edges: &[(u16, u16)], history: &[u16]) -> bool {
    edges
        .iter()
        .any(|(source, _)| history.iter().rev().take(WAVE_WINDOW).any(|token| token == source))
}

fn evaluate_future_edges(
    model: &Model,
    examples: &[SleepExample],
    baselines: &[WaveTrace],
    edges: &[(u16, u16)],
    dose: u8,
    base: &[f64],
    total: f64,
    limit: usize,
) -> (DeferredEvidence, HashMap<(u16, u16), f64>) {
    let overlay = edges_overlay(edges, dose);
    let mut evidence = DeferredEvidence::default();
    let mut edge_value = HashMap::new();
    for (example, baseline) in examples.iter().zip(baselines) {
        if evidence.evaluated == limit || !proposal_relevant(edges, &example.history) {
            continue;
        }
        let after = wave_trace_modified(
            model,
            &example.history,
            base,
            total,
            Mode::FrequencyGradient,
            None,
            Some(&overlay),
        );
        let before = trace_outcome(baseline, example.gold);
        let next = trace_outcome(&after, example.gold);
        let gold_delta = next.0 - before.0;
        let full_delta = next.1 - before.1;
        let coverage_delta = next.2 - before.2;
        let margin_delta = next.3 - before.3;
        let wrong_promotion = next.4 - before.4;
        let value = 4.0 * full_delta
            + 2.0 * gold_delta
            + 0.25 * coverage_delta
            + margin_delta.clamp(-0.25, 0.25)
            - wrong_promotion.max(0.0);
        evidence.evaluated += 1;
        evidence.gold_delta += gold_delta;
        evidence.full_delta += full_delta;
        evidence.coverage_delta += coverage_delta;
        evidence.margin_delta += margin_delta;
        evidence.wrong_promotion += wrong_promotion;
        evidence.value += value;
        if value.abs() > 1e-12 {
            let support: HashSet<_> = after
                .candidates
                .iter()
                .find(|candidate| candidate.token == example.gold)
                .map(|candidate| candidate.support_edges.iter().copied().collect())
                .unwrap_or_default();
            for &edge in edges {
                if support.contains(&edge) {
                    *edge_value.entry(edge).or_insert(0.0) += value;
                }
            }
        }
    }
    if evidence.evaluated > 0 {
        let tolerance = 0.01 * evidence.evaluated as f64;
        evidence.adverse = evidence.gold_delta < 0.0
            || evidence.full_delta < 0.0
            || evidence.wrong_promotion > tolerance;
        evidence.supported = !evidence.adverse
            && (evidence.gold_delta > 0.0
                || evidence.full_delta > 0.0
                || evidence.coverage_delta > 0.0
                || evidence.margin_delta > 0.01);
    }
    (evidence, edge_value)
}

fn proposals_related(left: &DeferredProposal, right: &DeferredProposal) -> bool {
    left.edges.iter().any(|a| {
        right
            .edges
            .iter()
            .any(|b| a == b || a.0 == b.0 || a.1 == b.1)
    })
}

fn evaluate_temporal_metagraph(
    model: &Model,
    examples: &[SleepExample],
    baselines: &[WaveTrace],
    proposals: &mut [DeferredProposal],
    base: &[f64],
    total: f64,
    stats: &mut TrainStats,
) {
    let mut evaluated = 0usize;
    'outer: for left_index in 0..proposals.len() {
        for right_index in left_index + 1..proposals.len() {
            if evaluated == METAGRAPH_MAX_PAIRS {
                break 'outer;
            }
            if !proposals_related(&proposals[left_index], &proposals[right_index]) {
                continue;
            }
            let left = proposals[left_index].clone();
            let right = proposals[right_index].clone();
            let mut combined = left.edges.clone();
            for edge in &right.edges {
                if !combined.contains(edge) {
                    combined.push(*edge);
                }
            }
            let (left_evidence, _) = evaluate_future_edges(
                model, examples, baselines, &left.edges, left.dose, base, total,
                METAGRAPH_PAIR_EXAMPLES,
            );
            let (right_evidence, _) = evaluate_future_edges(
                model, examples, baselines, &right.edges, right.dose, base, total,
                METAGRAPH_PAIR_EXAMPLES,
            );
            let (joint, _) = evaluate_future_edges(
                model, examples, baselines, &combined, left.dose.max(right.dose), base, total,
                METAGRAPH_PAIR_EXAMPLES,
            );
            stats.deferred_overlay_evaluations +=
                (left_evidence.evaluated + right_evidence.evaluated + joint.evaluated) as u64;
            let interaction = joint.value - left_evidence.value - right_evidence.value;
            stats.metagraph_pairs_evaluated += 1;
            stats.metagraph_interaction_sum += interaction;
            let (left_mut, right_mut) = {
                let (head, tail) = proposals.split_at_mut(right_index);
                (&mut head[left_index], &mut tail[0])
            };
            if interaction > 0.05 {
                stats.metagraph_synergy += 1;
                left_mut.synergy = left_mut.synergy.saturating_add(1);
                right_mut.synergy = right_mut.synergy.saturating_add(1);
            } else if interaction < -0.05 {
                stats.metagraph_conflict += 1;
                left_mut.conflict = left_mut.conflict.saturating_add(1);
                right_mut.conflict = right_mut.conflict.saturating_add(1);
            } else {
                stats.metagraph_neutral += 1;
            }
            evaluated += 1;
        }
    }
}

fn deferred_origin_journal(
    model: &Model,
    examples: &[SleepExample],
    policy: SleepPolicy,
    stats: &mut TrainStats,
    pending: &mut Vec<DeferredProposal>,
) -> Vec<BiasChange> {
    let (base, total, _) = base_weights(model);
    let stride = (examples.len() / DEFERRED_FUTURE_SAMPLE.max(1)).max(1);
    let future_examples: Vec<_> = examples
        .iter()
        .step_by(stride)
        .take(DEFERRED_FUTURE_SAMPLE)
        .cloned()
        .collect();
    let baselines: Vec<_> = future_examples
        .iter()
        .map(|example| wave_trace(model, &example.history, &base, total, Mode::FrequencyGradient))
        .collect();

    for proposal in pending.iter_mut() {
        proposal.age = proposal.age.saturating_add(1);
        stats.deferred_future_examples_screened += future_examples.len() as u64;
        let (evidence, edge_values) = evaluate_future_edges(
            model,
            &future_examples,
            &baselines,
            &proposal.edges,
            proposal.dose,
            &base,
            total,
            DEFERRED_RELEVANT_LIMIT,
        );
        stats.deferred_overlay_evaluations += evidence.evaluated as u64;
        stats.deferred_future_examples_relevant += evidence.evaluated as u64;
        if evidence.evaluated == 0 {
            proposal.last_value = 0.0;
            continue;
        }
        proposal.evidence_windows = proposal.evidence_windows.saturating_add(1);
        proposal.evaluated_examples = proposal
            .evaluated_examples
            .saturating_add(evidence.evaluated as u32);
        proposal.gold_delta += evidence.gold_delta;
        proposal.full_delta += evidence.full_delta;
        proposal.coverage_delta += evidence.coverage_delta;
        proposal.margin_delta += evidence.margin_delta;
        proposal.wrong_promotion += evidence.wrong_promotion;
        proposal.value += evidence.value;
        proposal.last_value = evidence.value;
        for (edge, value) in edge_values {
            *proposal.edge_value.entry(edge).or_insert(0.0) += value;
        }
        stats.deferred_evidence_windows += 1;
        stats.deferred_gold_delta_sum += evidence.gold_delta;
        stats.deferred_full_delta_sum += evidence.full_delta;
        stats.deferred_coverage_delta_sum += evidence.coverage_delta;
        stats.deferred_margin_delta_sum += evidence.margin_delta;
        stats.deferred_wrong_promotion_sum += evidence.wrong_promotion;
        stats.deferred_value_sum += evidence.value;
        if evidence.supported {
            proposal.supported_windows = proposal.supported_windows.saturating_add(1);
            stats.deferred_supported_windows += 1;
        } else if evidence.adverse {
            proposal.adverse_windows = proposal.adverse_windows.saturating_add(1);
            stats.deferred_adverse_windows += 1;
        } else {
            stats.deferred_neutral_windows += 1;
        }
    }

    if policy == SleepPolicy::DeferredMetagraph {
        evaluate_temporal_metagraph(
            model,
            &future_examples,
            &baselines,
            pending,
            &base,
            total,
            stats,
        );
    }

    let mut credit = HashMap::new();
    let mut retained = Vec::new();
    for proposal in pending.drain(..) {
        let mature = proposal.evidence_windows >= DEFERRED_REQUIRED_WINDOWS;
        let tolerance = 0.01 * proposal.evaluated_examples as f64;
        let structural_gain = proposal.gold_delta > 0.0
            || proposal.full_delta > 0.0
            || proposal.coverage_delta > 0.0
            || proposal.margin_delta > 0.01;
        let metagraph_ok = policy != SleepPolicy::DeferredMetagraph
            || proposal.conflict <= proposal.synergy;
        let eligible = mature
            && proposal.supported_windows >= DEFERRED_REQUIRED_WINDOWS
            && proposal.adverse_windows == 0
            && proposal.gold_delta >= 0.0
            && proposal.full_delta >= 0.0
            && proposal.wrong_promotion <= tolerance
            && structural_gain
            && metagraph_ok;
        if mature {
            stats.deferred_matured += 1;
        }
        if eligible {
            let committed_edges: Vec<_> = if policy == SleepPolicy::DeferredEdge {
                proposal
                    .edges
                    .iter()
                    .copied()
                    .filter(|edge| proposal.edge_value.get(edge).copied().unwrap_or(0.0) > 0.0)
                    .collect()
            } else {
                proposal.edges.clone()
            };
            if !committed_edges.is_empty() {
                add_path_credit(&mut credit, &committed_edges, -1.5);
                stats.deferred_committed += 1;
                stats.deferred_journal_edges += committed_edges.len() as u64;
                if policy == SleepPolicy::DeferredEdge {
                    stats.deferred_edge_commits += committed_edges.len() as u64;
                } else {
                    stats.deferred_route_commits += proposal.routes.len() as u64;
                }
                continue;
            }
        }
        if proposal.age >= DEFERRED_MAX_AGE {
            stats.deferred_expired += 1;
        } else {
            retained.push(proposal);
        }
    }

    let shuffled: Vec<u16> = if examples.is_empty() {
        Vec::new()
    } else {
        (0..examples.len())
            .map(|index| examples[(index * 17 + 7) % examples.len()].gold)
            .collect()
    };
    for (index, example) in examples.iter().enumerate() {
        if retained.len() >= DEFERRED_LIMIT {
            break;
        }
        let gold = if policy.shuffled() { shuffled[index] } else { example.gold };
        let baseline = wave_trace(model, &example.history, &base, total, Mode::FrequencyGradient);
        let Some(good) = baseline
            .candidates
            .iter()
            .find(|candidate| candidate.token == gold && !candidate.full)
        else {
            continue;
        };
        let window = WAVE_WINDOW.min(example.history.len());
        let input = &example.history[example.history.len() - window..];
        let mut routes = Vec::new();
        for origin in 0..window {
            if good.origin_present[origin] {
                continue;
            }
            stats.origin_attempts += 1;
            let Some(path) = find_origin_path(model, input[origin], gold, &base, total) else {
                continue;
            };
            stats.origin_paths_found += 1;
            if !routes.contains(&path) && routes.len() < ROUTE_SET_MAX_ROUTES {
                routes.push(path);
            }
        }
        if routes.is_empty() {
            continue;
        }
        stats.route_set_groups += 1;
        stats.route_set_candidates += routes.len() as u64;
        let selection_policy = if policy.shuffled() {
            SleepPolicy::RouteSetShuffle
        } else {
            SleepPolicy::RouteSetBeam
        };
        let choice = select_route_set(
            model,
            &example.history,
            gold,
            &baseline,
            &routes,
            selection_policy,
            &base,
            total,
            stats,
        );
        if let Some(choice) = choice {
            record_route_set_choice(
                model,
                &example.history,
                gold,
                &baseline,
                &routes,
                &choice,
                &base,
                total,
                stats,
            );
            let selected_routes: Vec<_> = choice
                .indices
                .iter()
                .map(|&route_index| routes[route_index].clone())
                .collect();
            let edges = unique_edges(&selected_routes);
            let duplicate = retained.iter().any(|proposal| proposal.edges == edges);
            if !edges.is_empty() && !duplicate {
                retained.push(DeferredProposal {
                    origin_history: example.history.clone(),
                    origin_gold: gold,
                    routes: selected_routes,
                    edges,
                    dose: choice.dose,
                    age: 0,
                    evidence_windows: 0,
                    supported_windows: 0,
                    adverse_windows: 0,
                    evaluated_examples: 0,
                    gold_delta: 0.0,
                    full_delta: 0.0,
                    coverage_delta: 0.0,
                    margin_delta: 0.0,
                    wrong_promotion: 0.0,
                    value: 0.0,
                    last_value: 0.0,
                    edge_value: HashMap::new(),
                    synergy: 0,
                    conflict: 0,
                });
                stats.deferred_proposed += 1;
            }
        } else {
            stats.route_set_rejected_no_joint_gain += 1;
        }
    }
    *pending = retained;
    stats.deferred_quarantine_peak = stats.deferred_quarantine_peak.max(pending.len());
    journal_from_credit(model, credit)
}

fn construction_masks(policy: SleepPolicy) -> Vec<u8> {
    let _ = policy;
    (1u8..(1u8 << CONSTRUCTION_INTERFACE))
        .filter(|mask| {
            let slots = mask.count_ones();
            slots == 1 || slots == 2
        })
        .collect()
}

fn construction_key(history: &[u16], target: u16, slot_mask: u8) -> Option<ConstructionKey> {
    if history.len() < CONSTRUCTION_INTERFACE {
        return None;
    }
    let input = &history[history.len() - CONSTRUCTION_INTERFACE..];
    let mut symbols = [0u16; CONSTRUCTION_INTERFACE];
    for (index, token) in input.iter().enumerate() {
        if slot_mask & (1 << index) == 0 {
            symbols[index] = *token;
        }
    }
    Some(ConstructionKey {
        len: CONSTRUCTION_INTERFACE as u8,
        slot_mask,
        symbols,
        target,
    })
}

fn commit_construction(model: &mut Model, construction: Construction) {
    if let Some(existing) = model
        .glp_constructions
        .iter_mut()
        .find(|row| row.key == construction.key)
    {
        existing.support = existing.support.saturating_add(construction.support);
        existing.confirmations = existing
            .confirmations
            .saturating_add(construction.confirmations);
        let old = existing.value as f64;
        existing.value = (0.5 * old + 0.5 * construction.value as f64) as f32;
    } else if model.glp_constructions.len() < CONSTRUCTION_LIMIT {
        model.glp_constructions.push(construction);
    }
}

fn induce_constructions(
    examples: &[SleepExample],
    policy: SleepPolicy,
    model: &Model,
    pending: &[PendingConstruction],
    stats: &mut TrainStats,
) -> Vec<Construction> {
    let masks = construction_masks(policy);
    let shuffled: Vec<u16> = if examples.is_empty() {
        Vec::new()
    } else {
        (0..examples.len())
            .map(|index| examples[(index * 17 + 7) % examples.len()].gold)
            .collect()
    };
    let mut groups: HashMap<ConstructionKey, (u32, [HashSet<u16>; CONSTRUCTION_INTERFACE])> =
        HashMap::new();
    for (index, example) in examples.iter().enumerate() {
        if example.history.len() < CONSTRUCTION_INTERFACE {
            continue;
        }
        let target = if policy.shuffled() {
            shuffled[index]
        } else {
            example.gold
        };
        let input = &example.history[example.history.len() - CONSTRUCTION_INTERFACE..];
        for &mask in &masks {
            let Some(key) = construction_key(&example.history, target, mask) else {
                continue;
            };
            let row = groups
                .entry(key)
                .or_insert_with(|| (0, std::array::from_fn(|_| HashSet::new())));
            row.0 = row.0.saturating_add(1);
            for position in 0..CONSTRUCTION_INTERFACE {
                if mask & (1 << position) != 0 {
                    row.1[position].insert(input[position]);
                }
            }
        }
    }
    stats.construction_candidates += groups.len() as u64;
    let mut rows: Vec<_> = groups
        .into_iter()
        .filter_map(|(key, (support, fillings))| {
            let minimum = if key.slot_mask == 0 { 2 } else { 3 };
            let diverse = (0..CONSTRUCTION_INTERFACE).all(|position| {
                key.slot_mask & (1 << position) == 0 || fillings[position].len() >= 2
            });
            if support < minimum || !diverse {
                return None;
            }
            if model.glp_constructions.iter().any(|row| row.key == key)
                || pending.iter().any(|row| row.construction.key == key)
            {
                return None;
            }
            let fill_count = (0..CONSTRUCTION_INTERFACE)
                .filter(|position| key.slot_mask & (1 << position) != 0)
                .map(|position| fillings[position].len() as u64)
                .sum::<u64>();
            Some((
                Construction::new(key, support, policy.construction_dag()),
                fill_count,
            ))
        })
        .collect();
    rows.sort_unstable_by(|left, right| {
        right
            .0
            .support
            .cmp(&left.0.support)
            .then_with(|| left.0.key.slot_mask.count_ones().cmp(&right.0.key.slot_mask.count_ones()))
            .then_with(|| left.0.key.cmp(&right.0.key))
    });
    rows.truncate(CONSTRUCTION_NEW_LIMIT);
    for (construction, fillings) in &rows {
        stats.construction_proposed += 1;
        if construction.key.slot_mask == 0 {
            stats.construction_fixed_proposed += 1;
        } else {
            stats.construction_slot_proposed += 1;
            stats.construction_slot_fillings += *fillings;
        }
    }
    rows.into_iter().map(|row| row.0).collect()
}

fn finalize_construction_trace(
    history: &[u16],
    mut trace: WaveTrace,
    construction: &Construction,
    edge_application: u8,
) -> WaveTrace {
    let (active, adjustments) = inject_construction_candidates(
        history,
        std::slice::from_ref(construction),
        &mut trace.candidates,
        edge_application,
    );
    trace.early_active_constructions = trace.early_active_constructions.max(active);
    trace.early_gr_adjustments += adjustments;
    trace.candidates.sort_unstable_by(|a, b| {
        b.bonus
            .partial_cmp(&a.bonus)
            .unwrap()
            .then_with(|| a.token.cmp(&b.token))
    });
    trace.candidates.truncate(WAVE_MAX_GRS);
    trace.full = trace.candidates.iter().any(|candidate| candidate.full);
    trace.max_coverage = trace
        .candidates
        .iter()
        .map(|candidate| candidate.coverage as f64 / WAVE_WINDOW as f64)
        .fold(0.0, f64::max);
    trace
}

fn construction_trace_utility(trace: &WaveTrace, gold: u16) -> f64 {
    let row = trace_outcome(trace, gold);
    4.0 * row.1
        + 2.0 * row.0
        + 0.25 * row.2
        + row.3.clamp(-0.25, 0.25)
        - row.4.max(0.0)
}

fn evaluate_construction_future(
    model: &Model,
    examples: &[SleepExample],
    construction: &Construction,
    policy: SleepPolicy,
) -> ConstructionEvidence {
    let (base, total, _) = base_weights(model);
    let mut evidence = ConstructionEvidence::default();
    for example in examples {
        if evidence.matched == CONSTRUCTION_FUTURE_LIMIT
            || !construction_matches(&construction.key, &example.history)
        {
            continue;
        }
        let baseline = wave_trace(
            model,
            &example.history,
            &base,
            total,
            Mode::FrequencyGradient,
        );
        let after = finalize_construction_trace(
            &example.history,
            baseline.clone(),
            construction,
            model.edge_application,
        );
        let before = trace_outcome(&baseline, example.gold);
        let next = trace_outcome(&after, example.gold);
        let gold_delta = next.0 - before.0;
        let full_delta = next.1 - before.1;
        let coverage_delta = next.2 - before.2;
        let margin_delta = next.3 - before.3;
        let wrong_promotion = next.4 - before.4;
        let value = 4.0 * full_delta
            + 2.0 * gold_delta
            + 0.25 * coverage_delta
            + margin_delta.clamp(-0.25, 0.25)
            - wrong_promotion.max(0.0);
        evidence.matched += 1;
        evidence.gold_delta += gold_delta;
        evidence.full_delta += full_delta;
        evidence.coverage_delta += coverage_delta;
        evidence.margin_delta += margin_delta;
        evidence.wrong_promotion += wrong_promotion;
        evidence.value += value;
        if policy.construction_edge_role_credit() {
            let input = &example.history[example.history.len() - CONSTRUCTION_INTERFACE..];
            for position in 0..CONSTRUCTION_INTERFACE {
                let semantic = (input[position], construction.key.target);
                let masked_base = wave_trace_masked(
                    model,
                    &example.history,
                    &base,
                    total,
                    Mode::FrequencyGradient,
                    Some(semantic),
                );
                let mut weakened = *construction;
                weakened.dag_edges[position].weight = 0.0;
                let masked = finalize_construction_trace(
                    &example.history,
                    masked_base,
                    &weakened,
                    model.edge_application,
                );
                let delta = construction_trace_utility(&after, example.gold)
                    - construction_trace_utility(&masked, example.gold);
                evidence.edge_value[position] += delta;
                evidence.edge_value_sq[position] += delta * delta;
                evidence.edge_tests[position] = evidence.edge_tests[position].saturating_add(1);
            }
        }
        if policy.construction_edge_prediction_credit() {
            let mut weakened = *construction;
            weakened.dag_edges[CONSTRUCTION_INTERFACE].weight = 0.0;
            let masked = finalize_construction_trace(
                &example.history,
                baseline.clone(),
                &weakened,
                model.edge_application,
            );
            let delta = construction_trace_utility(&after, example.gold)
                - construction_trace_utility(&masked, example.gold);
            evidence.edge_value[CONSTRUCTION_INTERFACE] += delta;
            evidence.edge_value_sq[CONSTRUCTION_INTERFACE] += delta * delta;
            evidence.edge_tests[CONSTRUCTION_INTERFACE] = evidence.edge_tests[CONSTRUCTION_INTERFACE].saturating_add(1);
        }
    }
    if evidence.matched > 0 {
        let tolerance = 0.01 * evidence.matched as f64;
        evidence.adverse = evidence.gold_delta < 0.0
            || evidence.full_delta < 0.0
            || evidence.wrong_promotion > tolerance;
        evidence.supported = !evidence.adverse
            && (evidence.gold_delta > 0.0
                || evidence.full_delta > 0.0
                || evidence.coverage_delta > 0.0
                || evidence.margin_delta > 0.01);
    }
    evidence
}

fn apply_directional_edge_credit(
    construction: &mut Construction,
    proposal: &PendingConstruction,
    policy: SleepPolicy,
) {
    for index in 0..DAG_EDGE_COUNT {
        let enabled = (index < CONSTRUCTION_INTERFACE
            && policy.construction_edge_role_credit())
            || (index == CONSTRUCTION_INTERFACE
                && policy.construction_edge_prediction_credit());
        let tests = proposal.edge_tests[index];
        if !enabled || tests == 0 {
            continue;
        }
        let mean_value = proposal.edge_value[index] / tests as f64;
        let gradient = -mean_value;
        let edge = &mut construction.dag_edges[index];
        edge.weight = (-0.20 * gradient).exp().clamp(0.65, 1.35) as f32;
        edge.gradient_sum = (-proposal.edge_value[index]) as f32;
        edge.gradient_sq = proposal.edge_value_sq[index] as f32;
        edge.credit_count = tests;
    }
}

fn construction_sleep(
    model: &mut Model,
    examples: &[SleepExample],
    policy: SleepPolicy,
    stats: &mut TrainStats,
    pending: &mut Vec<PendingConstruction>,
) {
    let mut retained = Vec::new();
    for mut proposal in pending.drain(..) {
        proposal.age = proposal.age.saturating_add(1);
        let evidence = evaluate_construction_future(model, examples, &proposal.construction, policy);
        stats.construction_future_matches += evidence.matched as u64;
        if evidence.matched > 0 {
            proposal.evidence_windows = proposal.evidence_windows.saturating_add(1);
            proposal.evaluated_examples = proposal
                .evaluated_examples
                .saturating_add(evidence.matched as u32);
            proposal.gold_delta += evidence.gold_delta;
            proposal.full_delta += evidence.full_delta;
            proposal.coverage_delta += evidence.coverage_delta;
            proposal.margin_delta += evidence.margin_delta;
            proposal.wrong_promotion += evidence.wrong_promotion;
            proposal.value += evidence.value;
            for index in 0..DAG_EDGE_COUNT {
                proposal.edge_value[index] += evidence.edge_value[index];
                proposal.edge_value_sq[index] += evidence.edge_value_sq[index];
                proposal.edge_tests[index] = proposal.edge_tests[index]
                    .saturating_add(evidence.edge_tests[index]);
                if evidence.edge_tests[index] > 0 {
                    stats.directional_edge_tests += evidence.edge_tests[index] as u64;
                    stats.directional_edge_value_sum += evidence.edge_value[index];
                    if index < CONSTRUCTION_INTERFACE {
                        stats.directional_role_edge_tests += evidence.edge_tests[index] as u64;
                    } else {
                        stats.directional_prediction_edge_tests += evidence.edge_tests[index] as u64;
                    }
                    if evidence.edge_value[index] > 1e-12 {
                        stats.directional_edge_positive += 1;
                    } else if evidence.edge_value[index] < -1e-12 {
                        stats.directional_edge_negative += 1;
                    } else {
                        stats.directional_edge_neutral += 1;
                    }
                }
            }
            stats.construction_evidence_windows += 1;
            stats.construction_gold_delta_sum += evidence.gold_delta;
            stats.construction_full_delta_sum += evidence.full_delta;
            stats.construction_coverage_delta_sum += evidence.coverage_delta;
            stats.construction_margin_delta_sum += evidence.margin_delta;
            stats.construction_wrong_promotion_sum += evidence.wrong_promotion;
            stats.construction_value_sum += evidence.value;
            if evidence.supported {
                proposal.supported_windows = proposal.supported_windows.saturating_add(1);
                stats.construction_supported_windows += 1;
            } else if evidence.adverse {
                proposal.adverse_windows = proposal.adverse_windows.saturating_add(1);
                stats.construction_adverse_windows += 1;
            } else {
                stats.construction_neutral_windows += 1;
            }
        }
        let mature = proposal.evidence_windows >= CONSTRUCTION_REQUIRED_WINDOWS;
        let tolerance = 0.01 * proposal.evaluated_examples as f64;
        let structural_gain = proposal.gold_delta > 0.0
            || proposal.full_delta > 0.0
            || proposal.coverage_delta > 0.0
            || proposal.margin_delta > 0.01;
        let eligible = mature
            && proposal.supported_windows >= CONSTRUCTION_REQUIRED_WINDOWS
            && proposal.adverse_windows == 0
            && proposal.gold_delta >= 0.0
            && proposal.full_delta >= 0.0
            && proposal.wrong_promotion <= tolerance
            && structural_gain;
        if mature {
            stats.construction_matured += 1;
        }
        if eligible {
            proposal.construction.confirmations = proposal.supported_windows as u16;
            proposal.construction.value = proposal.value as f32;
            let snapshot = proposal.clone();
            apply_directional_edge_credit(&mut proposal.construction, &snapshot, policy);
            commit_construction(model, proposal.construction);
            stats.construction_committed += 1;
        } else if proposal.age >= CONSTRUCTION_MAX_AGE {
            stats.construction_expired += 1;
        } else {
            retained.push(proposal);
        }
    }
    let proposals = induce_constructions(examples, policy, model, &retained, stats);
    if !policy.construction_deferred() {
        for construction in proposals {
            commit_construction(model, construction);
            stats.construction_immediate_committed += 1;
        }
    } else {
        for construction in proposals {
            if retained.len() == CONSTRUCTION_LIMIT {
                break;
            }
            retained.push(PendingConstruction {
                construction,
                age: 0,
                evidence_windows: 0,
                supported_windows: 0,
                adverse_windows: 0,
                evaluated_examples: 0,
                gold_delta: 0.0,
                full_delta: 0.0,
                coverage_delta: 0.0,
                margin_delta: 0.0,
                wrong_promotion: 0.0,
                value: 0.0,
                edge_value: [0.0; DAG_EDGE_COUNT],
                edge_value_sq: [0.0; DAG_EDGE_COUNT],
                edge_tests: [0; DAG_EDGE_COUNT],
            });
        }
    }
    *pending = retained;
    stats.construction_quarantine_peak = stats.construction_quarantine_peak.max(pending.len());
    stats.construction_glp_peak = stats.construction_glp_peak.max(model.glp_constructions.len());
}

fn origin_completion_journal(
    model: &Model,
    examples: &[SleepExample],
    policy: SleepPolicy,
    stats: &mut TrainStats,
    pending_origin: &mut Vec<PendingOrigin>,
    pending_deferred: &mut Vec<DeferredProposal>,
) -> Vec<BiasChange> {
    if policy.deferred() {
        return deferred_origin_journal(model, examples, policy, stats, pending_deferred);
    }
    if policy.route_set() {
        return route_set_origin_journal(model, examples, policy, stats, pending_origin);
    }
    let (base, total, _) = base_weights(model);
    let mut credit = HashMap::new();
    let mut next_origin = Vec::new();
    if policy.origin_bank() {
        for mut hypothesis in pending_origin.drain(..) {
            stats.origin_bank_revalidated += 1;
            let trace = wave_trace(
                model,
                &hypothesis.history,
                &base,
                total,
                Mode::FrequencyGradient,
            );
            let current = trace
                .candidates
                .iter()
                .find(|candidate| candidate.token == hypothesis.gold);
            if current
                .map(|candidate| {
                    candidate.coverage > hypothesis.baseline_coverage || candidate.full
                })
                .unwrap_or(false)
            {
                stats.origin_bank_incorporated += 1;
                continue;
            }
            let base_coverage = current.map(|candidate| candidate.coverage).unwrap_or(0);
            let base_bonus = current.map(|candidate| candidate.bonus).unwrap_or(-0.15);
            let baseline_wrong = trace
                .candidates
                .iter()
                .filter(|candidate| candidate.token != hypothesis.gold)
                .map(|candidate| candidate.bonus)
                .fold(-0.15, f64::max);
            let mut passed = false;
            for dose in [1u8, 2, 4] {
                let overlay: HashMap<_, _> = hypothesis
                    .path
                    .iter()
                    .map(|&edge| (edge, -0.03 * dose as f64))
                    .collect();
                let after_trace = wave_trace_modified(
                    model,
                    &hypothesis.history,
                    &base,
                    total,
                    Mode::FrequencyGradient,
                    None,
                    Some(&overlay),
                );
                let after = after_trace
                    .candidates
                    .iter()
                    .find(|candidate| candidate.token == hypothesis.gold);
                let wrong = after_trace
                    .candidates
                    .iter()
                    .filter(|candidate| candidate.token != hypothesis.gold)
                    .map(|candidate| candidate.bonus)
                    .fold(-0.15, f64::max);
                if after
                    .map(|candidate| {
                        candidate.coverage > base_coverage && candidate.bonus + 0.01 >= base_bonus
                    })
                    .unwrap_or(false)
                    && wrong - baseline_wrong <= 0.01
                {
                    for &edge in &hypothesis.path {
                        let row = credit.entry(edge).or_insert((0.0, 0));
                        row.0 -= 1.5;
                        row.1 += 1;
                    }
                    stats.origin_bank_recredited += 1;
                    hypothesis.age += 1;
                    if hypothesis.age < 4 && next_origin.len() < 4096 {
                        next_origin.push(hypothesis);
                    } else {
                        stats.origin_bank_expired += 1;
                    }
                    passed = true;
                    break;
                }
            }
            if !passed {
                stats.origin_bank_expired += 1;
            }
        }
    } else {
        pending_origin.clear();
    }
    let shuffled: Vec<u16> = (0..examples.len())
        .map(|i| examples[(i * 17 + 7) % examples.len()].gold)
        .collect();
    for (index, example) in examples.iter().enumerate() {
        let gold = if policy.shuffled() {
            shuffled[index]
        } else {
            example.gold
        };
        let trace = wave_trace(
            model,
            &example.history,
            &base,
            total,
            Mode::FrequencyGradient,
        );
        let Some(good) = trace
            .candidates
            .iter()
            .find(|candidate| candidate.token == gold && !candidate.full)
        else {
            continue;
        };
        let window = WAVE_WINDOW.min(example.history.len());
        let input = &example.history[example.history.len() - window..];
        let baseline_wrong = trace
            .candidates
            .iter()
            .filter(|candidate| candidate.token != gold)
            .map(|candidate| candidate.bonus)
            .fold(-0.15, f64::max);
        let mut accepted = false;
        for origin in 0..window {
            if good.origin_present[origin] {
                continue;
            }
            stats.origin_attempts += 1;
            let Some(path) = find_origin_path(model, input[origin], gold, &base, total) else {
                continue;
            };
            stats.origin_paths_found += 1;
            for dose in [1u8, 2, 4] {
                let delta = -0.03 * dose as f64;
                let overlay: HashMap<_, _> = path.iter().map(|&edge| (edge, delta)).collect();
                let after_trace = wave_trace_modified(
                    model,
                    &example.history,
                    &base,
                    total,
                    Mode::FrequencyGradient,
                    None,
                    Some(&overlay),
                );
                let after = after_trace
                    .candidates
                    .iter()
                    .find(|candidate| candidate.token == gold);
                let wrong = after_trace
                    .candidates
                    .iter()
                    .filter(|candidate| candidate.token != gold)
                    .map(|candidate| candidate.bonus)
                    .fold(-0.15, f64::max);
                let pass = after
                    .map(|candidate| {
                        candidate.coverage > good.coverage && candidate.bonus + 0.01 >= good.bonus
                    })
                    .unwrap_or(false)
                    && wrong - baseline_wrong <= 0.01;
                if pass {
                    stats.origin_validated += 1;
                    stats.origin_full_created +=
                        after.map(|candidate| candidate.full).unwrap_or(false) as u64;
                    match dose {
                        1 => stats.origin_dose1_pass += 1,
                        2 => stats.origin_dose2_pass += 1,
                        4 => stats.origin_dose4_pass += 1,
                        _ => unreachable!(),
                    }
                    for &edge in &path {
                        let row = credit.entry(edge).or_insert((0.0, 0));
                        row.0 -= 1.5;
                        row.1 += 1;
                    }
                    stats.origin_edges_credited += path.len() as u64;
                    if policy.origin_bank() && next_origin.len() < 4096 {
                        next_origin.push(PendingOrigin {
                            history: example.history.clone(),
                            gold,
                            path: path.clone(),
                            baseline_coverage: good.coverage,
                            age: 1,
                        });
                        stats.origin_bank_enqueued += 1;
                    }
                    accepted = true;
                    break;
                }
            }
            if accepted {
                break;
            }
            stats.origin_rejected_no_gain += 1;
        }
    }
    *pending_origin = next_origin;
    stats.origin_bank_peak = stats.origin_bank_peak.max(pending_origin.len());
    journal_from_credit(model, credit)
}

fn separate_phase_journal(
    model: &Model,
    examples: &[SleepExample],
    policy: SleepPolicy,
    stats: &mut TrainStats,
) -> Vec<BiasChange> {
    let (base, total, _) = base_weights(model);
    let mut credit = HashMap::new();
    let shuffled: Vec<u16> = (0..examples.len())
        .map(|i| examples[(i * 17 + 7) % examples.len()].gold)
        .collect();
    for (index, example) in examples.iter().enumerate() {
        let gold = if policy.shuffled() {
            shuffled[index]
        } else {
            example.gold
        };
        let trace = wave_trace(
            model,
            &example.history,
            &base,
            total,
            Mode::FrequencyGradient,
        );
        let Some(good) = trace
            .candidates
            .iter()
            .find(|candidate| candidate.token == gold && !candidate.full)
        else {
            continue;
        };
        stats.phase_attempts += 1;
        let window = WAVE_WINDOW.min(example.history.len());
        if good.coverage == window {
            stats.phase_full_coverage_attempts += 1;
        } else {
            stats.phase_missing_coverage_attempts += 1;
        }
        stats.phase_vector_edges += good.phase_deltas.len() as u64;
        stats.phase_dispersion_before_sum += good.dispersion;
        let validation =
            validate_phase_vector(model, &example.history, gold, &trace, good, &base, total);
        if validation.accepted {
            stats.phase_accepted += 1;
            stats.phase_full_created += validation.full_created as u64;
            stats.phase_dispersion_after_sum += validation.after_dispersion;
            for &(edge, delta) in &good.phase_deltas {
                let row = credit.entry(edge).or_insert((0.0, 0));
                row.0 += delta / 0.02;
                row.1 += 1;
                stats.phase_abs_delta_sum += delta.abs();
            }
        } else if !validation.gold_found {
            stats.phase_rejected_gold_lost += 1;
        } else if !validation.coverage_preserved {
            stats.phase_rejected_coverage += 1;
        } else if !validation.improved {
            stats.phase_rejected_no_improvement += 1;
        } else if !validation.bonus_preserved {
            stats.phase_rejected_bonus += 1;
        } else {
            stats.phase_rejected_wrong_promotion += 1;
        }
    }
    journal_from_credit(model, credit)
}


// EXP152: the model is the immutable sleep snapshot. Workers emit ordered
// prejournals; only the authoritative thread reduces and commits them.
struct UniformPrejournal {
    events: [((u16, u16), f64); 64],
    event_count: usize,
    gold_candidate: bool,
    full_gold: bool,
}

fn parallel_train_workers() -> usize {
    env::var("NGU_TRAIN_WORKERS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(1)
        .clamp(1, 64)
}

fn uniform_prejournal(
    model: &Model,
    example: &SleepExample,
    base: &[f64],
    total: f64,
) -> UniformPrejournal {
    let trace = wave_trace(
        model,
        &example.history,
        base,
        total,
        Mode::FrequencyGradient,
    );
    let good = trace
        .candidates
        .iter()
        .find(|candidate| candidate.token == example.gold);
    let bad = trace
        .candidates
        .iter()
        .find(|candidate| candidate.token != example.gold);
    fn append(
        events: &mut [((u16, u16), f64); 64],
        event_count: &mut usize,
        edges: &[(u16, u16)],
        value: f64,
    ) {
        for &edge in edges {
            assert!(*event_count < events.len(), "uniform prejournal overflow");
            events[*event_count] = (edge, value);
            *event_count += 1;
        }
    }
    let mut events = [((0, 0), 0.0); 64];
    let mut event_count = 0usize;
    if let (Some(good), Some(bad)) = (good, bad) {
        let magnitude = ((0.05 - good.bonus + bad.bonus) / 0.05).clamp(0.0, 2.0);
        if magnitude > 0.0 {
            append(&mut events, &mut event_count, &good.support_edges, -magnitude);
            append(&mut events, &mut event_count, &bad.support_edges, magnitude);
        }
    }
    if let Some(good) = good {
        let coverage =
            good.coverage as f64 / WAVE_WINDOW.min(example.history.len()) as f64;
        let reward =
            1.0 + good.dispersion / MEETING_TOLERANCE + (1.0 - coverage);
        append(&mut events, &mut event_count, &good.support_edges, -reward);
    } else {
        let mut seen = HashSet::new();
        for &source in &example.history {
            if seen.insert(source) && direct_edge_exists(model, source, example.gold) {
                assert!(event_count < events.len(), "uniform prejournal overflow");
                events[event_count] = ((source, example.gold), -1.0);
                event_count += 1;
            }
        }
    }
    if let Some(bad) = bad {
        append(&mut events, &mut event_count, &bad.support_edges, 0.5);
    }
    UniformPrejournal {
        events,
        event_count,
        gold_candidate: good.is_some(),
        full_gold: good.map(|candidate| candidate.full).unwrap_or(false),
    }
}

fn parallel_uniform_journal(
    model: &Model,
    examples: &[SleepExample],
    stats: &mut TrainStats,
    workers: usize,
) -> Vec<BiasChange> {
    let (base, total, _) = base_weights(model);
    let workers = workers.min(examples.len().max(1));
    let chunk_size = (examples.len() + workers - 1) / workers;
    let groups: Vec<Vec<UniformPrejournal>> = std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for chunk in examples.chunks(chunk_size.max(1)) {
            let base = &base;
            handles.push(scope.spawn(move || {
                wave_successor_cache_begin();
                chunk
                    .iter()
                    .map(|example| uniform_prejournal(model, example, base, total))
                    .collect()
            }));
        }
        handles
            .into_iter()
            .map(|handle| handle.join().expect("EXP152 sleep worker panicked"))
            .collect()
    });

    let mut credit: HashMap<(u16, u16), (f64, u32)> = HashMap::new();
    for row in groups.into_iter().flatten() {
        stats.sleep_candidate_traces_before_feedback += 1;
        stats.sleep_examples += 1;
        stats.sleep_gold_candidates_before_feedback += row.gold_candidate as u64;
        stats.sleep_full_gold_candidates_before_feedback += row.full_gold as u64;
        // Preserve the exact serial order of every floating-point addition.
        for &(edge, value) in &row.events[..row.event_count] {
            let entry = credit.entry(edge).or_insert((0.0, 0));
            entry.0 += value;
            entry.1 += 1;
        }
    }
    journal_from_credit(model, credit)
}

fn gr_loss_journal(
    model: &Model,
    examples: &[SleepExample],
    policy: SleepPolicy,
    stats: &mut TrainStats,
    pending: &mut Vec<PendingBridge>,
) -> Vec<BiasChange> {
    let workers = parallel_train_workers();
    if policy == SleepPolicy::Uniform && workers > 1 {
        pending.clear();
        let journal = parallel_uniform_journal(model, examples, stats, workers);
        stats.sleep_transaction_edges += journal.len() as u64;
        stats.replay_updates += journal.len() as u64;
        stats.pending_peak = stats.pending_peak.max(pending.len());
        return journal;
    }
    let (base, total, _) = base_weights(model);
    let mut credit: HashMap<(u16, u16), (f64, u32)> = HashMap::new();
    let mut next_pending = Vec::new();
    if policy.cumulative() {
        for mut hypothesis in pending.drain(..) {
            stats.pending_revalidated += 1;
            let baseline = wave_trace(
                model,
                &hypothesis.history,
                &base,
                total,
                Mode::FrequencyGradient,
            );
            if baseline
                .candidates
                .iter()
                .any(|candidate| candidate.token == hypothesis.gold)
            {
                stats.pending_resolved += 1;
                continue;
            }
            let validation = validate_bridge_multidose(
                model,
                &hypothesis.history,
                hypothesis.gold,
                &baseline,
                &hypothesis.bridge,
                hypothesis.magnitude,
                &base,
                total,
                stats,
            );
            if validation.accepted {
                add_path_credit(&mut credit, &hypothesis.bridge.edges, -hypothesis.magnitude);
                stats.pending_recredited += 1;
                stats.bridge_edges_credited += hypothesis.bridge.edges.len() as u64;
                hypothesis.age += 1;
                if validation.dose > 1 && hypothesis.age < 4 && next_pending.len() < 4096 {
                    next_pending.push(hypothesis);
                } else {
                    stats.pending_resolved += 1;
                }
            } else {
                stats.pending_expired += 1;
            }
        }
    } else {
        pending.clear();
    }
    let reverse = if policy.bridge() {
        Some(build_reverse_index(model, &base, total))
    } else {
        None
    };
    let shuffled: Vec<u16> = (0..examples.len())
        .map(|i| examples[(i * 17 + 7) % examples.len()].gold)
        .collect();
    for (index, example) in examples.iter().enumerate() {
        // Candidate construction has no gold argument: feedback is consulted only after this trace exists.
        let trace = wave_trace(
            model,
            &example.history,
            &base,
            total,
            Mode::FrequencyGradient,
        );
        stats.sleep_candidate_traces_before_feedback += 1;
        let gold = if policy.shuffled() {
            shuffled[index]
        } else {
            example.gold
        };
        stats.sleep_examples += 1;
        let gold_gr = trace.candidates.iter().find(|gr| gr.token == gold);
        let wrong = trace.candidates.iter().find(|gr| gr.token != gold);
        if let Some(gr) = gold_gr {
            stats.sleep_gold_candidates_before_feedback += 1;
            if gr.full {
                stats.sleep_full_gold_candidates_before_feedback += 1;
            }
        }
        if policy == SleepPolicy::Uniform {
            if let (Some(good), Some(bad)) = (gold_gr, wrong) {
                let magnitude = ((0.05 - good.bonus + bad.bonus) / 0.05).clamp(0.0, 2.0);
                if magnitude > 0.0 {
                    add_path_credit(&mut credit, &good.support_edges, -magnitude);
                    add_path_credit(&mut credit, &bad.support_edges, magnitude);
                }
            }
            if let Some(good) = gold_gr {
                let coverage = good.coverage as f64 / WAVE_WINDOW.min(example.history.len()) as f64;
                let reward = 1.0 + good.dispersion / MEETING_TOLERANCE + (1.0 - coverage);
                add_path_credit(&mut credit, &good.support_edges, -reward);
            } else {
                let mut seen = HashSet::new();
                for &source in &example.history {
                    if seen.insert(source) && direct_edge_exists(model, source, gold) {
                        let row = credit.entry((source, gold)).or_insert((0.0, 0));
                        row.0 -= 1.0;
                        row.1 += 1;
                    }
                }
            }
            if let Some(bad) = wrong {
                add_path_credit(&mut credit, &bad.support_edges, 0.5);
            }
            continue;
        }
        if policy.bridge() && gold_gr.is_none() {
            stats.bridge_attempts += 1;
            if let Some(bridge) = bidirectional_bridge(
                model,
                &example.history,
                gold,
                &base,
                total,
                reverse.as_ref().unwrap(),
            ) {
                stats.bridge_found += 1;
                stats.bridge_full_coverage += bridge.full as u64;
                stats.bridge_coverage_sum += bridge.coverage;
                stats.bridge_dispersion_sum += bridge.dispersion;
                stats.bridge_cost_sum += bridge.cost;
                let magnitude =
                    (1.0 + bridge.coverage + (1.0 - bridge.dispersion / 2.5)).clamp(1.0, 3.0);
                let accepted = if policy.multidose() {
                    stats.bridge_validations += 1;
                    let validation = validate_bridge_multidose(
                        model,
                        &example.history,
                        gold,
                        &trace,
                        &bridge,
                        magnitude,
                        &base,
                        total,
                        stats,
                    );
                    stats.prospective_gold_bonus_sum += validation.last.gold_bonus;
                    stats.prospective_wrong_promotion_sum += validation.last.wrong_promotion;
                    if validation.last.margin.is_finite() {
                        stats.prospective_margin_sum += validation.last.margin;
                    }
                    if validation.accepted {
                        stats.bridge_validated += 1;
                        if policy.cumulative() && validation.dose > 1 {
                            if next_pending.len() < 4096 {
                                next_pending.push(PendingBridge {
                                    history: example.history.clone(),
                                    gold,
                                    bridge: bridge.clone(),
                                    magnitude,
                                    age: 1,
                                });
                                stats.pending_enqueued += 1;
                            } else {
                                stats.pending_expired += 1;
                            }
                        }
                        true
                    } else if !validation.last.gold_found {
                        stats.bridge_rejected_no_gold += 1;
                        false
                    } else if validation.last.wrong_promotion > PROSPECTIVE_MAX_WRONG_PROMOTION {
                        stats.bridge_rejected_wrong_promotion += 1;
                        false
                    } else {
                        stats.bridge_rejected_margin += 1;
                        false
                    }
                } else if policy.validated() {
                    stats.bridge_validations += 1;
                    let validation = validate_bridge_prospectively(
                        model,
                        &example.history,
                        gold,
                        &trace,
                        &bridge,
                        magnitude,
                        &base,
                        total,
                    );
                    stats.prospective_gold_bonus_sum += validation.gold_bonus;
                    stats.prospective_wrong_promotion_sum += validation.wrong_promotion;
                    if validation.margin.is_finite() {
                        stats.prospective_margin_sum += validation.margin;
                    }
                    if validation.accepted {
                        stats.bridge_validated += 1;
                        true
                    } else if !validation.gold_found {
                        stats.bridge_rejected_no_gold += 1;
                        false
                    } else if validation.wrong_promotion > PROSPECTIVE_MAX_WRONG_PROMOTION {
                        stats.bridge_rejected_wrong_promotion += 1;
                        false
                    } else {
                        stats.bridge_rejected_margin += 1;
                        false
                    }
                } else {
                    true
                };
                if accepted {
                    stats.bridge_edges_credited += bridge.edges.len() as u64;
                    add_path_credit(&mut credit, &bridge.edges, -magnitude);
                }
            }
        }
        if policy.phase() {
            if let Some(good) = gold_gr.filter(|candidate| !candidate.full) {
                stats.phase_attempts += 1;
                let window = WAVE_WINDOW.min(example.history.len());
                if good.coverage == window {
                    stats.phase_full_coverage_attempts += 1;
                } else {
                    stats.phase_missing_coverage_attempts += 1;
                }
                stats.phase_vector_edges += good.phase_deltas.len() as u64;
                stats.phase_dispersion_before_sum += good.dispersion;
                let validation = validate_phase_vector(
                    model,
                    &example.history,
                    gold,
                    &trace,
                    good,
                    &base,
                    total,
                );
                if validation.accepted {
                    stats.phase_accepted += 1;
                    stats.phase_full_created += validation.full_created as u64;
                    stats.phase_dispersion_after_sum += validation.after_dispersion;
                    for &(edge, delta) in &good.phase_deltas {
                        let row = credit.entry(edge).or_insert((0.0, 0));
                        row.0 += delta / 0.02;
                        row.1 += 1;
                        stats.phase_abs_delta_sum += delta.abs();
                    }
                } else if !validation.gold_found {
                    stats.phase_rejected_gold_lost += 1;
                } else if !validation.coverage_preserved {
                    stats.phase_rejected_coverage += 1;
                } else if !validation.improved {
                    stats.phase_rejected_no_improvement += 1;
                } else if !validation.bonus_preserved {
                    stats.phase_rejected_bonus += 1;
                } else {
                    stats.phase_rejected_wrong_promotion += 1;
                }
            }
        }
        if policy.contrast() {
            if let (Some(good), Some(bad)) = (gold_gr, wrong) {
                let mut union: Vec<_> = good
                    .support_edges
                    .iter()
                    .chain(&bad.support_edges)
                    .copied()
                    .collect();
                union.sort_unstable();
                union.dedup();
                union.truncate(16);
                for edge in union {
                    stats.counterfactual_interventions += 1;
                    stats.contrast_edges_tested += 1;
                    let intervened = wave_trace_masked(
                        model,
                        &example.history,
                        &base,
                        total,
                        Mode::FrequencyGradient,
                        Some(edge),
                    );
                    let gold_need = necessity_from_trace(good, &intervened);
                    let wrong_need = necessity_from_trace(bad, &intervened);
                    let difference = gold_need - wrong_need;
                    stats.contrast_gold_necessity_sum += gold_need;
                    stats.contrast_wrong_necessity_sum += wrong_need;
                    stats.contrast_abs_difference_sum += difference.abs();
                    if difference > 0.10 {
                        stats.contrast_gold_favored += 1;
                        add_path_credit(&mut credit, &[edge], -difference);
                    } else if difference < -0.10 {
                        stats.contrast_wrong_favored += 1;
                        add_path_credit(&mut credit, &[edge], -difference);
                    } else {
                        stats.contrast_neutral += 1;
                    }
                }
            }
        }
    }
    let deltas: HashMap<_, _> = credit
        .into_iter()
        .map(|(edge, (sum, count))| (edge, (0.02 * sum / count.max(1) as f64).clamp(-0.03, 0.03)))
        .collect();
    let mut journal = Vec::new();
    for (gcp, graph) in [(false, &model.glp), (true, &model.gcp)] {
        for (key, node) in graph {
            if key.len() != 1 {
                continue;
            }
            let source = key.tokens()[0];
            for edge in &node.edges {
                if let Some(delta) = deltas.get(&(source, edge.token)) {
                    journal.push(BiasChange {
                        gcp,
                        key: *key,
                        token: edge.token,
                        old: edge.sleep_bias,
                        new: (edge.sleep_bias as f64 + delta).clamp(-0.60, 0.60) as f32,
                    });
                }
            }
        }
    }
    stats.sleep_transaction_edges += journal.len() as u64;
    stats.replay_updates += journal.len() as u64;
    *pending = next_pending;
    stats.pending_peak = stats.pending_peak.max(pending.len());
    journal
}

fn apply_journal(model: &mut Model, journal: &[BiasChange], commit: bool) {
    let mut changed_glp = BTreeSet::new();
    for change in journal {
        let graph = if change.gcp {
            &mut model.gcp
        } else {
            changed_glp.insert(change.key);
            &mut model.glp
        };
        if let Some(edge) = graph.get_mut(&change.key).and_then(|node| {
            node.edges
                .iter_mut()
                .find(|edge| edge.token == change.token)
        }) {
            edge.sleep_bias = if commit { change.new } else { change.old };
        }
    }
    for key in changed_glp {
        sync_exp171_shard_node(model, key);
    }
}


fn dominant_token(node: Option<&Node>) -> Option<u16> {
    node.and_then(|node| {
        node.edges
            .iter()
            .max_by_key(|edge| (edge.count, std::cmp::Reverse(edge.token)))
            .map(|edge| edge.token)
    })
}

fn materialize_medial_dream(
    model: &mut Model,
    candidate: &MedialDreamCandidate,
    credit: f64,
    options: &Options,
    stats: &mut TrainStats,
) {
    // Feedback always enters GCP. The ordinary deep reflection later moves it
    // to GM1; the promotion phase has already run, preventing a direct GLP
    // write in this cycle.
    let key = candidate.key.context;
    let target = candidate.key.target;
    let node = model
        .gcp
        .entry(key)
        .or_insert_with(|| Node::new(key.len() as usize));
    let existed = node
        .edge_position(target, options.successor_index)
        .is_some();
    for _ in 0..4 {
        let _ = node.observe(
            target,
            training_step(stats.tokens),
            options.max_successors,
            options.capacity_policy,
            options.successor_index,
        );
    }
    if let Some(position) = node.edge_position(target, options.successor_index) {
        let edge = &mut node.edges[position];
        edge.weight = (edge.weight + (0.02 + 0.03 * credit.clamp(0.0, 1.0)) as f32)
            .clamp(-4.0, 4.0);
        edge.sleep_bias = (edge.sleep_bias + 0.01).clamp(-0.60, 0.60);
    }
    stats.medial_dream_materialized += 1;
    stats.medial_dream_credit_sum += credit;
    if existed {
        stats.medial_dream_reinforced_edges += 1;
    } else {
        stats.medial_dream_created_edges += 1;
    }
}

fn validate_medial_dreams(
    model: &mut Model,
    episode: &[u16],
    options: &Options,
    stats: &mut TrainStats,
) {
    if model.medial_mode < 3 {
        return;
    }
    let mut pending = std::mem::take(&mut model.medial_pending);
    if pending.is_empty() {
        return;
    }
    pending.sort_unstable_by_key(|row| row.key);
    let mut positions: ContextMap<usize> = ContextMap::default();
    for (index, candidate) in pending.iter().enumerate() {
        positions.insert(candidate.key.context, index);
    }
    let mut matches = vec![0u32; pending.len()];
    let mut correct = vec![0u32; pending.len()];
    let mut history = Vec::with_capacity(model.max_order);
    for (index, &gold) in episode.iter().enumerate() {
        if index % 512 == 0 || gold == 0 {
            history.clear();
        }
        for order in 2..=model.max_order.min(history.len()) {
            let key = make_key(&history, order);
            if let Some(&position) = positions.get(&key) {
                matches[position] += 1;
                correct[position] += (pending[position].key.target == gold) as u32;
            }
        }
        history.push(gold);
        if history.len() > model.max_order {
            history.remove(0);
        }
    }
    let mut keep = Vec::new();
    for (index, mut candidate) in pending.into_iter().enumerate() {
        candidate.age = candidate.age.saturating_add(1);
        if matches[index] >= 4 {
            let credit =
                (2.0 * correct[index] as f64 - matches[index] as f64)
                    / matches[index] as f64;
            candidate.evaluated = candidate.evaluated.saturating_add(matches[index]);
            candidate.credit_sum += credit;
            stats.medial_dream_future_windows += 1;
            if credit > 0.0 {
                candidate.supported_windows = candidate.supported_windows.saturating_add(1);
                stats.medial_dream_supported_windows += 1;
            } else if credit < 0.0 {
                candidate.adverse_windows = candidate.adverse_windows.saturating_add(1);
                stats.medial_dream_adverse_windows += 1;
            }
        }
        if candidate.supported_windows >= 2 && candidate.adverse_windows == 0 {
            let credit = candidate.credit_sum / candidate.supported_windows as f64;
            materialize_medial_dream(model, &candidate, credit, options, stats);
        } else if candidate.age < 5 && candidate.adverse_windows == 0 {
            keep.push(candidate);
        } else {
            stats.medial_dream_expired += 1;
        }
    }
    model.medial_pending = keep;
}

fn propose_medial_dreams(
    model: &mut Model,
    options: &Options,
    stats: &mut TrainStats,
) {
    if model.medial_mode < 2 {
        return;
    }
    let mut existing: HashSet<MedialDreamKey> =
        model.medial_pending.iter().map(|row| row.key).collect();
    let mut keys: Vec<_> = model
        .gcp
        .iter()
        .filter(|(key, node)| key.len() >= 2 && node.total >= 2)
        .map(|(key, _)| *key)
        .collect();
    keys.sort_unstable();
    let budget = 1024usize.min(4096usize.saturating_sub(model.medial_pending.len()));
    let mut proposed = Vec::new();
    for key in keys {
        if proposed.len() == budget {
            break;
        }
        let Some(recent_top) = dominant_token(model.gcp.get(&key)) else { continue };
        let medial_top = dominant_token(model.gm1.get(&key)).unwrap_or(u16::MAX);
        let long_top = dominant_token(model.glp.get(&key)).unwrap_or(u16::MAX);
        if medial_top == recent_top && long_top == recent_top {
            continue;
        }
        let dream_key = MedialDreamKey { context: key, target: recent_top };
        if !existing.insert(dream_key) {
            continue;
        }
        proposed.push(MedialDreamCandidate {
            key: dream_key,
            recent_top,
            medial_top,
            long_top,
            age: 0,
            supported_windows: 0,
            adverse_windows: 0,
            evaluated: 0,
            credit_sum: 0.0,
        });
    }
    stats.medial_dream_proposed += proposed.len() as u64;
    if model.medial_mode == 2 {
        for candidate in proposed {
            stats.medial_dream_immediate += 1;
            materialize_medial_dream(model, &candidate, 1.0, options, stats);
        }
    } else {
        model.medial_pending.extend(proposed);
        model.medial_pending.sort_unstable_by_key(|row| row.key);
        model.medial_pending.truncate(4096);
        stats.medial_dream_pending_peak =
            stats.medial_dream_pending_peak.max(model.medial_pending.len());
    }
}

fn medial_dream_cycle(
    model: &mut Model,
    episode: &[u16],
    options: &Options,
    stats: &mut TrainStats,
) {
    validate_medial_dreams(model, episode, options, stats);
    propose_medial_dreams(model, options, stats);
}

fn sleep_cycle_inner(
    model: &mut Model,
    episode: &[u16],
    options: &Options,
    stats: &mut TrainStats,
    pending: &mut Vec<PendingBridge>,
    pending_origin: &mut Vec<PendingOrigin>,
    pending_deferred: &mut Vec<DeferredProposal>,
    pending_construction: &mut Vec<PendingConstruction>,
) {
    medial_dream_cycle(model, episode, options, stats);
    stats.sleeps_attempted += 1;
    let policy = options.sleep_policy;
    if !policy.active() {
        stats.sleeps_accepted += 1;
        deep_reflection(model, options, stats);
        return;
    }
    let sample_limit = if policy.construction() {
        CONSTRUCTION_SLEEP_EXAMPLES
    } else {
        4096
    };
    let examples = sleep_examples(episode, sample_limit, model.max_order);
    if policy.construction() {
        // Evidence is evaluated on the new wake block before its sleep-derived
        // structural updates. New constructions are proposed only afterwards.
        construction_sleep(model, &examples, policy, stats, pending_construction);
    }
    if policy.deferred() {
        // Delayed evidence is measured before any sleep update derived from the
        // future block. This keeps the comparison base-versus-overlay causal.
        let origin = origin_completion_journal(
            model,
            &examples,
            policy,
            stats,
            pending_origin,
            pending_deferred,
        );
        stats.sequential_origin_edges += origin.len() as u64;
        stats.sleep_transaction_edges += origin.len() as u64;
        stats.replay_updates += origin.len() as u64;
        apply_journal(model, &origin, true);
    }
    let journal = gr_loss_journal(model, &examples, policy, stats, pending);
    if policy.sequential() {
        stats.sequential_semantic_edges += journal.len() as u64;
    }
    apply_journal(model, &journal, true);
    if policy.origin() && !policy.deferred() {
        let origin = origin_completion_journal(
            model,
            &examples,
            policy,
            stats,
            pending_origin,
            pending_deferred,
        );
        stats.sequential_origin_edges += origin.len() as u64;
        stats.sleep_transaction_edges += origin.len() as u64;
        stats.replay_updates += origin.len() as u64;
        apply_journal(model, &origin, true);
    }
    if policy.separate_phase() {
        let phase = separate_phase_journal(model, &examples, policy, stats);
        stats.sequential_phase_edges += phase.len() as u64;
        stats.sleep_transaction_edges += phase.len() as u64;
        stats.replay_updates += phase.len() as u64;
        apply_journal(model, &phase, true);
    }
    stats.sleeps_accepted += 1;
    stats.sleep_passes += 1;
    deep_reflection(model, options, stats);
}


fn sleep_cycle(
    model: &mut Model,
    episode: &[u16],
    options: &Options,
    stats: &mut TrainStats,
    pending: &mut Vec<PendingBridge>,
    pending_origin: &mut Vec<PendingOrigin>,
    pending_deferred: &mut Vec<DeferredProposal>,
    pending_construction: &mut Vec<PendingConstruction>,
) {
    let timer = profile_timer();
    sleep_cycle_inner(
        model,
        episode,
        options,
        stats,
        pending,
        pending_origin,
        pending_deferred,
        pending_construction,
    );
    profile_finish(timer, &PROFILE_SLEEP_NS, &PROFILE_SLEEP_CALLS);
}

fn train_serial(model: &mut Model, tokens: &[u16], options: &Options) -> TrainStats {
    let mut stats = TrainStats::default();
    let mut pending = Vec::new();
    let mut pending_origin = Vec::new();
    let mut pending_deferred = Vec::new();
    let mut pending_construction = Vec::new();
    let mut history = Vec::with_capacity(model.max_order);
    let (mut base, mut total, mut top) = base_weights(model);
    let mut wake_start = 0usize;
    let mut wake_fingerprint = model.glp_version;
    for (index, &gold) in tokens.iter().enumerate() {
        stats.tokens = index + 1;
        if index % 512 == 0 || gold == 0 {
            history.clear();
        }
        if !history.is_empty() {
            let (p, ranked, features, _) =
                training_distribution(model, &history, gold, &base, total, &top);
            stats.nll -= p.ln();
            stats.predictions += 1;
            update(
                model, &history, gold, p, &ranked, &features, options, &mut stats,
            );
            recycle_ranked_buffer(ranked);
            recycle_feature_buffer(features);
        } else {
            model.gcp_tokens[gold as usize].count =
                model.gcp_tokens[gold as usize].count.saturating_add(1);
        }
        history.push(gold);
        if history.len() > model.max_order {
            history.remove(0);
        }
        let sleep_boundary = stats.tokens % options.sleep_policy.interval(options) == 0;
        if sleep_boundary {
            stats.wake_integrity_checks += 1;
            if model.glp_version != wake_fingerprint {
                stats.wake_glp_mutation_violations += 1;
            }
            sleep_cycle(
                model,
                &tokens[wake_start..stats.tokens],
                options,
                &mut stats,
                &mut pending,
                &mut pending_origin,
                &mut pending_deferred,
                &mut pending_construction,
            );
            wake_start = stats.tokens;
            wake_fingerprint = model.glp_version;
            let rebuilt = base_weights(model);
            base = rebuilt.0;
            total = rebuilt.1;
            top = rebuilt.2;
        } else if stats.tokens % options.light_interval == 0 {
            light_reflection(model, options, &mut stats);
        } else if stats.tokens % 100_000 == 0 {
            let rebuilt = base_weights(model);
            base = rebuilt.0;
            total = rebuilt.1;
            top = rebuilt.2;
        }
        stats.max_gcp = stats.max_gcp.max(model.gcp.len());
    }
    if !model.gcp.is_empty() || !model.gcp_credit.is_empty() {
        stats.wake_integrity_checks += 1;
        if model.glp_version != wake_fingerprint {
            stats.wake_glp_mutation_violations += 1;
        }
        sleep_cycle(
            model,
            &tokens[wake_start..],
            options,
            &mut stats,
            &mut pending,
            &mut pending_origin,
            &mut pending_deferred,
            &mut pending_construction,
        );
    }
    stats
}


#[derive(Clone)]
struct SpecEvent {
    index: usize,
    gold: u16,
    history: [u16; 4],
    history_len: u8,
    speculative: bool,
    route_keys: [ContextKey; 4],
}

impl SpecEvent {
    fn history(&self) -> &[u16] {
        &self.history[..self.history_len as usize]
    }
}

struct SpecResult {
    position: usize,
    p_gold: f64,
    ranked: Vec<(u16, f64)>,
    features: Vec<Feature>,
}

#[derive(Default)]
struct SpecAudit {
    windows: u64,
    events: u64,
    predicted: u64,
    speculative: u64,
    replay: u64,
    empty_history: u64,
    window_sum: u64,
    window_min: usize,
    window_max: usize,
}

fn exp154_workers() -> usize {
    std::env::var("NGU_SPEC_WORKERS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or_else(parallel_train_workers)
        .clamp(1, 64)
}

fn exp154_window() -> usize {
    std::env::var("NGU_SPEC_WINDOW")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(0)
}

fn exp154_adaptive() -> bool {
    std::env::var("NGU_SPEC_ADAPTIVE")
        .map(|value| value == "1" || value.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

fn exp154_compute_enabled() -> bool {
    std::env::var("NGU_SPEC_COMPUTE")
        .map(|value| value != "0" && !value.eq_ignore_ascii_case("false"))
        .unwrap_or(true)
}

fn build_spec_events(
    tokens: &[u16],
    start: usize,
    end: usize,
    max_order: usize,
    history: &mut Vec<u16>,
) -> Vec<SpecEvent> {
    let mut seen = std::collections::BTreeSet::new();
    let mut events = Vec::with_capacity(end - start);
    for index in start..end {
        let gold = tokens[index];
        if index % 512 == 0 || gold == 0 {
            history.clear();
        }
        let mut stored = [0u16; 4];
        stored[..history.len()].copy_from_slice(history);
        let mut conflict = false;
        let mut route_keys = [ContextKey::new(0, [0; 4]); 4];
        for order in 1..=max_order.min(history.len()) {
            let key = make_key(history, order);
            route_keys[order - 1] = key;
            if !seen.insert(key) {
                conflict = true;
            }
        }
        events.push(SpecEvent {
            index,
            gold,
            history: stored,
            history_len: history.len() as u8,
            speculative: !history.is_empty() && !conflict,
            route_keys,
        });
        history.push(gold);
        if history.len() > max_order {
            history.remove(0);
        }
    }
    events
}

fn calculate_speculative(
    model: &Model,
    events: &[SpecEvent],
    base: &[f64],
    total: f64,
    top: &[u16],
    workers: usize,
) -> Vec<SpecResult> {
    let selected: Vec<usize> = events
        .iter()
        .enumerate()
        .filter_map(|(position, event)| event.speculative.then_some(position))
        .collect();
    if selected.is_empty() {
        return Vec::new();
    }
    let workers = workers.min(selected.len()).max(1);
    let workgroups: Vec<Vec<usize>> = if model.shard_mode == 15 {
        GROUPED_SPEC_WINDOWS.fetch_add(1, Ordering::Relaxed);
        let mut buckets: Vec<Vec<usize>> =
            (0..EXP170_SHARDS).map(|_| Vec::new()).collect();
        for &position in &selected {
            let event = &events[position];
            let owner = (1..=event.history().len())
                .rev()
                .find_map(|order| {
                    let key = make_key(event.history(), order);
                    model.shard_assignments.get(&key).copied()
                })
                .unwrap_or_else(|| {
                    event.history().last().copied().unwrap_or(0) as u8
                        % EXP170_SHARDS as u8
                });
            buckets[owner as usize].push(position);
        }
        let total_selected: usize = buckets.iter().map(Vec::len).sum();
        let target = (total_selected + workers - 1) / workers;
        let mut tasks = Vec::new();
        for bucket in buckets {
            if bucket.is_empty() {
                continue;
            }
            for chunk in bucket.chunks(target.max(1)) {
                tasks.push(chunk.to_vec());
            }
        }
        GROUPED_SPEC_TASKS.fetch_add(tasks.len() as u64, Ordering::Relaxed);
        tasks
    } else {
        let chunk = (selected.len() + workers - 1) / workers;
        selected.chunks(chunk).map(|rows| rows.to_vec()).collect()
    };
    let mut groups = std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for positions in &workgroups {
            handles.push(scope.spawn(move || {
                let mut local = Vec::with_capacity(positions.len());
                for &position in positions {
                    let event = &events[position];
                    let (p_gold, ranked, features, _) = training_distribution(
                        model,
                        event.history(),
                        event.gold,
                        base,
                        total,
                        top,
                    );
                    local.push(SpecResult {
                        position,
                        p_gold,
                        ranked,
                        features,
                    });
                }
                local
            }));
        }
        handles
            .into_iter()
            .map(|handle| handle.join().expect("EXP173 shard search worker"))
            .collect::<Vec<_>>()
    });
    let mut results = Vec::with_capacity(selected.len());
    for group in groups.drain(..) {
        results.extend(group);
    }
    results.sort_unstable_by_key(|result| result.position);
    results
}

fn train_speculative(
    model: &mut Model,
    tokens: &[u16],
    options: &Options,
    initial_window: usize,
) -> TrainStats {
    let mut stats = TrainStats::default();
    let mut pending = Vec::new();
    let mut pending_origin = Vec::new();
    let mut pending_deferred = Vec::new();
    let mut pending_construction = Vec::new();
    let mut history = Vec::with_capacity(model.max_order);
    let (mut base, mut total, mut top) = base_weights(model);
    let mut wake_start = 0usize;
    let mut wake_fingerprint = model.glp_version;
    let mut index = 0usize;
    let mut window = initial_window.max(1);
    let adaptive = exp154_adaptive();
    let compute_enabled = exp154_compute_enabled();
    let workers = exp154_workers();
    let mut audit = SpecAudit {
        window_min: usize::MAX,
        ..SpecAudit::default()
    };

    while index < tokens.len() {
        let next_light = ((index / options.light_interval) + 1) * options.light_interval;
        let sleep_interval = options.sleep_policy.interval(options);
        let next_sleep = ((index / sleep_interval) + 1) * sleep_interval;
        let end = (index + window)
            .min(tokens.len())
            .min(next_light)
            .min(next_sleep);
        let events = build_spec_events(tokens, index, end, model.max_order, &mut history);
        let spec_count = events.iter().filter(|event| event.speculative).count();
        let predicted = events.iter().filter(|event| event.history_len > 0).count();
        let conflict_count = predicted - spec_count;
        audit.windows += 1;
        audit.events += events.len() as u64;
        audit.predicted += predicted as u64;
        audit.speculative += spec_count as u64;
        audit.replay += conflict_count as u64;
        audit.empty_history += (events.len() - predicted) as u64;
        audit.window_sum += events.len() as u64;
        audit.window_min = audit.window_min.min(events.len());
        audit.window_max = audit.window_max.max(events.len());

        let results = if compute_enabled {
            calculate_speculative(model, &events, &base, total, &top, workers)
        } else {
            Vec::new()
        };
        let mut result_iter = results.into_iter().peekable();

        for (position, event) in events.iter().enumerate() {
            stats.tokens = event.index + 1;
            if event.history_len > 0 {
                let (p_gold, ranked, features) = if compute_enabled && event.speculative {
                    let result = result_iter
                        .next()
                        .expect("EXP154 missing speculative result");
                    assert_eq!(result.position, position);
                    (result.p_gold, result.ranked, result.features)
                } else {
                    let (p_gold, ranked, features, _) = training_distribution(
                        model,
                        event.history(),
                        event.gold,
                        &base,
                        total,
                        &top,
                    );
                    (p_gold, ranked, features)
                };
                stats.nll -= p_gold.ln();
                stats.predictions += 1;
                update(
                    model,
                    event.history(),
                    event.gold,
                    p_gold,
                    &ranked,
                    &features,
                    options,
                    &mut stats,
                );
                recycle_ranked_buffer(ranked);
                recycle_feature_buffer(features);
            } else {
                model.gcp_tokens[event.gold as usize].count = model.gcp_tokens
                    [event.gold as usize]
                    .count
                    .saturating_add(1);
            }
            let token_boundary = stats.tokens % sleep_interval == 0
                || stats.tokens % options.light_interval == 0
                || stats.tokens % 100_000 == 0;
            if !token_boundary {
                stats.max_gcp = stats.max_gcp.max(model.gcp.len());
            }
        }
        assert!(result_iter.next().is_none());
        index = end;

        let sleep_boundary = stats.tokens % sleep_interval == 0;
        if sleep_boundary {
            stats.wake_integrity_checks += 1;
            if model.glp_version != wake_fingerprint {
                stats.wake_glp_mutation_violations += 1;
            }
            sleep_cycle(
                model,
                &tokens[wake_start..stats.tokens],
                options,
                &mut stats,
                &mut pending,
                &mut pending_origin,
                &mut pending_deferred,
                &mut pending_construction,
            );
            wake_start = stats.tokens;
            wake_fingerprint = model.glp_version;
            let rebuilt = base_weights(model);
            base = rebuilt.0;
            total = rebuilt.1;
            top = rebuilt.2;
        } else if stats.tokens % options.light_interval == 0 {
            light_reflection(model, options, &mut stats);
        } else if stats.tokens % 100_000 == 0 {
            let rebuilt = base_weights(model);
            base = rebuilt.0;
            total = rebuilt.1;
            top = rebuilt.2;
        }
        // The serial reference samples this diagnostic after any boundary
        // reflection. Within a window GCP size is monotonic, so sampling once
        // here preserves the exact maximum without changing learned state.
        stats.max_gcp = stats.max_gcp.max(model.gcp.len());

        if adaptive && predicted > 0 {
            let rate = conflict_count as f64 / predicted as f64;
            if rate > 0.25 {
                window = (window / 2).max(64);
            } else if rate < 0.05 {
                window = (window * 2).min(4096);
            }
        }
    }
    if !model.gcp.is_empty() || !model.gcp_credit.is_empty() {
        stats.wake_integrity_checks += 1;
        if model.glp_version != wake_fingerprint {
            stats.wake_glp_mutation_violations += 1;
        }
        sleep_cycle(
            model,
            &tokens[wake_start..],
            options,
            &mut stats,
            &mut pending,
            &mut pending_origin,
            &mut pending_deferred,
            &mut pending_construction,
        );
    }
    let conflict_rate = if audit.predicted > 0 {
        audit.replay as f64 / audit.predicted as f64
    } else {
        0.0
    };
    eprintln!(
        "EXP154_SPEC window_initial={} adaptive={} compute={} workers={} windows={} \
events={} predicted={} speculative={} replay={} empty={} conflict_rate={:.9} \
window_mean={:.3} window_min={} window_max={}",
        initial_window,
        adaptive,
        compute_enabled,
        workers,
        audit.windows,
        audit.events,
        audit.predicted,
        audit.speculative,
        audit.replay,
        audit.empty_history,
        conflict_rate,
        audit.window_sum as f64 / audit.windows.max(1) as f64,
        if audit.window_min == usize::MAX { 0 } else { audit.window_min },
        audit.window_max,
    );
    stats
}



enum PersistentMessage {
    Run {
        events: Arc<Vec<SpecEvent>>,
        positions: Vec<usize>,
        base: Arc<Vec<f64>>,
        total: f64,
        top: Arc<Vec<u16>>,
        reply: mpsc::Sender<Vec<SpecResult>>,
    },
    Stop,
}

fn persistent_worker(
    model: &RwLock<&mut Model>,
    receiver: mpsc::Receiver<PersistentMessage>,
) {
    while let Ok(message) = receiver.recv() {
        match message {
            PersistentMessage::Run {
                events,
                positions,
                base,
                total,
                top,
                reply,
            } => {
                let timer = profile_timer();
                let guard = model.read().expect("EXP174 model read lock");
                let mut local = Vec::with_capacity(positions.len());
                for position in positions {
                    let event = &events[position];
                    let (p_gold, ranked, features, _) = training_distribution(
                        &**guard,
                        event.history(),
                        event.gold,
                        base.as_slice(),
                        total,
                        top.as_slice(),
                    );
                    local.push(SpecResult {
                        position,
                        p_gold,
                        ranked,
                        features,
                    });
                }
                drop(guard);
                profile_finish(timer, &POOL_WORK_NS, &POOL_WORK_CALLS);
                reply.send(local).expect("EXP174 result receiver");
            }
            PersistentMessage::Stop => break,
        }
    }
}

fn exp174_event_owner(model: &Model, event: &SpecEvent, descriptors: bool) -> usize {
    let owner = if descriptors {
        (1..=event.history_len as usize)
            .rev()
            .find_map(|order| {
                model
                    .shard_assignments
                    .get(&event.route_keys[order - 1])
                    .copied()
            })
    } else {
        (1..=event.history().len()).rev().find_map(|order| {
            let key = make_key(event.history(), order);
            model.shard_assignments.get(&key).copied()
        })
    };
    owner
        .unwrap_or_else(|| {
            event.history().last().copied().unwrap_or(0) as u8
                % EXP170_SHARDS as u8
        }) as usize
}

fn exp174_workgroups(
    model: &Model,
    events: &[SpecEvent],
    workers: usize,
    pool_mode: u8,
) -> Vec<Vec<usize>> {
    let selected: Vec<usize> = events
        .iter()
        .enumerate()
        .filter_map(|(position, event)| event.speculative.then_some(position))
        .collect();
    if selected.is_empty() {
        return Vec::new();
    }
    let workers = workers.min(selected.len()).max(1);
    if pool_mode == 1 {
        let chunk = (selected.len() + workers - 1) / workers;
        return selected
            .chunks(chunk)
            .map(|positions| positions.to_vec())
            .collect();
    }
    let descriptors = pool_mode == 3;
    let mut groups: Vec<Vec<usize>> = (0..workers).map(|_| Vec::new()).collect();
    for position in selected {
        let owner = exp174_event_owner(model, &events[position], descriptors);
        groups[owner % workers].push(position);
    }
    groups.retain(|group| !group.is_empty());
    groups
}

fn exp174_dispatch(
    model: &RwLock<&mut Model>,
    senders: &[mpsc::Sender<PersistentMessage>],
    events: &Arc<Vec<SpecEvent>>,
    base: &Arc<Vec<f64>>,
    total: f64,
    top: &Arc<Vec<u16>>,
    pool_mode: u8,
) -> Vec<SpecResult> {
    let prep_timer = profile_timer();
    let events = Arc::clone(events);
    let groups = {
        let guard = model.read().expect("EXP174 schedule read lock");
        exp174_workgroups(&**guard, events.as_slice(), senders.len(), pool_mode)
    };
    profile_finish(prep_timer, &POOL_PREP_NS, &POOL_PREP_CALLS);
    POOL_WINDOWS.fetch_add(1, Ordering::Relaxed);
    POOL_TASKS.fetch_add(groups.len() as u64, Ordering::Relaxed);
    POOL_POSITIONS.fetch_add(
        groups.iter().map(Vec::len).sum::<usize>() as u64,
        Ordering::Relaxed,
    );
    if groups.is_empty() {
        return Vec::new();
    }

    let (reply, receiver) = mpsc::channel();
    let dispatch_timer = profile_timer();
    for (sender, positions) in senders.iter().zip(groups.into_iter()) {
        sender
            .send(PersistentMessage::Run {
                events: Arc::clone(&events),
                positions,
                base: Arc::clone(base),
                total,
                top: Arc::clone(top),
                reply: reply.clone(),
            })
            .expect("EXP174 persistent worker");
    }
    profile_finish(
        dispatch_timer,
        &POOL_DISPATCH_NS,
        &POOL_DISPATCH_CALLS,
    );
    drop(reply);

    let wait_timer = profile_timer();
    let mut chunks = Vec::new();
    while let Ok(result) = receiver.recv() {
        chunks.push(result);
    }
    profile_finish(wait_timer, &POOL_WAIT_NS, &POOL_WAIT_CALLS);

    let reduce_timer = profile_timer();
    let total_results: usize = chunks.iter().map(Vec::len).sum();
    let mut results = Vec::with_capacity(total_results);
    for chunk in chunks {
        results.extend(chunk);
    }
    results.sort_unstable_by_key(|result| result.position);
    profile_finish(reduce_timer, &POOL_REDUCE_NS, &POOL_REDUCE_CALLS);
    results
}

fn train_persistent(
    model: &mut Model,
    tokens: &[u16],
    options: &Options,
    initial_window: usize,
    pool_mode: u8,
) -> TrainStats {
    let mut stats = TrainStats::default();
    let mut pending = Vec::new();
    let mut pending_origin = Vec::new();
    let mut pending_deferred = Vec::new();
    let mut pending_construction = Vec::new();
    let mut history = Vec::with_capacity(model.max_order);
    let initial_base = base_weights(model);
    let mut base = Arc::new(initial_base.0);
    let mut total = initial_base.1;
    let mut top = Arc::new(initial_base.2);
    let mut wake_start = 0usize;
    let mut wake_fingerprint = model.glp_version;
    let mut index = 0usize;
    let mut window = initial_window.max(1);
    let adaptive = exp154_adaptive();
    let compute_enabled = exp154_compute_enabled();
    let workers = exp154_workers();
    let mut audit = SpecAudit {
        window_min: usize::MAX,
        ..SpecAudit::default()
    };
    let model_lock = RwLock::new(model);

    std::thread::scope(|scope| {
        let mut senders = Vec::with_capacity(workers);
        for _ in 0..workers {
            let (sender, receiver) = mpsc::channel();
            senders.push(sender);
            let shared = &model_lock;
            scope.spawn(move || persistent_worker(shared, receiver));
        }

        while index < tokens.len() {
            let next_light =
                ((index / options.light_interval) + 1) * options.light_interval;
            let sleep_interval = options.sleep_policy.interval(options);
            let next_sleep = ((index / sleep_interval) + 1) * sleep_interval;
            let end = (index + window)
                .min(tokens.len())
                .min(next_light)
                .min(next_sleep);
            let max_order = model_lock
                .read()
                .expect("EXP174 max order")
                .max_order;
            let events = Arc::new(
                build_spec_events(tokens, index, end, max_order, &mut history)
            );
            let spec_count =
                events.iter().filter(|event| event.speculative).count();
            let predicted =
                events.iter().filter(|event| event.history_len > 0).count();
            let conflict_count = predicted - spec_count;
            audit.windows += 1;
            audit.events += events.len() as u64;
            audit.predicted += predicted as u64;
            audit.speculative += spec_count as u64;
            audit.replay += conflict_count as u64;
            audit.empty_history += (events.len() - predicted) as u64;
            audit.window_sum += events.len() as u64;
            audit.window_min = audit.window_min.min(events.len());
            audit.window_max = audit.window_max.max(events.len());

            let results = if compute_enabled {
                exp174_dispatch(
                    &model_lock,
                    &senders,
                    &events,
                    &base,
                    total,
                    &top,
                    pool_mode,
                )
            } else {
                Vec::new()
            };
            let mut result_iter = results.into_iter().peekable();
            let mut guard = model_lock.write().expect("EXP174 update lock");

            for (position, event) in events.iter().enumerate() {
                stats.tokens = event.index + 1;
                if event.history_len > 0 {
                    let (p_gold, ranked, features) =
                        if compute_enabled && event.speculative {
                            let result = result_iter
                                .next()
                                .expect("EXP174 missing persistent result");
                            assert_eq!(result.position, position);
                            (result.p_gold, result.ranked, result.features)
                        } else {
                            let (p_gold, ranked, features, _) =
                                training_distribution(
                                    &**guard,
                                    event.history(),
                                    event.gold,
                                    base.as_slice(),
                                    total,
                                    top.as_slice(),
                                );
                            (p_gold, ranked, features)
                        };
                    stats.nll -= p_gold.ln();
                    stats.predictions += 1;
                    update(
                        &mut **guard,
                        event.history(),
                        event.gold,
                        p_gold,
                        &ranked,
                        &features,
                        options,
                        &mut stats,
                    );
                    recycle_ranked_buffer(ranked);
                    recycle_feature_buffer(features);
                } else {
                    guard.gcp_tokens[event.gold as usize].count = guard
                        .gcp_tokens[event.gold as usize]
                        .count
                        .saturating_add(1);
                }
                let token_boundary = stats.tokens % sleep_interval == 0
                    || stats.tokens % options.light_interval == 0
                    || stats.tokens % 100_000 == 0;
                if !token_boundary {
                    stats.max_gcp = stats.max_gcp.max(guard.gcp.len());
                }
            }
            assert!(result_iter.next().is_none());
            index = end;

            let sleep_boundary = stats.tokens % sleep_interval == 0;
            if sleep_boundary {
                stats.wake_integrity_checks += 1;
                if guard.glp_version != wake_fingerprint {
                    stats.wake_glp_mutation_violations += 1;
                }
                sleep_cycle(
                    &mut **guard,
                    &tokens[wake_start..stats.tokens],
                    options,
                    &mut stats,
                    &mut pending,
                    &mut pending_origin,
                    &mut pending_deferred,
                    &mut pending_construction,
                );
                wake_start = stats.tokens;
                wake_fingerprint = guard.glp_version;
                let rebuilt = base_weights(&**guard);
                base = Arc::new(rebuilt.0);
                total = rebuilt.1;
                top = Arc::new(rebuilt.2);
            } else if stats.tokens % options.light_interval == 0 {
                light_reflection(&mut **guard, options, &mut stats);
            } else if stats.tokens % 100_000 == 0 {
                let rebuilt = base_weights(&**guard);
                base = Arc::new(rebuilt.0);
                total = rebuilt.1;
                top = Arc::new(rebuilt.2);
            }
            stats.max_gcp = stats.max_gcp.max(guard.gcp.len());
            drop(guard);

            if adaptive && predicted > 0 {
                let rate = conflict_count as f64 / predicted as f64;
                if rate > 0.25 {
                    window = (window / 2).max(64);
                } else if rate < 0.05 {
                    window = (window * 2).min(4096);
                }
            }
        }

        {
            let mut guard = model_lock.write().expect("EXP174 final sleep lock");
            if !guard.gcp.is_empty() || !guard.gcp_credit.is_empty() {
                stats.wake_integrity_checks += 1;
                if guard.glp_version != wake_fingerprint {
                    stats.wake_glp_mutation_violations += 1;
                }
                sleep_cycle(
                    &mut **guard,
                    &tokens[wake_start..],
                    options,
                    &mut stats,
                    &mut pending,
                    &mut pending_origin,
                    &mut pending_deferred,
                    &mut pending_construction,
                );
            }
        }
        for sender in &senders {
            sender
                .send(PersistentMessage::Stop)
                .expect("EXP174 stop worker");
        }
    });

    let conflict_rate = if audit.predicted > 0 {
        audit.replay as f64 / audit.predicted as f64
    } else {
        0.0
    };
    eprintln!(
        "EXP174_POOL mode={} window={} adaptive={} compute={} workers={} \
windows={} events={} predicted={} speculative={} replay={} conflict_rate={:.9}",
        pool_mode,
        initial_window,
        adaptive,
        compute_enabled,
        workers,
        audit.windows,
        audit.events,
        audit.predicted,
        audit.speculative,
        audit.replay,
        conflict_rate,
    );
    stats
}


struct DependencyResult {
    position: usize,
    p_gold: f64,
    ranked: Vec<(u16, f64)>,
    features: Vec<Feature>,
    plan: Option<PreparedUpdate>,
}

fn dependency_waves(events: &[SpecEvent]) -> Vec<Vec<usize>> {
    let mut last_level: HashMap<ContextKey, usize> = HashMap::new();
    let mut waves: Vec<Vec<usize>> = Vec::new();
    for (position, event) in events.iter().enumerate() {
        if event.history_len == 0 {
            continue;
        }
        let mut level = 0usize;
        for order in 0..event.history_len as usize {
            level = level.max(
                last_level
                    .get(&event.route_keys[order])
                    .map(|value| value + 1)
                    .unwrap_or(0),
            );
        }
        if waves.len() <= level {
            waves.resize_with(level + 1, Vec::new);
        }
        waves[level].push(position);
        for order in 0..event.history_len as usize {
            last_level.insert(event.route_keys[order], level);
        }
    }
    waves
}

fn calculate_dependency_wave(
    model: &Model,
    events: &[SpecEvent],
    positions: &[usize],
    base: &[f64],
    total: f64,
    top: &[u16],
    workers: usize,
    prepared: bool,
) -> Vec<DependencyResult> {
    if positions.is_empty() {
        return Vec::new();
    }
    let workers = workers.min(positions.len()).max(1);
    let chunk = (positions.len() + workers - 1) / workers;
    let timer = profile_timer();
    let mut groups = std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for rows in positions.chunks(chunk) {
            handles.push(scope.spawn(move || {
                let mut local = Vec::with_capacity(rows.len());
                for &position in rows {
                    let event = &events[position];
                    let (p_gold, ranked, features, _) = training_distribution(
                        model,
                        event.history(),
                        event.gold,
                        base,
                        total,
                        top,
                    );
                    let plan = prepared.then(|| {
                        let plan_timer = profile_timer();
                        let value = prepare_update(
                            event.history(),
                            event.gold,
                            &ranked,
                            model.max_order,
                        );
                        profile_finish(
                            plan_timer,
                            &PHASE_PLAN_NS,
                            &PHASE_PLAN_CALLS,
                        );
                        value
                    });
                    local.push(DependencyResult {
                        position,
                        p_gold,
                        ranked,
                        features,
                        plan,
                    });
                }
                local
            }));
        }
        handles
            .into_iter()
            .map(|handle| handle.join().expect("EXP175 wave worker"))
            .collect::<Vec<_>>()
    });
    profile_finish(
        timer,
        &PHASE_DISTRIBUTION_NS,
        &PHASE_DISTRIBUTION_CALLS,
    );
    let mut results = Vec::with_capacity(positions.len());
    for group in groups.drain(..) {
        results.extend(group);
    }
    results.sort_unstable_by_key(|result| result.position);
    results
}

fn train_dependency_waves(
    model: &mut Model,
    tokens: &[u16],
    options: &Options,
    initial_window: usize,
    prepared: bool,
) -> TrainStats {
    let mut stats = TrainStats::default();
    let mut pending = Vec::new();
    let mut pending_origin = Vec::new();
    let mut pending_deferred = Vec::new();
    let mut pending_construction = Vec::new();
    let mut history = Vec::with_capacity(model.max_order);
    let (mut base, mut total, mut top) = base_weights(model);
    let mut wake_start = 0usize;
    let mut wake_fingerprint = model.glp_version;
    let mut index = 0usize;
    let mut window = initial_window.max(1);
    let adaptive = exp154_adaptive();
    let workers = exp154_workers();

    while index < tokens.len() {
        let next_light =
            ((index / options.light_interval) + 1) * options.light_interval;
        let sleep_interval = options.sleep_policy.interval(options);
        let next_sleep = ((index / sleep_interval) + 1) * sleep_interval;
        let end = (index + window)
            .min(tokens.len())
            .min(next_light)
            .min(next_sleep);
        let events =
            build_spec_events(tokens, index, end, model.max_order, &mut history);
        let waves = dependency_waves(&events);
        PHASE_WINDOWS.fetch_add(1, Ordering::Relaxed);
        PHASE_WAVES.fetch_add(waves.len() as u64, Ordering::Relaxed);
        PHASE_MAX_WAVES.fetch_max(waves.len() as u64, Ordering::Relaxed);
        PHASE_POSITIONS.fetch_add(
            events.iter().filter(|event| event.history_len > 0).count() as u64,
            Ordering::Relaxed,
        );
        let predicted = events.iter().filter(|event| event.history_len > 0).count();
        let mut nll_by_position = vec![0.0f64; events.len()];

        for positions in &waves {
            let results = calculate_dependency_wave(
                model,
                &events,
                positions,
                &base,
                total,
                &top,
                workers,
                prepared,
            );
            let commit_timer = profile_timer();
            for result in results {
                let event = &events[result.position];
                stats.tokens = event.index + 1;
                nll_by_position[result.position] = -result.p_gold.ln();
                stats.predictions += 1;
                if let Some(plan) = &result.plan {
                    apply_prepared_update(
                        model,
                        event.gold,
                        result.p_gold,
                        &result.features,
                        plan,
                        options,
                        &mut stats,
                    );
                } else {
                    update(
                        model,
                        event.history(),
                        event.gold,
                        result.p_gold,
                        &result.ranked,
                        &result.features,
                        options,
                        &mut stats,
                    );
                }
                recycle_ranked_buffer(result.ranked);
                recycle_feature_buffer(result.features);
                stats.max_gcp = stats.max_gcp.max(model.gcp.len());
            }
            profile_finish(
                commit_timer,
                &PHASE_COMMIT_NS,
                &PHASE_COMMIT_CALLS,
            );
        }
        for (position, event) in events.iter().enumerate() {
            if event.history_len == 0 {
                model.gcp_tokens[event.gold as usize].count = model.gcp_tokens
                    [event.gold as usize]
                    .count
                    .saturating_add(1);
            } else {
                stats.nll += nll_by_position[position];
            }
        }
        stats.tokens = end;
        index = end;

        let sleep_boundary = stats.tokens % sleep_interval == 0;
        if sleep_boundary {
            stats.wake_integrity_checks += 1;
            if model.glp_version != wake_fingerprint {
                stats.wake_glp_mutation_violations += 1;
            }
            sleep_cycle(
                model,
                &tokens[wake_start..stats.tokens],
                options,
                &mut stats,
                &mut pending,
                &mut pending_origin,
                &mut pending_deferred,
                &mut pending_construction,
            );
            wake_start = stats.tokens;
            wake_fingerprint = model.glp_version;
            let rebuilt = base_weights(model);
            base = rebuilt.0;
            total = rebuilt.1;
            top = rebuilt.2;
        } else if stats.tokens % options.light_interval == 0 {
            light_reflection(model, options, &mut stats);
        } else if stats.tokens % 100_000 == 0 {
            let rebuilt = base_weights(model);
            base = rebuilt.0;
            total = rebuilt.1;
            top = rebuilt.2;
        }
        stats.max_gcp = stats.max_gcp.max(model.gcp.len());

        if adaptive && predicted > 0 {
            let repeated = events
                .iter()
                .filter(|event| event.history_len > 0 && !event.speculative)
                .count();
            let rate = repeated as f64 / predicted as f64;
            if rate > 0.25 {
                window = (window / 2).max(64);
            } else if rate < 0.05 {
                window = (window * 2).min(4096);
            }
        }
    }
    if !model.gcp.is_empty() || !model.gcp_credit.is_empty() {
        stats.wake_integrity_checks += 1;
        if model.glp_version != wake_fingerprint {
            stats.wake_glp_mutation_violations += 1;
        }
        sleep_cycle(
            model,
            &tokens[wake_start..],
            options,
            &mut stats,
            &mut pending,
            &mut pending_origin,
            &mut pending_deferred,
            &mut pending_construction,
        );
    }
    eprintln!(
        "EXP175_PHASE prepared={} workers={} windows={} waves={} max_waves={}",
        prepared,
        workers,
        PHASE_WINDOWS.load(Ordering::Relaxed),
        PHASE_WAVES.load(Ordering::Relaxed),
        PHASE_MAX_WAVES.load(Ordering::Relaxed),
    );
    stats
}

fn train(model: &mut Model, tokens: &[u16], options: &Options) -> TrainStats {
    let window = exp154_window();
    let pool_mode = exp174_pool_mode();
    let phase_mode = exp175_phase_mode();
    if window == 0 {
        train_serial(model, tokens, options)
    } else if matches!(phase_mode, 1 | 2 | 4) {
        train_dependency_waves(
            model,
            tokens,
            options,
            window,
            matches!(phase_mode, 2 | 4),
        )
    } else if pool_mode == 0 {
        train_speculative(model, tokens, options, window)
    } else {
        assert!((1..=3).contains(&pool_mode));
        train_persistent(model, tokens, options, window, pool_mode)
    }
}

fn tail_score_mask(tokens: &[u16], tail: usize) -> Vec<bool> {
    if tail == 0 {
        return vec![true; tokens.len()];
    }
    let mut mask = vec![false; tokens.len()];
    let mut document_start = 0usize;
    for (index, &token) in tokens.iter().enumerate() {
        if token == 0 {
            let from = index.saturating_sub(tail).max(document_start);
            for value in &mut mask[from..index] {
                *value = true;
            }
            document_start = index + 1;
        }
    }
    if document_start < tokens.len() {
        let from = tokens.len().saturating_sub(tail).max(document_start);
        for value in &mut mask[from..] {
            *value = true;
        }
    }
    mask
}

fn evaluate(model: &Model, tokens: &[u16], mode: Mode, score_tail: usize) -> EvalStats {
    let mut stats = EvalStats::default();
    let score_mask = tail_score_mask(tokens, score_tail);
    let (base, total, top) = base_weights(model);
    // EXP151 model-view stage: an inactive comparison policy needs no
    // duplicate graph. Active policies retain the exact reference path until
    // the full borrowed ModelView is introduced and differentially gated.
    let late_model = if model.edge_application > 0 {
        let mut late = model.clone();
        late.edge_application = 0;
        Some(late)
    } else {
        None
    };
    let count_total: f64 = (0..model.vocab_size)
        .map(|t| combined_token(model, t).count as f64)
        .sum();
    let unigram_denom = count_total + ALPHA * model.vocab_size as f64;
    let mut history = Vec::with_capacity(model.max_order);
    for (index, &gold) in tokens.iter().enumerate() {
        if (score_tail == 0 && index % 512 == 0) || gold == 0 {
            history.clear();
        }
        if !history.is_empty() && score_mask[index] {
            let (p, ranked, _, trace) =
                distribution(model, &history, gold, &base, total, &top, mode);
            for order in 1..=model.max_order.min(history.len()).min(4) {
                let key = make_key(&history, order);
                stats.context_queries_by_order[order - 1] += 1;
                stats.context_hits_by_order[order - 1] +=
                    (model.gcp.contains_key(&key)
                        || (model.gm1_mode >= 2 && model.gm1.contains_key(&key))
                        || model.glp.contains_key(&key)) as u64;
            }
            if model.edge_application > 0 {
                let (late_p, late_ranked, _, late_trace) =
                    distribution(
                        late_model.as_ref().expect("active comparison view"),
                        &history,
                        gold,
                        &base,
                        total,
                        &top,
                        mode,
                    );
                let rank = ranked
                    .iter()
                    .position(|row| row.0 == gold)
                    .unwrap_or(ranked.len()) + 1;
                let late_rank = late_ranked
                    .iter()
                    .position(|row| row.0 == gold)
                    .unwrap_or(late_ranked.len()) + 1;
                if rank < late_rank {
                    stats.paired_gold_rank_improved += 1;
                } else if rank > late_rank {
                    stats.paired_gold_rank_worsened += 1;
                } else {
                    stats.paired_gold_rank_equal += 1;
                }
                let wrong = ranked
                    .iter()
                    .find(|row| row.0 != gold)
                    .map(|row| row.1)
                    .unwrap_or(1e-30);
                let late_wrong = late_ranked
                    .iter()
                    .find(|row| row.0 != gold)
                    .map(|row| row.1)
                    .unwrap_or(1e-30);
                stats.paired_gold_margin_delta_sum +=
                    (p.max(1e-30).ln() - wrong.max(1e-30).ln())
                        - (late_p.max(1e-30).ln() - late_wrong.max(1e-30).ln());
                stats.paired_gold_probability_delta_sum += p - late_p;
                stats.paired_top1_changed +=
                    (ranked.first().map(|row| row.0)
                        != late_ranked.first().map(|row| row.0)) as u64;
                let current_set: HashSet<u16> =
                    trace.candidates.iter().map(|row| row.token).collect();
                let late_set: HashSet<u16> =
                    late_trace.candidates.iter().map(|row| row.token).collect();
                stats.paired_gr_set_changed += (current_set != late_set) as u64;
                stats.paired_predictions += 1;
            }
            stats.nll -= p.ln();
            let count = combined_token(model, gold as usize).count;
            let frequency_bin = if count <= 10 {
                0
            } else if count <= 1_000 {
                1
            } else {
                2
            };
            stats.frequency_bin_predictions[frequency_bin] += 1;
            stats.frequency_bin_nll[frequency_bin] -= p.ln();
            stats.frequency_bin_top1[frequency_bin] +=
                (ranked.first().map(|r| r.0) == Some(gold)) as u64;
            if trace.candidates.iter().any(|gr| gr.token == gold) {
                let best_wrong = ranked
                    .iter()
                    .find(|row| row.0 != gold)
                    .map(|row| row.1)
                    .unwrap_or(1e-30);
                stats.correct_gr_margin_sum +=
                    p.max(1e-30).ln() - best_wrong.max(1e-30).ln();
                stats.correct_gr_margin_count += 1;
            }
            let unigram =
                (combined_token(model, gold as usize).count as f64 + ALPHA) / unigram_denom;
            stats.unigram_nll -= unigram.max(1e-30).ln();
            stats.top1 += (ranked.first().map(|r| r.0) == Some(gold)) as usize;
            stats.top5 += ranked.iter().take(5).any(|r| r.0 == gold) as usize;
            stats.predictions += 1;
            if mode.waves() {
                stats.wave_predictions += 1;
                stats.wave_candidate_sum += trace.candidates.len() as u64;
                stats.wave_full += trace.full as usize;
                stats.wave_coverage_sum += trace.max_coverage;
                stats.gold_candidate += trace.candidates.iter().any(|gr| gr.token == gold) as usize;
                stats.gold_full_candidate += trace
                    .candidates
                    .iter()
                    .any(|gr| gr.token == gold && gr.full) as usize;
                stats.dispersion_sum +=
                    trace.candidates.iter().map(|gr| gr.dispersion).sum::<f64>();
                stats.cost_sum += trace.candidates.iter().map(|gr| gr.mean_cost).sum::<f64>();
                stats.signal_sum += trace
                    .candidates
                    .iter()
                    .map(|gr| gr.mean_signal)
                    .sum::<f64>();
                stats.conflict_sum += trace.candidates.iter().map(|gr| gr.conflict).sum::<f64>();
                stats.energy_sum += trace.candidates.iter().map(|gr| gr.energy).sum::<f64>();
                stats.candidate_count += trace.candidates.len() as u64;
                stats.early_active_constructions += trace.early_active_constructions as u64;
                stats.early_modified_transitions += trace.early_modified_transitions as u64;
                stats.early_route_reorders += trace.early_route_reorders as u64;
                stats.early_gr_adjustments += trace.early_gr_adjustments as u64;
            }
        }
        history.push(gold);
        if history.len() > model.max_order {
            history.remove(0);
        }
    }
    stats
}

fn random_unit(state: &mut u64) -> f64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    (*state >> 11) as f64 / ((1u64 << 53) as f64)
}

fn cycle_agreement(sequence: &[u16], candidate: u16, max_period: usize) -> f64 {
    let mut extended = sequence.to_vec();
    extended.push(candidate);
    let n = extended.len();
    let mut best: f64 = 0.0;
    for period in 1..=max_period.min(n / 2) {
        let comparisons = (n - period).min(4 * period).max(period);
        let start = n - comparisons;
        let matches = (start..n)
            .filter(|&i| i >= period && extended[i] == extended[i - period])
            .count();
        let valid = (start..n).filter(|&i| i >= period).count();
        if valid > 0 {
            best = best.max(matches as f64 / valid as f64);
        }
    }
    best
}

fn diversity_safe(sequence: &[u16], candidate: u16) -> bool {
    if sequence.len() < 8 {
        return true;
    }
    let token_seen = sequence.contains(&candidate);
    let pair_seen = sequence
        .windows(2)
        .any(|w| w[0] == *sequence.last().unwrap() && w[1] == candidate);
    let mut unique: HashSet<u16> = sequence.iter().copied().collect();
    let current_unique = unique.len() as f64 / sequence.len() as f64;
    unique.insert(candidate);
    let next_unique = unique.len() as f64 / (sequence.len() + 1) as f64;
    let mut pairs: HashSet<(u16, u16)> = sequence.windows(2).map(|w| (w[0], w[1])).collect();
    let current_distinct = pairs.len() as f64 / (sequence.len() - 1) as f64;
    pairs.insert((*sequence.last().unwrap(), candidate));
    let next_distinct = pairs.len() as f64 / sequence.len() as f64;
    (if current_unique < 0.30 {
        !token_seen
    } else {
        next_unique >= 0.30 || !token_seen
    }) && (if current_distinct < 0.50 {
        !pair_seen
    } else {
        next_distinct >= 0.50 || !pair_seen
    })
}

fn segment_fingerprint(tokens: &[u16]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for &token in tokens {
        hash ^= token as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}
fn hierarchy_penalty(sequence: &[u16], candidate: u16) -> f64 {
    let segment_len = 4;
    let recent = &sequence[sequence.len().saturating_sub(64)..];
    let completed: Vec<u64> = recent
        .chunks(segment_len)
        .filter(|x| x.len() == segment_len)
        .map(segment_fingerprint)
        .collect();
    let partial_start = sequence.len() - sequence.len() % segment_len;
    let mut prospective = sequence[partial_start..].to_vec();
    prospective.push(candidate);
    if prospective.len() != segment_len {
        return 0.0;
    }
    let fingerprint = segment_fingerprint(&prospective);
    let repeats = completed.iter().filter(|&&x| x == fingerprint).count();
    3.0 * repeats as f64
}

fn sample_calibrated(
    candidates: &[(u16, f64, usize, f64, f64)],
    state: &mut u64,
) -> (u16, f64, usize, f64, f64) {
    if candidates.len() <= 1 {
        return candidates
            .first()
            .copied()
            .unwrap_or((0, 1e-30, 1, 0.0, 0.0));
    }
    let scores: Vec<f64> = candidates
        .iter()
        .map(|c| (c.1.max(1e-30).ln() - c.3 - c.4) / 0.80)
        .collect();
    let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let weights: Vec<f64> = scores.iter().map(|s| (s - maximum).exp()).collect();
    let total: f64 = weights.iter().sum();
    let mut cutoff = weights.len();
    let mut cumulative = 0.0;
    for (i, w) in weights.iter().enumerate() {
        cumulative += w / total;
        if cumulative >= 0.90 {
            cutoff = i + 1;
            break;
        }
    }
    let retained: f64 = weights[..cutoff].iter().sum();
    let target = random_unit(state) * retained;
    let mut running = 0.0;
    for i in 0..cutoff {
        running += weights[i];
        if running >= target {
            return candidates[i];
        }
    }
    candidates[cutoff - 1]
}

fn generate(
    model: &Model,
    tokens: &[u16],
    mode: Mode,
    samples: usize,
    prompt: usize,
    steps: usize,
    seed: u64,
) -> GenerationStats {
    let (base, total, top) = base_weights(model);
    let mut stats = GenerationStats::default();
    for sample in 0..samples {
        let start = sample * 512;
        if start + prompt > tokens.len() {
            break;
        }
        let mut history = tokens[start..start + prompt]
            .iter()
            .rev()
            .take(model.max_order)
            .copied()
            .collect::<Vec<_>>();
        history.reverse();
        let mut output = Vec::new();
        let mut state = seed ^ ((sample as u64 + 1).wrapping_mul(0x9e3779b97f4a7c15));
        for _ in 0..steps {
            let (_, ranked, _, trace) = distribution(model, &history, 0, &base, total, &top, mode);
            let baseline = (ranked.len() * 24 + output.len() * 2) as u64;
            stats.baseline_peak = stats.baseline_peak.max(baseline);
            stats.gr_peak = stats
                .gr_peak
                .max(baseline + (trace.candidates.len() * 128) as u64);
            let candidates: Vec<(u16, f64, usize, f64, f64)> = ranked
                .iter()
                .enumerate()
                .map(|(rank, &(token, p))| {
                    let agreement = cycle_agreement(&output, token, 8);
                    let cycle = if agreement >= 0.5 {
                        6.0 * (agreement - 0.5) / 0.5
                    } else {
                        0.0
                    };
                    (token, p, rank + 1, cycle, hierarchy_penalty(&output, token))
                })
                .collect();
            let safe: Vec<_> = candidates
                .iter()
                .take(16)
                .filter(|row| {
                    cycle_agreement(&output, row.0, 8) < 0.80 && diversity_safe(&output, row.0)
                })
                .copied()
                .collect();
            let pool = if !safe.is_empty() {
                safe
            } else {
                candidates
                    .iter()
                    .skip(16)
                    .take(48)
                    .find(|row| {
                        cycle_agreement(&output, row.0, 8) < 0.80 && diversity_safe(&output, row.0)
                    })
                    .copied()
                    .into_iter()
                    .collect()
            };
            let next = sample_calibrated(
                if pool.is_empty() { &candidates } else { &pool },
                &mut state,
            )
            .0;
            output.push(next);
            history.push(next);
            if history.len() > model.max_order {
                history.remove(0);
            }
        }
        stats.continuations.push(output);
    }
    stats
}

fn geometry_stats(model: &Model, mode: Mode) -> (usize, f64, f64, f64, f64, f64) {
    if !mode.waves() {
        return (0, 0.0, 0.0, 0.0, 0.0, 0.0);
    }
    let mut lengths = Vec::new();
    let mut frequencies = Vec::new();
    let mut utilities = Vec::new();
    let mut keys: Vec<_> = model
        .glp
        .keys()
        .filter(|key| key.len() == 1)
        .copied()
        .collect();
    keys.sort_unstable();
    for key in keys {
        if let Some((_, _, mut edges)) = aggregate_node(model, &key) {
            edges.sort_unstable_by_key(|e| e.token);
            for i in 0..edges.len() {
                lengths.push(assigned_length(&edges, i, key.tokens()[0], mode));
                frequencies.push((1.0 + edges[i].count as f64).ln());
                utilities.push(utility(&edges[i]));
            }
        }
    }
    if lengths.is_empty() {
        return (0, 0.0, 0.0, 0.0, 0.0, 0.0);
    }
    let mean = lengths.iter().sum::<f64>() / lengths.len() as f64;
    let variance =
        lengths.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / lengths.len() as f64;
    let correlation = |a: &[f64], b: &[f64]| {
        let ma = a.iter().sum::<f64>() / a.len() as f64;
        let mb = b.iter().sum::<f64>() / b.len() as f64;
        let covariance = a
            .iter()
            .zip(b)
            .map(|(x, y)| (x - ma) * (y - mb))
            .sum::<f64>();
        let da = a.iter().map(|x| (x - ma) * (x - ma)).sum::<f64>().sqrt();
        let db = b.iter().map(|x| (x - mb) * (x - mb)).sum::<f64>().sqrt();
        if da * db > 0.0 {
            covariance / (da * db)
        } else {
            0.0
        }
    };
    let inverse: Vec<f64> = lengths.iter().map(|x| 1.0 / x).collect();
    let mut sorted = lengths.clone();
    sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    (
        lengths.len(),
        mean,
        variance.sqrt(),
        sorted[sorted.len() / 10],
        sorted[sorted.len() * 9 / 10],
        0.5 * (correlation(&frequencies, &inverse) + correlation(&utilities, &inverse)),
    )
}

fn logical_bytes(model: &Model) -> (u64, u64, u64) {
    let token = (model.vocab_size * std::mem::size_of::<TokenVertex>()) as u64;
    let graph = |g: &ContextMap<Node>| {
        g.values()
            .map(|n| {
                std::mem::size_of::<ContextKey>() as u64
                    + 24
                    + (n.edges.len() * std::mem::size_of::<Edge>()) as u64
            })
            .sum::<u64>()
    };
    let constructions =
        (model.glp_constructions.len() * std::mem::size_of::<Construction>()) as u64;
    (
        token + graph(&model.gcp),
        token + graph(&model.gm1) + graph(&model.gm1_cache),
        token + graph(&model.glp) + constructions,
    )
}

fn directed_construction_audit(model: &Model) -> (u64, u64, u64, u64, u64, u64) {
    let mut role_edges = 0u64;
    let mut prediction_edges = 0u64;
    let mut invalid_edges = 0u64;
    let mut self_loops = 0u64;
    let mut dags = 0u64;
    let mut flats = 0u64;
    for construction in &model.glp_constructions {
        if construction.dag_edge_count == 0 {
            flats += 1;
            continue;
        }
        dags += 1;
        if !construction.dag_is_forward_only() {
            invalid_edges += 1;
        }
        for edge in &construction.dag_edges[..construction.dag_edge_count as usize] {
            self_loops += (edge.source == edge.target) as u64;
            if edge.relation == DAG_ROLE_EDGE {
                role_edges += 1;
            } else if edge.relation == DAG_PREDICTION_EDGE {
                prediction_edges += 1;
            } else {
                invalid_edges += 1;
            }
        }
    }
    (role_edges, prediction_edges, invalid_edges, self_loops, dags, flats)
}

fn directional_edge_credit_audit(model: &Model) -> (u64, u64, usize, f64, f64, f64, f64, f64) {
    let mut weights = Vec::new();
    let mut credited = 0u64;
    let mut nonunit = 0u64;
    let mut gradient_max = 0.0f64;
    for construction in &model.glp_constructions {
        for edge in &construction.dag_edges[..construction.dag_edge_count as usize] {
            let weight = edge.weight as f64;
            weights.push(weight);
            credited += (edge.credit_count > 0) as u64;
            nonunit += ((weight - 1.0).abs() > 1e-6) as u64;
            if edge.credit_count > 0 {
                gradient_max = gradient_max.max(
                    (edge.gradient_sum as f64 / edge.credit_count as f64).abs(),
                );
            }
        }
    }
    if weights.is_empty() {
        return (0, 0, 0, 1.0, 1.0, 1.0, 0.0, 0.0);
    }
    let distinct: HashSet<_> = weights.iter().map(|x| (*x as f32).to_bits()).collect();
    let mean = weights.iter().sum::<f64>() / weights.len() as f64;
    let sd = (weights.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
        / weights.len() as f64)
        .sqrt();
    let minimum = weights.iter().copied().fold(f64::INFINITY, f64::min);
    let maximum = weights.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    (credited, nonunit, distinct.len(), minimum, maximum, mean, sd, gradient_max)
}

fn peak_rss_kib() -> u64 {
    std::fs::read_to_string("/proc/self/status")
        .unwrap_or_default()
        .lines()
        .find(|line| line.starts_with("VmHWM:"))
        .and_then(|line| line.split_whitespace().nth(1))
        .and_then(|x| x.parse().ok())
        .unwrap_or(0)
}

fn write_u32<W: Write>(w: &mut W, x: u32) {
    w.write_all(&x.to_le_bytes()).unwrap();
}
fn write_u64<W: Write>(w: &mut W, x: u64) {
    w.write_all(&x.to_le_bytes()).unwrap();
}
fn write_f32<W: Write>(w: &mut W, x: f32) {
    w.write_all(&x.to_le_bytes()).unwrap();
}
fn read_u32<R: Read>(r: &mut R) -> u32 {
    let mut b = [0; 4];
    r.read_exact(&mut b).unwrap();
    u32::from_le_bytes(b)
}
fn read_u64<R: Read>(r: &mut R) -> u64 {
    let mut b = [0; 8];
    r.read_exact(&mut b).unwrap();
    u64::from_le_bytes(b)
}
fn read_f32<R: Read>(r: &mut R) -> f32 {
    let mut b = [0; 4];
    r.read_exact(&mut b).unwrap();
    f32::from_le_bytes(b)
}

fn save_model(model: &Model, path: &str) {
    assert!(
        model.gcp_credit.is_empty(),
        "cannot serialize uncommitted GCP credit"
    );
    let mut w = BufWriter::new(File::create(path).unwrap());
    w.write_all(MAGIC).unwrap();
    write_u32(&mut w, model.vocab_size as u32);
    write_u32(&mut w, model.max_order as u32);
    for vertices in [&model.gcp_tokens, &model.gm1_tokens, &model.glp_tokens] {
        for vertex in vertices {
            write_u32(&mut w, vertex.count);
            write_f32(&mut w, vertex.potential);
        }
    }
    for graph in [&model.gcp, &model.gm1, &model.gm1_cache, &model.glp] {
        let mut rows: Vec<_> = graph.iter().collect();
        rows.sort_unstable_by_key(|row| *row.0);
        write_u32(&mut w, rows.len() as u32);
        for (key, node) in rows {
            w.write_all(&[key.len()]).unwrap();
            for token in key.tokens() {
                w.write_all(&token.to_le_bytes()).unwrap();
            }
            write_u32(&mut w, node.total);
            write_u64(&mut w, node.last_seen);
            write_f32(&mut w, node.potential);
            let mut edges = node.edges.clone();
            edges.sort_unstable_by_key(|e| e.token);
            write_u32(&mut w, edges.len() as u32);
            for edge in edges {
                w.write_all(&edge.token.to_le_bytes()).unwrap();
                write_u32(&mut w, edge.count);
                write_f32(&mut w, edge.weight);
                write_f32(&mut w, edge.gradient_sum);
                write_f32(&mut w, edge.gradient_sq);
                write_u32(&mut w, edge.conflicts);
                write_u64(&mut w, edge.last_seen);
                write_f32(&mut w, edge.sleep_bias);
            }
        }
    }
    let mut constructions = model.glp_constructions.clone();
    constructions.sort_unstable_by_key(|row| row.key);
    write_u32(&mut w, constructions.len() as u32);
    for construction in constructions {
        w.write_all(&[construction.key.len, construction.key.slot_mask])
            .unwrap();
        for symbol in construction.key.symbols {
            w.write_all(&symbol.to_le_bytes()).unwrap();
        }
        w.write_all(&construction.key.target.to_le_bytes()).unwrap();
        write_u32(&mut w, construction.support);
        w.write_all(&construction.confirmations.to_le_bytes()).unwrap();
        write_f32(&mut w, construction.value);
        w.write_all(&[construction.dag_edge_count]).unwrap();
        for edge in &construction.dag_edges[..construction.dag_edge_count as usize] {
            w.write_all(&[edge.source, edge.target, edge.relation]).unwrap();
            write_f32(&mut w, edge.weight);
            write_f32(&mut w, edge.gradient_sum);
            write_f32(&mut w, edge.gradient_sq);
            write_u32(&mut w, edge.credit_count);
        }
    }
    if model.shard_mode >= 3 {
        w.write_all(b"NGR170V1").unwrap();
        w.write_all(&[model.shard_mode]).unwrap();
        write_u32(&mut w, model.token_themes.len() as u32);
        w.write_all(&model.token_themes).unwrap();
        let mut routes: Vec<_> = model.gm1_routes.iter().collect();
        routes.sort_unstable_by_key(|row| *row.0);
        write_u32(&mut w, routes.len() as u32);
        for (key, shard) in routes {
            w.write_all(&[key.len()]).unwrap();
            for token in key.tokens() {
                w.write_all(&token.to_le_bytes()).unwrap();
            }
            w.write_all(&[*shard]).unwrap();
        }
        let mut mediators: Vec<_> = model.gm1_route_mediators.values().collect();
        mediators.sort_unstable_by_key(|row| row.signature);
        write_u32(&mut w, mediators.len() as u32);
        for mediator in mediators {
            w.write_all(&[mediator.signature.len]).unwrap();
            w.write_all(&mediator.signature.themes).unwrap();
            for vote in mediator.votes {
                write_u32(&mut w, vote);
            }
            w.write_all(&[mediator.top, mediator.confirmations]).unwrap();
        }
    }
}

fn load_model(path: &str) -> Model {
    let mut r = BufReader::new(File::open(path).unwrap());
    let mut magic = [0; 8];
    r.read_exact(&mut magic).unwrap();
    assert_eq!(&magic, MAGIC);
    let vocab = read_u32(&mut r) as usize;
    let order = read_u32(&mut r) as usize;
    let mut vertices = Vec::new();
    for _ in 0..3 {
        let mut list = Vec::with_capacity(vocab);
        for _ in 0..vocab {
            list.push(TokenVertex {
                count: read_u32(&mut r),
                potential: read_f32(&mut r),
            });
        }
        vertices.push(list);
    }
    let mut graphs = Vec::new();
    for _ in 0..4 {
        let mut graph = ContextMap::default();
        for _ in 0..read_u32(&mut r) {
            let mut len = [0];
            r.read_exact(&mut len).unwrap();
            let mut tokens = [0u16; 4];
            for token in &mut tokens {
                let mut b = [0; 2];
                r.read_exact(&mut b).unwrap();
                *token = u16::from_le_bytes(b);
            }
            let total = read_u32(&mut r);
            let last_seen = read_u64(&mut r);
            let potential = read_f32(&mut r);
            let mut edges = Vec::new();
            for _ in 0..read_u32(&mut r) {
                let mut b = [0; 2];
                r.read_exact(&mut b).unwrap();
                edges.push(Edge {
                    token: u16::from_le_bytes(b),
                    count: read_u32(&mut r),
                    weight: read_f32(&mut r),
                    gradient_sum: read_f32(&mut r),
                    gradient_sq: read_f32(&mut r),
                    conflicts: read_u32(&mut r),
                    last_seen: read_u64(&mut r),
                    sleep_bias: read_f32(&mut r),
                });
            }
            graph.insert(
                ContextKey::new(len[0], tokens),
                Node {
                    total,
                    last_seen,
                    potential,
                    edges,
                    edge_index: None,
                },
            );
        }
        graphs.push(graph);
    }
    let mut constructions = Vec::new();
    for _ in 0..read_u32(&mut r) {
        let mut header = [0u8; 2];
        r.read_exact(&mut header).unwrap();
        let mut symbols = [0u16; CONSTRUCTION_INTERFACE];
        for symbol in &mut symbols {
            let mut bytes = [0u8; 2];
            r.read_exact(&mut bytes).unwrap();
            *symbol = u16::from_le_bytes(bytes);
        }
        let mut target = [0u8; 2];
        r.read_exact(&mut target).unwrap();
        let support = read_u32(&mut r);
        let mut confirmations = [0u8; 2];
        r.read_exact(&mut confirmations).unwrap();
        let value = read_f32(&mut r);
        let mut count = [0u8; 1];
        r.read_exact(&mut count).unwrap();
        assert!((count[0] as usize) <= DAG_EDGE_COUNT);
        let mut dag_edges = [DirectedConstructionEdge::default(); DAG_EDGE_COUNT];
        for edge in &mut dag_edges[..count[0] as usize] {
            let mut header = [0u8; 3];
            r.read_exact(&mut header).unwrap();
            *edge = DirectedConstructionEdge {
                source: header[0],
                target: header[1],
                relation: header[2],
                weight: read_f32(&mut r),
                gradient_sum: read_f32(&mut r),
                gradient_sq: read_f32(&mut r),
                credit_count: read_u32(&mut r),
            };
        }
        constructions.push(Construction {
            key: ConstructionKey {
                len: header[0],
                slot_mask: header[1],
                symbols,
                target: u16::from_le_bytes(target),
            },
            support,
            confirmations: u16::from_le_bytes(confirmations),
            value,
            dag_edge_count: count[0],
            dag_edges,
        });
    }
    let mut loaded_shard_mode = 0u8;
    let mut loaded_themes = Vec::new();
    let mut loaded_routes = ContextMap::default();
    let mut loaded_mediators = RouteMap::default();
    let mut marker = [0u8; 8];
    if r.read_exact(&mut marker).is_ok() {
        assert_eq!(&marker, b"NGR170V1");
        let mut mode = [0u8; 1];
        r.read_exact(&mut mode).unwrap();
        loaded_shard_mode = mode[0];
        loaded_themes.resize(read_u32(&mut r) as usize, 0);
        r.read_exact(&mut loaded_themes).unwrap();
        for _ in 0..read_u32(&mut r) {
            let mut len = [0u8; 1];
            r.read_exact(&mut len).unwrap();
            let mut tokens = [0u16; 4];
            for token in &mut tokens {
                let mut bytes = [0u8; 2];
                r.read_exact(&mut bytes).unwrap();
                *token = u16::from_le_bytes(bytes);
            }
            let mut shard = [0u8; 1];
            r.read_exact(&mut shard).unwrap();
            loaded_routes.insert(ContextKey::new(len[0], tokens), shard[0]);
        }
        for _ in 0..read_u32(&mut r) {
            let mut len = [0u8; 1];
            r.read_exact(&mut len).unwrap();
            let mut themes = [0u8; 4];
            r.read_exact(&mut themes).unwrap();
            let mut votes = [0u32; EXP170_SHARDS];
            for vote in &mut votes {
                *vote = read_u32(&mut r);
            }
            let mut tail = [0u8; 2];
            r.read_exact(&mut tail).unwrap();
            let signature = RouteSignature { len: len[0], themes };
            loaded_mediators.insert(signature, RouteMediator {
                signature, votes, top: tail[0], confirmations: tail[1],
            });
        }
    }
    Model {
        vocab_size: vocab,
        max_order: order,
        gcp_tokens: vertices.remove(0),
        gm1_tokens: vertices.remove(0),
        glp_tokens: vertices.remove(0),
        gcp: graphs.remove(0),
        gm1: graphs.remove(0),
        gm1_cache: graphs.remove(0),
        glp: graphs.remove(0),
        layout_mode: 1,
        flat_glp: FlatGraph::default(),
        active_successor_limit: active_successor_limit(),
        active_glp: ContextMap::default(),
        gcp_credit: ContextMap::default(),
        glp_constructions: constructions,
        edge_application: 0,
        glp_version: 0,
        gm1_mode: 0,
        medial_mode: 0,
        medial_pending: Vec::new(),
        shard_mode: loaded_shard_mode,
        glp_shards: Vec::new(),
        token_themes: loaded_themes,
        gcp_routes: ContextMap::default(),
        gcp_route_mediators: RouteMap::default(),
        gm1_routes: loaded_routes,
        gm1_route_mediators: loaded_mediators,
        shard_assignments: ContextMap::default(),
        route_support: ContextMap::default(),
        route_order: BTreeSet::new(),
        compact_routes: Vec::new(),
        glp_shard_views: Vec::new(),
        eviction_rank: ContextMap::default(),
        eviction_order: BTreeSet::new(),
    }
}

fn read_tokens(path: &str, limit: usize) -> Vec<u16> {
    let mut r = BufReader::new(File::open(path).unwrap());
    let mut bytes = Vec::new();
    r.read_to_end(&mut bytes).unwrap();
    bytes
        .chunks_exact(4)
        .take(limit)
        .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as u16)
        .collect()
}

fn next_xorshift64(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

fn permute_documents(tokens: &[u16], seed: u64) -> Vec<u16> {
    let mut ranges = Vec::new();
    let mut start = 0usize;
    for (index, &token) in tokens.iter().enumerate() {
        if token == 0 {
            ranges.push((start, index + 1));
            start = index + 1;
        }
    }
    if start < tokens.len() {
        ranges.push((start, tokens.len()));
    }
    let mut state = seed.max(1);
    for index in (1..ranges.len()).rev() {
        let swap = (next_xorshift64(&mut state) as usize) % (index + 1);
        ranges.swap(index, swap);
    }
    let mut output = Vec::with_capacity(tokens.len());
    for (from, to) in ranges {
        output.extend_from_slice(&tokens[from..to]);
    }
    output
}

fn order_documents(tokens: &[u16], seed: u64, preserve: bool) -> Vec<u16> {
    if preserve {
        tokens.to_vec()
    } else {
        permute_documents(tokens, seed)
    }
}

fn arg(name: &str, default: Option<&str>) -> String {
    let args: Vec<String> = env::args().collect();
    args.windows(2)
        .find(|w| w[0] == name)
        .map(|w| w[1].clone())
        .or_else(|| default.map(str::to_string))
        .unwrap_or_else(|| panic!("missing {}", name))
}
fn has(name: &str) -> bool {
    env::args().any(|value| value == name)
}
fn requested_mode() -> Mode {
    if has("--w-unit") {
        Mode::Unit
    } else if has("--w-freq") {
        Mode::Frequency
    } else if has("--w-grad") {
        Mode::Gradient
    } else if has("--w-fg") {
        Mode::FrequencyGradient
    } else if has("--w-fgd") {
        Mode::Full
    } else if has("--shuffle") {
        Mode::Shuffle
    } else {
        Mode::Base
    }
}


fn online_warm_gcp(model: &mut Model, tokens: &[u16], options: &Options) {
    let (base, total, top) = base_weights(model);
    let mut history = Vec::with_capacity(model.max_order);
    let mut stats = TrainStats::default();
    for (index, &gold) in tokens.iter().enumerate() {
        if index % 512 == 0 || gold == 0 {
            history.clear();
        }
        stats.tokens += 1;
        if history.is_empty() {
            model.gcp_tokens[gold as usize].count =
                model.gcp_tokens[gold as usize].count.saturating_add(1);
        } else {
            let (p, ranked, features, _) =
                training_distribution(model, &history, gold, &base, total, &top);
            update(model, &history, gold, p, &ranked, &features, options, &mut stats);
        }
        history.push(gold);
        if history.len() > model.max_order {
            history.remove(0);
        }
    }
    prune_graph(&mut model.gcp, options.gcp_max);
}

fn main() {
    let start = Instant::now();
    reset_layer_access();
    let mut train_elapsed_seconds = 0.0f64;
    let command = env::args().nth(1).expect("train or eval");
    let eval_path = arg("--eval", None);
    let output = arg("--output", None);
    let mut train_stats = TrainStats::default();
    let mut model;
    let policy;
    let train_order_seed: u64;
    if command == "train" {
        let options = Options {
            vocab_size: arg("--vocab", Some("50280")).parse().unwrap(),
            max_order: arg("--max-order", Some("4")).parse().unwrap(),
            train_limit: arg("--train-limit", Some("10000000")).parse().unwrap(),
            gcp_max: arg("--gcp-max", Some("150000")).parse().unwrap(),
            gm1_max: arg("--gm1-max", Some("0")).parse().unwrap(),
            glp_max: arg("--glp-max", Some("1500000")).parse().unwrap(),
            max_successors: arg("--max-successors", Some("32")).parse().unwrap(),
            capacity_policy: CapacityPolicy::parse(&arg(
                "--capacity-policy",
                Some("fixed"),
            )),
            successor_index: arg("--successor-index", Some("on")) == "on",
            light_interval: arg("--light-interval", Some("250000")).parse().unwrap(),
            deep_interval: arg("--deep-interval", Some("1000000")).parse().unwrap(),
            wake_interval: arg("--wake-interval", Some("250000")).parse().unwrap(),
            lr: arg("--lr", Some("0.025")).parse().unwrap(),
            sleep_policy: SleepPolicy::parse(&arg("--sleep-policy", Some("cycle"))),
        };
        policy = options.sleep_policy;
        train_order_seed = arg("--train-order-seed", Some("8801")).parse().unwrap();
        let mut value = Model {
            vocab_size: options.vocab_size,
            max_order: options.max_order,
            gcp_tokens: vec![TokenVertex::default(); options.vocab_size],
            gm1_tokens: vec![TokenVertex::default(); options.vocab_size],
            glp_tokens: vec![TokenVertex::default(); options.vocab_size],
            gcp: ContextMap::default(),
            gm1: ContextMap::default(),
            gm1_cache: ContextMap::default(),
            glp: ContextMap::default(),
            layout_mode: exp176_layout_mode(),
            flat_glp: FlatGraph::default(),
        active_successor_limit: active_successor_limit(),
        active_glp: ContextMap::default(),
            gcp_credit: ContextMap::default(),
            glp_constructions: Vec::new(),
            edge_application: policy.edge_application(),
            glp_version: 0,
            gm1_mode: arg("--gm1-mode", Some("0")).parse().unwrap(),
            medial_mode: arg("--medial-mode", Some("0")).parse().unwrap(),
            medial_pending: Vec::new(),
            shard_mode: arg("--shard-mode", Some("0")).parse().unwrap(),
            glp_shards: Vec::new(),
            token_themes: Vec::new(),
            gcp_routes: ContextMap::default(),
            gcp_route_mediators: RouteMap::default(),
            gm1_routes: ContextMap::default(),
            gm1_route_mediators: RouteMap::default(),
            shard_assignments: ContextMap::default(),
            route_support: ContextMap::default(),
            route_order: BTreeSet::new(),
            compact_routes: Vec::new(),
        glp_shard_views: Vec::new(),
        eviction_rank: ContextMap::default(),
        eviction_order: BTreeSet::new(),
        };
        TRAIN_STEP_OFFSET.store(0, Ordering::Relaxed);
        let raw_tokens = read_tokens(&arg("--train", None), options.train_limit);
        let tokens = order_documents(&raw_tokens, train_order_seed, has("--preserve-order"));
        train_stats = train(&mut value, &tokens, &options);
        train_elapsed_seconds = start.elapsed().as_secs_f64();
        save_model(&value, &arg("--model-out", None));
        model = value;

    } else if command == "continue" {
        let options = Options {
            vocab_size: arg("--vocab", Some("4096")).parse().unwrap(),
            max_order: arg("--max-order", Some("4")).parse().unwrap(),
            train_limit: arg("--train-limit", Some("100000")).parse().unwrap(),
            gcp_max: arg("--gcp-max", Some("50000")).parse().unwrap(),
            gm1_max: arg("--gm1-max", Some("200000")).parse().unwrap(),
            glp_max: arg("--glp-max", Some("500000")).parse().unwrap(),
            max_successors: arg("--max-successors", Some("512")).parse().unwrap(),
            capacity_policy: CapacityPolicy::parse(&arg(
                "--capacity-policy",
                Some("fixed"),
            )),
            successor_index: arg("--successor-index", Some("on")) == "on",
            light_interval: arg("--light-interval", Some("25000")).parse().unwrap(),
            deep_interval: arg("--deep-interval", Some("100000")).parse().unwrap(),
            wake_interval: arg("--wake-interval", Some("50000")).parse().unwrap(),
            lr: arg("--lr", Some("0.025")).parse().unwrap(),
            sleep_policy: SleepPolicy::parse(&arg("--sleep-policy", Some("uniform"))),
        };
        policy = options.sleep_policy;
        train_order_seed = arg("--train-order-seed", Some("22201")).parse().unwrap();
        let mut value = load_model(&arg("--model", None));
        value.edge_application = policy.edge_application();
        value.gm1_mode = arg("--gm1-mode", Some("0")).parse().unwrap();
        value.medial_mode = arg("--medial-mode", Some("1")).parse().unwrap();
        value.shard_mode = arg("--shard-mode", Some("14")).parse().unwrap();
        value.layout_mode = exp176_layout_mode();
        initialize_exp171_after_load(&mut value);
        refresh_active_glp_all(&mut value);
        refresh_flat_layout(&mut value);
        TRAIN_STEP_OFFSET.store(model_last_step(&value), Ordering::Relaxed);
        let raw_tokens = read_tokens(&arg("--train", None), options.train_limit);
        let tokens = order_documents(
            &raw_tokens,
            train_order_seed,
            has("--preserve-order"),
        );
        train_stats = train(&mut value, &tokens, &options);
        train_elapsed_seconds = start.elapsed().as_secs_f64();
        save_model(&value, &arg("--model-out", None));
        model = value;
    } else if command == "eval" || command == "eval-online" {
        policy = SleepPolicy::parse(&arg("--sleep-policy", Some("cycle")));
        train_order_seed = arg("--train-order-seed", Some("8801")).parse().unwrap();
        model = load_model(&arg("--model", None));
        model.edge_application = policy.edge_application();
        model.gm1_mode = arg("--gm1-mode", Some("0")).parse().unwrap();
        model.medial_mode = arg("--medial-mode", Some("0")).parse().unwrap();
        model.shard_mode = arg("--shard-mode", Some("0")).parse().unwrap();
        model.layout_mode = exp176_layout_mode();
        initialize_exp171_after_load(&mut model);
        refresh_active_glp_all(&mut model);
        refresh_flat_layout(&mut model);
    } else {
        panic!("train or eval");
    }
    let train_layer_access = layer_access();
    let train_route_access = route_access();
    let train_profile = profile_snapshot();
    let train_pool_profile = pool_snapshot();
    let train_phase_profile = phase_snapshot();
    let train_layout_profile = layout_snapshot();
    reset_layer_access();
    let mode = if command == "train" || command == "continue" {
        Mode::FrequencyGradient
    } else {
        requested_mode()
    };
    let mut tokens = read_tokens(&eval_path, usize::MAX);
    let online_warm_tokens = if command == "eval-online" {
        let count: usize = arg("--online-warm-tokens", Some("25000"))
            .parse::<usize>().unwrap().min(tokens.len());
        let options = Options {
            vocab_size: model.vocab_size,
            max_order: model.max_order,
            train_limit: count,
            gcp_max: arg("--gcp-max", Some("50000")).parse().unwrap(),
            gm1_max: arg("--gm1-max", Some("200000")).parse().unwrap(),
            glp_max: arg("--glp-max", Some("500000")).parse().unwrap(),
            max_successors: arg("--max-successors", Some("512")).parse().unwrap(),
            capacity_policy: CapacityPolicy::Fixed,
            successor_index: true,
            light_interval: 25000,
            deep_interval: 100000,
            wake_interval: 50000,
            lr: 0.025,
            sleep_policy: policy,
        };
        online_warm_gcp(&mut model, &tokens[..count], &options);
        tokens.drain(..count);
        reset_layer_access();
        count
    } else {
        0
    };
    let score_tail: usize = arg("--score-tail", Some("0")).parse().unwrap();
    let eval = evaluate(&model, &tokens, mode, score_tail);
    let eval_layer_access = layer_access();
    let eval_route_access = route_access();
    let samples = arg("--generation-samples", Some("128")).parse().unwrap();
    let prompt = arg("--generation-prompt", Some("32")).parse().unwrap();
    let steps = arg("--generation-steps", Some("64")).parse().unwrap();
    let generation = generate(
        &model,
        &tokens,
        mode,
        samples,
        prompt,
        steps,
        arg("--sampling-seed", Some("69041")).parse().unwrap(),
    );
    let (gcp_bytes, gm1_bytes, glp_bytes) = logical_bytes(&model);
    let model_path = if command == "train" || command == "continue" {
        arg("--model-out", None)
    } else {
        arg("--model", None)
    };
    let model_bytes = std::fs::metadata(model_path).unwrap().len();
    let (length_edges, length_mean, length_sd, length_p10, length_p90, length_alignment) =
        geometry_stats(&model, mode);
    let overhead = if generation.baseline_peak > 0 {
        generation.gr_peak as f64 / generation.baseline_peak as f64 - 1.0
    } else {
        0.0
    };
    let continuations = format!(
        "[{}]",
        generation
            .continuations
            .iter()
            .map(|sequence| format!(
                "[{}]",
                sequence
                    .iter()
                    .map(|x| x.to_string())
                    .collect::<Vec<_>>()
                    .join(",")
            ))
            .collect::<Vec<_>>()
            .join(",")
    );
    let predictions = eval.predictions.max(1) as f64;
    let wave_predictions = eval.wave_predictions.max(1) as f64;
    let candidates = eval.candidate_count.max(1) as f64;
    let json=format!("{{\n  \"experiment\": \"EXP96\",\n  \"command\": \"{}\",\n  \"condition\": \"{}\",\n  \"geometry\": \"{}\",\n  \"train_tokens\": {},\n  \"train_predictions\": {},\n  \"train_nll_per_token\": {:.12},\n  \"eval_predictions\": {},\n  \"nll_per_token\": {:.12},\n  \"perplexity\": {:.12},\n  \"bits_per_token\": {:.12},\n  \"top1_accuracy\": {:.12},\n  \"top5_accuracy\": {:.12},\n  \"unigram_nll_per_token\": {:.12},\n  \"gcp_contexts\": {},\n  \"glp_contexts\": {},\n  \"gcp_logical_bytes\": {},\n  \"glp_logical_bytes\": {},\n  \"model_serialized_bytes\": {},\n  \"peak_rss_kib\": {},\n  \"elapsed_seconds\": {:.6},\n  \"created_edges\": {},\n  \"edge_updates\": {},\n  \"conflicts\": {},\n  \"light_reflections\": {},\n  \"deep_reflections\": {},\n  \"pruned_gcp\": {},\n  \"pruned_glp\": {},\n  \"max_gcp_contexts\": {},\n  \"gradient_audit_max_abs\": {:.12},\n  \"sleeps_attempted\": {},\n  \"sleeps_accepted\": {},\n  \"sleeps_rolled_back\": {},\n  \"sleep_passes\": {},\n  \"replay_updates\": {},\n  \"sleep_examples\": {},\n  \"sleep_gold_candidates_before_feedback\": {},\n  \"sleep_full_gold_candidates_before_feedback\": {},\n  \"sleep_transaction_edges\": {},\n  \"sleep_candidate_traces_before_feedback\": {},\n  \"counterfactual_interventions\": {},\n  \"cf_gold_edges_tested\": {},\n  \"cf_gold_edges_causal\": {},\n  \"cf_wrong_edges_tested\": {},\n  \"cf_wrong_edges_causal\": {},\n  \"bootstrap_updates\": {},\n  \"contrast_edges_tested\": {},\n  \"contrast_gold_favored\": {},\n  \"contrast_wrong_favored\": {},\n  \"contrast_neutral\": {},\n  \"contrast_gold_necessity_sum\": {:.12},\n  \"contrast_wrong_necessity_sum\": {:.12},\n  \"contrast_abs_difference_sum\": {:.12},\n  \"bridge_attempts\": {},\n  \"bridge_found\": {},\n  \"bridge_full_coverage\": {},\n  \"bridge_edges_credited\": {},\n  \"bridge_coverage_sum\": {:.12},\n  \"bridge_dispersion_sum\": {:.12},\n  \"bridge_cost_sum\": {:.12},\n  \"bridge_validations\": {},\n  \"bridge_validated\": {},\n  \"bridge_rejected_no_gold\": {},\n  \"bridge_rejected_margin\": {},\n  \"bridge_rejected_wrong_promotion\": {},\n  \"prospective_gold_bonus_sum\": {:.12},\n  \"prospective_margin_sum\": {:.12},\n  \"prospective_wrong_promotion_sum\": {:.12},\n  \"multidose_virtual_traces\": {},\n  \"multidose_dose1_pass\": {},\n  \"multidose_dose2_pass\": {},\n  \"multidose_dose4_pass\": {},\n  \"multidose_rejected\": {},\n  \"multidose_monotonic_rejected\": {},\n  \"pending_enqueued\": {},\n  \"pending_revalidated\": {},\n  \"pending_recredited\": {},\n  \"pending_resolved\": {},\n  \"pending_expired\": {},\n  \"pending_peak\": {},\n  \"wake_integrity_checks\": {},\n  \"wake_glp_mutation_violations\": {},\n  \"gcp_credit_entries_after_train\": {},\n  \"wave_predictions\": {},\n  \"wave_candidate_mean\": {:.12},\n  \"wave_full_convergence_fraction\": {:.12},\n  \"wave_maximum_coverage_mean\": {:.12},\n  \"wave_gold_candidate_fraction\": {:.12},\n  \"wave_dispersion_mean\": {:.12},\n  \"wave_path_cost_mean\": {:.12},\n  \"wave_signal_mean\": {:.12},\n  \"wave_conflict_mean\": {:.12},\n  \"candidate_gr_energy_mean\": {:.12},\n  \"length_edges\": {},\n  \"length_mean\": {:.12},\n  \"length_sd\": {:.12},\n  \"length_p10\": {:.12},\n  \"length_p90\": {:.12},\n  \"length_alignment\": {:.12},\n  \"generation_gr_baseline_peak_bytes_approx\": {},\n  \"generation_gr_peak_bytes_approx\": {},\n  \"generation_gr_memory_overhead_ratio\": {:.12},\n  \"greedy_continuations\": {}\n}}\n",
        command,policy.name(),mode.name(),train_stats.tokens,train_stats.predictions,if train_stats.predictions>0{train_stats.nll/train_stats.predictions as f64}else{0.0},eval.predictions,eval.nll/predictions,(eval.nll/predictions).exp(),eval.nll/predictions/std::f64::consts::LN_2,eval.top1 as f64/predictions,eval.top5 as f64/predictions,eval.unigram_nll/predictions,
        model.gcp.len(),model.glp.len(),gcp_bytes,glp_bytes,model_bytes,peak_rss_kib(),start.elapsed().as_secs_f64(),train_stats.created_edges,train_stats.edge_updates,train_stats.conflicts,train_stats.light_reflections,train_stats.deep_reflections,train_stats.pruned_gcp,train_stats.pruned_glp,train_stats.max_gcp,train_stats.gradient_audit_max_abs,
        train_stats.sleeps_attempted,train_stats.sleeps_accepted,train_stats.sleeps_rolled_back,train_stats.sleep_passes,train_stats.replay_updates,
        train_stats.sleep_examples,train_stats.sleep_gold_candidates_before_feedback,train_stats.sleep_full_gold_candidates_before_feedback,
        train_stats.sleep_transaction_edges,train_stats.sleep_candidate_traces_before_feedback,
        train_stats.counterfactual_interventions,train_stats.cf_gold_edges_tested,train_stats.cf_gold_edges_causal,
        train_stats.cf_wrong_edges_tested,train_stats.cf_wrong_edges_causal,
        train_stats.bootstrap_updates,train_stats.contrast_edges_tested,train_stats.contrast_gold_favored,
        train_stats.contrast_wrong_favored,train_stats.contrast_neutral,train_stats.contrast_gold_necessity_sum,
        train_stats.contrast_wrong_necessity_sum,train_stats.contrast_abs_difference_sum,
        train_stats.bridge_attempts,train_stats.bridge_found,train_stats.bridge_full_coverage,train_stats.bridge_edges_credited,
        train_stats.bridge_coverage_sum,train_stats.bridge_dispersion_sum,train_stats.bridge_cost_sum,
        train_stats.bridge_validations,train_stats.bridge_validated,train_stats.bridge_rejected_no_gold,
        train_stats.bridge_rejected_margin,train_stats.bridge_rejected_wrong_promotion,
        train_stats.prospective_gold_bonus_sum,train_stats.prospective_margin_sum,
        train_stats.prospective_wrong_promotion_sum,
        train_stats.multidose_virtual_traces,train_stats.multidose_dose1_pass,
        train_stats.multidose_dose2_pass,train_stats.multidose_dose4_pass,
        train_stats.multidose_rejected,train_stats.multidose_monotonic_rejected,
        train_stats.pending_enqueued,train_stats.pending_revalidated,train_stats.pending_recredited,
        train_stats.pending_resolved,train_stats.pending_expired,train_stats.pending_peak,
        train_stats.wake_integrity_checks,train_stats.wake_glp_mutation_violations,model.gcp_credit.len(),
        eval.wave_predictions,eval.wave_candidate_sum as f64/wave_predictions,eval.wave_full as f64/wave_predictions,eval.wave_coverage_sum/wave_predictions,eval.gold_candidate as f64/wave_predictions,eval.dispersion_sum/candidates,eval.cost_sum/candidates,eval.signal_sum/candidates,eval.conflict_sum/candidates,eval.energy_sum/candidates,length_edges,length_mean,length_sd,length_p10,length_p90,length_alignment,generation.baseline_peak,generation.gr_peak,overhead,continuations);
    let json = json.replacen("\"experiment\": \"EXP96\"", "\"experiment\": \"EXP150\"", 1);
    let phase_fields = format!(
        "  \"phase_attempts\": {},\n  \"phase_full_coverage_attempts\": {},\n  \"phase_missing_coverage_attempts\": {},\n  \"phase_vector_edges\": {},\n  \"phase_accepted\": {},\n  \"phase_full_created\": {},\n  \"phase_rejected_gold_lost\": {},\n  \"phase_rejected_coverage\": {},\n  \"phase_rejected_no_improvement\": {},\n  \"phase_rejected_bonus\": {},\n  \"phase_rejected_wrong_promotion\": {},\n  \"phase_dispersion_before_sum\": {:.12},\n  \"phase_dispersion_after_sum\": {:.12},\n  \"phase_abs_delta_sum\": {:.12},\n",
        train_stats.phase_attempts,
        train_stats.phase_full_coverage_attempts,
        train_stats.phase_missing_coverage_attempts,
        train_stats.phase_vector_edges,
        train_stats.phase_accepted,
        train_stats.phase_full_created,
        train_stats.phase_rejected_gold_lost,
        train_stats.phase_rejected_coverage,
        train_stats.phase_rejected_no_improvement,
        train_stats.phase_rejected_bonus,
        train_stats.phase_rejected_wrong_promotion,
        train_stats.phase_dispersion_before_sum,
        train_stats.phase_dispersion_after_sum,
        train_stats.phase_abs_delta_sum,
    );
    let json = json.replacen(
        "  \"wake_integrity_checks\"",
        &(phase_fields + "  \"wake_integrity_checks\""),
        1,
    );
    let origin_fields = format!(
        "  \"origin_attempts\": {},\n  \"origin_paths_found\": {},\n  \"origin_validated\": {},\n  \"origin_full_created\": {},\n  \"origin_dose1_pass\": {},\n  \"origin_dose2_pass\": {},\n  \"origin_dose4_pass\": {},\n  \"origin_rejected_no_gain\": {},\n  \"origin_edges_credited\": {},\n  \"sequential_semantic_edges\": {},\n  \"sequential_origin_edges\": {},\n  \"sequential_phase_edges\": {},\n",
        train_stats.origin_attempts,
        train_stats.origin_paths_found,
        train_stats.origin_validated,
        train_stats.origin_full_created,
        train_stats.origin_dose1_pass,
        train_stats.origin_dose2_pass,
        train_stats.origin_dose4_pass,
        train_stats.origin_rejected_no_gain,
        train_stats.origin_edges_credited,
        train_stats.sequential_semantic_edges,
        train_stats.sequential_origin_edges,
        train_stats.sequential_phase_edges,
    );
    let json = json.replacen(
        "  \"wake_integrity_checks\"",
        &(origin_fields + "  \"wake_integrity_checks\""),
        1,
    );
    let bank_fields = format!(
        "  \"origin_bank_enqueued\": {},\n  \"origin_bank_revalidated\": {},\n  \"origin_bank_recredited\": {},\n  \"origin_bank_incorporated\": {},\n  \"origin_bank_expired\": {},\n  \"origin_bank_peak\": {},\n",
        train_stats.origin_bank_enqueued,
        train_stats.origin_bank_revalidated,
        train_stats.origin_bank_recredited,
        train_stats.origin_bank_incorporated,
        train_stats.origin_bank_expired,
        train_stats.origin_bank_peak,
    );
    let json = json.replacen(
        "  \"wake_integrity_checks\"",
        &(bank_fields + "  \"wake_integrity_checks\""),
        1,
    );
    let route_set_fields = format!(
        "  \"route_set_groups\": {},\n  \"route_set_candidates\": {},\n  \"route_set_states_evaluated\": {},\n  \"route_set_selected_groups\": {},\n  \"route_set_selected_routes\": {},\n  \"route_set_selected_edges\": {},\n  \"route_set_full_created\": {},\n  \"route_set_revalidated_groups\": {},\n  \"route_set_rejected_no_joint_gain\": {},\n  \"route_set_negative_interactions\": {},\n  \"route_set_nonnegative_interactions\": {},\n  \"route_set_joint_gain_sum\": {:.12},\n  \"route_set_individual_gain_sum\": {:.12},\n  \"route_set_interference_sum\": {:.12},\n  \"route_set_selected_bonus_sum\": {:.12},\n  \"route_set_selected_margin_sum\": {:.12},\n  \"route_set_wrong_promotion_sum\": {:.12},\n",
        train_stats.route_set_groups,
        train_stats.route_set_candidates,
        train_stats.route_set_states_evaluated,
        train_stats.route_set_selected_groups,
        train_stats.route_set_selected_routes,
        train_stats.route_set_selected_edges,
        train_stats.route_set_full_created,
        train_stats.route_set_revalidated_groups,
        train_stats.route_set_rejected_no_joint_gain,
        train_stats.route_set_negative_interactions,
        train_stats.route_set_nonnegative_interactions,
        train_stats.route_set_joint_gain_sum,
        train_stats.route_set_individual_gain_sum,
        train_stats.route_set_interference_sum,
        train_stats.route_set_selected_bonus_sum,
        train_stats.route_set_selected_margin_sum,
        train_stats.route_set_wrong_promotion_sum,
    );
    let json = json.replacen(
        "  \"wake_integrity_checks\"",
        &(route_set_fields + "  \"wake_integrity_checks\""),
        1,
    );
    let deferred_fields = format!(
        "  \"deferred_proposed\": {},\n  \"deferred_evidence_windows\": {},\n  \"deferred_future_examples_screened\": {},\n  \"deferred_future_examples_relevant\": {},\n  \"deferred_overlay_evaluations\": {},\n  \"deferred_supported_windows\": {},\n  \"deferred_adverse_windows\": {},\n  \"deferred_neutral_windows\": {},\n  \"deferred_matured\": {},\n  \"deferred_committed\": {},\n  \"deferred_expired\": {},\n  \"deferred_route_commits\": {},\n  \"deferred_edge_commits\": {},\n  \"deferred_quarantine_peak\": {},\n  \"deferred_gold_delta_sum\": {:.12},\n  \"deferred_full_delta_sum\": {:.12},\n  \"deferred_coverage_delta_sum\": {:.12},\n  \"deferred_margin_delta_sum\": {:.12},\n  \"deferred_wrong_promotion_sum\": {:.12},\n  \"deferred_value_sum\": {:.12},\n  \"deferred_journal_edges\": {},\n  \"metagraph_pairs_evaluated\": {},\n  \"metagraph_synergy\": {},\n  \"metagraph_conflict\": {},\n  \"metagraph_neutral\": {},\n  \"metagraph_interaction_sum\": {:.12},\n",
        train_stats.deferred_proposed,
        train_stats.deferred_evidence_windows,
        train_stats.deferred_future_examples_screened,
        train_stats.deferred_future_examples_relevant,
        train_stats.deferred_overlay_evaluations,
        train_stats.deferred_supported_windows,
        train_stats.deferred_adverse_windows,
        train_stats.deferred_neutral_windows,
        train_stats.deferred_matured,
        train_stats.deferred_committed,
        train_stats.deferred_expired,
        train_stats.deferred_route_commits,
        train_stats.deferred_edge_commits,
        train_stats.deferred_quarantine_peak,
        train_stats.deferred_gold_delta_sum,
        train_stats.deferred_full_delta_sum,
        train_stats.deferred_coverage_delta_sum,
        train_stats.deferred_margin_delta_sum,
        train_stats.deferred_wrong_promotion_sum,
        train_stats.deferred_value_sum,
        train_stats.deferred_journal_edges,
        train_stats.metagraph_pairs_evaluated,
        train_stats.metagraph_synergy,
        train_stats.metagraph_conflict,
        train_stats.metagraph_neutral,
        train_stats.metagraph_interaction_sum,
    );
    let json = json.replacen(
        "  \"wake_integrity_checks\"",
        &(deferred_fields + "  \"wake_integrity_checks\""),
        1,
    );
    let directed = directed_construction_audit(&model);
    let edge_credit = directional_edge_credit_audit(&model);
    let construction_fields = format!(
        "  \"construction_candidates\": {},\n  \"construction_proposed\": {},\n  \"construction_fixed_proposed\": {},\n  \"construction_slot_proposed\": {},\n  \"construction_slot_fillings\": {},\n  \"construction_future_matches\": {},\n  \"construction_evidence_windows\": {},\n  \"construction_supported_windows\": {},\n  \"construction_adverse_windows\": {},\n  \"construction_neutral_windows\": {},\n  \"construction_matured\": {},\n  \"construction_committed\": {},\n  \"construction_immediate_committed\": {},\n  \"construction_expired\": {},\n  \"construction_quarantine_peak\": {},\n  \"construction_glp_peak\": {},\n  \"construction_gold_delta_sum\": {:.12},\n  \"construction_full_delta_sum\": {:.12},\n  \"construction_coverage_delta_sum\": {:.12},\n  \"construction_margin_delta_sum\": {:.12},\n  \"construction_wrong_promotion_sum\": {:.12},\n  \"construction_value_sum\": {:.12},\n  \"model_glp_constructions\": {},\n  \"directed_role_edges\": {},\n  \"directed_prediction_edges\": {},\n  \"directed_invalid_edges\": {},\n  \"directed_self_loops\": {},\n  \"directed_dag_constructions\": {},\n  \"directed_flat_constructions\": {},\n  \"directed_reverse_traversals\": {},\n  \"directed_automatic_reverse_edges\": 0,\n  \"directional_edge_tests\": {},\n  \"directional_role_edge_tests\": {},\n  \"directional_prediction_edge_tests\": {},\n  \"directional_edge_positive\": {},\n  \"directional_edge_negative\": {},\n  \"directional_edge_neutral\": {},\n  \"directional_edge_value_sum\": {:.12},\n  \"model_directional_credited_edges\": {},\n  \"model_directional_nonunit_edges\": {},\n  \"model_directional_distinct_weights\": {},\n  \"model_directional_weight_min\": {:.12},\n  \"model_directional_weight_max\": {:.12},\n  \"model_directional_weight_mean\": {:.12},\n  \"model_directional_weight_sd\": {:.12},\n  \"model_directional_gradient_max_abs\": {:.12},\n  \"wave_gold_full_convergence_fraction\": {:.12},\n",
        train_stats.construction_candidates,
        train_stats.construction_proposed,
        train_stats.construction_fixed_proposed,
        train_stats.construction_slot_proposed,
        train_stats.construction_slot_fillings,
        train_stats.construction_future_matches,
        train_stats.construction_evidence_windows,
        train_stats.construction_supported_windows,
        train_stats.construction_adverse_windows,
        train_stats.construction_neutral_windows,
        train_stats.construction_matured,
        train_stats.construction_committed,
        train_stats.construction_immediate_committed,
        train_stats.construction_expired,
        train_stats.construction_quarantine_peak,
        train_stats.construction_glp_peak,
        train_stats.construction_gold_delta_sum,
        train_stats.construction_full_delta_sum,
        train_stats.construction_coverage_delta_sum,
        train_stats.construction_margin_delta_sum,
        train_stats.construction_wrong_promotion_sum,
        train_stats.construction_value_sum,
        model.glp_constructions.len(),
        directed.0,
        directed.1,
        directed.2,
        directed.3,
        directed.4,
        directed.5,
        train_stats.bridge_attempts,
        train_stats.directional_edge_tests,
        train_stats.directional_role_edge_tests,
        train_stats.directional_prediction_edge_tests,
        train_stats.directional_edge_positive,
        train_stats.directional_edge_negative,
        train_stats.directional_edge_neutral,
        train_stats.directional_edge_value_sum,
        edge_credit.0,
        edge_credit.1,
        edge_credit.2,
        edge_credit.3,
        edge_credit.4,
        edge_credit.5,
        edge_credit.6,
        edge_credit.7,
        eval.gold_full_candidate as f64 / wave_predictions,
    );
    let json = json.replacen(
        "  \"wake_integrity_checks\"",
        &(construction_fields + "  \"wake_integrity_checks\""),
        1,
    );
    let paired = eval.paired_predictions.max(1) as f64;
    let early_fields = format!(
        "  \"edge_application_stage\": {},\n  \"early_active_constructions\": {},\n  \"early_modified_transitions\": {},\n  \"early_route_reorders\": {},\n  \"early_gr_adjustments\": {},\n  \"paired_predictions\": {},\n  \"paired_top1_changed\": {},\n  \"paired_top1_changed_fraction\": {:.12},\n  \"paired_gold_rank_improved\": {},\n  \"paired_gold_rank_worsened\": {},\n  \"paired_gold_rank_equal\": {},\n  \"paired_gold_margin_delta_mean\": {:.12},\n  \"paired_gold_probability_delta_mean\": {:.12},\n  \"paired_gr_set_changed\": {},\n  \"paired_gr_set_changed_fraction\": {:.12},\n",
        model.edge_application,
        eval.early_active_constructions,
        eval.early_modified_transitions,
        eval.early_route_reorders,
        eval.early_gr_adjustments,
        eval.paired_predictions,
        eval.paired_top1_changed,
        eval.paired_top1_changed as f64 / paired,
        eval.paired_gold_rank_improved,
        eval.paired_gold_rank_worsened,
        eval.paired_gold_rank_equal,
        eval.paired_gold_margin_delta_sum / paired,
        eval.paired_gold_probability_delta_sum / paired,
        eval.paired_gr_set_changed,
        eval.paired_gr_set_changed as f64 / paired,
    );
    let json = json.replacen(
        "  \"wake_integrity_checks\"",
        &(early_fields + "  \"wake_integrity_checks\""),
        1,
    );
    let seed_field = format!(
        "  \"train_order_seed\": {},\n  \"preserve_document_order\": {},\n  \"score_tail_tokens\": {},\n",
        train_order_seed,
        has("--preserve-order"),
        score_tail
    );
    let json = json.replacen(
        "  \"wake_integrity_checks\"",
        &(seed_field + "  \"wake_integrity_checks\""),
        1,
    );
    let graph_nodes: Vec<&Node> = model
        .gcp.values().chain(model.gm1.values()).chain(model.glp.values()).collect();
    let successor_edges: usize = graph_nodes.iter().map(|node| node.edges.len()).sum();
    let successor_max = graph_nodes
        .iter()
        .map(|node| node.edges.len())
        .max()
        .unwrap_or(0);
    let successor_bins = [
        graph_nodes.iter().filter(|node| node.edges.len() <= 32).count(),
        graph_nodes
            .iter()
            .filter(|node| (33..=64).contains(&node.edges.len()))
            .count(),
        graph_nodes
            .iter()
            .filter(|node| (65..=128).contains(&node.edges.len()))
            .count(),
        graph_nodes
            .iter()
            .filter(|node| (129..=256).contains(&node.edges.len()))
            .count(),
        graph_nodes.iter().filter(|node| node.edges.len() > 256).count(),
    ];
    let capacity_fields = format!(
        "  \"configured_max_successors\": {},\n  \"capacity_policy\": \"{}\",\n  \"successor_index_enabled\": {},\n  \"indexed_contexts_after_train\": {},\n  \"successor_edges_after_train\": {},\n  \"successors_per_context_mean\": {:.12},\n  \"successors_per_context_max\": {},\n  \"successor_context_bins\": [{},{},{},{},{}],\n  \"successor_capacity_rejections\": {},\n  \"successor_capacity_rejections_order_1\": {},\n  \"successor_capacity_rejections_order_2\": {},\n  \"successor_capacity_rejections_order_3\": {},\n  \"successor_capacity_rejections_order_4\": {},\n  \"sleep_successor_capacity_rejections\": {},\n  \"context_queries_by_order\": [{},{},{},{}],\n  \"context_hits_by_order\": [{},{},{},{}],\n  \"frequency_bin_predictions\": [{},{},{}],\n  \"frequency_bin_top1\": [{},{},{}],\n  \"frequency_bin_nll\": [{:.12},{:.12},{:.12}],\n  \"correct_gr_margin_count\": {},\n  \"correct_gr_margin_mean\": {:.12},\n",
        if command == "train" {
            arg("--max-successors", Some("32")).parse::<usize>().unwrap()
        } else {
            0
        },
        if command == "train" {
            arg("--capacity-policy", Some("fixed"))
        } else {
            "checkpoint".to_string()
        },
        if command == "train" {
            arg("--successor-index", Some("on")) == "on"
        } else {
            false
        },
        model
            .gcp
            .values()
            .chain(model.gm1.values())
            .chain(model.glp.values())
            .filter(|node| node.edge_index.is_some())
            .count(),
        successor_edges,
        successor_edges as f64 / graph_nodes.len().max(1) as f64,
        successor_max,
        successor_bins[0],
        successor_bins[1],
        successor_bins[2],
        successor_bins[3],
        successor_bins[4],
        train_stats.successor_capacity_rejections,
        train_stats.successor_capacity_rejections_by_order[0],
        train_stats.successor_capacity_rejections_by_order[1],
        train_stats.successor_capacity_rejections_by_order[2],
        train_stats.successor_capacity_rejections_by_order[3],
        train_stats.sleep_successor_capacity_rejections,
        eval.context_queries_by_order[0],
        eval.context_queries_by_order[1],
        eval.context_queries_by_order[2],
        eval.context_queries_by_order[3],
        eval.context_hits_by_order[0],
        eval.context_hits_by_order[1],
        eval.context_hits_by_order[2],
        eval.context_hits_by_order[3],
        eval.frequency_bin_predictions[0],
        eval.frequency_bin_predictions[1],
        eval.frequency_bin_predictions[2],
        eval.frequency_bin_top1[0],
        eval.frequency_bin_top1[1],
        eval.frequency_bin_top1[2],
        eval.frequency_bin_nll[0],
        eval.frequency_bin_nll[1],
        eval.frequency_bin_nll[2],
        eval.correct_gr_margin_count,
        if eval.correct_gr_margin_count > 0 {
            eval.correct_gr_margin_sum / eval.correct_gr_margin_count as f64
        } else {
            0.0
        },
    );
    let json = json.replacen(
        "  \"wake_integrity_checks\"",
        &(capacity_fields + "  \"wake_integrity_checks\""),
        1,
    );
    let gm1_fields = format!(
        "  \"gm1_mode\": {},\n  \"gm1_contexts\": {},\n  \"gm1_cache_contexts\": {},\n  \"gm1_logical_bytes\": {},\n  \"gm1_support_profile\": [{},{},{},{},{},{}],\n  \"train_elapsed_seconds\": {:.9},\n  \"legacy_glp_fingerprint\": {},\n  \"pruned_gm1\": {},\n  \"gcp_to_gm1_contexts\": {},\n  \"gm1_to_glp_contexts\": {},\n  \"gm1_peak_contexts\": {},\n  \"gm1_cache_refreshes\": {},\n  \"gm1_cache_contexts_accumulated\": {},\n  \"gm1_cache_peak_contexts\": {},\n  \"gm1_dream_candidates\": {},\n  \"gm1_dream_reinforced\": {},\n  \"gm1_dream_inhibited\": {},\n  \"train_gcp_reads\": {},\n  \"train_gm1_reads\": {},\n  \"train_glp_reads\": {},\n  \"train_gcp_stops\": {},\n  \"train_gm1_stops\": {},\n  \"eval_gcp_reads\": {},\n  \"eval_gm1_reads\": {},\n  \"eval_glp_reads\": {},\n  \"eval_gcp_stops\": {},\n  \"eval_gm1_stops\": {},\n",
        model.gm1_mode, model.gm1.len(), model.gm1_cache.len(), gm1_bytes,
        gm1_support_profile(&model)[0], gm1_support_profile(&model)[1],
        gm1_support_profile(&model)[2], gm1_support_profile(&model)[3],
        gm1_support_profile(&model)[4], gm1_support_profile(&model)[5],
        train_elapsed_seconds,
        glp_fingerprint(&model), train_stats.pruned_gm1,
        train_stats.gcp_to_gm1_contexts, train_stats.gm1_to_glp_contexts,
        train_stats.gm1_peak, train_stats.gm1_cache_refreshes,
        train_stats.gm1_cache_contexts, train_stats.gm1_cache_peak,
        train_stats.gm1_dream_candidates,
        train_stats.gm1_dream_reinforced, train_stats.gm1_dream_inhibited,
        train_layer_access[0], train_layer_access[1], train_layer_access[2],
        train_layer_access[3], train_layer_access[4],
        eval_layer_access[0], eval_layer_access[1], eval_layer_access[2],
        eval_layer_access[3], eval_layer_access[4],
    );
    let json = json
        .replacen("\"experiment\": \"EXP150\"", "\"experiment\": \"EXP168\"", 1)
        .replacen(
            "  \"wake_integrity_checks\"",
            &(gm1_fields + "  \"wake_integrity_checks\""),
            1,
        );
    let medial_fields = format!(
        "  \"medial_mode\": {},\n  \"online_warm_tokens\": {},\n  \"online_gcp_contexts\": {},\n  \"medial_pending_after_train\": {},\n  \"medial_dream_proposed\": {},\n  \"medial_dream_immediate\": {},\n  \"medial_dream_future_windows\": {},\n  \"medial_dream_supported_windows\": {},\n  \"medial_dream_adverse_windows\": {},\n  \"medial_dream_materialized\": {},\n  \"medial_dream_created_edges\": {},\n  \"medial_dream_reinforced_edges\": {},\n  \"medial_dream_expired\": {},\n  \"medial_dream_pending_peak\": {},\n  \"medial_dream_credit_sum\": {:.12},\n",
        model.medial_mode, online_warm_tokens, model.gcp.len(),
        model.medial_pending.len(), train_stats.medial_dream_proposed,
        train_stats.medial_dream_immediate, train_stats.medial_dream_future_windows,
        train_stats.medial_dream_supported_windows,
        train_stats.medial_dream_adverse_windows,
        train_stats.medial_dream_materialized,
        train_stats.medial_dream_created_edges,
        train_stats.medial_dream_reinforced_edges,
        train_stats.medial_dream_expired,
        train_stats.medial_dream_pending_peak,
        train_stats.medial_dream_credit_sum,
    );
    let json = json
        .replacen("\"experiment\": \"EXP168\"", "\"experiment\": \"EXP169\"", 1)
        .replacen(
            "  \"wake_integrity_checks\"",
            &(medial_fields + "  \"wake_integrity_checks\""),
            1,
        );
    let sizes = shard_sizes(&model);
    let active_mediators = model.gm1_route_mediators
        .values().filter(|row| row.confirmations >= 2).count();
    let route_fields = format!(
        "  \"shard_mode\": {},\n  \"shard_count\": {},\n  \"shard_sizes\": [{},{},{},{},{},{},{},{}],\n  \"gcp_direct_routes\": {},\n  \"gcp_route_mediators\": {},\n  \"gm1_direct_routes\": {},\n  \"gm1_route_mediators\": {},\n  \"gm1_active_route_mediators\": {},\n  \"shard_rebuilds\": {},\n  \"shard_contexts_materialized\": {},\n  \"route_mediators_proposed\": {},\n  \"route_mediators_confirmed\": {},\n  \"route_mediators_reset\": {},\n  \"route_direct_entries\": {},\n  \"train_shard_queries\": {},\n  \"train_shard_probes\": {},\n  \"train_route_predictions\": {},\n  \"train_route_correct\": {},\n  \"train_route_wrong\": {},\n  \"train_route_fallbacks\": {},\n  \"train_lost_contexts\": {},\n  \"eval_shard_queries\": {},\n  \"eval_shard_probes\": {},\n  \"eval_route_predictions\": {},\n  \"eval_route_correct\": {},\n  \"eval_route_wrong\": {},\n  \"eval_route_fallbacks\": {},\n  \"eval_lost_contexts\": {},\n  \"eval_found_contexts\": {},\n",
        model.shard_mode, model.glp_shards.len(),
        sizes[0], sizes[1], sizes[2], sizes[3],
        sizes[4], sizes[5], sizes[6], sizes[7],
        model.gcp_routes.len(), model.gcp_route_mediators.len(), model.gm1_routes.len(),
        model.gm1_route_mediators.len(), active_mediators,
        train_stats.shard_rebuilds, train_stats.shard_contexts_materialized,
        train_stats.route_mediators_proposed,
        train_stats.route_mediators_confirmed,
        train_stats.route_mediators_reset,
        train_stats.route_direct_entries,
        train_route_access[0], train_route_access[1], train_route_access[2],
        train_route_access[3], train_route_access[4], train_route_access[5],
        train_route_access[6],
        eval_route_access[0], eval_route_access[1], eval_route_access[2],
        eval_route_access[3], eval_route_access[4], eval_route_access[5],
        eval_route_access[6], eval_route_access[7],
    );
    let json = json
        .replacen("\"experiment\": \"EXP169\"", "\"experiment\": \"EXP170\"", 1)
        .replacen(
            "  \"wake_integrity_checks\"",
            &(route_fields + "  \"wake_integrity_checks\""),
            1,
        );
    let exp171_fields = format!(
        "  \"incremental_initial_splits\": {},\n  \"incremental_refreshes\": {},\n  \"incremental_journal_keys\": {},\n  \"incremental_contexts_examined\": {},\n  \"incremental_migrations\": {},\n  \"incremental_removals\": {},\n  \"incremental_full_scanned_contexts\": {},\n  \"compact_route_rebuilds\": {},\n  \"shard_assignments\": {},\n  \"route_support_entries\": {},\n  \"route_order_entries\": {},\n  \"compact_routes\": {},\n",
        train_stats.incremental_initial_splits,
        train_stats.incremental_refreshes,
        train_stats.incremental_journal_keys,
        train_stats.incremental_contexts_examined,
        train_stats.incremental_migrations,
        train_stats.incremental_removals,
        train_stats.incremental_full_scanned_contexts,
        train_stats.compact_route_rebuilds,
        model.shard_assignments.len(),
        model.route_support.len(),
        model.route_order.len(),
        model.compact_routes.len(),
    );
    let json = json
        .replacen("\"experiment\": \"EXP170\"", "\"experiment\": \"EXP171\"", 1)
        .replacen(
            "  \"wake_integrity_checks\"",
            &(exp171_fields + "  \"wake_integrity_checks\""),
            1,
        );
    let exp172_fields = format!(
        "  \"eviction_index_builds\": {},\n  \"eviction_rank_updates\": {},\n  \"indexed_prune_calls\": {},\n  \"indexed_evictions\": {},\n  \"global_prune_calls\": {},\n  \"global_prune_contexts_scanned\": {},\n  \"victim_events\": {},\n  \"victim_fingerprint\": {},\n  \"view_memberships_written\": {},\n  \"view_shard_count\": {},\n  \"eviction_rank_entries\": {},\n  \"eviction_order_entries\": {},\n",
        train_stats.eviction_index_builds,
        train_stats.eviction_rank_updates,
        train_stats.indexed_prune_calls,
        train_stats.indexed_evictions,
        train_stats.global_prune_calls,
        train_stats.global_prune_contexts_scanned,
        train_stats.victim_events,
        train_stats.victim_fingerprint,
        train_stats.view_memberships_written,
        model.glp_shard_views.len(),
        model.eviction_rank.len(),
        model.eviction_order.len(),
    );
    let json = json
        .replacen("\"experiment\": \"EXP171\"", "\"experiment\": \"EXP172\"", 1)
        .replacen(
            "  \"wake_integrity_checks\"",
            &(exp172_fields + "  \"wake_integrity_checks\""),
            1,
        );
    let exp173_fields = format!(
        "  \"profile_enabled\": {},\n  \"profile_distribution_ns\": {},\n  \"profile_distribution_calls\": {},\n  \"profile_routed_get_ns\": {},\n  \"profile_routed_get_calls\": {},\n  \"profile_merge_ns\": {},\n  \"profile_merge_calls\": {},\n  \"profile_prune_ns\": {},\n  \"profile_prune_calls\": {},\n  \"profile_route_refresh_ns\": {},\n  \"profile_route_refresh_calls\": {},\n  \"profile_sleep_ns\": {},\n  \"profile_sleep_calls\": {},\n  \"grouped_spec_windows\": {},\n  \"grouped_spec_tasks\": {},\n  \"full_directory\": {},\n  \"owner_storage\": {},\n",
        profile_enabled(),
        train_profile[0],
        train_profile[1],
        train_profile[2],
        train_profile[3],
        train_profile[4],
        train_profile[5],
        train_profile[6],
        train_profile[7],
        train_profile[8],
        train_profile[9],
        train_profile[10],
        train_profile[11],
        train_profile[12],
        train_profile[13],
        exp173_full_directory(model.shard_mode),
        exp173_owner_storage(model.shard_mode),
    );
    let json = json
        .replacen("\"experiment\": \"EXP172\"", "\"experiment\": \"EXP173\"", 1)
        .replacen(
            "  \"wake_integrity_checks\"",
            &(exp173_fields + "  \"wake_integrity_checks\""),
            1,
        );
    let exp174_fields = format!(
        "  \"pool_mode\": {},\n  \"pool_workers\": {},\n  \"pool_windows\": {},\n  \"pool_tasks\": {},\n  \"pool_positions\": {},\n  \"pool_prep_ns\": {},\n  \"pool_prep_calls\": {},\n  \"pool_dispatch_ns\": {},\n  \"pool_dispatch_calls\": {},\n  \"pool_wait_ns\": {},\n  \"pool_wait_calls\": {},\n  \"pool_reduce_ns\": {},\n  \"pool_reduce_calls\": {},\n  \"pool_work_ns\": {},\n  \"pool_work_calls\": {},\n",
        exp174_pool_mode(),
        exp154_workers(),
        train_pool_profile[0],
        train_pool_profile[1],
        train_pool_profile[2],
        train_pool_profile[3],
        train_pool_profile[4],
        train_pool_profile[5],
        train_pool_profile[6],
        train_pool_profile[7],
        train_pool_profile[8],
        train_pool_profile[9],
        train_pool_profile[10],
        train_pool_profile[11],
        train_pool_profile[12],
    );
    let json = json
        .replacen("\"experiment\": \"EXP173\"", "\"experiment\": \"EXP174\"", 1)
        .replacen(
            "  \"wake_integrity_checks\"",
            &(exp174_fields + "  \"wake_integrity_checks\""),
            1,
        );
    let exp175_fields = format!(
        "  \"phase_mode\": {},\n  \"phase_sleep_workers\": {},\n  \"phase_windows\": {},\n  \"phase_waves\": {},\n  \"phase_max_waves\": {},\n  \"phase_positions\": {},\n  \"phase_plan_ns\": {},\n  \"phase_plan_calls\": {},\n  \"phase_commit_ns\": {},\n  \"phase_commit_calls\": {},\n  \"phase_distribution_ns\": {},\n  \"phase_distribution_calls\": {},\n",
        exp175_phase_mode(),
        parallel_train_workers(),
        train_phase_profile[0],
        train_phase_profile[1],
        train_phase_profile[2],
        train_phase_profile[3],
        train_phase_profile[4],
        train_phase_profile[5],
        train_phase_profile[6],
        train_phase_profile[7],
        train_phase_profile[8],
        train_phase_profile[9],
    );
    let json = json
        .replacen("\"experiment\": \"EXP174\"", "\"experiment\": \"EXP175\"", 1)
        .replacen(
            "  \"wake_integrity_checks\"",
            &(exp175_fields + "  \"wake_integrity_checks\""),
            1,
        );
    let layout_fields = format!(
        "  \"layout_mode\": {},\n  \"layout_context_key_bytes\": {},\n  \"layout_flat_nodes\": {},\n  \"layout_flat_edges\": {},\n  \"layout_flat_bytes\": {},\n  \"layout_lookup_ns\": {},\n  \"layout_lookup_calls\": {},\n  \"layout_lookup_hits\": {},\n  \"layout_rebuild_ns\": {},\n  \"layout_rebuild_calls\": {},\n",
        model.layout_mode,
        std::mem::size_of::<ContextKey>(),
        model.flat_glp.keys.len(),
        model.flat_glp.edges.len(),
        model.flat_glp.bytes(),
        train_layout_profile[0],
        train_layout_profile[1],
        train_layout_profile[2],
        train_layout_profile[3],
        train_layout_profile[4],
    );
    let json = json
        .replacen("\"experiment\": \"EXP175\"", "\"experiment\": \"EXP176\"", 1)
        .replacen(
            "  \"wake_integrity_checks\"",
            &(layout_fields + "  \"wake_integrity_checks\""),
            1,
        );
    std::fs::write(&output, &json).unwrap();
    print!("{}", json);
}

#[cfg(test)]
mod tests {
    use super::*;
    pub(super) fn tiny() -> Model {
        Model {
            vocab_size: 16,
            max_order: 4,
            gcp_tokens: vec![TokenVertex::default(); 16],
            gm1_tokens: vec![TokenVertex::default(); 16],
            glp_tokens: vec![TokenVertex::default(); 16],
            gcp: ContextMap::default(),
            gm1: ContextMap::default(),
            gm1_cache: ContextMap::default(),
            glp: ContextMap::default(),
            layout_mode: exp176_layout_mode(),
            flat_glp: FlatGraph::default(),
        active_successor_limit: active_successor_limit(),
        active_glp: ContextMap::default(),
            gcp_credit: ContextMap::default(),
            glp_constructions: Vec::new(),
            edge_application: 0,
        glp_version: 0,
        gm1_mode: 0,
        medial_mode: 0,
        medial_pending: Vec::new(),
        shard_mode: 0,
        glp_shards: Vec::new(),
        token_themes: Vec::new(),
        gcp_routes: ContextMap::default(),
        gcp_route_mediators: RouteMap::default(),
        gm1_routes: ContextMap::default(),
        gm1_route_mediators: RouteMap::default(),
        shard_assignments: ContextMap::default(),
        route_support: ContextMap::default(),
        route_order: BTreeSet::new(),
        compact_routes: Vec::new(),
        glp_shard_views: Vec::new(),
        eviction_rank: ContextMap::default(),
        eviction_order: BTreeSet::new(),
        }
    }
    fn edge(count: u32, gradient: f32, square: f32, conflicts: u32) -> EdgeAggregate {
        EdgeAggregate {
            token: 2,
            count,
            weight: 0.0,
            gradient_sum: gradient as f64,
            gradient_sq: square as f64,
            conflicts,
            last_seen: 0,
            sleep_bias: 0.0,
        }
    }
    fn chain(model: &mut Model, a: u16, b: u16) {
        let key = ContextKey::new(1, [a, 0, 0, 0]);
        let node = model.glp.entry(key).or_insert_with(|| Node::new(1));
        for i in 0..8 {
            node.observe(
                b,
                i,
                DEFAULT_MAX_SUCCESSORS,
                CapacityPolicy::Fixed,
                true,
            );
        }
        model.glp_tokens[b as usize].count += 8;
    }
    #[test]
    fn successor_index_preserves_edge_order_and_updates() {
        let mut indexed = Node::new(1);
        let mut linear = Node::new(1);
        for step in 0..80 {
            let token = ((step * 37) % 41) as u16;
            let a = indexed.observe(token, step, 64, CapacityPolicy::Fixed, true);
            let b = linear.observe(token, step, 64, CapacityPolicy::Fixed, false);
            assert_eq!(a, b);
        }
        assert!(indexed.edge_index.is_some());
        assert!(linear.edge_index.is_none());
        assert_eq!(
            indexed.edges.iter().map(|edge| edge.token).collect::<Vec<_>>(),
            linear.edges.iter().map(|edge| edge.token).collect::<Vec<_>>()
        );
        for (a, b) in indexed.edges.iter().zip(&linear.edges) {
            assert_eq!(a.token, b.token);
            assert_eq!(a.count, b.count);
            assert_eq!(a.last_seen, b.last_seen);
        }
    }
    #[test]
    fn adaptive_capacity_expands_only_with_support_and_diversity() {
        let mut node = Node::new(1);
        for token in 0..32 {
            assert_eq!(
                node.observe(token, token as u64, 512, CapacityPolicy::Adaptive, true),
                ObserveOutcome::Created
            );
        }
        assert_eq!(
            node.observe(32, 32, 512, CapacityPolicy::Adaptive, true),
            ObserveOutcome::RejectedAtCapacity
        );
        for step in 33..65 {
            node.observe(0, step, 512, CapacityPolicy::Adaptive, true);
        }
        assert_eq!(
            node.observe(32, 65, 512, CapacityPolicy::Adaptive, true),
            ObserveOutcome::Created
        );
        assert_eq!(node.edges.len(), 33);
    }
    #[test]
    fn pruning_ties_are_independent_of_hashmap_insertion_order() {
        let mut forward = HashMap::new();
        let mut reverse = HashMap::new();
        for token in 0..20u16 {
            let key = ContextKey::new(1, [token, 0, 0, 0]);
            let mut node = Node::new(1);
            node.total = 1;
            node.last_seen = 7;
            forward.insert(key, node);
        }
        for token in (0..20u16).rev() {
            let key = ContextKey::new(1, [token, 0, 0, 0]);
            let mut node = Node::new(1);
            node.total = 1;
            node.last_seen = 7;
            reverse.insert(key, node);
        }
        prune_graph(&mut forward, 10);
        prune_graph(&mut reverse, 10);
        let mut a: Vec<_> = forward.keys().copied().collect();
        let mut b: Vec<_> = reverse.keys().copied().collect();
        a.sort_unstable();
        b.sort_unstable();
        assert!(a == b);
    }
    #[test]
    fn frequency_shortens_edges() {
        assert!(
            raw_length(&edge(100, 0.0, 0.0, 0), Mode::Frequency)
                < raw_length(&edge(1, 0.0, 0.0, 0), Mode::Frequency)
        );
    }
    #[test]
    fn useful_gradient_shortens_edges() {
        assert!(
            raw_length(&edge(10, -2.0, 1.0, 0), Mode::Gradient)
                < raw_length(&edge(10, 2.0, 1.0, 0), Mode::Gradient)
        );
    }
    #[test]
    fn conflict_lengthens_edges() {
        assert!(
            raw_length(&edge(10, -1.0, 1.0, 5), Mode::Full)
                > raw_length(&edge(10, -1.0, 1.0, 0), Mode::Full)
        );
    }
    #[test]
    fn shuffle_preserves_multiset() {
        let rows = vec![
            edge(1, 0.0, 1.0, 0),
            edge(10, -1.0, 1.0, 0),
            edge(100, -2.0, 1.0, 0),
        ];
        let mut a: Vec<_> = (0..3).map(|i| raw_length(&rows[i], Mode::Full)).collect();
        let mut b: Vec<_> = (0..3)
            .map(|i| assigned_length(&rows, i, 7, Mode::Shuffle))
            .collect();
        a.sort_by(|x, y| x.partial_cmp(y).unwrap());
        b.sort_by(|x, y| x.partial_cmp(y).unwrap());
        assert_eq!(a, b);
    }
    #[test]
    fn unit_chain_converges_in_virtual_time() {
        let mut model = tiny();
        for (a, b) in [(1, 2), (2, 3), (3, 4), (4, 5)] {
            chain(&mut model, a, b);
        }
        let (base, total, _) = base_weights(&model);
        let trace = wave_trace(&model, &[1, 2, 3, 4], &base, total, Mode::Unit);
        assert!(trace.full);
        assert_eq!(trace.candidates.first().map(|x| x.token), Some(5));
    }
    #[test]
    fn serialization_roundtrip() {
        let mut model = tiny();
        chain(&mut model, 1, 2);
        model.glp.values_mut().next().unwrap().edges[0].sleep_bias = -0.2;
        let mut construction = Construction::new(
            ConstructionKey {
                len: 4,
                slot_mask: 2,
                symbols: [1, 0, 3, 4],
                target: 5,
            },
            7,
            true,
        );
        construction.confirmations = 2;
        construction.value = 1.25;
        model.glp_constructions.push(construction);
        let path = format!("/tmp/ngu76-test-{}.model", std::process::id());
        save_model(&model, &path);
        let loaded = load_model(&path);
        std::fs::remove_file(path).unwrap();
        let tokens = [1, 2, 1, 2, 1, 2, 1, 2];
        let before = evaluate(&model, &tokens, Mode::FrequencyGradient, 0);
        let after = evaluate(&loaded, &tokens, Mode::FrequencyGradient, 0);
        assert_eq!(before.nll.to_bits(), after.nll.to_bits());
        assert_eq!(before.top1, after.top1);
        assert_eq!(before.top5, after.top5);
        assert_eq!(loaded.glp.len(), 1);
        assert_eq!(loaded.glp_tokens[2].count, 8);
        assert!((loaded.glp.values().next().unwrap().edges[0].sleep_bias + 0.2).abs() < 1e-6);
        assert!(loaded.gcp_credit.is_empty());
        assert_eq!(loaded.glp_constructions.len(), 1);
        assert_eq!(loaded.glp_constructions[0].key.slot_mask, 2);
        assert_eq!(loaded.glp_constructions[0].support, 7);
    }
    #[test]
    fn global_ties_are_broken_by_token_id() {
        let model = tiny();
        let (_, _, top) = base_weights(&model);
        assert_eq!(top, (0u16..16).collect::<Vec<_>>());
    }
    #[test]
    fn deep_reflection_moves_gcp_to_glp() {
        let mut model = tiny();
        chain(&mut model, 1, 2);
        let node = model
            .glp
            .remove(&ContextKey::new(1, [1, 0, 0, 0]))
            .unwrap();
        model.gcp.insert(
            ContextKey::new(1, [1, 0, 0, 0]),
            node,
        );
        let options = Options {
            vocab_size: 16,
            max_order: 4,
            train_limit: 10,
            gcp_max: 8,
            gm1_max: 12,
            glp_max: 16,
            max_successors: DEFAULT_MAX_SUCCESSORS,
            capacity_policy: CapacityPolicy::Fixed,
            successor_index: true,
            light_interval: 5,
            deep_interval: 10,
            wake_interval: 5,
            lr: 0.01,
            sleep_policy: SleepPolicy::Cycle,
        };
        let mut stats = TrainStats::default();
        deep_reflection(&mut model, &options, &mut stats);
        assert!(model.gcp.is_empty());
        assert_eq!(model.glp.len(), 1);
    }
    #[test]
    fn gr_candidates_are_constructed_without_gold() {
        let mut model = tiny();
        for (a, b) in [(1, 2), (2, 3), (3, 4), (4, 5)] {
            chain(&mut model, a, b);
        }
        let (base, total, _) = base_weights(&model);
        let a = wave_trace(&model, &[1, 2, 3, 4], &base, total, Mode::FrequencyGradient);
        let b = wave_trace(&model, &[1, 2, 3, 4], &base, total, Mode::FrequencyGradient);
        assert_eq!(
            a.candidates.iter().map(|x| x.token).collect::<Vec<_>>(),
            b.candidates.iter().map(|x| x.token).collect::<Vec<_>>()
        );
    }
    #[test]
    fn sleep_sampler_uses_training_prefix_only() {
        let tokens: Vec<u16> = (1..=100).collect();
        let rows = sleep_examples(&tokens, 8, 4);
        assert!(!rows.is_empty());
        assert!(rows.iter().all(|row| row.gold <= 90));
    }
    #[test]
    fn wake_update_changes_only_gcp() {
        let mut model = tiny();
        chain(&mut model, 1, 2);
        let before = glp_fingerprint(&model);
        let (base, total, top) = base_weights(&model);
        let (p, ranked, features, _) =
            distribution(&model, &[1], 2, &base, total, &top, Mode::Base);
        let options = Options {
            vocab_size: 16,
            max_order: 4,
            train_limit: 10,
            gcp_max: 8,
            gm1_max: 12,
            glp_max: 16,
            max_successors: DEFAULT_MAX_SUCCESSORS,
            capacity_policy: CapacityPolicy::Fixed,
            successor_index: true,
            light_interval: 5,
            deep_interval: 10,
            wake_interval: 5,
            lr: 0.01,
            sleep_policy: SleepPolicy::Cycle,
        };
        let mut stats = TrainStats {
            tokens: 1,
            ..TrainStats::default()
        };
        update(
            &mut model,
            &[1],
            2,
            p,
            &ranked,
            &features,
            &options,
            &mut stats,
        );
        assert_eq!(before, glp_fingerprint(&model));
        assert!(!model.gcp_credit.is_empty());
        assert!(!model.gcp.is_empty());
    }
    #[test]
    fn rejected_journal_restores_glp() {
        let mut model = tiny();
        for (a, b) in [(1, 2), (2, 3), (3, 4), (4, 5)] {
            chain(&mut model, a, b);
        }
        let before = glp_fingerprint(&model);
        let mut stats = TrainStats::default();
        let examples = vec![SleepExample {
            history: vec![1, 2, 3, 4],
            gold: 5,
        }];
        let journal = gr_loss_journal(
            &model,
            &examples,
            SleepPolicy::Uniform,
            &mut stats,
            &mut Vec::new(),
        );
        assert!(!journal.is_empty());
        apply_journal(&mut model, &journal, true);
        assert_ne!(before, glp_fingerprint(&model));
        apply_journal(&mut model, &journal, false);
        assert_eq!(before, glp_fingerprint(&model));
    }
    #[test]
    fn masking_necessary_edge_removes_candidate() {
        let mut model = tiny();
        for (a, b) in [(1, 2), (2, 3), (3, 4), (4, 5)] {
            chain(&mut model, a, b);
        }
        let (base, total, _) = base_weights(&model);
        let normal = wave_trace(&model, &[1, 2, 3, 4], &base, total, Mode::Unit);
        let masked = wave_trace_masked(
            &model,
            &[1, 2, 3, 4],
            &base,
            total,
            Mode::Unit,
            Some((4, 5)),
        );
        assert!(normal.candidates.iter().any(|gr| gr.token == 5));
        assert!(!masked.candidates.iter().any(|gr| gr.token == 5));
    }
    #[test]
    fn common_fragility_cancels_in_difference() {
        let good = CandidateGr {
            token: 5,
            bonus: 0.2,
            ..CandidateGr::default()
        };
        let bad = CandidateGr {
            token: 6,
            bonus: 0.2,
            ..CandidateGr::default()
        };
        let intervened = WaveTrace::default();
        assert_eq!(
            necessity_from_trace(&good, &intervened) - necessity_from_trace(&bad, &intervened),
            0.0
        );
    }
    #[test]
    fn bidirectional_bridge_joins_multiple_origins() {
        let mut model = tiny();
        for (a, b) in [(1, 3), (2, 3), (3, 4), (4, 5)] {
            chain(&mut model, a, b);
        }
        let (base, total, _) = base_weights(&model);
        let reverse = build_reverse_index(&model, &base, total);
        let bridge = bidirectional_bridge(&model, &[1, 2], 5, &base, total, &reverse).unwrap();
        assert!(bridge.full);
        assert!(bridge.edges.contains(&(3, 4)));
        assert_eq!(bridge.coverage, 1.0);
    }
    #[test]
    fn prospective_overlay_is_transient_and_shortens_only_selected_edge() {
        let mut model = tiny();
        chain(&mut model, 1, 2);
        chain(&mut model, 1, 3);
        let (base, total, _) = base_weights(&model);
        let before = glp_fingerprint(&model);
        let normal =
            successors_modified(&model, 1, &base, total, Mode::FrequencyGradient, None, None);
        let overlay = HashMap::from([((1, 2), -0.03)]);
        let changed = successors_modified(
            &model,
            1,
            &base,
            total,
            Mode::FrequencyGradient,
            None,
            Some(&overlay),
        );
        let length = |rows: &[(u16, f64, f64, f64)], token| {
            rows.iter().find(|row| row.0 == token).unwrap().2
        };
        assert!(length(&changed, 2) < length(&normal, 2));
        assert_eq!(length(&changed, 3), length(&normal, 3));
        assert_eq!(before, glp_fingerprint(&model));
    }
    #[test]
    fn prospective_validation_rejects_when_gold_does_not_emerge() {
        let mut model = tiny();
        for (a, b) in [(1, 3), (2, 3), (3, 4), (4, 5)] {
            chain(&mut model, a, b);
        }
        let (base, total, _) = base_weights(&model);
        let baseline = wave_trace(&model, &[1, 2], &base, total, Mode::FrequencyGradient);
        let bridge = Bridge {
            edges: vec![(1, 3), (2, 3)],
            coverage: 1.0,
            dispersion: 0.0,
            cost: 1.0,
            full: true,
        };
        let result = validate_bridge_prospectively(
            &model,
            &[1, 2],
            6,
            &baseline,
            &bridge,
            2.0,
            &base,
            total,
        );
        assert!(!result.gold_found);
        assert!(!result.accepted);
    }
    #[test]
    fn multidose_policies_separate_memory_ablation() {
        assert!(SleepPolicy::MultiDoseContrast.multidose());
        assert!(!SleepPolicy::MultiDoseContrast.cumulative());
        assert!(SleepPolicy::CumulativeContrast.multidose());
        assert!(SleepPolicy::CumulativeContrast.cumulative());
        assert!(SleepPolicy::CumulativeShuffle.shuffled());
    }
    #[test]
    fn four_doses_shorten_more_than_one_without_mutation() {
        let mut model = tiny();
        chain(&mut model, 1, 2);
        let (base, total, _) = base_weights(&model);
        let before = glp_fingerprint(&model);
        let one = HashMap::from([((1, 2), -0.03)]);
        let four = HashMap::from([((1, 2), -0.12)]);
        let length = |overlay: &HashMap<(u16, u16), f64>| {
            successors_modified(
                &model,
                1,
                &base,
                total,
                Mode::FrequencyGradient,
                None,
                Some(overlay),
            )[0]
            .2
        };
        assert!(length(&four) < length(&one));
        assert_eq!(before, glp_fingerprint(&model));
    }
    #[test]
    fn phase_policies_preserve_ablation_and_shuffle() {
        assert!(SleepPolicy::PhaseContrast.phase());
        assert!(!SleepPolicy::PhaseContrast.bridge());
        assert!(SleepPolicy::CumulativePhase.phase());
        assert!(SleepPolicy::CumulativePhase.bridge());
        assert!(SleepPolicy::CumulativePhase.cumulative());
        assert!(SleepPolicy::CumulativePhaseShuffle.shuffled());
    }
    #[test]
    fn phase_vector_is_bounded_and_transient() {
        let mut model = tiny();
        for (a, b) in [(1, 3), (2, 3), (3, 4), (4, 5)] {
            chain(&mut model, a, b);
        }
        let (base, total, _) = base_weights(&model);
        let before = glp_fingerprint(&model);
        let trace = wave_trace(&model, &[1, 2], &base, total, Mode::FrequencyGradient);
        for candidate in &trace.candidates {
            assert!(candidate
                .phase_deltas
                .iter()
                .all(|(_, delta)| delta.abs() <= 0.03 + 1e-12));
        }
        assert_eq!(before, glp_fingerprint(&model));
    }
    #[test]
    fn sequential_policies_keep_journals_separate() {
        assert!(SleepPolicy::SequentialPhase.sequential());
        assert!(!SleepPolicy::SequentialPhase.origin());
        assert!(SleepPolicy::OriginSequential.sequential());
        assert!(SleepPolicy::OriginSequential.origin());
        assert!(SleepPolicy::OriginSequentialShuffle.shuffled());
        assert!(!SleepPolicy::OriginSequential.phase());
    }
    #[test]
    fn missing_origin_path_is_found_without_mutation() {
        let mut model = tiny();
        for (a, b) in [(1, 2), (2, 3), (3, 4), (4, 5)] {
            chain(&mut model, a, b);
        }
        let (base, total, _) = base_weights(&model);
        let before = glp_fingerprint(&model);
        let path = find_origin_path(&model, 1, 5, &base, total).unwrap();
        assert_eq!(path, vec![(1, 2), (2, 3), (3, 4), (4, 5)]);
        assert_eq!(before, glp_fingerprint(&model));
    }
    #[test]
    fn origin_bank_ablation_controls_phase() {
        assert!(SleepPolicy::OriginBankNoPhase.origin_bank());
        assert!(!SleepPolicy::OriginBankNoPhase.separate_phase());
        assert!(SleepPolicy::OriginBankPhase.origin_bank());
        assert!(SleepPolicy::OriginBankPhase.separate_phase());
        assert!(SleepPolicy::OriginBankShuffle.shuffled());
    }
    #[test]
    fn route_set_policies_are_joint_and_phase_free() {
        for policy in [
            SleepPolicy::RouteSetGreedy,
            SleepPolicy::RouteSetBeam,
            SleepPolicy::RouteSetNoInterference,
            SleepPolicy::RouteSetShuffle,
        ] {
            assert!(policy.route_set());
            assert!(policy.origin_bank());
            assert!(policy.origin());
            assert!(policy.sequential());
            assert!(!policy.separate_phase());
        }
        assert!(SleepPolicy::RouteSetShuffle.shuffled());
        assert!(!SleepPolicy::RouteSetBeam.shuffled());
    }
    #[test]
    fn route_set_overlay_unions_shared_edges_without_double_credit() {
        let routes = vec![vec![(1, 2), (2, 3)], vec![(2, 3), (3, 4)]];
        let overlay = route_set_overlay(&routes, &[0, 1], 4);
        assert_eq!(overlay.len(), 3);
        assert!((overlay[&(2, 3)] + 0.12).abs() < 1e-12);
    }
    #[test]
    fn route_set_evaluation_is_transient() {
        let mut model = tiny();
        for (a, b) in [(1, 3), (2, 4), (4, 5), (5, 3)] {
            chain(&mut model, a, b);
        }
        let (base, total, _) = base_weights(&model);
        let before = glp_fingerprint(&model);
        let baseline = wave_trace(&model, &[1, 2], &base, total, Mode::FrequencyGradient);
        let routes = vec![vec![(2, 4), (4, 5), (5, 3)]];
        let _ = evaluate_route_set(
            &model,
            &[1, 2],
            3,
            &baseline,
            &routes,
            &[0],
            4,
            &base,
            total,
        );
        assert_eq!(before, glp_fingerprint(&model));
    }
    #[test]
    fn route_set_beam_never_returns_an_unaccepted_terminal_state() {
        let model = tiny();
        let (base, total, _) = base_weights(&model);
        let baseline = wave_trace(&model, &[1], &base, total, Mode::FrequencyGradient);
        let routes = vec![vec![(1, 2)]];
        let mut stats = TrainStats::default();
        let selected = select_route_set(
            &model,
            &[1],
            15,
            &baseline,
            &routes,
            SleepPolicy::RouteSetBeam,
            &base,
            total,
            &mut stats,
        );
        assert!(selected.is_none());
    }
    #[test]
    fn deferred_policies_use_quarantine_without_phase() {
        for policy in [
            SleepPolicy::DeferredRoute,
            SleepPolicy::DeferredEdge,
            SleepPolicy::DeferredMetagraph,
            SleepPolicy::DeferredShuffle,
        ] {
            assert!(policy.deferred());
            assert!(policy.origin());
            assert!(policy.origin_bank());
            assert!(policy.sequential());
            assert!(!policy.route_set());
            assert!(!policy.separate_phase());
        }
        assert!(SleepPolicy::DeferredShuffle.shuffled());
    }
    #[test]
    fn future_overlay_is_transient() {
        let mut model = tiny();
        for (a, b) in [(1, 2), (2, 3), (3, 4)] {
            chain(&mut model, a, b);
        }
        let examples = vec![SleepExample {
            history: vec![1, 2],
            gold: 3,
        }];
        let (base, total, _) = base_weights(&model);
        let baselines = vec![wave_trace(
            &model,
            &examples[0].history,
            &base,
            total,
            Mode::FrequencyGradient,
        )];
        let before = glp_fingerprint(&model);
        let _ = evaluate_future_edges(
            &model,
            &examples,
            &baselines,
            &[(1, 2), (2, 3)],
            2,
            &base,
            total,
            32,
        );
        assert_eq!(before, glp_fingerprint(&model));
    }
    #[test]
    fn irrelevant_future_context_is_not_evidence() {
        let model = tiny();
        let examples = vec![SleepExample {
            history: vec![8, 9],
            gold: 10,
        }];
        let (base, total, _) = base_weights(&model);
        let baselines = vec![wave_trace(
            &model,
            &examples[0].history,
            &base,
            total,
            Mode::FrequencyGradient,
        )];
        let (evidence, _) = evaluate_future_edges(
            &model,
            &examples,
            &baselines,
            &[(1, 2)],
            1,
            &base,
            total,
            32,
        );
        assert_eq!(evidence.evaluated, 0);
        assert!(!evidence.supported);
    }
    #[test]
    fn route_credit_requires_two_future_windows() {
        let mut model = tiny();
        chain(&mut model, 1, 2);
        let proposal = DeferredProposal {
            origin_history: vec![1],
            origin_gold: 2,
            routes: vec![vec![(1, 2)]],
            edges: vec![(1, 2)],
            dose: 1,
            age: 1,
            evidence_windows: 1,
            supported_windows: 1,
            adverse_windows: 0,
            evaluated_examples: 8,
            gold_delta: 0.0,
            full_delta: 0.0,
            coverage_delta: 1.0,
            margin_delta: 0.1,
            wrong_promotion: 0.0,
            value: 0.35,
            last_value: 0.35,
            edge_value: HashMap::new(),
            synergy: 0,
            conflict: 0,
        };
        let mut pending = vec![proposal.clone()];
        let mut stats = TrainStats::default();
        let journal = deferred_origin_journal(
            &model,
            &[],
            SleepPolicy::DeferredRoute,
            &mut stats,
            &mut pending,
        );
        assert!(journal.is_empty());
        assert_eq!(pending.len(), 1);

        let mut mature = proposal;
        mature.evidence_windows = 2;
        mature.supported_windows = 2;
        mature.age = 2;
        let mut pending = vec![mature];
        let journal = deferred_origin_journal(
            &model,
            &[],
            SleepPolicy::DeferredRoute,
            &mut stats,
            &mut pending,
        );
        assert_eq!(journal.len(), 1);
        assert!(pending.is_empty());
    }
    #[test]
    fn edge_credit_commits_only_positive_supported_edges() {
        let mut model = tiny();
        chain(&mut model, 1, 2);
        chain(&mut model, 2, 3);
        let mut values = HashMap::new();
        values.insert((1, 2), 1.0);
        values.insert((2, 3), -1.0);
        let proposal = DeferredProposal {
            origin_history: vec![1],
            origin_gold: 3,
            routes: vec![vec![(1, 2), (2, 3)]],
            edges: vec![(1, 2), (2, 3)],
            dose: 1,
            age: 2,
            evidence_windows: 2,
            supported_windows: 2,
            adverse_windows: 0,
            evaluated_examples: 16,
            gold_delta: 0.0,
            full_delta: 0.0,
            coverage_delta: 1.0,
            margin_delta: 0.1,
            wrong_promotion: 0.0,
            value: 0.35,
            last_value: 0.35,
            edge_value: values,
            synergy: 0,
            conflict: 0,
        };
        let mut pending = vec![proposal];
        let mut stats = TrainStats::default();
        let journal = deferred_origin_journal(
            &model,
            &[],
            SleepPolicy::DeferredEdge,
            &mut stats,
            &mut pending,
        );
        assert_eq!(journal.len(), 1);
        assert_eq!(journal[0].key.tokens()[0], 1);
        assert_eq!(journal[0].token, 2);
    }
    #[test]
    fn construction_policies_separate_slots_quarantine_and_shuffle() {
        for policy in [
            SleepPolicy::ConstructionFixed,
            SleepPolicy::ConstructionSlotsImmediate,
            SleepPolicy::ConstructionDeferred,
            SleepPolicy::ConstructionBoth,
            SleepPolicy::ConstructionShuffle,
        ] {
            assert!(policy.construction());
            assert!(policy.sequential());
            assert!(!policy.origin());
            assert!(!policy.deferred());
            assert!(!policy.separate_phase());
            assert!(!policy.bridge());
            assert!(policy.construction_deferred());
            assert!(policy.construction_dag());
        }
        assert!(SleepPolicy::ConstructionShuffle.shuffled());
        assert!(SleepPolicy::ConstructionFixed.construction_edge_role_credit());
        assert!(SleepPolicy::ConstructionSlotsImmediate.construction_edge_role_credit());
        assert!(SleepPolicy::ConstructionSlotsImmediate.construction_edge_prediction_credit());
        assert!(SleepPolicy::ConstructionDeferred.construction_edge_prediction_credit());
        assert_eq!(SleepPolicy::ConstructionFixed.edge_application(), 0b0010);
        assert_eq!(SleepPolicy::ConstructionSlotsImmediate.edge_application(), 0b0110);
        assert_eq!(SleepPolicy::ConstructionDeferred.edge_application(), 0b1010);
        assert_eq!(SleepPolicy::ConstructionBoth.edge_application(), 0b1110);
        assert!(construction_masks(SleepPolicy::ConstructionFixed)
            .iter()
            .any(|mask| mask.count_ones() == 2));
    }
    #[test]
    fn directed_construction_is_forward_only_dag() {
        let construction = Construction::new(
            ConstructionKey {
                len: 4,
                slot_mask: 2,
                symbols: [1, 0, 3, 4],
                target: 7,
            },
            3,
            true,
        );
        assert!(construction.dag_is_forward_only());
        assert_eq!(construction.dag_edge_count as usize, DAG_EDGE_COUNT);
        for edge in &construction.dag_edges[..CONSTRUCTION_INTERFACE] {
            assert_eq!(edge.target, DAG_CONSTRUCTION_VERTEX);
            assert_eq!(edge.relation, DAG_ROLE_EDGE);
            assert_ne!(edge.source, edge.target);
        }
        let edge = construction.dag_edges[CONSTRUCTION_INTERFACE];
        assert_eq!(edge.source, DAG_CONSTRUCTION_VERTEX);
        assert_eq!(edge.target, DAG_TARGET_VERTEX);
        assert_eq!(edge.relation, DAG_PREDICTION_EDGE);
    }
    #[test]
    fn reverse_or_self_loop_dag_is_inert() {
        let mut construction = Construction::new(
            ConstructionKey {
                len: 4,
                slot_mask: 2,
                symbols: [1, 0, 3, 4],
                target: 7,
            },
            3,
            true,
        );
        construction.dag_edges[0].source = DAG_CONSTRUCTION_VERTEX;
        construction.dag_edges[0].target = 0;
        assert!(!construction.dag_is_forward_only());
        assert_eq!(construction.forward_gain(), 0.0);
        let mut candidates = Vec::new();
        inject_construction_candidates(&[1, 9, 3, 4], &[construction], &mut candidates, 3);
        assert!(candidates.is_empty());
    }
    #[test]
    fn directional_edge_credit_updates_enabled_edges_independently() {
        let construction = Construction::new(
            ConstructionKey {
                len: 4,
                slot_mask: 2,
                symbols: [1, 0, 3, 4],
                target: 7,
            },
            3,
            true,
        );
        let mut proposal = PendingConstruction {
            construction,
            age: 2,
            evidence_windows: 2,
            supported_windows: 2,
            adverse_windows: 0,
            evaluated_examples: 8,
            gold_delta: 1.0,
            full_delta: 1.0,
            coverage_delta: 1.0,
            margin_delta: 0.1,
            wrong_promotion: 0.0,
            value: 1.0,
            edge_value: [0.0; DAG_EDGE_COUNT],
            edge_value_sq: [0.0; DAG_EDGE_COUNT],
            edge_tests: [8; DAG_EDGE_COUNT],
        };
        proposal.edge_value = [4.0, -4.0, 2.0, -2.0, 6.0];
        let mut signal = construction;
        apply_directional_edge_credit(
            &mut signal,
            &proposal,
            SleepPolicy::ConstructionSlotsImmediate,
        );
        assert!(signal.dag_edges[0].weight > 1.0);
        assert!(signal.dag_edges[1].weight < 1.0);
        assert!(signal.dag_edges[CONSTRUCTION_INTERFACE].weight > 1.0);
        let mut all = construction;
        apply_directional_edge_credit(&mut all, &proposal, SleepPolicy::ConstructionDeferred);
        assert!(all.dag_edges[CONSTRUCTION_INTERFACE].weight > 1.0);
        assert_eq!(all.dag_edges[0].credit_count, 8);
        assert!(all.dag_edges.iter().all(|edge| edge.weight >= 0.65 && edge.weight <= 1.35));
    }
    #[test]
    fn active_role_weight_changes_signal_and_length_without_reverse_edge() {
        let mut model = tiny();
        chain(&mut model, 1, 7);
        let mut construction = Construction::new(
            ConstructionKey {
                len: 4,
                slot_mask: 0,
                symbols: [1, 2, 3, 4],
                target: 7,
            },
            4,
            true,
        );
        construction.dag_edges[0].weight = 1.35;
        model.glp_constructions.push(construction);
        model.edge_application = 0b0010;
        let overlay = active_construction_overlay(&model, &[1, 2, 3, 4]);
        assert_eq!(overlay.active_constructions, 1);
        assert!(
            (overlay.role_weights.get(&(1, 7)).copied().unwrap() - 1.35).abs() < 1e-6
        );
        assert!(!overlay.role_weights.contains_key(&(7, 1)));
        let (base, total, _) = base_weights(&model);
        let ordinary =
            successors_modified(&model, 1, &base, total, Mode::FrequencyGradient, None, None);
        let (early, modified, _) = successors_early(
            &model,
            1,
            &base,
            total,
            Mode::FrequencyGradient,
            None,
            None,
            &overlay,
        );
        assert_eq!(modified, 1);
        assert!(early[0].1 > ordinary[0].1);
        assert!(early[0].2 < ordinary[0].2);
    }
    #[test]
    fn full_application_adjusts_gr_before_truncation() {
        let mut construction = Construction::new(
            ConstructionKey {
                len: 4,
                slot_mask: 0,
                symbols: [1, 2, 3, 4],
                target: 7,
            },
            4,
            true,
        );
        for edge in &mut construction.dag_edges {
            edge.weight = 1.35;
        }
        let mut late = Vec::new();
        let mut early = Vec::new();
        inject_construction_candidates(&[1, 2, 3, 4], &[construction], &mut late, 0);
        let (_, adjustments) =
            inject_construction_candidates(&[1, 2, 3, 4], &[construction], &mut early, 0b1110);
        assert_eq!(adjustments, 1);
        assert!(early[0].bonus > late[0].bonus);
    }
    #[test]
    fn role_and_prediction_gr_terms_are_independently_ablatable() {
        let mut construction = Construction::new(
            ConstructionKey {
                len: 4,
                slot_mask: 0,
                symbols: [1, 2, 3, 4],
                target: 7,
            },
            4,
            true,
        );
        for edge in &mut construction.dag_edges[..CONSTRUCTION_INTERFACE] {
            edge.weight = 1.35;
        }
        construction.dag_edges[CONSTRUCTION_INTERFACE].weight = 0.65;
        let late = construction_bonus_applied(&construction, 0b0010);
        let role = construction_bonus_applied(&construction, 0b0110);
        let prediction = construction_bonus_applied(&construction, 0b1010);
        let both = construction_bonus_applied(&construction, 0b1110);
        assert!(role > late);
        assert!(prediction < late);
        assert!((both - (role + prediction - late)).abs() < 1e-12);
    }
    #[test]
    fn document_permutation_preserves_documents_and_is_seeded() {
        let tokens = vec![1, 2, 0, 3, 0, 4, 5, 6, 0];
        let first = permute_documents(&tokens, 8801);
        let again = permute_documents(&tokens, 8801);
        let second = permute_documents(&tokens, 8802);
        assert_eq!(first, again);
        assert_ne!(first, second);
        let documents = |rows: &[u16]| {
            let mut output = Vec::new();
            let mut start = 0;
            for (index, token) in rows.iter().enumerate() {
                if *token == 0 {
                    output.push(rows[start..=index].to_vec());
                    start = index + 1;
                }
            }
            output.sort();
            output
        };
        assert_eq!(documents(&tokens), documents(&first));
        assert_eq!(documents(&tokens), documents(&second));
    }
    #[test]
    fn matched_pretraining_preserves_the_frozen_dolma_order() {
        let tokens = vec![1, 2, 0, 3, 0, 4, 5, 6, 0];
        assert_eq!(order_documents(&tokens, 9601, true), tokens);
        assert_ne!(order_documents(&tokens, 9601, false), tokens);
    }
    #[test]
    fn functional_evaluation_scores_only_document_tails() {
        let tokens = vec![1, 2, 3, 0, 4, 5, 0];
        assert_eq!(
            tail_score_mask(&tokens, 1),
            vec![false, false, true, false, false, true, false]
        );
        assert_eq!(
            tail_score_mask(&tokens, 2),
            vec![false, true, true, false, true, true, false]
        );
    }
    #[test]
    fn slot_matching_ignores_only_variable_roles() {
        let key = ConstructionKey {
            len: 4,
            slot_mask: 2,
            symbols: [1, 0, 3, 4],
            target: 5,
        };
        assert!(construction_matches(&key, &[1, 9, 3, 4]));
        assert!(construction_matches(&key, &[7, 1, 8, 3, 4]));
        assert!(!construction_matches(&key, &[1, 9, 8, 4]));
    }
    #[test]
    fn induction_anti_unifies_diverse_fillings() {
        let model = tiny();
        let examples = vec![
            SleepExample { history: vec![1, 2, 3, 4], gold: 7 },
            SleepExample { history: vec![1, 5, 3, 4], gold: 7 },
            SleepExample { history: vec![1, 6, 3, 4], gold: 7 },
        ];
        let mut stats = TrainStats::default();
        let rows = induce_constructions(
            &examples,
            SleepPolicy::ConstructionDeferred,
            &model,
            &[],
            &mut stats,
        );
        assert!(rows.iter().any(|row| {
            row.key.slot_mask == 2 && row.key.symbols == [1, 0, 3, 4] && row.key.target == 7
        }));
        assert!(stats.construction_slot_fillings >= 3);
    }
    #[test]
    fn committed_construction_creates_candidate_without_graph_mutation() {
        let mut model = tiny();
        let mut construction = Construction::new(
            ConstructionKey {
                len: 4,
                slot_mask: 2,
                symbols: [1, 0, 3, 4],
                target: 7,
            },
            4,
            true,
        );
        construction.confirmations = 2;
        construction.value = 1.0;
        model.glp_constructions.push(construction);
        let before = glp_fingerprint(&model);
        let (base, total, _) = base_weights(&model);
        let trace = wave_trace(&model, &[1, 9, 3, 4], &base, total, Mode::FrequencyGradient);
        let candidate = trace.candidates.iter().find(|row| row.token == 7).unwrap();
        assert!(candidate.full);
        assert_eq!(candidate.coverage, 4);
        assert_eq!(before, glp_fingerprint(&model));
    }
    #[test]
    fn construction_commit_requires_two_supported_windows() {
        let mut model = tiny();
        let construction = Construction::new(
            ConstructionKey {
                len: 4,
                slot_mask: 2,
                symbols: [1, 0, 3, 4],
                target: 7,
            },
            3,
            true,
        );
        let mut pending = vec![PendingConstruction {
            construction,
            age: 1,
            evidence_windows: 1,
            supported_windows: 1,
            adverse_windows: 0,
            evaluated_examples: 4,
            gold_delta: 1.0,
            full_delta: 1.0,
            coverage_delta: 1.0,
            margin_delta: 0.1,
            wrong_promotion: 0.0,
            value: 7.0,
            edge_value: [0.0; DAG_EDGE_COUNT],
            edge_value_sq: [0.0; DAG_EDGE_COUNT],
            edge_tests: [0; DAG_EDGE_COUNT],
        }];
        let mut stats = TrainStats::default();
        construction_sleep(
            &mut model,
            &[],
            SleepPolicy::ConstructionDeferred,
            &mut stats,
            &mut pending,
        );
        assert!(model.glp_constructions.is_empty());
        assert_eq!(pending.len(), 1);
        pending[0].evidence_windows = 2;
        pending[0].supported_windows = 2;
        construction_sleep(
            &mut model,
            &[],
            SleepPolicy::ConstructionDeferred,
            &mut stats,
            &mut pending,
        );
        assert_eq!(model.glp_constructions.len(), 1);
        assert!(pending.is_empty());
    }
    #[test]
    fn exp89_uses_preregistered_dense_construction_budget() {
        assert_eq!(CONSTRUCTION_SLEEP_EXAMPLES, 8192);
        assert_eq!(CONSTRUCTION_NEW_LIMIT, 512);
        assert_eq!(CONSTRUCTION_LIMIT, 8192);
        assert_eq!(CONSTRUCTION_FUTURE_LIMIT, 64);
        assert_eq!(CONSTRUCTION_REQUIRED_WINDOWS, 2);
        assert_eq!(CONSTRUCTION_MAX_AGE, 4);
    }
}



#[cfg(test)]
mod exp168_tests {
    use super::*;

    fn node(total: u32, dominant: u32) -> Node {
        let mut row = Node::new(1);
        row.total = total;
        row.edges.push(Edge { token: 2, count: dominant, ..Edge::default() });
        row
    }

    #[test]
    fn staged_lookup_stops_at_the_first_confident_level() {
        reset_layer_access();
        let mut model = super::tests::tiny();
        model.gm1_mode = 3;
        let key = ContextKey::new(1, [1, 0, 0, 0]);
        model.gm1.insert(key, node(10, 8));
        model.glp.insert(key, node(100, 90));
        assert_eq!(mediated_long_view(&model, &key, None).unwrap().total, 10);
    }

    #[test]
    fn fresh_gcp_cannot_jump_directly_to_glp() {
        let mut model = super::tests::tiny();
        model.gm1_mode = 3;
        let key = ContextKey::new(1, [1, 0, 0, 0]);
        model.gcp.insert(key, node(12, 9));
        let options = Options {
            vocab_size: 16, max_order: 4, train_limit: 10,
            gcp_max: 8, gm1_max: 12, glp_max: 16,
            max_successors: DEFAULT_MAX_SUCCESSORS,
            capacity_policy: CapacityPolicy::Fixed, successor_index: true,
            light_interval: 5, deep_interval: 10, wake_interval: 5,
            lr: 0.01, sleep_policy: SleepPolicy::Cycle,
        };
        deep_reflection(&mut model, &options, &mut TrainStats::default());
        assert!(model.gcp.is_empty());
        assert!(model.gm1.contains_key(&key));
        assert!(!model.glp.contains_key(&key));
    }

    #[test]
    fn observer_never_participates_in_lookup() {
        reset_layer_access();
        let mut model = super::tests::tiny();
        model.gm1_mode = 1;
        let key = ContextKey::new(1, [1, 0, 0, 0]);
        model.gm1.insert(key, node(10, 9));
        assert!(mediated_long_view(&model, &key, None).is_none());
    }
}









#[cfg(test)]
mod exp176_tests {
    use super::*;

    fn sample_graph() -> ContextMap<Node> {
        let mut graph = ContextMap::default();
        for source in 1..16u16 {
            let key = ContextKey::new(2, [source, source + 1, 0, 0]);
            let mut node = Node::new(2);
            for token in [7u16, 3, 11, 5] {
                node.observe(token, token as u64, 32, CapacityPolicy::Fixed, true);
            }
            graph.insert(key, node);
        }
        graph
    }

    #[test]
    fn packed_key_is_eight_bytes_and_preserves_order() {
        assert_eq!(std::mem::size_of::<ContextKey>(), 8);
        let a = ContextKey::new(1, [9, 0, 0, 0]);
        let b = ContextKey::new(2, [1, 2, 0, 0]);
        assert!(a < b);
        assert_eq!(b.tokens(), [1, 2, 0, 0]);
    }

    #[test]
    fn flat_index_finds_every_canonical_key() {
        let graph = sample_graph();
        let mut flat = FlatGraph::default();
        flat.rebuild(&graph, 2);
        for (key, node) in &graph {
            let view = flat.view(key, 2).expect("flat key");
            assert_eq!(view.total, node.total);
            assert_eq!(view.edges.len(), node.edges.len());
        }
    }

    #[test]
    fn contiguous_and_compacted_edges_preserve_sorted_view() {
        let graph = sample_graph();
        let mut contiguous = FlatGraph::default();
        let mut compacted = FlatGraph::default();
        contiguous.rebuild(&graph, 3);
        compacted.rebuild(&graph, 4);
        for key in graph.keys() {
            let left = EdgeOrder::new(contiguous.view(key, 3).unwrap());
            let right = EdgeOrder::new(compacted.view(key, 4).unwrap());
            let left_tokens: Vec<_> =
                (0..left.len()).map(|i| left.get(i).unwrap().token).collect();
            let right_tokens: Vec<_> =
                (0..right.len()).map(|i| right.get(i).unwrap().token).collect();
            assert_eq!(left_tokens, right_tokens);
        }
    }
}

#[cfg(test)]
mod exp175_tests {
    use super::*;

    fn options() -> Options {
        Options {
            vocab_size: 16,
            max_order: 4,
            train_limit: 256,
            gcp_max: 32,
            gm1_max: 64,
            glp_max: 128,
            max_successors: 32,
            capacity_policy: CapacityPolicy::Fixed,
            successor_index: true,
            light_interval: 64,
            deep_interval: 128,
            wake_interval: 64,
            lr: 0.025,
            sleep_policy: SleepPolicy::Uniform,
        }
    }

    fn model() -> Model {
        let mut model = super::tests::tiny();
        model.shard_mode = 14;
        model.token_themes = (0..model.vocab_size)
            .map(|token| (token % EXP170_SHARDS) as u8)
            .collect();
        model
    }

    #[test]
    fn dependency_waves_have_disjoint_keys() {
        let mut history = Vec::new();
        let events = build_spec_events(
            &[1, 2, 3, 1, 2, 4, 5, 1, 2, 6],
            0,
            10,
            4,
            &mut history,
        );
        for wave in dependency_waves(&events) {
            let mut keys = HashSet::new();
            for position in wave {
                let event = &events[position];
                for order in 0..event.history_len as usize {
                    assert!(keys.insert(event.route_keys[order]));
                }
            }
        }
    }

    #[test]
    fn wave_and_prepared_modes_preserve_serial_speculative_state() {
        std::env::set_var("NGU_SPEC_WORKERS", "4");
        std::env::set_var("NGU_TRAIN_WORKERS", "1");
        let tokens: Vec<u16> = (0..256)
            .map(|position| {
                if position % 32 == 0 {
                    0
                } else {
                    ((position * 7 + position / 5) % 15 + 1) as u16
                }
            })
            .collect();
        let mut reference = model();
        let reference_stats =
            train_speculative(&mut reference, &tokens, &options(), 64);
        for prepared in [false, true] {
            let mut candidate = model();
            let stats = train_dependency_waves(
                &mut candidate,
                &tokens,
                &options(),
                64,
                prepared,
            );
            assert_eq!(glp_fingerprint(&candidate), glp_fingerprint(&reference));
            assert_eq!(stats.predictions, reference_stats.predictions);
            assert_eq!(stats.nll.to_bits(), reference_stats.nll.to_bits());
            assert_eq!(
                stats.victim_fingerprint,
                reference_stats.victim_fingerprint
            );
        }
    }
}

#[cfg(test)]
mod exp174_tests {
    use super::*;

    fn options() -> Options {
        Options {
            vocab_size: 16,
            max_order: 4,
            train_limit: 256,
            gcp_max: 32,
            gm1_max: 64,
            glp_max: 128,
            max_successors: 32,
            capacity_policy: CapacityPolicy::Fixed,
            successor_index: true,
            light_interval: 64,
            deep_interval: 128,
            wake_interval: 64,
            lr: 0.025,
            sleep_policy: SleepPolicy::Uniform,
        }
    }

    fn model() -> Model {
        let mut model = super::tests::tiny();
        model.shard_mode = 14;
        model.token_themes = (0..model.vocab_size)
            .map(|token| (token % EXP170_SHARDS) as u8)
            .collect();
        model
    }

    #[test]
    fn persistent_modes_preserve_learned_state() {
        std::env::set_var("NGU_SPEC_WORKERS", "4");
        let tokens: Vec<u16> = (0..256)
            .map(|position| {
                if position % 32 == 0 {
                    0
                } else {
                    ((position * 7 + position / 5) % 15 + 1) as u16
                }
            })
            .collect();
        let mut reference = model();
        let reference_stats =
            train_speculative(&mut reference, &tokens, &options(), 64);
        for mode in 1..=3 {
            let mut candidate = model();
            let stats =
                train_persistent(&mut candidate, &tokens, &options(), 64, mode);
            assert_eq!(glp_fingerprint(&candidate), glp_fingerprint(&reference));
            assert_eq!(stats.predictions, reference_stats.predictions);
            assert_eq!(stats.nll.to_bits(), reference_stats.nll.to_bits());
            assert_eq!(stats.victim_fingerprint, reference_stats.victim_fingerprint);
        }
    }

    #[test]
    fn route_descriptors_match_dynamic_owner() {
        let mut model = model();
        let key = ContextKey::new(2, [3, 4, 0, 0]);
        model.shard_assignments.insert(key, 6);
        let mut history = Vec::new();
        let events = build_spec_events(&[3, 4, 5], 0, 3, 4, &mut history);
        let event = &events[2];
        assert_eq!(
            exp174_event_owner(&model, event, false),
            exp174_event_owner(&model, event, true)
        );
    }
}

#[cfg(test)]
mod exp173_tests {
    use super::*;

    fn node(total: u32, target: u16) -> Node {
        Node {
            total,
            last_seen: total as u64,
            potential: 0.0,
            edges: vec![Edge {
                token: target,
                count: total,
                weight: 0.0,
                gradient_sum: 0.0,
                gradient_sq: 0.0,
                conflicts: 0,
                last_seen: total as u64,
                sleep_bias: 0.0,
            }],
            edge_index: None,
        }
    }

    fn model(mode: u8) -> Model {
        let mut model = super::tests::tiny();
        model.shard_mode = mode;
        model.token_themes = (0..model.vocab_size)
            .map(|token| (token % EXP170_SHARDS) as u8)
            .collect();
        model
    }

    #[test]
    fn full_directory_contains_every_glp_context() {
        let mut model = model(13);
        for token in 0..16u16 {
            let key = ContextKey::new(1, [token, 0, 0, 0]);
            model.glp.insert(key, node(token as u32 + 1, token));
        }
        full_exp171_partition(&mut model, &mut TrainStats::default());
        assert_eq!(model.gcp_routes.len(), model.glp.len());
    }

    #[test]
    fn owner_storage_has_no_node_or_view_copies() {
        let key = ContextKey::new(1, [3, 0, 0, 0]);
        let mut model = model(14);
        model.glp.insert(key, node(7, 3));
        full_exp171_partition(&mut model, &mut TrainStats::default());
        assert!(model.glp_shards.iter().all(|shard| shard.is_empty()));
        assert!(model.glp_shard_views.is_empty());
        let shard = model.shard_assignments[&key];
        assert_eq!(probe_shard(&model, shard, &key).unwrap().total, 7);
    }

    #[test]
    fn full_directory_absence_uses_no_fallback() {
        reset_layer_access();
        let mut model = model(14);
        full_exp171_partition(&mut model, &mut TrainStats::default());
        let key = ContextKey::new(1, [9, 0, 0, 0]);
        assert!(routed_glp_get(&model, &key, None).is_none());
        assert_eq!(route_access()[5], 0);
    }

    #[test]
    fn owner_and_view_return_same_authoritative_node() {
        let key = ContextKey::new(1, [3, 0, 0, 0]);
        let mut view = model(13);
        view.glp.insert(key, node(7, 3));
        full_exp171_partition(&mut view, &mut TrainStats::default());
        let mut owner = view.clone();
        owner.shard_mode = 14;
        full_exp171_partition(&mut owner, &mut TrainStats::default());
        let shard = owner.shard_assignments[&key];
        assert_eq!(
            probe_shard(&view, shard, &key).unwrap().total,
            probe_shard(&owner, shard, &key).unwrap().total,
        );
    }
}

#[cfg(test)]
mod exp172_tests {
    use super::*;

    fn node(total: u32, last_seen: u64, target: u16) -> Node {
        Node {
            total,
            last_seen,
            potential: 0.0,
            edges: vec![Edge {
                token: target,
                count: total,
                weight: 0.0,
                gradient_sum: 0.0,
                gradient_sq: 0.0,
                conflicts: 0,
                last_seen,
                sleep_bias: 0.0,
            }],
            edge_index: None,
        }
    }

    fn model(mode: u8) -> Model {
        let mut model = super::tests::tiny();
        model.shard_mode = mode;
        model.token_themes = (0..model.vocab_size)
            .map(|token| (token % EXP170_SHARDS) as u8)
            .collect();
        model
    }

    #[test]
    fn indexed_pruning_selects_exact_global_victim_set() {
        let mut global = model(6);
        for token in 0..12u16 {
            let key = ContextKey::new(1, [token, 0, 0, 0]);
            global.glp.insert(key, node((token % 3) as u32, (token % 2) as u64, token));
        }
        let mut indexed = global.clone();
        indexed.shard_mode = 10;
        let mut stats = TrainStats::default();
        rebuild_eviction_index(&mut indexed, &mut stats);
        let mut expected = prune_graph_collect(&mut global.glp, 7);
        let mut actual = prune_graph_indexed(&mut indexed, 7, &mut stats);
        expected.sort_unstable();
        actual.sort_unstable();
        assert!(actual == expected);
        assert_eq!(indexed.glp.len(), global.glp.len());
        for (key, node) in &global.glp {
            let indexed_node = indexed.glp.get(key).expect("same retained key");
            assert_eq!(indexed_node.total, node.total);
            assert_eq!(indexed_node.last_seen, node.last_seen);
        }
    }

    #[test]
    fn eviction_rank_update_preserves_exact_order() {
        let key = ContextKey::new(1, [3, 0, 0, 0]);
        let mut model = model(10);
        model.glp.insert(key, node(2, 4, 3));
        let mut stats = TrainStats::default();
        rebuild_eviction_index(&mut model, &mut stats);
        model.glp.insert(key, node(9, 11, 3));
        update_eviction_index(&mut model, &[key], &mut stats);
        assert!(!model.eviction_order.contains(&(2, 4, key)));
        assert!(model.eviction_order.contains(&(9, 11, key)));
        assert_eq!(model.eviction_order.len(), 1);
    }

    #[test]
    fn canonical_view_returns_authoritative_node() {
        let key = ContextKey::new(1, [3, 0, 0, 0]);
        let mut model = model(11);
        model.glp.insert(key, node(7, 9, 3));
        full_exp171_partition(&mut model, &mut TrainStats::default());
        assert!(model.glp_shards.iter().all(|shard| shard.is_empty()));
        let shard = model.shard_assignments[&key];
        assert!(model.glp_shard_views[shard as usize].contains_key(&key));
        model.glp.get_mut(&key).unwrap().edges[0].sleep_bias = 0.375;
        assert_eq!(probe_shard(&model, shard, &key).unwrap().edges[0].sleep_bias, 0.375);
    }

    #[test]
    fn physical_and_view_partitions_have_same_assignments_and_sizes() {
        let mut physical = model(10);
        for token in 0..16u16 {
            let key = ContextKey::new(1, [token, 0, 0, 0]);
            physical.glp.insert(key, node(token as u32 + 1, token as u64, token));
        }
        full_exp171_partition(&mut physical, &mut TrainStats::default());
        let mut view = physical.clone();
        view.shard_mode = 12;
        full_exp171_partition(&mut view, &mut TrainStats::default());
        assert!(physical.shard_assignments == view.shard_assignments);
        assert_eq!(shard_sizes(&physical), shard_sizes(&view));
    }
}

#[cfg(test)]
mod exp171_tests {
    use super::*;

    fn edge(token: u16, count: u32) -> Edge {
        Edge {
            token, count, weight: 0.0, gradient_sum: 0.0, gradient_sq: 0.0,
            conflicts: 0, last_seen: 0, sleep_bias: 0.0,
        }
    }

    fn node(token: u16, count: u32) -> Node {
        Node {
            total: count, last_seen: 0, potential: 0.0,
            edges: vec![edge(token, count)], edge_index: None,
        }
    }

    fn model(mode: u8) -> Model {
        let mut model = super::tests::tiny();
        model.shard_mode = mode;
        model.token_themes = (0..model.vocab_size)
            .map(|token| (token % EXP170_SHARDS) as u8)
            .collect();
        model
    }

    #[test]
    fn incremental_update_matches_full_frozen_partition() {
        let key1 = ContextKey::new(2, [1, 2, 0, 0]);
        let key2 = ContextKey::new(2, [2, 3, 0, 0]);
        let mut inc = model(6);
        inc.glp.insert(key1, node(3, 10));
        full_exp171_partition(&mut inc, &mut TrainStats::default());
        inc.glp.insert(key1, node(4, 12));
        inc.glp.insert(key2, node(5, 8));
        incremental_exp171_partition(
            &mut inc, &[key1, key2], &[], &mut TrainStats::default(),
        );
        let mut full = model(9);
        full.glp = inc.glp.clone();
        full_exp171_partition(&mut full, &mut TrainStats::default());
        assert!(inc.shard_assignments == full.shard_assignments);
        assert!(inc.gcp_routes == full.gcp_routes);
        assert_eq!(shard_sizes(&inc), shard_sizes(&full));
    }

    #[test]
    fn incremental_removal_clears_all_indexes() {
        let key = ContextKey::new(2, [1, 2, 0, 0]);
        let mut model = model(6);
        model.glp.insert(key, node(3, 10));
        full_exp171_partition(&mut model, &mut TrainStats::default());
        model.glp.remove(&key);
        incremental_exp171_partition(
            &mut model, &[], &[key], &mut TrainStats::default(),
        );
        assert!(!model.shard_assignments.contains_key(&key));
        assert!(!model.route_support.contains_key(&key));
        assert!(model.glp_shards.iter().all(|graph| !graph.contains_key(&key)));
    }

    #[test]
    fn compact_and_hash_routes_are_identical() {
        let key = ContextKey::new(2, [1, 2, 0, 0]);
        let mut hash = model(6);
        hash.glp.insert(key, node(3, 10));
        full_exp171_partition(&mut hash, &mut TrainStats::default());
        let mut compact = model(8);
        compact.glp = hash.glp.clone();
        full_exp171_partition(&mut compact, &mut TrainStats::default());
        assert_eq!(hash.gcp_routes.get(&key).copied(), compact_route(&compact, &key));
    }

    #[test]
    fn incremental_path_scans_full_glp_only_once() {
        let key = ContextKey::new(2, [1, 2, 0, 0]);
        let mut model = model(6);
        model.glp.insert(key, node(3, 10));
        let mut stats = TrainStats::default();
        full_exp171_partition(&mut model, &mut stats);
        let scanned = stats.incremental_full_scanned_contexts;
        incremental_exp171_partition(&mut model, &[key], &[], &mut stats);
        assert_eq!(stats.incremental_full_scanned_contexts, scanned);
        assert_eq!(stats.incremental_contexts_examined, 1);
    }

    #[test]
    fn sleep_bias_transaction_updates_physical_shard_clone() {
        let key = ContextKey::new(2, [1, 2, 0, 0]);
        let mut model = model(6);
        model.glp.insert(key, node(3, 10));
        full_exp171_partition(&mut model, &mut TrainStats::default());
        apply_journal(
            &mut model,
            &[BiasChange {
                gcp: false, key, token: 3, old: 0.0, new: 0.25,
            }],
            true,
        );
        let shard = model.shard_assignments[&key] as usize;
        let master = model.glp[&key].edges[0].sleep_bias;
        let clone = model.glp_shards[shard][&key].edges[0].sleep_bias;
        assert_eq!(master, 0.25);
        assert_eq!(clone, master);
    }
}

#[cfg(test)]
mod exp170_tests {
    use super::*;

    fn edge(token: u16, count: u32) -> Edge {
        Edge {
            token, count, weight: 0.0, gradient_sum: 0.0, gradient_sq: 0.0,
            conflicts: 0, last_seen: 0, sleep_bias: 0.0,
        }
    }

    fn routed_model(mode: u8) -> Model {
        let mut model = super::tests::tiny();
        model.shard_mode = mode;
        let key = ContextKey::new(2, [1, 2, 0, 0]);
        model.glp.insert(key, Node {
            total: 10, last_seen: 0, potential: 0.0,
            edges: vec![edge(3, 10)], edge_index: None,
        });
        let mut stats = TrainStats::default();
        refresh_exp170_routing(&mut model, &mut stats);
        model
    }

    #[test]
    fn every_glp_context_enters_exactly_one_shard() {
        let model = routed_model(2);
        assert_eq!(shard_sizes(&model).iter().sum::<usize>(), model.glp.len());
        let key = *model.glp.keys().next().unwrap();
        assert_eq!(
            model.glp_shards.iter().filter(|graph| graph.contains_key(&key)).count(),
            1
        );
    }

    #[test]
    fn reconstructed_gcp_router_is_not_persistent_gm1_state() {
        let model = routed_model(2);
        assert!(!model.gcp_routes.is_empty());
        assert!(model.gm1_routes.is_empty());
        assert!(model.gm1_route_mediators.is_empty());
    }

    #[test]
    fn abstract_mediator_requires_two_sleep_confirmations() {
        let mut model = routed_model(4);
        assert!(model.gm1_route_mediators.values().all(|row| row.confirmations == 1));
        refresh_exp170_routing(&mut model, &mut TrainStats::default());
        assert!(model.gm1_route_mediators.values().all(|row| row.confirmations >= 2));
    }

    #[test]
    fn top2_fallback_never_loses_an_authoritative_context() {
        let model = routed_model(5);
        let key = *model.glp.keys().next().unwrap();
        reset_layer_access();
        assert!(routed_glp_get(&model, &key, None).is_some());
        assert_eq!(route_access()[6], 0);
    }

    #[test]
    fn direct_router_is_bounded() {
        let model = routed_model(3);
        assert!(model.gm1_routes.len() <= EXP170_DIRECT_ROUTES);
    }
}

#[cfg(test)]
mod exp169_tests {
    use super::*;

    fn options() -> Options {
        Options {
            vocab_size: 16, max_order: 4, train_limit: 100,
            gcp_max: 8, gm1_max: 12, glp_max: 16,
            max_successors: DEFAULT_MAX_SUCCESSORS,
            capacity_policy: CapacityPolicy::Fixed, successor_index: true,
            light_interval: 25, deep_interval: 100, wake_interval: 50,
            lr: 0.01, sleep_policy: SleepPolicy::Uniform,
        }
    }

    fn candidate() -> MedialDreamCandidate {
        MedialDreamCandidate {
            key: MedialDreamKey {
                context: ContextKey::new(2, [1, 2, 0, 0]),
                target: 3,
            },
            recent_top: 3, medial_top: 4, long_top: 4, age: 0,
            supported_windows: 0, adverse_windows: 0, evaluated: 0,
            credit_sum: 0.0,
        }
    }

    #[test]
    fn causal_dream_requires_two_future_windows() {
        let mut model = super::tests::tiny();
        model.gm1_mode = 3;
        model.medial_mode = 3;
        model.medial_pending.push(candidate());
        let episode = vec![1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3];
        let mut stats = TrainStats::default();
        validate_medial_dreams(&mut model, &episode, &options(), &mut stats);
        assert_eq!(model.medial_pending[0].supported_windows, 1);
        assert!(model.gcp.is_empty());
        validate_medial_dreams(&mut model, &episode, &options(), &mut stats);
        assert!(model.medial_pending.is_empty());
        assert!(model.gcp.contains_key(&candidate().key.context));
    }

    #[test]
    fn immediate_dream_returns_to_gcp_first() {
        let mut model = super::tests::tiny();
        model.gm1_mode = 3;
        let mut stats = TrainStats::default();
        materialize_medial_dream(&mut model, &candidate(), 1.0, &options(), &mut stats);
        assert!(model.gcp.contains_key(&candidate().key.context));
        assert!(!model.gm1.contains_key(&candidate().key.context));
        assert!(!model.glp.contains_key(&candidate().key.context));
    }

    #[test]
    fn online_wake_fills_gcp_without_mutating_glp() {
        let mut model = super::tests::tiny();
        let before = glp_fingerprint(&model);
        online_warm_gcp(&mut model, &[1, 2, 3, 1, 2, 3], &options());
        assert!(!model.gcp.is_empty());
        assert_eq!(before, glp_fingerprint(&model));
    }

    #[test]
    fn semantic_only_mode_clears_composed_fallback() {
        let mut model = super::tests::tiny();
        model.gm1_mode = 3;
        model.medial_mode = 1;
        let key = candidate().key.context;
        model.gm1_cache.insert(key, Node::new(2));
        let mut stats = TrainStats::default();
        deep_reflection(&mut model, &options(), &mut stats);
        assert!(model.gm1_cache.is_empty());
    }
}

#[cfg(test)]
mod exp154_tests {
    use super::*;

    #[test]
    fn repeated_context_requires_serial_replay() {
        let tokens = [1u16, 2, 1, 3];
        let mut history = Vec::new();
        let events = build_spec_events(&tokens, 0, tokens.len(), 4, &mut history);
        assert_eq!(events[0].history_len, 0);
        assert!(events[1].speculative);
        assert!(events[2].speculative);
        assert!(!events[3].speculative);
    }

    #[test]
    fn document_boundary_resets_speculative_history() {
        let tokens = [5u16, 6, 0, 7];
        let mut history = Vec::new();
        let events = build_spec_events(&tokens, 0, tokens.len(), 4, &mut history);
        assert_eq!(events[0].history_len, 0);
        assert_eq!(events[2].history_len, 0);
        assert_eq!(events[3].history(), &[0]);
        assert!(events[3].speculative);
    }
}
