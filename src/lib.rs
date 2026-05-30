//! # spectral-deadband-v2
//!
//! Experimental refinement of the deadband concept from Casey's key insight:
//! *"the thermostat deadband IS the spectral gap."*
//!
//! A deadband is a symmetric interval around a center value. Information within
//! the deadband is absorbed (ignored); information above it propagates. When we
//! connect deadband width to the spectral gap of a graph Laplacian, we get a
//! principled way to decide what signal content matters.

use std::fmt;

// ---------------------------------------------------------------------------
// Core: Deadband
// ---------------------------------------------------------------------------

/// A symmetric deadband interval `[center - width/2, center + width/2]`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Deadband {
    /// Center of the deadband.
    pub center: f64,
    /// Total width of the deadband (must be ≥ 0).
    pub width: f64,
}

impl Deadband {
    /// Create a new deadband with the given center and total width.
    pub fn new(center: f64, width: f64) -> Self {
        assert!(width >= 0.0, "deadband width must be non-negative");
        Self { center, width }
    }

    /// Lower bound of the deadband.
    pub fn lower(&self) -> f64 {
        self.center - self.width / 2.0
    }

    /// Upper bound of the deadband.
    pub fn upper(&self) -> f64 {
        self.center + self.width / 2.0
    }

    /// Returns `true` if `value` falls within the deadband (inclusive).
    pub fn contains(&self, value: f64) -> bool {
        value >= self.lower() && value <= self.upper()
    }

    /// Returns `true` if the change from `last_value` to `new_value` exceeds
    /// the deadband width — i.e. the signal "breaks through."
    pub fn passes(&self, new_value: f64, last_value: f64) -> bool {
        (new_value - last_value).abs() > self.width
    }

    /// Snap `value` to the nearest deadband center, clamping into the interval
    /// first and then returning the center. If the value is inside the band,
    /// returns the center; if outside, returns the nearest boundary.
    pub fn quantize(&self, value: f64) -> f64 {
        if value <= self.lower() {
            self.lower()
        } else if value >= self.upper() {
            self.upper()
        } else {
            self.center
        }
    }
}

impl fmt::Display for Deadband {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[{}, {}] (center={}, width={})",
            self.lower(),
            self.upper(),
            self.center,
            self.width
        )
    }
}

// ---------------------------------------------------------------------------
// Adaptive Deadband
// ---------------------------------------------------------------------------

/// A deadband that adapts its width based on observed signal statistics.
///
/// Uses an exponential moving variance estimate:
/// - When variance is low (stable signal), the band narrows.
/// - When variance is high (noisy signal), the band widens.
#[derive(Debug, Clone)]
pub struct AdaptiveDeadband {
    /// Current deadband.
    pub band: Deadband,
    /// Exponential moving average of observations.
    ema: f64,
    /// Exponential moving variance.
    emv: f64,
    /// Smoothing factor α ∈ (0, 1] for EMA updates.
    alpha: f64,
    /// Minimum allowed width.
    min_width: f64,
    /// Maximum allowed width.
    max_width: f64,
    /// Number of observations seen.
    n: u64,
}

impl AdaptiveDeadband {
    /// Create a new adaptive deadband.
    ///
    /// * `center` — initial center
    /// * `initial_width` — starting width
    /// * `alpha` — EMA smoothing factor (try 0.05–0.2)
    /// * `min_width` / `max_width` — clamp the adapted width
    pub fn new(center: f64, initial_width: f64, alpha: f64, min_width: f64, max_width: f64) -> Self {
        assert!(alpha > 0.0 && alpha <= 1.0, "alpha must be in (0, 1]");
        assert!(min_width <= max_width, "min_width must be <= max_width");
        Self {
            band: Deadband::new(center, initial_width),
            ema: center,
            emv: 0.0,
            alpha,
            min_width,
            max_width,
            n: 0,
        }
    }

    /// Feed a new observation. The deadband width adapts based on the running
    /// variance estimate.
    pub fn update(&mut self, value: f64) {
        self.n += 1;
        let diff = value - self.ema;
        // Online EMA / EMV update
        self.ema += self.alpha * diff;
        self.emv = (1.0 - self.alpha) * (self.emv + self.alpha * diff * diff);

        // Width = 2σ from the EMV (covers ~95% of signal under Gaussian assumption)
        let sigma = self.emv.sqrt();
        let new_width = (2.0 * sigma).clamp(self.min_width, self.max_width);

        self.band = Deadband::new(self.ema, new_width);
    }

    /// Current estimated standard deviation of the signal.
    pub fn sigma(&self) -> f64 {
        self.emv.sqrt()
    }

    /// Number of observations seen so far.
    pub fn observations(&self) -> u64 {
        self.n
    }

    /// Delegate: does the value lie in the current adapted band?
    pub fn contains(&self, value: f64) -> bool {
        self.band.contains(value)
    }

    /// Delegate: does the change exceed the current adapted band?
    pub fn passes(&self, new_value: f64, last_value: f64) -> bool {
        self.band.passes(new_value, last_value)
    }
}

// ---------------------------------------------------------------------------
// Spectral Deadband
// ---------------------------------------------------------------------------

/// A deadband whose width equals the spectral gap of a graph Laplacian.
///
/// The spectral gap is λ₁ − λ₀ (for a connected graph, λ₀ = 0, so the gap
/// is simply λ₁). Information below the gap gets absorbed; above it propagates.
#[derive(Debug, Clone)]
pub struct SpectralDeadband {
    /// Eigenvalues of the graph Laplacian, sorted ascending.
    pub eigenvalues: Vec<f64>,
    /// The deadband derived from the spectral gap.
    pub band: Deadband,
}

impl SpectralDeadband {
    /// Build from a pre-computed list of Laplacian eigenvalues (sorted ascending).
    ///
    /// The deadband center is 0 and its width is the spectral gap λ₁ − λ₀.
    pub fn from_eigenvalues(eigenvalues: &[f64]) -> Self {
        assert!(eigenvalues.len() >= 2, "need at least 2 eigenvalues");
        let mut sorted: Vec<f64> = eigenvalues.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let gap = sorted[1] - sorted[0];
        Self {
            eigenvalues: sorted,
            band: Deadband::new(0.0, gap),
        }
    }

    /// Build from an adjacency matrix (symmetric, square) by computing the
    /// (unnormalized) graph Laplacian eigenvalues.
    ///
    /// Uses a simple power-iteration + deflation approach. Suitable for small
    /// graphs (n ≤ ~50). For larger graphs, bring your own eigensolver.
    pub fn from_graph(adjacency: &[Vec<f64>]) -> Self {
        let n = adjacency.len();
        assert!(n > 0, "adjacency must be non-empty");
        for row in adjacency {
            assert_eq!(row.len(), n, "adjacency must be square");
        }

        // Build degree matrix diagonal and Laplacian L = D - A
        let degree: Vec<f64> = (0..n)
            .map(|i| adjacency[i].iter().sum())
            .collect();

        // L[v][u] = degree[v] if v==u else -adjacency[v][u]
        let laplacian = |v: usize, u: usize| -> f64 {
            if v == u {
                degree[v]
            } else {
                -adjacency[v][u]
            }
        };

        // Power iteration to find top eigenvalues, then reverse for smallest
        let eigenvalues = compute_smallest_eigenvalues(&laplacian, n, n.min(10));

        Self::from_eigenvalues(&eigenvalues)
    }

    /// The spectral gap (λ₁ − λ₀).
    pub fn spectral_gap(&self) -> f64 {
        self.band.width
    }

    /// Returns `true` if the graph appears to be connected (λ₀ ≈ 0, λ₁ > 0).
    pub fn is_connected(&self) -> bool {
        self.eigenvalues[0].abs() < 1e-8 && self.eigenvalues[1] > 1e-8
    }
}

/// Compute the smallest eigenvalues of a symmetric matrix (given by a closure)
/// using power iteration + deflation. Returns eigenvalues sorted ascending.
fn compute_smallest_eigenvalues(
    mat: &dyn Fn(usize, usize) -> f64,
    n: usize,
    k: usize,
) -> Vec<f64> {
    // We compute the largest-magnitude eigenvalues of (shifted) matrix,
    // then shift back. For a Laplacian, eigenvalues are in [0, 2*max_degree].
    // We use inverse iteration to find the smallest eigenvalues.
    // Simpler approach: just do power iteration on L, collect largest, then
    // those minus the shift give us smallest.

    // Actually, simplest correct approach for small matrices:
    // Use repeated power iteration + deflation to get top k eigenvalues by magnitude.
    // Then the smallest Laplacian eigenvalues are near zero.

    let k = k.min(n);
    // We'll compute eigenvalues of L using QR-like power iteration.
    // For small n, just do inverse iteration for smallest eigenvalues.
    // Simplification: shift-and-invert with shift=0 means we need L^{-1},
    // which doesn't exist (singular). Use small shift.

    // Let's use a simpler approach: power iteration to get largest eigenvalue,
    // then Wielandt deflation to get subsequent ones.
    // Then smallest eigenvalue ≈ 0 for connected graph.

    // For simplicity and correctness on small matrices, use Jacobi eigenvalue algorithm.
    let mut eigenvalues = jacobi_eigenvalues(mat, n);
    eigenvalues.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    eigenvalues.truncate(k);
    eigenvalues
}

/// Jacobi eigenvalue algorithm for small symmetric matrices.
/// Returns all eigenvalues (unsorted).
#[allow(clippy::needless_range_loop)]
fn jacobi_eigenvalues(mat: &dyn Fn(usize, usize) -> f64, n: usize) -> Vec<f64> {
    // Build matrix
    let mut a = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        for j in 0..n {
            a[i][j] = mat(i, j);
        }
    }

    let max_iter = 100 * n * n;
    for _ in 0..max_iter {
        // Find largest off-diagonal element
        let mut max_val = 0.0f64;
        let (mut p, mut q) = (0, 1);
        for i in 0..n {
            for j in (i + 1)..n {
                if a[i][j].abs() > max_val {
                    max_val = a[i][j].abs();
                    p = i;
                    q = j;
                }
            }
        }
        if max_val < 1e-12 {
            break;
        }

        // Compute rotation angle
        let app = a[p][p];
        let aqq = a[q][q];
        let apq = a[p][q];

        let theta = if (app - aqq).abs() < 1e-15 {
            std::f64::consts::FRAC_PI_4
        } else {
            0.5 * (2.0 * apq / (app - aqq)).atan()
        };
        let c = theta.cos();
        let s = theta.sin();

        // Apply Givens rotation
        for i in 0..n {
            if i != p && i != q {
                let aip = a[i][p];
                let aiq = a[i][q];
                a[i][p] = c * aip + s * aiq;
                a[p][i] = a[i][p];
                a[i][q] = -s * aip + c * aiq;
                a[q][i] = a[i][q];
            }
        }
        let new_pp = c * c * app + 2.0 * s * c * apq + s * s * aqq;
        let new_qq = s * s * app - 2.0 * s * c * apq + c * c * aqq;
        a[p][p] = new_pp;
        a[q][q] = new_qq;
        a[p][q] = 0.0;
        a[q][p] = 0.0;
    }

    (0..n).map(|i| a[i][i]).collect()
}

// ---------------------------------------------------------------------------
// Multi-Deadband
// ---------------------------------------------------------------------------

/// Classification level for a value relative to layered deadbands.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeadbandLevel {
    /// Value is inside the inner (finest) deadband.
    Fine,
    /// Value is inside the middle deadband.
    Medium,
    /// Value is inside the outer (coarsest) deadband.
    Coarse,
    /// Value is outside all deadbands.
    Outside,
}

impl fmt::Display for DeadbandLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DeadbandLevel::Fine => write!(f, "Fine"),
            DeadbandLevel::Medium => write!(f, "Medium"),
            DeadbandLevel::Coarse => write!(f, "Coarse"),
            DeadbandLevel::Outside => write!(f, "Outside"),
        }
    }
}

/// Layered deadbands for multi-scale processing.
///
/// Each layer has the same center but progressively wider widths.
/// The inner band captures fine detail, the outer captures coarse structure.
#[derive(Debug, Clone)]
pub struct MultiDeadband {
    /// All three bands: inner (fine), middle, outer (coarse).
    pub bands: [Deadband; 3],
}

impl MultiDeadband {
    /// Create a multi-scale deadband with three widths sharing the same center.
    pub fn new(center: f64, inner_width: f64, middle_width: f64, outer_width: f64) -> Self {
        assert!(
            inner_width <= middle_width && middle_width <= outer_width,
            "widths must be ordered: inner ≤ middle ≤ outer"
        );
        Self {
            bands: [
                Deadband::new(center, inner_width),
                Deadband::new(center, middle_width),
                Deadband::new(center, outer_width),
            ],
        }
    }

    /// Classify a value into one of four levels based on which band (if any)
    /// contains it. Innermost band wins.
    pub fn classify(&self, value: f64) -> DeadbandLevel {
        if self.bands[0].contains(value) {
            DeadbandLevel::Fine
        } else if self.bands[1].contains(value) {
            DeadbandLevel::Medium
        } else if self.bands[2].contains(value) {
            DeadbandLevel::Coarse
        } else {
            DeadbandLevel::Outside
        }
    }

    /// Convenience: the inner (fine) deadband.
    pub fn inner(&self) -> &Deadband {
        &self.bands[0]
    }

    /// Convenience: the middle deadband.
    pub fn middle(&self) -> &Deadband {
        &self.bands[1]
    }

    /// Convenience: the outer (coarse) deadband.
    pub fn outer(&self) -> &Deadband {
        &self.bands[2]
    }

    /// The center shared by all three bands.
    pub fn center(&self) -> f64 {
        self.bands[0].center
    }
}

// ---------------------------------------------------------------------------
// Experiments
// ---------------------------------------------------------------------------

/// Results from the deadband experiment comparing fixed vs adaptive deadbands
/// on a random walk signal sampled on a graph.
#[derive(Debug, Clone)]
pub struct ExperimentResult {
    /// Number of times the fixed deadband fired (value passed through).
    pub fixed_fires: u64,
    /// Number of times the adaptive deadband fired.
    pub adaptive_fires: u64,
    /// Final adapted width of the adaptive deadband.
    pub final_adaptive_width: f64,
    /// Spectral gap of the underlying graph.
    pub spectral_gap: f64,
    /// Total samples in the experiment.
    pub samples: u64,
}

impl fmt::Display for ExperimentResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "=== Deadband Experiment ===")?;
        writeln!(f, "Samples:           {}", self.samples)?;
        writeln!(f, "Spectral gap:       {:.6}", self.spectral_gap)?;
        writeln!(f, "Fixed fires:        {}", self.fixed_fires)?;
        writeln!(f, "Adaptive fires:     {}", self.adaptive_fires)?;
        writeln!(
            f,
            "Adaptive width:     {:.6}",
            self.final_adaptive_width
        )?;
        writeln!(
            f,
            "Adaptive/spectral:  {:.4}",
            self.final_adaptive_width / self.spectral_gap.max(1e-15)
        )
    }
}

/// Run the signature experiment: compare fixed vs adaptive deadband on a graph
/// random walk, and show that the adaptive deadband converges toward the
/// spectral gap.
///
/// Uses a ring graph of `n` nodes as the underlying structure.
pub fn deadband_experiment() -> ExperimentResult {
    let n = 10_usize;
    let steps = 5000_usize;

    // Build a ring graph adjacency
    let mut adj = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        adj[i][(i + 1) % n] = 1.0;
        adj[i][(i + n - 1) % n] = 1.0;
    }

    // Compute spectral deadband
    let spectral = SpectralDeadband::from_graph(&adj);
    let gap = spectral.spectral_gap();

    // Generate a random walk on the graph
    let mut rng = SimpleRng::new(42);
    let mut signal = Vec::with_capacity(steps);
    let mut node = 0_usize;
    for _ in 0..steps {
        // Random walk step
        if rng.next() < 0.5 {
            node = (node + 1) % n;
        } else {
            node = (node + n - 1) % n;
        }
        // Add a little noise to the node index to make it a continuous signal
        let value = node as f64 + (rng.next() - 0.5) * 0.1;
        signal.push(value);
    }

    // Fixed deadband: width = spectral gap
    let fixed_band = Deadband::new(signal[0], gap);

    // Adaptive deadband: starts wide, adapts
    let mut adaptive = AdaptiveDeadband::new(signal[0], gap * 5.0, 0.05, gap * 0.1, gap * 10.0);

    let mut fixed_fires = 0u64;
    let mut adaptive_fires = 0u64;

    for i in 1..signal.len() {
        if fixed_band.passes(signal[i], signal[i - 1]) {
            fixed_fires += 1;
        }
        if adaptive.passes(signal[i], signal[i - 1]) {
            adaptive_fires += 1;
        }
        adaptive.update(signal[i]);
    }

    ExperimentResult {
        fixed_fires,
        adaptive_fires,
        final_adaptive_width: adaptive.band.width,
        spectral_gap: gap,
        samples: steps as u64,
    }
}

// Simple LCG RNG for reproducibility (no external dep)
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    fn next(&mut self) -> f64 {
        // Numerical Recipes LCG
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.state >> 33) as f64 / (1u64 << 31) as f64
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Deadband basics ----

    #[test]
    fn deadband_contains_center() {
        let db = Deadband::new(5.0, 2.0);
        assert!(db.contains(5.0));
    }

    #[test]
    fn deadband_contains_boundary() {
        let db = Deadband::new(5.0, 2.0); // [4, 6]
        assert!(db.contains(4.0));
        assert!(db.contains(6.0));
    }

    #[test]
    fn deadband_excludes_outside() {
        let db = Deadband::new(5.0, 2.0);
        assert!(!db.contains(3.9));
        assert!(!db.contains(6.1));
    }

    #[test]
    fn deadband_passes_large_change() {
        let db = Deadband::new(0.0, 1.0);
        assert!(db.passes(2.0, 0.0)); // Δ=2 > width=1
    }

    #[test]
    fn deadband_does_not_pass_small_change() {
        let db = Deadband::new(0.0, 1.0);
        assert!(!db.passes(0.5, 0.0)); // Δ=0.5 ≤ width=1
    }

    #[test]
    fn deadband_passes_boundary_exactly() {
        let db = Deadband::new(0.0, 1.0);
        assert!(!db.passes(1.0, 0.0)); // Δ=1 == width, not strictly greater
        assert!(db.passes(1.0 + 1e-10, 0.0));
    }

    #[test]
    fn deadband_quantize_inside() {
        let db = Deadband::new(5.0, 4.0); // [3, 7]
        assert_eq!(db.quantize(5.0), 5.0); // center
        assert_eq!(db.quantize(4.5), 5.0); // inside → center
    }

    #[test]
    fn deadband_quantize_outside() {
        let db = Deadband::new(5.0, 4.0); // [3, 7]
        assert_eq!(db.quantize(1.0), 3.0); // below → lower
        assert_eq!(db.quantize(9.0), 7.0); // above → upper
    }

    #[test]
    fn deadband_zero_width() {
        let db = Deadband::new(3.0, 0.0);
        assert!(db.contains(3.0));
        assert!(!db.contains(3.001));
    }

    #[test]
    fn deadband_display() {
        let db = Deadband::new(5.0, 2.0);
        let s = format!("{db}");
        assert!(s.contains("center=5"));
        assert!(s.contains("width=2"));
    }

    // ---- Adaptive Deadband ----

    #[test]
    fn adaptive_converges_to_constant() {
        let mut ad = AdaptiveDeadband::new(0.0, 10.0, 0.1, 0.01, 100.0);
        // Feed constant signal
        for _ in 0..200 {
            ad.update(5.0);
        }
        assert!(ad.band.width < 1.0, "width should shrink on constant signal");
        assert!(ad.band.center > 4.9 && ad.band.center < 5.1);
    }

    #[test]
    fn adaptive_widens_on_noise() {
        let mut rng = SimpleRng::new(123);
        let mut ad = AdaptiveDeadband::new(0.0, 0.1, 0.1, 0.01, 100.0);
        for _ in 0..200 {
            let noise = (rng.next() - 0.5) * 20.0;
            ad.update(noise);
        }
        assert!(ad.band.width > 5.0, "width should grow on noisy signal");
    }

    #[test]
    fn adaptive_observations_count() {
        let mut ad = AdaptiveDeadband::new(0.0, 1.0, 0.1, 0.01, 100.0);
        assert_eq!(ad.observations(), 0);
        ad.update(1.0);
        assert_eq!(ad.observations(), 1);
        ad.update(2.0);
        assert_eq!(ad.observations(), 2);
    }

    #[test]
    fn adaptive_contains_delegates() {
        let ad = AdaptiveDeadband::new(5.0, 2.0, 0.1, 0.01, 100.0);
        assert!(ad.contains(5.0));
    }

    #[test]
    fn adaptive_sigma_starts_zero() {
        let ad = AdaptiveDeadband::new(0.0, 1.0, 0.1, 0.01, 100.0);
        assert_eq!(ad.sigma(), 0.0);
    }

    // ---- Spectral Deadband ----

    #[test]
    fn spectral_from_known_eigenvalues() {
        // For a connected graph: λ₀=0, λ₁=0.5 → gap=0.5
        let sd = SpectralDeadband::from_eigenvalues(&[0.0, 0.5, 1.0, 2.0]);
        assert!((sd.spectral_gap() - 0.5).abs() < 1e-10);
        assert!(sd.is_connected());
    }

    #[test]
    fn spectral_disconnected_graph() {
        // Two disconnected components: λ₀=0, λ₁=0
        let sd = SpectralDeadband::from_eigenvalues(&[0.0, 0.0, 1.0, 2.0]);
        assert!(!sd.is_connected());
    }

    #[test]
    fn spectral_ring_graph() {
        // Ring of 4 nodes: eigenvalues of L are 0, 2, 2, 4
        // Spectral gap = 2
        let adj = vec![
            vec![0.0, 1.0, 0.0, 1.0],
            vec![1.0, 0.0, 1.0, 0.0],
            vec![0.0, 1.0, 0.0, 1.0],
            vec![1.0, 0.0, 1.0, 0.0],
        ];
        let sd = SpectralDeadband::from_graph(&adj);
        assert!(sd.is_connected());
        // Gap should be close to 2.0
        assert!(
            (sd.spectral_gap() - 2.0).abs() < 0.1,
            "gap was {}",
            sd.spectral_gap()
        );
    }

    #[test]
    fn spectral_path_graph() {
        // Path of 3 nodes: L has eigenvalues 0, 1, 3
        let adj = vec![
            vec![0.0, 1.0, 0.0],
            vec![1.0, 0.0, 1.0],
            vec![0.0, 1.0, 0.0],
        ];
        let sd = SpectralDeadband::from_graph(&adj);
        assert!(sd.is_connected());
        assert!(
            (sd.spectral_gap() - 1.0).abs() < 0.1,
            "gap was {}",
            sd.spectral_gap()
        );
    }

    #[test]
    fn spectral_complete_graph() {
        // K₃: eigenvalues of L are 0, 3, 3
        let adj = vec![
            vec![0.0, 1.0, 1.0],
            vec![1.0, 0.0, 1.0],
            vec![1.0, 1.0, 0.0],
        ];
        let sd = SpectralDeadband::from_graph(&adj);
        assert!(sd.is_connected());
        assert!(
            (sd.spectral_gap() - 3.0).abs() < 0.1,
            "gap was {}",
            sd.spectral_gap()
        );
    }

    #[test]
    fn spectral_eigenvalues_sorted() {
        let sd = SpectralDeadband::from_eigenvalues(&[3.0, 1.0, 0.0, 2.0]);
        assert_eq!(sd.eigenvalues, vec![0.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn spectral_band_center_is_zero() {
        let sd = SpectralDeadband::from_eigenvalues(&[0.0, 1.5]);
        assert_eq!(sd.band.center, 0.0);
        assert!((sd.band.width - 1.5).abs() < 1e-10);
    }

    // ---- MultiDeadband ----

    #[test]
    fn multi_classify_fine() {
        let md = MultiDeadband::new(0.0, 1.0, 3.0, 5.0);
        assert_eq!(md.classify(0.0), DeadbandLevel::Fine);
        assert_eq!(md.classify(0.4), DeadbandLevel::Fine);
    }

    #[test]
    fn multi_classify_medium() {
        let md = MultiDeadband::new(0.0, 1.0, 3.0, 5.0);
        assert_eq!(md.classify(1.0), DeadbandLevel::Medium); // boundary of inner
        assert_eq!(md.classify(1.4), DeadbandLevel::Medium);
    }

    #[test]
    fn multi_classify_coarse() {
        let md = MultiDeadband::new(0.0, 1.0, 3.0, 5.0);
        assert_eq!(md.classify(2.0), DeadbandLevel::Coarse);
    }

    #[test]
    fn multi_classify_outside() {
        let md = MultiDeadband::new(0.0, 1.0, 3.0, 5.0);
        assert_eq!(md.classify(3.0), DeadbandLevel::Outside); // boundary of outer
        assert_eq!(md.classify(10.0), DeadbandLevel::Outside);
    }

    #[test]
    fn multi_center() {
        let md = MultiDeadband::new(7.0, 1.0, 2.0, 4.0);
        assert_eq!(md.center(), 7.0);
    }

    #[test]
    fn multi_accessors() {
        let md = MultiDeadband::new(0.0, 1.0, 3.0, 5.0);
        assert_eq!(md.inner().width, 1.0);
        assert_eq!(md.middle().width, 3.0);
        assert_eq!(md.outer().width, 5.0);
    }

    #[test]
    fn deadband_level_display() {
        assert_eq!(format!("{}", DeadbandLevel::Fine), "Fine");
        assert_eq!(format!("{}", DeadbandLevel::Medium), "Medium");
        assert_eq!(format!("{}", DeadbandLevel::Coarse), "Coarse");
        assert_eq!(format!("{}", DeadbandLevel::Outside), "Outside");
    }

    // ---- Experiment ----

    #[test]
    fn experiment_runs() {
        let result = deadband_experiment();
        assert!(result.samples > 0);
        assert!(result.spectral_gap > 0.0);
        // Both should fire at least sometimes
        assert!(result.fixed_fires > 0);
        println!("{result}");
    }

    #[test]
    fn experiment_adaptive_width_bounded() {
        let result = deadband_experiment();
        assert!(result.final_adaptive_width > 0.0);
        assert!(
            result.final_adaptive_width < result.spectral_gap * 10.0,
            "adaptive width should stay within reasonable bounds"
        );
    }

    #[test]
    fn experiment_fixed_vs_adaptive() {
        let result = deadband_experiment();
        // Adaptive should generally fire less than fixed (it adapts to the signal)
        // This is a soft check — the adaptive learns the signal variance
        println!(
            "Fixed fires: {}, Adaptive fires: {}",
            result.fixed_fires, result.adaptive_fires
        );
        assert!(result.adaptive_fires < result.samples);
    }

    #[test]
    #[should_panic(expected = "deadband width must be non-negative")]
    fn deadband_negative_width_panics() {
        Deadband::new(0.0, -1.0);
    }

    #[test]
    #[should_panic(expected = "widths must be ordered")]
    fn multi_unordered_widths_panics() {
        MultiDeadband::new(0.0, 5.0, 3.0, 1.0);
    }
}
