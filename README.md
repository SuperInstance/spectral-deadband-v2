# spectral-deadband-v2

> Spectral deadband analysis for the SuperInstance fleet — connecting thermostat deadbands to graph Laplacian spectral gaps.

[![Part of SuperInstance](https://img.shields.io/badge/Part%20of-SuperInstance-blue)](https://github.com/SuperInstance)

Part of the **SuperInstance** fleet.

## Overview

Experimental refinement of the deadband concept inspired by the insight: *"the thermostat deadband IS the spectral gap."*

A deadband is a symmetric interval around a center value — information within the band is absorbed, information above it propagates. By connecting deadband width to the spectral gap of a graph Laplacian, this crate provides a principled way to decide what signal content matters.

## Core Types

### `Deadband`

A symmetric interval `[center − width/2, center + width/2]` with operations:

- **`contains(value)`** — Is the value inside the deadband?
- **`passes(new, last)`** — Did the signal break through the band?
- **`quantize(value)`** — Snap to nearest boundary or center

## Installation

```bash
cargo add spectral-deadband-v2
```

Or build from source:

```bash
git clone https://github.com/SuperInstance/spectral-deadband-v2.git
cd spectral-deadband-v2
cargo build --release
```

## Usage

```rust
use spectral_deadband_v2::Deadband;

// Create a deadband centered at 0 with width 2.0
let db = Deadband::new(0.0, 2.0);

assert!(db.contains(0.5));    // inside the band
assert!(!db.contains(2.5));   // outside the band
assert!(db.passes(3.0, 0.0)); // signal broke through
assert_eq!(db.quantize(0.5), 0.0);  // snaps to center
```

## License

MIT
