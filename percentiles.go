package chronos2

// The named percentile accessors cover every native level of the base model.
// Each returns a copy of the native line, or nil when the producing model
// does not predict that level natively; a native line always holds at least
// one step, so nil never means an empty result. Code that must work with any
// model checks for nil or calls Percentile instead.

// Percentile1 returns the 1st-percentile line, or nil when the producing
// model does not predict that level natively. Both published models predict it natively.
func (f Forecast) Percentile1() []float64 { return f.native(1) }

// Percentile5 returns the 5th-percentile line, or nil when the producing
// model does not predict that level natively. Both published models predict it natively.
func (f Forecast) Percentile5() []float64 { return f.native(5) }

// Percentile10 returns the 10th-percentile line, or nil when the producing
// model does not predict that level natively. Both published models predict it natively.
func (f Forecast) Percentile10() []float64 { return f.native(10) }

// Percentile15 returns the 15th-percentile line, or nil when the producing
// model does not predict that level natively. The small model does not predict it, so it returns nil there.
func (f Forecast) Percentile15() []float64 { return f.native(15) }

// Percentile20 returns the 20th-percentile line, or nil when the producing
// model does not predict that level natively. Both published models predict it natively.
func (f Forecast) Percentile20() []float64 { return f.native(20) }

// Percentile25 returns the 25th-percentile line, or nil when the producing
// model does not predict that level natively. The small model does not predict it, so it returns nil there.
func (f Forecast) Percentile25() []float64 { return f.native(25) }

// Percentile30 returns the 30th-percentile line, or nil when the producing
// model does not predict that level natively. Both published models predict it natively.
func (f Forecast) Percentile30() []float64 { return f.native(30) }

// Percentile35 returns the 35th-percentile line, or nil when the producing
// model does not predict that level natively. The small model does not predict it, so it returns nil there.
func (f Forecast) Percentile35() []float64 { return f.native(35) }

// Percentile40 returns the 40th-percentile line, or nil when the producing
// model does not predict that level natively. Both published models predict it natively.
func (f Forecast) Percentile40() []float64 { return f.native(40) }

// Percentile45 returns the 45th-percentile line, or nil when the producing
// model does not predict that level natively. The small model does not predict it, so it returns nil there.
func (f Forecast) Percentile45() []float64 { return f.native(45) }

// Percentile50 returns the 50th-percentile line, or nil when the producing
// model does not predict that level natively. Both published models predict it natively.
func (f Forecast) Percentile50() []float64 { return f.native(50) }

// Percentile55 returns the 55th-percentile line, or nil when the producing
// model does not predict that level natively. The small model does not predict it, so it returns nil there.
func (f Forecast) Percentile55() []float64 { return f.native(55) }

// Percentile60 returns the 60th-percentile line, or nil when the producing
// model does not predict that level natively. Both published models predict it natively.
func (f Forecast) Percentile60() []float64 { return f.native(60) }

// Percentile65 returns the 65th-percentile line, or nil when the producing
// model does not predict that level natively. The small model does not predict it, so it returns nil there.
func (f Forecast) Percentile65() []float64 { return f.native(65) }

// Percentile70 returns the 70th-percentile line, or nil when the producing
// model does not predict that level natively. Both published models predict it natively.
func (f Forecast) Percentile70() []float64 { return f.native(70) }

// Percentile75 returns the 75th-percentile line, or nil when the producing
// model does not predict that level natively. The small model does not predict it, so it returns nil there.
func (f Forecast) Percentile75() []float64 { return f.native(75) }

// Percentile80 returns the 80th-percentile line, or nil when the producing
// model does not predict that level natively. Both published models predict it natively.
func (f Forecast) Percentile80() []float64 { return f.native(80) }

// Percentile85 returns the 85th-percentile line, or nil when the producing
// model does not predict that level natively. The small model does not predict it, so it returns nil there.
func (f Forecast) Percentile85() []float64 { return f.native(85) }

// Percentile90 returns the 90th-percentile line, or nil when the producing
// model does not predict that level natively. Both published models predict it natively.
func (f Forecast) Percentile90() []float64 { return f.native(90) }

// Percentile95 returns the 95th-percentile line, or nil when the producing
// model does not predict that level natively. Both published models predict it natively.
func (f Forecast) Percentile95() []float64 { return f.native(95) }

// Percentile99 returns the 99th-percentile line, or nil when the producing
// model does not predict that level natively. Both published models predict it natively.
func (f Forecast) Percentile99() []float64 { return f.native(99) }
