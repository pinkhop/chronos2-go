//go:build goexperiment.simd && (amd64 || arm64)

package kernel

// Constants of the vector exponential. The range reduction writes
// x = n·ln 2 + r with n = round(x·log₂e) and |r| ≤ ln 2 / 2, subtracting
// n·ln 2 in two parts (Cody–Waite): expLn2High has few enough significant
// bits that n·expLn2High is exact for every n the clamp allows, and
// expLn2Low carries the rest of ln 2. The polynomial is Cephes' expf
// minimax fit of (eʳ − 1 − r) / r² on that interval. The NEON and AVX2
// exponentials share these constants.
const (
	// expLog2E is log₂ e, which turns x into a count of powers of two.
	expLog2E = 1.44269504088896341
	// expLn2High is the leading part of ln 2.
	expLn2High = 0.693359375
	// expLn2Low is ln 2 − expLn2High.
	expLn2Low = -2.12194440e-4
	// expCoefficient0 is the polynomial's constant term, close to the
	// Taylor coefficient 1/2!.
	expCoefficient0 = 5.0000001201e-1
	// expCoefficient1 multiplies r, close to the Taylor coefficient 1/3!.
	expCoefficient1 = 1.6666665459e-1
	// expCoefficient2 multiplies r², close to the Taylor coefficient 1/4!.
	expCoefficient2 = 4.1665795894e-2
	// expCoefficient3 multiplies r³, close to the Taylor coefficient 1/5!.
	expCoefficient3 = 8.3334519073e-3
	// expCoefficient4 multiplies r⁴, close to the Taylor coefficient 1/6!.
	expCoefficient4 = 1.3981999507e-3
	// expCoefficient5 multiplies r⁵, close to the Taylor coefficient 1/7!.
	expCoefficient5 = 1.9875691500e-4
	// expUnderflow is ln 2⁻¹⁵⁰: below it, eˣ is under half the smallest
	// subnormal float32 and rounds to zero.
	expUnderflow = -103.972077083991796
	// expClamp bounds x from below so n stays at least -150 and the scale
	// factors stay normal; lanes below expUnderflow are zeroed afterward.
	expClamp = -104
	// expExponentBias is added to an integer exponent to form a float32's
	// biased exponent field.
	expExponentBias = 127
	// expMantissaBits is the width of a float32's significand field, the
	// shift that moves a biased exponent into place.
	expMantissaBits = 23
)
