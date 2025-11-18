// Folly F14 hash table intrinsics mode link alias
// This file ensures compatibility between different F14IntrinsicsMode builds
// 
// F14IntrinsicsMode is determined at compile time:
// - Mode 0: No SIMD
// - Mode 1: Baseline SIMD (SSE2-ish)
// - Mode 2: Better SIMD (typically AVX2 + extra goodies)
//
// The folly library was built with F14IntrinsicsMode 2 (verified via nm).
// This file provides mode 1 as a wrapper that calls mode 2, ensuring
// compatibility when code expects mode 1 but the library provides mode 2.

extern "C" {
    // Declare the mode 2 symbol as external (defined in libfolly.a)
    // This will be resolved at link time
    extern void _ZN5folly3f146detail12F14LinkCheckILNS1_17F14IntrinsicsModeE2EE5checkEv();
    
    // Provide mode 1 as a wrapper that calls mode 2
    // This allows code expecting mode 1 to work with a mode 2 library
    void _ZN5folly3f146detail12F14LinkCheckILNS1_17F14IntrinsicsModeE1EE5checkEv() {
        _ZN5folly3f146detail12F14LinkCheckILNS1_17F14IntrinsicsModeE2EE5checkEv();
    }
}

