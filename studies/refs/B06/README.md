# B06 two-pulse echo (self-consistent VP + external impulsive kick)
k1=1.0 (seed density 0.01 at t=0), k2=1.5 (kick f(x,v)->f(x,v-0.05 cos k2 x) at tau=10), echo at k3=0.5; ballistic estimate t=k2 tau/k3=30.
Grid (Nx=64; Nx=32 aliased the k1 tail at 1e-7, echo unaffected): echo peak |E_k3| = 1.15873e-3 at t=29.05 (converged Nv 512/1024, dt 0.025/0.0125); half kick -> 0.511x amplitude (linear in kick).
SPECTRAX (Nx=33, kick by exact truncated-basis shift expm(sR) at restart): N=64: t=28.4, amp -21%, curve dev 56% of echo peak; N=128: 29.2, -7.4%, 22%;
N=256: 29.05, +0.0002%, 8% (post-echo recurrence); N=512: 29.05, 0.0019%, 2.9e-4. k1-mode Hermite recurrence (spurious |E_k1| up to 4e-3) at t≈15 (N=64), 17 (128), 27 (256), 38 (512).
Phase-mixing index of k1 at tau: (k1 tau)^2 = 100 -> N must exceed a few hundred.
