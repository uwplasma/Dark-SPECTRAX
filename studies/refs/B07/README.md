# B07 free streaming, force off (qs=0), k lambda_D=0.5
Exact: C_n(t)=C_0(0)(-ib)^n e^{-b^2/4}/sqrt(2^n n!), b=sqrt2 k v_t t, i.e. |g_n|^2 ∝ (k v_t t)^{2n}/n! e^{-(k v_t t)^2} (parent alpha = sqrt2 v_t).
For t < 0.5 sqrt(N)/(k v_t) the full coefficient vector matches to <=2.5e-8 C_0(0) for N>=64 (5e-6 at N=32, 9e-4 at N=16).
First low-mode (n<=3) contamination > 1e-6 C_0(0): N=16,32,64,128,256,512 -> t_c = 5.65, 11.7, 20.7, 33.7, 52.3, 78.7;
t_c k v_t/sqrt(N) = 0.71, 1.03, 1.29, 1.49, 1.63, 1.74; power fit t_c ∝ N^0.75 (not N^0.5) over this range.
