| SW run | vs grid | dK_e | W_ext | field spectrum |
|---|---|---|---|---|
| sw_vq0.1_Nx16_Nn128_dt0.02_T1000 | g_vq0.1_Nx64_Nv16384 | >=703.5 | >=703.5 | 461.5 (k3) |
| sw_vq0.1_Nx16_Nn128_dt0.02_T1000 | g_vq0.1_Nx32_Nv16384 | >=703.5 | >=703.5 | 461.5 (k3) |
| sw_vq0.1_Nx16_Nn64_dt0.02_T1000 | g_vq0.1_Nx64_Nv16384 | 634.5 | 637.5 | 445.5 (k3) |
| sw_vq0.1_Nx16_Nn64_dt0.02_T1000 | g_vq0.1_Nx32_Nv16384 | 634.5 | 637.5 | 445.5 (k3) |
| sw_vq0.1_Nx32_Nn128_dt0.02_T1000 | g_vq0.1_Nx64_Nv16384 | 737.5 | >=738 | 619.5 (k4) |
| sw_vq0.1_Nx32_Nn128_dt0.02_T1000 | g_vq0.1_Nx32_Nv16384 | 737.5 | >=738 | 633 (k4) |
| sw_vq0.1_Nx32_Nn256_dt0.02_T1000 | g_vq0.1_Nx64_Nv16384 | >=685 | >=685 | 622.5 (k4) |
| sw_vq0.1_Nx32_Nn256_dt0.02_T1000 | g_vq0.1_Nx32_Nv16384 | >=685 | >=685 | 638 (k5) |
| sw_vq0.1_Nx32_Nn64_dt0.02_T1000 | g_vq0.1_Nx64_Nv16384 | 681.0 | >=681.5 | 557 (k4) |
| sw_vq0.1_Nx32_Nn64_dt0.02_T1000 | g_vq0.1_Nx32_Nv16384 | 681.0 | >=681.5 | 557 (k4) |

| SW pair (reference first) | dK_e | W_ext | field spectrum |
|---|---|---|---|
| Nx32_Nn128 vs Nx32_Nn64 | 680.5 | >=681.5 | 557 (k4) |
| Nx32_Nn256 vs Nx32_Nn128 | >=685 | >=685 | 633.5 (k4) |
| Nx16_Nn128 vs Nx16_Nn64 | 634.0 | 637.5 | 456 (k4) |
| Nx32_Nn64 vs Nx16_Nn64 | 634.0 | 637.5 | 445.5 (k3) |
| Nx32_Nn128 vs Nx16_Nn128 | >=703.5 | >=703.5 | 464 (k3) |

| SW run | max ledger/W (50<t<=600) | first ledger/W > 1e-3 | max mass drift e | max momentum drift (t<=600) | max L2 drift | max top-quarter frac | wall s | RHS evals |
|---|---|---|---|---|---|---|---|---|
| sw_vq0.1_Nx16_Nn128_dt0.02_T1000 | 4.1e-04 | 629.0 | nan | 2.5e-04 | nan | nan | 44 | 140700 |
| sw_vq0.1_Nx16_Nn64_dt0.02_T1000 | 1.2e-03 | 590.5 | nan | 1.1e-03 | nan | nan | 18 | 127600 |
| sw_vq0.1_Nx32_Nn128_dt0.02_T1000 | 7.9e-06 | 704.5 | nan | 1.7e-06 | nan | nan | 57 | 147600 |
| sw_vq0.1_Nx32_Nn256_dt0.02_T1000 | 7.9e-06 | 682.0 | nan | 3.8e-08 | nan | nan | 179 | 137000 |
| sw_vq0.1_Nx32_Nn64_dt0.02_T1000 | 1.9e-05 | 658.0 | nan | 1.5e-05 | nan | nan | 23 | 136300 |

AW c=4 cost: [{"case": "vq0.1_Nx32_Nn64_c4_nf1e-14", "T": 820.0, "steps": 35619, "rejected": 13663, "run_time": 67.55672412470449, "wall_time": 115.8456641669618}, {"case": "vq0.1_Nx64_Nn64_c4_nf1e-14", "T": 760.0, "steps": 28022, "rejected": 11689, "run_time": 87.5358027057955, "wall_time": 128.92536954197567}]
{
 "G3_t_vs_grid": {
  "sw_vq0.1_Nx16_Nn128_dt0.02_T1000": 703.5,
  "sw_vq0.1_Nx16_Nn64_dt0.02_T1000": 634.5,
  "sw_vq0.1_Nx32_Nn128_dt0.02_T1000": 737.5,
  "sw_vq0.1_Nx32_Nn256_dt0.02_T1000": 685.0,
  "sw_vq0.1_Nx32_Nn64_dt0.02_T1000": 681.0
 },
 "G3_pairs_min": 634.0,
 "G3_pass": false,
 "G3_reject_before_650": true,
 "G4_max": {
  "sw_vq0.1_Nx16_Nn128_dt0.02_T1000": 0.00040729988675480947,
  "sw_vq0.1_Nx16_Nn64_dt0.02_T1000": 0.0011569920769267889,
  "sw_vq0.1_Nx32_Nn128_dt0.02_T1000": 7.866735992198566e-06,
  "sw_vq0.1_Nx32_Nn256_dt0.02_T1000": 7.866735992198566e-06,
  "sw_vq0.1_Nx32_Nn64_dt0.02_T1000": 1.8506696506832714e-05
 },
 "G4_pass": false
}
