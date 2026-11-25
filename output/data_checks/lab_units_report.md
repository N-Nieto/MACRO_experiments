# Lab units report (aggregates only)

Groups with fewer than 10 values show the count only.

## CULPRIT-SHOCK
### Unit-code columns (value counts)
- `icu_lab_tropiunit_c_x` (sheet: laboratory_data, patient_data): {3: 428, 4: 253, 1: 5}
- `icu_lab_hstrptunit_c_x` (sheet: laboratory_data, patient_data): {3: 436, 4: 246, 1: 4}
- `icu_lab_hstrpiunit_c_x` (sheet: laboratory_data, patient_data): {3: 433, 4: 245, 1: 5, 2: 3}
- `icu_lab_lactunit_c_x` (sheet: laboratory_data, patient_data): {1: 648, 2: 28, 3: 10}
- `icu_lab_lactunit_copy_c` (sheet: patient_data): {1: 648, 2: 28, 3: 10}
- `icu_lab_creaunit_c` (sheet: laboratory_data): {3: 345, 1: 340, 2: 1}
- `icu_lab_hbunit_c` (sheet: laboratory_data): {2: 454, 1: 170, 3: 62}
- `icu_lab_hctunit_c` (sheet: laboratory_data): {1: 474, 2: 212}
- `icu_lab_wbcunit_c` (sheet: laboratory_data): {1: 444, 2: 125, 3: 117}
- `icu_lab_crpunit_c` (sheet: laboratory_data): {2: 488, 1: 198}
- `icu_lab_glcunit_c` (sheet: laboratory_data): {1: 406, 2: 280}
- `icu_lab_hstrptunit_c_y` (sheet: laboratory_data, patient_data): {3: 436, 4: 246, 1: 4}
- `icu_lab_ckunit_c` (sheet: laboratory_data): {2: 540, 1: 146}
- `icu_lab_ckmbunit_c` (sheet: laboratory_data): {2: 438, 3: 133, 1: 115}
- `icu_lab_bununit_c` (sheet: laboratory_data): {2: 371, 1: 315}
- `icu_lab_troptunit_c` (sheet: laboratory_data): {3: 367, 4: 282, 2: 30, 1: 7}
- `icu_lab_tropiunit_c_y` (sheet: laboratory_data, patient_data): {3: 428, 4: 253, 1: 5}
- `icu_lab_hstrpiunit_c_y` (sheet: laboratory_data, patient_data): {3: 433, 4: 245, 1: 5, 2: 3}
- `icu_lab_lactunit_c_y` (sheet: laboratory_data, patient_data): {1: 648, 2: 28, 3: 10}

### Raw values per unit code
| unit column | value column | code | n | q5 | q25 | q50 | q75 | q95 |
|---|---|---|---|---|---|---|---|---|
| icu_lab_tropiunit_c_x | icu_lab_tropi_x | 1 | 0 |  |  |  |  |  |
| icu_lab_tropiunit_c_x | icu_lab_tropi_x | 3 | 128 | 0.0228 | 0.268 | 3.364 | 16.15 | 174.3 |
| icu_lab_tropiunit_c_x | icu_lab_tropi_x | 4 | 41 | 0 | 1.3 | 43 | 1.096e+04 | 5e+04 |
| icu_lab_lactunit_c_x | icu_lab_lactprepci_x | 1 | 414 | 1.2 | 2.646 | 5.03 | 8.1 | 15.87 |
| icu_lab_lactunit_c_x | icu_lab_lactprepci_x | 2 | 22 | 12 | 23.55 | 43 | 76 | 97.9 |
| icu_lab_lactunit_c_x | icu_lab_lactprepci_x | 3 | 1 |  |  |  |  |  |
| icu_lab_lactunit_c_x | icu_lab_lactpopci_x | 1 | 515 | 0.9 | 1.63 | 3.1 | 6.4 | 14.94 |
| icu_lab_lactunit_c_x | icu_lab_lactpopci_x | 2 | 28 | 9.05 | 18.75 | 33 | 77.5 | 109.5 |
| icu_lab_lactunit_c_x | icu_lab_lactpopci_x | 3 | 5 |  |  |  |  |  |
| icu_lab_lactunit_c_x | icu_lab_lact8hpci_x | 1 | 482 | 0.8 | 1.4 | 2.5 | 5.095 | 11.89 |
| icu_lab_lactunit_c_x | icu_lab_lact8hpci_x | 2 | 28 | 5.35 | 12.75 | 22.3 | 62.25 | 126.9 |
| icu_lab_lactunit_c_x | icu_lab_lact8hpci_x | 3 | 7 |  |  |  |  |  |
| icu_lab_lactunit_c_x | icu_lab_lact16hpci_x | 1 | 445 | 0.8 | 1.2 | 1.9 | 3.6 | 9.24 |
| icu_lab_lactunit_c_x | icu_lab_lact16hpci_x | 2 | 26 | 7 | 10 | 20 | 31.7 | 146 |
| icu_lab_lactunit_c_x | icu_lab_lact16hpci_x | 3 | 6 |  |  |  |  |  |
| icu_lab_lactunit_c_x | icu_lab_lact24hpci_x | 1 | 414 | 0.7855 | 1.2 | 1.8 | 3.2 | 8.035 |
| icu_lab_lactunit_c_x | icu_lab_lact24hpci_x | 2 | 25 | 7 | 11 | 16 | 21 | 100.6 |
| icu_lab_lactunit_c_x | icu_lab_lact24hpci_x | 3 | 6 |  |  |  |  |  |
| icu_lab_lactunit_c_x | icu_lab_lact32hpci_x | 1 | 390 | 0.7 | 1.103 | 1.6 | 2.5 | 8.1 |
| icu_lab_lactunit_c_x | icu_lab_lact32hpci_x | 2 | 24 | 5.3 | 8.075 | 12 | 18.25 | 120.5 |
| icu_lab_lactunit_c_x | icu_lab_lact32hpci_x | 3 | 3 |  |  |  |  |  |
| icu_lab_lactunit_c_x | icu_lab_lact40hpci_x | 1 | 376 | 0.6645 | 1 | 1.47 | 2.4 | 6.55 |
| icu_lab_lactunit_c_x | icu_lab_lact40hpci_x | 2 | 23 | 4.2 | 9 | 12 | 22.5 | 187.2 |
| icu_lab_lactunit_c_x | icu_lab_lact40hpci_x | 3 | 3 |  |  |  |  |  |
| icu_lab_lactunit_c_x | icu_lab_lact48hpci_x | 1 | 365 | 0.7 | 1 | 1.4 | 2 | 5.5 |
| icu_lab_lactunit_c_x | icu_lab_lact48hpci_x | 2 | 21 | 8 | 9 | 13 | 22 | 168 |
| icu_lab_lactunit_c_x | icu_lab_lact48hpci_x | 3 | 3 |  |  |  |  |  |
| icu_lab_lactunit_c_x | icu_lab_lactdone_y | 1 | 648 | 0 | 0 | 1 | 1 | 1 |
| icu_lab_lactunit_c_x | icu_lab_lactdone_y | 2 | 28 | 0 | 0 | 1 | 1 | 1 |
| icu_lab_lactunit_c_x | icu_lab_lactdone_y | 3 | 10 | 0.45 | 1 | 1 | 1 | 1 |
| icu_lab_lactunit_c_x | admission_lactate | 1 | 554 | 0.965 | 1.7 | 3.2 | 6.4 | 15 |
| icu_lab_lactunit_c_x | admission_lactate | 2 | 28 | 9.05 | 18.75 | 33 | 77.5 | 109.5 |
| icu_lab_lactunit_c_x | admission_lactate | 3 | 5 |  |  |  |  |  |
| icu_lab_lactunit_copy_c | icu_lab_lactprepci_x | 1 | 414 | 1.2 | 2.646 | 5.03 | 8.1 | 15.87 |
| icu_lab_lactunit_copy_c | icu_lab_lactprepci_x | 2 | 22 | 12 | 23.55 | 43 | 76 | 97.9 |
| icu_lab_lactunit_copy_c | icu_lab_lactprepci_x | 3 | 1 |  |  |  |  |  |
| icu_lab_lactunit_copy_c | icu_lab_lactpopci_x | 1 | 515 | 0.9 | 1.63 | 3.1 | 6.4 | 14.94 |
| icu_lab_lactunit_copy_c | icu_lab_lactpopci_x | 2 | 28 | 9.05 | 18.75 | 33 | 77.5 | 109.5 |
| icu_lab_lactunit_copy_c | icu_lab_lactpopci_x | 3 | 5 |  |  |  |  |  |
| icu_lab_lactunit_copy_c | icu_lab_lact8hpci_x | 1 | 482 | 0.8 | 1.4 | 2.5 | 5.095 | 11.89 |
| icu_lab_lactunit_copy_c | icu_lab_lact8hpci_x | 2 | 28 | 5.35 | 12.75 | 22.3 | 62.25 | 126.9 |
| icu_lab_lactunit_copy_c | icu_lab_lact8hpci_x | 3 | 7 |  |  |  |  |  |
| icu_lab_lactunit_copy_c | icu_lab_lact16hpci_x | 1 | 445 | 0.8 | 1.2 | 1.9 | 3.6 | 9.24 |
| icu_lab_lactunit_copy_c | icu_lab_lact16hpci_x | 2 | 26 | 7 | 10 | 20 | 31.7 | 146 |
| icu_lab_lactunit_copy_c | icu_lab_lact16hpci_x | 3 | 6 |  |  |  |  |  |
| icu_lab_lactunit_copy_c | icu_lab_lact24hpci_x | 1 | 414 | 0.7855 | 1.2 | 1.8 | 3.2 | 8.035 |
| icu_lab_lactunit_copy_c | icu_lab_lact24hpci_x | 2 | 25 | 7 | 11 | 16 | 21 | 100.6 |
| icu_lab_lactunit_copy_c | icu_lab_lact24hpci_x | 3 | 6 |  |  |  |  |  |
| icu_lab_lactunit_copy_c | icu_lab_lact32hpci_x | 1 | 390 | 0.7 | 1.103 | 1.6 | 2.5 | 8.1 |
| icu_lab_lactunit_copy_c | icu_lab_lact32hpci_x | 2 | 24 | 5.3 | 8.075 | 12 | 18.25 | 120.5 |
| icu_lab_lactunit_copy_c | icu_lab_lact32hpci_x | 3 | 3 |  |  |  |  |  |
| icu_lab_lactunit_copy_c | icu_lab_lact40hpci_x | 1 | 376 | 0.6645 | 1 | 1.47 | 2.4 | 6.55 |
| icu_lab_lactunit_copy_c | icu_lab_lact40hpci_x | 2 | 23 | 4.2 | 9 | 12 | 22.5 | 187.2 |
| icu_lab_lactunit_copy_c | icu_lab_lact40hpci_x | 3 | 3 |  |  |  |  |  |
| icu_lab_lactunit_copy_c | icu_lab_lact48hpci_x | 1 | 365 | 0.7 | 1 | 1.4 | 2 | 5.5 |
| icu_lab_lactunit_copy_c | icu_lab_lact48hpci_x | 2 | 21 | 8 | 9 | 13 | 22 | 168 |
| icu_lab_lactunit_copy_c | icu_lab_lact48hpci_x | 3 | 3 |  |  |  |  |  |
| icu_lab_lactunit_copy_c | icu_lab_lactdone_y | 1 | 648 | 0 | 0 | 1 | 1 | 1 |
| icu_lab_lactunit_copy_c | icu_lab_lactdone_y | 2 | 28 | 0 | 0 | 1 | 1 | 1 |
| icu_lab_lactunit_copy_c | icu_lab_lactdone_y | 3 | 10 | 0.45 | 1 | 1 | 1 | 1 |
| icu_lab_lactunit_copy_c | admission_lactate | 1 | 554 | 0.965 | 1.7 | 3.2 | 6.4 | 15 |
| icu_lab_lactunit_copy_c | admission_lactate | 2 | 28 | 9.05 | 18.75 | 33 | 77.5 | 109.5 |
| icu_lab_lactunit_copy_c | admission_lactate | 3 | 5 |  |  |  |  |  |
| icu_lab_creaunit_c | icu_lab_crea_x | 1 | 318 | 67.73 | 91 | 110 | 139 | 229 |
| icu_lab_creaunit_c | icu_lab_crea_x | 2 | 1 |  |  |  |  |  |
| icu_lab_creaunit_c | icu_lab_crea_x | 3 | 329 | 0.74 | 1.05 | 1.29 | 1.66 | 2.752 |
| icu_lab_hbunit_c | icu_lab_hb_x_x | 1 | 162 | 5.9 | 7.4 | 8.5 | 9.2 | 10.4 |
| icu_lab_hbunit_c | icu_lab_hb_x_x | 2 | 440 | 8.795 | 11.9 | 13.55 | 14.8 | 16.7 |
| icu_lab_hbunit_c | icu_lab_hb_x_x | 3 | 56 | 96.25 | 112 | 130.5 | 144.2 | 162 |
| icu_lab_hbunit_c | icu_lab_hb_x_y | 1 | 100 | 6 | 7.6 | 8.7 | 9.125 | 10.1 |
| icu_lab_hbunit_c | icu_lab_hb_x_y | 2 | 312 | 5.309 | 7.447 | 8.316 | 9.185 | 10.3 |
| icu_lab_hbunit_c | icu_lab_hb_x_y | 3 | 28 | 5.759 | 7.277 | 8.223 | 8.952 | 9.886 |
| icu_lab_hctunit_c | icu_lab_hct_x_x | 1 | 445 | 23.76 | 34.6 | 39.5 | 43.3 | 48 |
| icu_lab_hctunit_c | icu_lab_hct_x_x | 2 | 187 | 0.3 | 0.3865 | 0.43 | 0.519 | 44.67 |
| icu_lab_hctunit_c | icu_lab_hct_x_y | 1 | 285 | 23.76 | 35 | 39.7 | 43.5 | 47.92 |
| icu_lab_hctunit_c | icu_lab_hct_x_y | 2 | 141 | 29.2 | 37 | 40 | 45 | 49.8 |
| icu_lab_crpunit_c | icu_lab_crp_x | 1 | 177 | 0.098 | 0.3 | 0.81 | 3.1 | 23.82 |
| icu_lab_crpunit_c | icu_lab_crp_x | 2 | 376 | 0.8 | 2.9 | 6 | 26.88 | 151.2 |
| icu_lab_glcunit_c | icu_lab_glc_x | 1 | 299 | 101.9 | 157.5 | 222 | 311.5 | 484.1 |
| icu_lab_glcunit_c | icu_lab_glc_x | 2 | 215 | 5.773 | 8.28 | 12 | 16.8 | 26.09 |
| icu_lab_ckunit_c | icu_lab_ck_x | 1 | 128 | 1.09 | 2.328 | 5.22 | 17.27 | 156.2 |
| icu_lab_ckunit_c | icu_lab_ck_x | 2 | 469 | 53 | 145 | 288 | 843 | 3684 |
| icu_lab_ckunit_c | icu_lab_ckmb_x | 1 | 105 | 0.276 | 0.68 | 1.99 | 5.98 | 161.4 |
| icu_lab_ckunit_c | icu_lab_ckmb_x | 2 | 297 | 4.316 | 38 | 80 | 161 | 573.7 |
| icu_lab_ckmbunit_c | icu_lab_ckmb_x | 1 | 85 | 0.276 | 0.56 | 1.52 | 3.63 | 62.94 |
| icu_lab_ckmbunit_c | icu_lab_ckmb_x | 2 | 229 | 10.51 | 46 | 83 | 151 | 558.9 |
| icu_lab_ckmbunit_c | icu_lab_ckmb_x | 3 | 88 | 2.776 | 9.295 | 43.2 | 158.9 | 529.6 |
| icu_lab_bununit_c | icu_lab_bun_x | 1 | 187 | 3.13 | 5.51 | 7.37 | 11.25 | 29.52 |
| icu_lab_bununit_c | icu_lab_bun_x | 2 | 181 | 12.14 | 25 | 38 | 51.3 | 94 |
| icu_lab_troptunit_c | icu_lab_tropt_x | 1 | 0 |  |  |  |  |  |
| icu_lab_troptunit_c | icu_lab_tropt_x | 2 | 1 |  |  |  |  |  |
| icu_lab_troptunit_c | icu_lab_tropt_x | 3 | 48 | 0.1186 | 0.967 | 106.6 | 408.8 | 2504 |
| icu_lab_troptunit_c | icu_lab_tropt_x | 4 | 23 | 0 | 0.0065 | 0.231 | 172.8 | 6254 |
| icu_lab_tropiunit_c_y | icu_lab_tropi_x | 1 | 0 |  |  |  |  |  |
| icu_lab_tropiunit_c_y | icu_lab_tropi_x | 3 | 128 | 0.0228 | 0.268 | 3.364 | 16.15 | 174.3 |
| icu_lab_tropiunit_c_y | icu_lab_tropi_x | 4 | 41 | 0 | 1.3 | 43 | 1.096e+04 | 5e+04 |
| icu_lab_lactunit_c_y | icu_lab_lactprepci_x | 1 | 414 | 1.2 | 2.646 | 5.03 | 8.1 | 15.87 |
| icu_lab_lactunit_c_y | icu_lab_lactprepci_x | 2 | 22 | 12 | 23.55 | 43 | 76 | 97.9 |
| icu_lab_lactunit_c_y | icu_lab_lactprepci_x | 3 | 1 |  |  |  |  |  |
| icu_lab_lactunit_c_y | icu_lab_lactpopci_x | 1 | 515 | 0.9 | 1.63 | 3.1 | 6.4 | 14.94 |
| icu_lab_lactunit_c_y | icu_lab_lactpopci_x | 2 | 28 | 9.05 | 18.75 | 33 | 77.5 | 109.5 |
| icu_lab_lactunit_c_y | icu_lab_lactpopci_x | 3 | 5 |  |  |  |  |  |
| icu_lab_lactunit_c_y | icu_lab_lact8hpci_x | 1 | 482 | 0.8 | 1.4 | 2.5 | 5.095 | 11.89 |
| icu_lab_lactunit_c_y | icu_lab_lact8hpci_x | 2 | 28 | 5.35 | 12.75 | 22.3 | 62.25 | 126.9 |
| icu_lab_lactunit_c_y | icu_lab_lact8hpci_x | 3 | 7 |  |  |  |  |  |
| icu_lab_lactunit_c_y | icu_lab_lact16hpci_x | 1 | 445 | 0.8 | 1.2 | 1.9 | 3.6 | 9.24 |
| icu_lab_lactunit_c_y | icu_lab_lact16hpci_x | 2 | 26 | 7 | 10 | 20 | 31.7 | 146 |
| icu_lab_lactunit_c_y | icu_lab_lact16hpci_x | 3 | 6 |  |  |  |  |  |
| icu_lab_lactunit_c_y | icu_lab_lact24hpci_x | 1 | 414 | 0.7855 | 1.2 | 1.8 | 3.2 | 8.035 |
| icu_lab_lactunit_c_y | icu_lab_lact24hpci_x | 2 | 25 | 7 | 11 | 16 | 21 | 100.6 |
| icu_lab_lactunit_c_y | icu_lab_lact24hpci_x | 3 | 6 |  |  |  |  |  |
| icu_lab_lactunit_c_y | icu_lab_lact32hpci_x | 1 | 390 | 0.7 | 1.103 | 1.6 | 2.5 | 8.1 |
| icu_lab_lactunit_c_y | icu_lab_lact32hpci_x | 2 | 24 | 5.3 | 8.075 | 12 | 18.25 | 120.5 |
| icu_lab_lactunit_c_y | icu_lab_lact32hpci_x | 3 | 3 |  |  |  |  |  |
| icu_lab_lactunit_c_y | icu_lab_lact40hpci_x | 1 | 376 | 0.6645 | 1 | 1.47 | 2.4 | 6.55 |
| icu_lab_lactunit_c_y | icu_lab_lact40hpci_x | 2 | 23 | 4.2 | 9 | 12 | 22.5 | 187.2 |
| icu_lab_lactunit_c_y | icu_lab_lact40hpci_x | 3 | 3 |  |  |  |  |  |
| icu_lab_lactunit_c_y | icu_lab_lact48hpci_x | 1 | 365 | 0.7 | 1 | 1.4 | 2 | 5.5 |
| icu_lab_lactunit_c_y | icu_lab_lact48hpci_x | 2 | 21 | 8 | 9 | 13 | 22 | 168 |
| icu_lab_lactunit_c_y | icu_lab_lact48hpci_x | 3 | 3 |  |  |  |  |  |
| icu_lab_lactunit_c_y | icu_lab_lactdone_y | 1 | 648 | 0 | 0 | 1 | 1 | 1 |
| icu_lab_lactunit_c_y | icu_lab_lactdone_y | 2 | 28 | 0 | 0 | 1 | 1 | 1 |
| icu_lab_lactunit_c_y | icu_lab_lactdone_y | 3 | 10 | 0.45 | 1 | 1 | 1 | 1 |
| icu_lab_lactunit_c_y | admission_lactate | 1 | 554 | 0.965 | 1.7 | 3.2 | 6.4 | 15 |
| icu_lab_lactunit_c_y | admission_lactate | 2 | 28 | 9.05 | 18.75 | 33 | 77.5 | 109.5 |
| icu_lab_lactunit_c_y | admission_lactate | 3 | 5 |  |  |  |  |  |

### Raw lab columns (all codes pooled)
| column | sheet | n | q5 | q25 | q50 | q75 | q95 |
|---|---|---|---|---|---|---|---|
| icu_lab_ck_x | laboratory_data | 597 | 1.984 | 65 | 199 | 603 | 3243 |
| ck | clip | 425 | 1.344 | 3.17 | 7.02 | 17.33 | 73.76 |
| icu_lab_lactprepci_x | patient_data | 437 | 1.2 | 2.7 | 5.4 | 9 | 25.56 |
| icu_lab_lactpopci_x | patient_data | 548 | 1 | 1.7 | 3.35 | 8.025 | 26.39 |
| admission_lactate | derived/merged | 587 | 1 | 1.805 | 3.5 | 8 | 25.73 |
| icu_lab_lact8hpci_x | patient_data | 517 | 0.8 | 1.4 | 2.7 | 5.994 | 21.08 |
| icu_lab_lact16hpci_x | patient_data | 477 | 0.8 | 1.3 | 2.02 | 4.3 | 18.44 |
| icu_lab_lact24hpci_x | patient_data | 445 | 0.8 | 1.3 | 2 | 4.12 | 13.82 |
| lac | clip | 413 | 1.032 | 2.01 | 3.73 | 7.5 | 14.54 |
| icu_lab_crea_x | laboratory_data | 648 | 0.81 | 1.27 | 4.25 | 110 | 189 |
| cre | clip | 425 | 65 | 91 | 111 | 142 | 215.8 |
| icu_lab_wct_x_x | derived/merged | 646 | 7.718 | 11.49 | 15.5 | 20.64 | 6355 |
| icu_lab_hct_x_x | derived/merged | 632 | 0.36 | 21.07 | 36.8 | 42.4 | 48 |
| icu_lab_crp_x | laboratory_data | 553 | 0.15 | 1.2 | 4 | 15.8 | 138 |
| icu_lab_glc_x | laboratory_data | 514 | 6.509 | 13.53 | 134 | 243.5 | 424.8 |
| gluc | clip | 415 | 5.348 | 8.165 | 11.91 | 16.34 | 26.82 |
| pbnp | clip | 425 | 60.66 | 287.6 | 1423 | 5192 | 2.042e+04 |
| tnt | clip | 425 | 44.06 | 225.9 | 652.9 | 2033 | 1.246e+04 |
| alat | clip | 425 | 0.272 | 0.67 | 1.36 | 2.74 | 11.49 |
| icu_lab_inr_r | clip | 407 | 0.97 | 1.08 | 1.193 | 1.4 | 3.099 |
| cysc_s_1 | clip | 425 | 0.75 | 1.01 | 1.26 | 1.59 | 2.418 |
| il_6 | clip | 425 | 9.744 | 39.9 | 93.7 | 284 | 1327 |

### Model features after harmonization (Full model labs)
| feature | n | q5 | q25 | q50 | q75 | q95 |
|---|---|---|---|---|---|---|
| admission_lactate | 587 | 0.9 | 1.655 | 3.1 | 6.25 | 13.52 |
| icu_lab_lact8hpci_x | 517 | 0.8 | 1.4 | 2.5 | 4.9 | 11.7 |
| icu_lab_lact16hpci_x | 477 | 0.8 | 1.28 | 2 | 3.6 | 8.92 |
| icu_lab_lact24hpci_x | 445 | 0.7415 | 1.2 | 1.8 | 3.3 | 7.98 |
| pbnp | 425 | 60.66 | 287.6 | 1423 | 5192 | 2.042e+04 |
| icu_lab_ck_x | 597 | 33.07 | 1084 | 3317 | 1.005e+04 | 5.406e+04 |
| tnt | 425 | 44.06 | 225.9 | 652.9 | 2033 | 1.246e+04 |
| creatine | 648 | 65.43 | 91.74 | 111 | 143 | 236.3 |
| white_cell_count | 646 | 6.885 | 11.01 | 14.9 | 19 | 26.5 |
| hematocrit | 632 | 26.5 | 35.1 | 40 | 43.92 | 48.9 |
| crp | 553 | 0.156 | 3.07 | 30 | 121 | 1302 |
| icu_lab_inr_r | 407 | 0.97 | 1.08 | 1.193 | 1.4 | 3.099 |
| glucose | 514 | 102.8 | 154.2 | 218 | 307.3 | 477.4 |
| alat | 425 | 0.272 | 0.67 | 1.36 | 2.74 | 11.49 |
| hpe_proc_mechs_yn | 686 | 0 | 0 | 0 | 1 | 1 |
| renal_replacement_therapy | 426 | 0 | 0 | 0 | 0 | 0 |
| cathecholamine_therapy | 686 | 0 | 0 | 0 | 0 | 0 |
| sepsis | 686 | 0 | 0 | 0 | 0 | 0 |
| ventricular_fibrillation | 686 | 0 | 0 | 0 | 0 | 1 |
| stroke | 686 | 0 | 0 | 0 | 0 | 0 |

`hpe_proc_mechs_yn` counts: {0: 492, 1: 194}

## eICU
Restricted to the 245 cohort patients in outcomes_CICU.csv, labs from 24 h before to 24 h after ICU admission.

### Lab names matching the search (n results)
- `ALT (SGPT)`: 285
- `AST (SGOT)`: 285
- `BNP`: 76
- `CPK`: 205
- `CPK-MB`: 165
- `CPK-MB INDEX`: 109
- `CRP`: 3
- `Hct`: 629
- `PT - INR`: 313
- `WBC x 1000`: 542
- `WBC's in body fluid`: 1
- `WBC's in pleural fluid`: 1
- `WBC's in urine`: 21
- `bedside glucose`: 1143
- `creatinine`: 755
- `glucose`: 790
- `lactate`: 336
- `total cholesterol`: 65
- `troponin - I`: 536
- `troponin - T`: 24
- `urinary creatinine`: 14

### Values per lab name and reported unit
| labname | unit | n | q5 | q25 | q50 | q75 | q95 |
|---|---|---|---|---|---|---|---|
| lactate | mmol/L | 336 | 1.1 | 2.078 | 3.7 | 7.6 | 14.7 |
| BNP | pg/mL | 75 | 110.1 | 385.5 | 991.6 | 2618 | 1.514e+04 |
| CPK | Units/L | 204 | 42.35 | 152 | 471.5 | 1672 | 5929 |
| troponin - T | ng/mL | 23 | 0.0666 | 0.167 | 1.97 | 5.925 | 30.65 |
| creatinine | mg/dL | 755 | 0.77 | 1.2 | 1.66 | 2.51 | 4.27 |
| WBC x 1000 | K/mcL | 542 | 5.005 | 9.5 | 13.5 | 18.98 | 31.27 |
| Hct | % | 629 | 24.64 | 30.9 | 35.8 | 40.4 | 48 |
| CRP | mg/dL | 3 |  |  |  |  |  |
| PT - INR | ratio | 311 | 1 | 1.18 | 1.4 | 1.95 | 3.75 |
| glucose | mg/dL | 790 | 84 | 120.2 | 161.5 | 242.8 | 458.5 |
| ALT (SGPT) | Units/L | 285 | 11 | 29 | 67 | 269 | 2441 |
| triglycerides | mg/dL | 87 | 58.6 | 94 | 125 | 186 | 320.7 |
| HDL | mg/dL | 65 | 13.2 | 24 | 32 | 38 | 63 |
| LDL | mg/dL | 59 | 25.3 | 48 | 76 | 102.5 | 144 |
| anion gap | NA | 463 | 5 | 9 | 13 | 17 | 25 |
| BUN | mg/dL | 741 | 12 | 21 | 29 | 43 | 82 |

### Model features after harmonization (X_Full_CICU.csv)
| feature | n | q5 | q25 | q50 | q75 | q95 |
|---|---|---|---|---|---|---|
| admission_lactate | 53 | 0.98 | 2.2 | 4.895 | 8.1 | 14.09 |
| icu_lab_lact8hpci_x | 91 | 1.1 | 2.275 | 3.88 | 6.758 | 14.25 |
| icu_lab_lact16hpci_x | 59 | 1.218 | 1.85 | 3 | 5.9 | 11.83 |
| icu_lab_lact24hpci_x | 37 | 1.232 | 1.9 | 2.9 | 7.5 | 14.11 |
| pbnp | 66 | 107.5 | 327.2 | 1058 | 2472 | 1.749e+04 |
| icu_lab_ck_x | 94 | 237.3 | 893 | 2557 | 8748 | 3.416e+04 |
| tnt | 8 |  |  |  |  |  |
| creatine | 241 | 63.65 | 99.45 | 141.1 | 221 | 366.2 |
| white_cell_count | 233 | 5.65 | 9.425 | 13 | 17.28 | 28.12 |
| hematocrit | 234 | 25.09 | 31.01 | 35.27 | 40 | 47 |
| crp | 3 |  |  |  |  |  |
| icu_lab_inr_r | 178 | 1 | 1.155 | 1.4 | 1.95 | 3.332 |
| glucose | 241 | 95 | 121 | 161.7 | 215.8 | 393 |
| alat | 182 | 0.1846 | 0.4333 | 1.021 | 3.485 | 38.71 |
| hpe_proc_mechs_yn | 245 | 0 | 0 | 0 | 0 | 1 |
| renal_replacement_therapy | 0 |  |  |  |  |  |
| cathecholamine_therapy | 0 |  |  |  |  |  |
| sepsis | 0 |  |  |  |  |  |
| ventricular_fibrillation | 0 |  |  |  |  |  |
| stroke | 0 |  |  |  |  |  |

