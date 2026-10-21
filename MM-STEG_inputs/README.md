# MM-STEG simulation and MM/PBSA input files

These input files accompany the MM-STEG study and are intended for the
`pinoji/WEIE` repository linked from https://pinoji.github.io/#weie.
The five input files are unchanged from the supplied originals; only upload
suffixes in filenames were removed.

| File | Purpose |
| --- | --- |
| `em.mdp` | Steepest-descent energy minimization |
| `nvt.mdp` | 100 ps NVT equilibration |
| `npt.mdp` | 100 ps NPT equilibration |
| `md.mdp` | 50 ns production MD, 2 fs step, coordinates every 10 ps |
| `mmpbsa.in` | gmx_MMPBSA input, including PB and ordinary IE settings |

## Scope and provenance

The supplied MD inputs contain the thermostat group `Protein_JZ0`;
the MM/PBSA input has `sys_name = "Protein CAQ"`. They are supplied as
received, without asserting that this single set describes every T4 or
CB7 system. System-specific topologies, coordinates and index groups are
required separately.

The MD thermostat reference is 298 K. NVT velocity generation uses 300 K,
and the MM/PBSA analysis temperature is 298.15 K. These values are preserved
exactly. The MM/PBSA input requests frames 1000 through 5000 at interval 1;
conversion to times depends on the trajectory supplied to postprocessing.
`ie_segment = 25` specifies the final quarter for ordinary IE.

## Original uploaded filenames

- `em(3).mdp` → `em.mdp`
- `md.mdp` → `md.mdp`
- `mmpbsa(3).in` → `mmpbsa.in`
- `npt(1).mdp` → `npt.mdp`
- `nvt(1).mdp` → `nvt.mdp`
