# CLI

`dotnet run --project src/DotFly.Cli -c Release -- <command>` (or the built `dotfly` executable).
Neuron selections everywhere accept body IDs, `set:NAME`, `type:NAME`, `class:NAME`,
`superclass:NAME` with globs, an optional `@L` / `@R` side suffix, and `;` for unions —
e.g. `"type:LC4@L;type:ORN_DM1"`.

| command | what it does | options |
|---|---|---|
| `info` | environment, SIMD width, versions | |
| `build malecns <data-dir>` | MaleCNS Feather release → `.dfb` checkpoint | `-o`, `--filter superclass\|traced`, `--weights <file>` |
| `build flywire <completeness.csv> <connectivity.parquet>` | FlyWire (Shiu et al. files) → `.dfb` | `-m 630\|783`, `-o` |
| `inspect <checkpoint>` | provenance and counts; a body, a type; who a neuron drives or **who drives it** | `--body`, `--type`, `--upstream-of <ID>` (with `--hops`, `--min-weight`, `--class`), `--top` |
| `run <checkpoint>` | stimulate populations, optionally silence others, report who fires — during and after the stimulus | `--stimulate`, `--hz`, `--silence`, `--seconds`, `--stim-seconds`, `--seed`, `--backend cpu\|reference`, `--threads`, `--gain`, `--top`, `--csv` |
| `bench <checkpoint>` | steps/s and real-time factor for the standard scenarios | `--scenario`, `--hz`, `--seconds`, `--threads 1,4,8`, `--seed` |
| `explore <checkpoint>` | live spike raster and statistics in the terminal; keys toggle recurrent transmission (`r`), external input (`e`) and the rate (`+`/`-`) | `--stimulate`, `--hz`, `--watch`, `--ratio`, `--threads`, `--seed`, `--duration`, `--rows` |

## Recipes that produced the findings

```powershell
# Who drives the proboscis motor neuron? (found the GRN → MN9 pathway)
dotfly inspect .data/malecns-v1.0/malecns-v1.0-superclass.dfb --upstream-of <MN9 id> --hops 2 --min-weight 5

# Does a readout return to baseline after the stimulus stops? (the runaway attractors)
dotfly run <ckpt> --stimulate "type:ORN_DM1" --hz 60 --seconds 3 --stim-seconds 1 --silence "class:Kenyon_Cell;type:lLN*"

# Is a readout lateralized? (wind: AMMC012 answers one antenna only)
dotfly run <ckpt> --stimulate "type:JO-C*@L;type:JO-E*@L" --hz 60 --seconds 2 --silence "class:Kenyon_Cell;type:lLN*" --csv left.csv
dotfly run <ckpt> --stimulate "type:JO-C*@R;type:JO-E*@R" --hz 60 --seconds 2 --silence "class:Kenyon_Cell;type:lLN*" --csv right.csv

# A response curve (DM1_lPN vs ORN input rate)
foreach ($hz in 5,10,20,40,80,120) { dotfly run <ckpt> --stimulate "type:ORN_DM1" --hz $hz --seconds 2 --silence "class:Kenyon_Cell;type:lLN*" --top 400 }

# Performance
dotfly bench <ckpt> --scenario type:T4a --threads 1,4,8
```

`--csv` writes every neuron that fired with its type and count, for scripting. `DOTFLY_PROFILE=1`
prints the kernel profile at the end of a `run`; `DOTFLY_PIN_CALLER=1` pins the calling thread
(never inside a host).
