# Golden fixtures

Reference traces for the float64 backend, generated offline by `tools/brian2_golden.py` with
Brian2 (versions recorded in each `.json`) from the Shiu et al. 2024 equations
(`philshiu/Drosophila_brain_model`, MIT).

- `chain3.*` — a synthetic 3-neuron chain.
- `sugar_2hop.*`, `sugar_3hop.*` — 130- and 2,001-neuron sub-graphs (body IDs, edges with contact
  counts and signs) of the **FlyWire v630 connectivity** as shipped with the Shiu et al. model:
  Dorkenwald et al. 2024, *Neuronal wiring diagram of an adult brain*, Nature; FlyWire data are
  **CC BY 4.0** (https://flywire.ai/). Retain this attribution if you reuse the fixtures.

`.f64` files hold the recorded membrane traces (little-endian float64); `.json` holds the graph,
the input events and the spike steps.
