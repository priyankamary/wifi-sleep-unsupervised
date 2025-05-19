
## Bayesian Sleep Pipeline

This repository provides a modular pipeline to estimate sleep and wake-up times from WiFi connectivity logs of a device using probabilistic modeling. It compares multiple Bayesian inference models (Normal, Uniform, and Hierarchical priors) and outputs per-user predictions with model-averaged estimates.

---

###  Project Structure

```
.
├── data_loader.py            # Loads and preprocesses input CSVs
├── trajectory_utils.py       # Extracts periods of stable activity from CHRT logs
├── bayesian_models.py        # Defines and runs probabilistic models in PyMC3
├── inference_pipeline.py     # Executes inference and aggregates predictions
├── main.py                   # CLI entrypoint to run the pipeline
```

---

### ⚙️ Requirements

- Python ≥ 3.7
- `pymc3`
- `pandas`
- `argparse`
- `matplotlib` (optional)

Install dependencies via:

```bash
pip install pymc3 pandas matplotlib
```

---



Prepare the following inputs:
- Fitbit user CSVs (per user) for ground truth
- WiFi connectivity logs
- Binned session-level data files

Run the script:

```bash
python main.py 
```

---

###Output

Each user's output CSV contains:
- `sleep_avg`, `wakeup_avg`: Weighted average predictions.
- `sleep_true`, `wakeup_true`: Ground-truth labels from Fitbit.
- `sleep_hier`, `wakeup_hier`: Predictions from the hierarchical model.
- `count`: Index of processed days.

---

###  Notes

- Each model uses PyMC3 sampling with 1000 posterior and 1000 tuning samples.
- Trajectory estimation helps narrow inference to plausible intervals.

---

###  Example Use Case

Originally developed for a research project on uncertainty-aware, mobile-based sleep monitoring using probabilistic models.

Can be extended to:
- Sleep analytics platforms
- Behavioral inference from network data
- Mobile health monitoring applications





