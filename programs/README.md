# Predictor programs

Programs are ordinary Python modules exporting the shared model interface.

| Directory | Program |
| :--- | :--- |
| [bbbc036/](bbbc036/) | Dose-scaled gated fusion, a control baseline and a factorized response readout. |
| [bbbc047/](bbbc047/) | Separate control, compound and dose encoders with compound-query attention and joint response prediction. |
| [demo/](demo/) | Compact control-context model for the synthetic feedback walkthrough. |

## Model interface

```python
def candidate_metadata():
    ...

def build_model(task_spec):
    ...

def compute_loss(prediction, target, state):
    ...

def build_optimizer(model, state):
    ...
```

A program can also define `build_scheduler(optimizer, state)` and `step_scheduler(scheduler, state)`.

The task specification supplies `context_dim`, `chemical_dim`, `dose_dim`, `target_dim`, `cp_dim` and `l1000_dim`.

The model's forward interface is:

```python
prediction = model(control, compound, dose)
```

The result is one joint tensor with shape `[samples, target_dim]`. Morphology comes first, followed by expression.

## Train a program

```bash
cytowright train \
  --task bbbc047 \
  --source programs/bbbc047/predictor.py \
  --output runs/bbbc047/base \
  --seed 41 \
  --device cuda:0
```

The fitted source, model parameters, normalization and metrics are saved together. The same source is used by the `correct`, `predict` and `analyze` commands.
