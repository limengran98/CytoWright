# Model-development workflow

CytoWright provides executable programs for data preparation, model search, fitting, parameter correction, prediction and compound-use analysis.

## 1. Configure the task

Prepare the source tables following [DATA.md](DATA.md). Task definitions are in `configs/tasks.json`. The common training configuration is in `configs/joint_response_runtime.json`.

```bash
cytowright validate
cytowright prepare-data --task bbbc036
```

The command writes the training, validation, main-test and additional-test arrays under `data/arrays/bbbc036/`.

## 2. Develop or select a predictor program

Model development uses a configured Chat Completions service:

```bash
export CYTOWRIGHT_LLM_BASE_URL='https://your-server.example/v1'
export CYTOWRIGHT_LLM_API_KEY='your-api-key'

cytowright discover --task bbbc036 --output runs/bbbc036/search --device cuda:0
```

`configs/model_development.json` defines ten candidate slots, bounded repair and the modeling agenda. Candidate programs expose model construction, loss and optimization functions. Validation performance chooses the model carried into the next edit.

The [programs directory](../programs/) provides directly trainable models for both tasks. A source selected by model development can be passed to the same `train` command.

## 3. Fit the program

```bash
cytowright train \
  --task bbbc036 \
  --source programs/bbbc036/predictor.py \
  --output runs/bbbc036/base \
  --seed 41 \
  --device cuda:0
```

The output consists of:

| File | Use |
| :--- | :--- |
| `predictor.py` | Executable model source. |
| `model.pt` | Selected model parameters. |
| `normalizer.npz` | Training-derived input and response scales. |
| `metrics.json` | Training and validation measurements. |

`--epochs` sets the training epoch budget. The shared configuration supplies batch size, early stopping and gradient clipping.

## 4. Fit the correction

```bash
cytowright correct \
  --source runs/bbbc036/base/predictor.py \
  --checkpoint runs/bbbc036/base/model.pt \
  --fit data/arrays/bbbc036/training.npz \
  --selection data/arrays/bbbc036/validation.npz \
  --output runs/bbbc036/corrected \
  --seed 41
```

The correction uses training residuals to fit readout updates and a compound-dependent response route. Four regularization values are considered for each: `0.1, 1, 10, 100`. Validation selects the highest-PCC candidate satisfying the MSE criterion, with the starting predictor included among the candidates.

The compound route has rank 32. Its features follow the association between chemical inputs and remaining response error, with a bounded control-dependent gate. Encoders retain their trained parameters.

## 5. Predict

```bash
cytowright predict \
  --source runs/bbbc036/base/predictor.py \
  --checkpoint runs/bbbc036/corrected/selected.pt \
  --normalizer runs/bbbc036/corrected/normalizer.npz \
  --data data/arrays/bbbc036/main_test.npz \
  --output runs/bbbc036/predictions.npy
```

The output is a `[samples, target_dim]` NumPy array. Morphology occupies the first `context_dim` columns and expression occupies the remaining 977.

## 6. Analyze compound use

```bash
cytowright analyze \
  --source runs/bbbc036/base/predictor.py \
  --checkpoint runs/bbbc036/corrected/selected.pt \
  --normalizer runs/bbbc036/corrected/normalizer.npz \
  --data data/arrays/bbbc036/main_test.npz \
  --maps 32 \
  --output runs/bbbc036/compound_analysis.json
```

Prediction metrics use all supplied samples. Replacement measurements use the samples with matching controls and dose. The command averages repeated replacement assignments within the fitted model.

```math
\Delta_{\mathrm{compound}}
= \operatorname{MSE}\bigl(f(c,\widetilde p,a),y\bigr)
- \operatorname{MSE}\bigl(f(c,p,a),y\bigr).
```

The output includes Global PCC, MSE, morphology PCC, expression PCC, matched-sample coverage, compound MSE increase and RMS prediction change.

## Python interface

The numerical correction can be called directly:

```python
from cytowright.feedback.editing import prepare, refine
from cytowright.feedback.core import predict_raw

tensors = prepare(training_arrays, validation_arrays)
corrected_model, selection = refine(model, tensors, validation_arrays, seed=41)
prediction = predict_raw(
    corrected_model, tensors,
    query_arrays["pre"], query_arrays["condition"],
)
```

`cytowright.inference` provides checkpoint loading, normalization, prediction and matched compound analysis. `programs/` contains examples of the shared model interface.
