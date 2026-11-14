# Data and array formats

## Measurement tables

Use the same workspace for source measurements, prepared arrays and model outputs. The default workspace is the current project directory; `CYTOWRIGHT_WORKSPACE` selects another location.

```text
data/
├── BBBC036/
│   ├── CellPainting/replicate_level_cp_normalized_variable_selected.csv.gz
│   └── L1000/replicate_level_l1k.csv.gz
└── BBBC047/
    ├── CellPainting/replicate_level_cp_normalized_variable_selected.csv.gz
    └── L1000/replicate_level_l1k.csv.gz
```

`configs/tasks.json` defines the columns used for plate identifiers, untreated controls, compound–dose joins, SMILES and assay features. Preparation aggregates replicate measurements by condition, constructs control-relative responses, encodes compounds with Morgan fingerprints and groups chemical families by molecular scaffold.

```bash
cytowright prepare-data --task bbbc036
cytowright prepare-data --task bbbc047
```

## Named subsets

| Subset | Use |
| :--- | :--- |
| Training | Fit the model, normalization and correction features. |
| Validation | Select model checkpoints and correction candidates. |
| Main test | Measure the selected predictor. |
| Additional test | Measure the selected predictor on another group of chemical families. |

Prepared files are saved to `data/arrays/bbbc036/` and `data/arrays/bbbc047/`. Each directory contains `training.npz`, `validation.npz`, `main_test.npz` and `additional_test.npz`.

## Array interface

Each NPZ contains row-aligned `float32` matrices:

| Array | Shape | Contents |
| :--- | :--- | :--- |
| `pre` | `[samples, context_dim]` | Untreated control morphology. |
| `condition` | `[samples, 2049]` | 2,048 compound features followed by encoded dose. |
| `target` | `[samples, target_dim]` | Morphology followed by expression features. |

| Task | Context / morphology | Expression | Joint response |
| :--- | ---: | ---: | ---: |
| BBBC036 | 591 | 977 | 1,568 |
| BBBC047 | 701 | 977 | 1,678 |

The `predict` command reads `pre` and `condition`. The `correct` and `analyze` commands also use `target`.

## Normalization

Training inputs determine feature means and scales. Each fitted model is distributed together with `normalizer.npz`, which contains:

```text
pre_mean, pre_scale
condition_mean, condition_scale
target_mean, target_scale
```

Inference normalizes inputs and transforms predictions back to the measurement scale. Parameter correction uses the same statistics as its starting checkpoint.

## Compound replacement

Analysis groups samples by their complete control vector and encoded dose. Within each group, a replacement is selected from samples with a different compound representation. The model weights, control, dose and target stay fixed.

The JSON result records the number of matched samples, coverage, prediction change and MSE increase after replacement. A positive MSE increase means that the supplied compound helps predict the observed response under these matched replacements.
