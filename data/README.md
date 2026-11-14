# Measurement data

Place the BBBC036 and BBBC047 Cell Painting and L1000 measurement tables in the layout documented in [docs/DATA.md](../docs/DATA.md).

```bash
cytowright prepare-data --task bbbc036
cytowright prepare-data --task bbbc047
```

Preparation writes named training, validation, main-test and additional-test arrays under `data/arrays/`. The [workflow guide](../docs/WORKFLOW.md) shows how these arrays connect to model fitting, correction and prediction.
