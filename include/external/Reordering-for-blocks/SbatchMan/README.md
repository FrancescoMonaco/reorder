# SbatchMan/

SLURM batch job management tool for running experiments on HPC clusters.

## Structure

```
SbatchMan/
├── configs/             # Shared SLURM configurations
├── experiments/         # Generated experiment jobs and outputs
│   └── {experiment_name}/
│       ├── jobs/
│       │   └── {job_id}/
│       │       ├── script.sh
│       │       └── output.txt
│       └── metadata.json
└── sbatchman.py         # Main tool (or similar)
```

## Usage

### Generate Jobs
```bash
sbatchman generate yamls/operations_symmetric_reorder.yaml
```

### Submit Jobs
```bash
sbatchman submit experiments/{name}
```

### Check Status
```bash
sbatchman status experiments/{name}
```

## Configuration Reference

Jobs are configured via YAML files in `yamls/`. See [yamls/README.md](../yamls/README.md).

### configs.yaml Entries

```yaml
configs:
  gpu:
    partition: "gpu"
    gres: "gpu:1"
    time: "02:00:00"
    mem: "32G"
  cpu:
    partition: "compute"
    time: "01:00:00"
    mem: "16G"
```

## Output Parsing

Job outputs are parsed by `scripts/parse_results.py`:
- Reads `experiments/*/jobs/*/output.txt`
- Extracts timing from `<Timer>[label] X.XXX ms` format
- Extracts JSON from analysis outputs
- Aggregates into `results/*.csv`

## Job Archival

Completed job outputs remain in `experiments/` for result parsing.
Use `scripts/job_check.py` to monitor job status.
