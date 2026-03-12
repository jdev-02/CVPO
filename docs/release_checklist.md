# CVPO Release Checklist

## Functional Readiness
- [ ] `python run_cvpo.py --workflow-demo --goal geese_tracking --format pretty` runs successfully.
- [ ] `python run_cvpo.py --benchmark --benchmark-workflow guided_geese` runs and saves report.
- [ ] CLI, Gradio, and notebook helper flows produce consistent guided payloads.

## Test + Quality
- [ ] `python3 -m pytest` passes.
- [ ] IDE lint diagnostics are clean.
- [ ] Benchmark regression check against baseline is documented.

## Hygiene
- [ ] `python tools/hygiene_check.py` passes.
- [ ] No class-specific personal identifiers in public-facing files.
- [ ] `libs/` remains excluded from distribution workflows.

## Documentation
- [ ] README includes install/run/benchmark/report commands.
- [ ] CITATIONS.md includes core papers/models used.
- [ ] Process logs are up to date for presentation evidence.

## Optional Cleanup (Post Presentation)
- [ ] Remove or disable `--save-report` feature if not needed.
- [ ] Archive presentation-only artifacts from `docs/process/reports/`.
