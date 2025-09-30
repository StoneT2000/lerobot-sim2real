# Tests

This directory contains unit tests for the lerobot_sim2real project.

## Running Tests

The project uses pytest for testing. Tests can be run using `uv`:

### Run all tests

```bash
uv run pytest tests/
```

### Run with verbose output

```bash
uv run pytest tests/ -v
```

### Run specific test file

```bash
uv run pytest tests/test_rb_solver_override.py -v
```

### Run with coverage report

```bash
uv run pytest tests/ --cov=lerobot_sim2real --cov-report=term-missing
```

### Run specific test

```bash
uv run pytest tests/test_rb_solver_override.py::test_solver_initialization -v
```

## Test Structure

### `test_rb_solver_override.py`

Essential tests for the RBSolver rendering-based calibration solver (10 tests, 95% coverage)

- Config creation and validation
- Solver initialization
- Forward pass (basic, with mount poses, with GT metrics)
- Predicted extrinsic computation
- Gradient flow verification
- Batch size variations (parametrized: 1, 2, 4)

### `test_optimize_with_better_logging.py`

Essential tests for the optimization function (8 tests, 71% coverage)

- Basic optimization workflow
- Batch processing (with/without batching)
- Early stopping functionality
- History tracking (return_history mode)
- Ground truth pose metrics
- Camera mount poses
- Integration test (all features combined)

### `test_camera.py`

Tests for camera utilities (12 tests, 100% coverage)

- No scaling (same dimensions)
- Uniform downscaling and upscaling
- Width and height cropping scenarios
- Aspect ratio changes
- Matrix structure validation (bottom row, off-diagonal zeros)
- Various common resolutions (parametrized)

### `test_urdf_utils.py`

Tests for URDF mesh extraction utilities (9 tests, 62% coverage)

- **Individual mesh extraction** (`extract_individual_meshes_with_origins`):
  - No meshes, single mesh, multiple visuals per link
  - Skipping non-mesh geometry
  - Visual origin transformation
- **Merged mesh extraction** (`extract_merged_meshes_per_link`):
  - Mesh extraction and merging
  - Handling merge failures
  - Visual origin applied before merge
  - Multiple visuals merged per link

## Adding New Tests

When adding new test files:

1. Name them `test_*.py`
2. Use pytest fixtures for reusable test data
3. Use `@pytest.mark.parametrize` for testing multiple scenarios
4. Mock external dependencies (especially GPU-dependent ones)
5. Run tests locally before committing

## Configuration

Test configuration is in `pyproject.toml`:

- Test discovery paths
- Coverage settings
- Pytest options

## Test Summary

As of the latest run:

- **Total tests**: 39
- **All passing**: ✅
- **Coverage**:
  - `lerobot_sim2real.optim` module: 80%
    - `rb_solver_override.py`: 95% coverage (10 tests)
    - `optimize_with_better_logging.py`: 71% coverage (8 tests)
  - `lerobot_sim2real.utils` module:
    - `camera.py`: 100% coverage (12 tests)
    - `urdf_utils.py`: 62% coverage (9 tests, focused on core mesh extraction)
