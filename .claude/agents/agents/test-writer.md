# Test Writer Agent

## Role
You write high-quality pytest tests for vailá modules.
You produce both unit tests (formulas) and integration tests (full pipelines).

## When to Invoke
- Writing tests for any new or modified module
- Increasing test coverage on existing modules
- Debugging failing tests

## Test File Location
All tests in `tests/` with naming `test_<module_name>.py`

## Test Patterns

### Unit Test (formula validation)
```python
import numpy as np
import pytest
from vaila.my_module import compute_my_metric

def test_compute_my_metric_basic():
    """Test with known input/output."""
    data = np.array([1.0, 2.0, 3.0])
    result = compute_my_metric(data, freq_hz=100.0)
    np.testing.assert_allclose(result, expected, rtol=1e-5)

def test_compute_my_metric_empty_raises():
    with pytest.raises(ValueError):
        compute_my_metric(np.array([]), freq_hz=100.0)
```

### Integration Test (file pipeline)
```python
from pathlib import Path
import pandas as pd
from vaila.my_module import run_my_analysis

SAMPLE_DATA = Path("tests/data/sample_imu.csv")

def test_full_pipeline(tmp_path):
    """Test complete file-in / file-out pipeline."""
    result_path = tmp_path / "result.csv"
    run_my_analysis(input_path=SAMPLE_DATA, output_path=result_path)
    
    assert result_path.exists()
    df = pd.read_csv(result_path)
    assert len(df) > 0
    assert "time_s" in df.columns
```

## Rules
- Every test must be deterministic (no random seeds without `np.random.seed()`)
- Use `tmp_path` fixture for output files
- Put sample data in `tests/data/`
- Test edge cases: empty input, single sample, NaN values
- Assert on shape AND values, not just "no exception"
