# Skill: Port MATLAB Algorithm to Python

Use this skill when converting a MATLAB biomechanics script to a vailá module.

---

## Translation Reference

| MATLAB | Python (NumPy/SciPy) |
|---|---|
| `size(A)` | `A.shape` |
| `zeros(m,n)` | `np.zeros((m, n))` |
| `ones(m,n)` | `np.ones((m, n))` |
| `A(:,1)` | `A[:, 0]` (0-indexed!) |
| `A(end,:)` | `A[-1, :]` |
| `A'` (transpose) | `A.T` |
| `A * B` (element-wise) | `A * B` |
| `A * B` (matrix mult) | `A @ B` |
| `inv(A)` | `np.linalg.inv(A)` |
| `norm(v)` | `np.linalg.norm(v)` |
| `cross(a,b)` | `np.cross(a, b)` |
| `dot(a,b)` | `np.dot(a, b)` |
| `linspace(a,b,n)` | `np.linspace(a, b, n)` |
| `cumtrapz(y,x)` | `scipy.integrate.cumulative_trapezoid(y, x)` |
| `filtfilt(b,a,x)` | `scipy.signal.filtfilt(b, a, x)` |
| `butter(n,Wn)` | `scipy.signal.butter(n, Wn)` |
| `interp1(x,y,xi)` | `np.interp(xi, x, y)` |
| `fft(x)` | `np.fft.fft(x)` |
| `mean(A,2)` | `np.mean(A, axis=1)` |
| `std(A,0,2)` | `np.std(A, axis=1, ddof=0)` |
| `find(x>0)` | `np.where(x > 0)[0]` |
| `isnan(x)` | `np.isnan(x)` |

## Index Offset — CRITICAL ⚠️
MATLAB is **1-indexed**. Python is **0-indexed**.
```matlab
% MATLAB
x(1)   → first element
x(end) → last element
A(1,2) → row 1, col 2
```
```python
# Python
x[0]    # first element
x[-1]   # last element
A[0, 1] # row 0, col 1
```

## Butterworth Filter Pattern
```matlab
% MATLAB
[b,a] = butter(4, fc/(fs/2), 'low');
x_filt = filtfilt(b, a, x);
```
```python
# Python
from scipy.signal import butter, filtfilt
b, a = butter(4, fc / (fs / 2), btype='low')
x_filt = filtfilt(b, a, x)
```

## Steps

1. Read the MATLAB code and identify:
   - Input data format and expected columns
   - Key algorithm steps
   - Output variables and units

2. Translate line by line using the table above

3. Pay special attention to:
   - Index offsets (1→0)
   - Column-major vs row-major arrays (MATLAB is column-major)
   - `size(A,1)` = rows = `A.shape[0]`
   - `size(A,2)` = cols = `A.shape[1]`

4. Validate output against MATLAB with the same sample data

5. Wrap in the standard vailá module pattern (see `create-analysis-module` skill)
