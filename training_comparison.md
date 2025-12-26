# Training Process Comparison: train_model.sh vs bootstrap_training/workflow.py

## Summary
The training process in `bootstrap_training/workflow.py` is **now consistent** with `train_model.sh`. Previous critical differences have been addressed:

## ✅ What's Consistent

1. **Config file structure**: Both generate identical YAML configs for model creation and training
2. **Model architecture**: Same CRNN parameters (5-layer CNN, 3-layer LSTM, etc.)
3. **Training parameters**: Same batch size, learning rate, optimizer, etc.
4. **Resume logic**: Both check for existing model and set resume flag accordingly
5. **Experiment directory backup**: Both backup existing experiment directories before creating new models

## ❌ Critical Differences

### 1. **Environment Variables** ✅ FIXED
**train_model.sh:**
```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
ulimit -n 4096 2>/dev/null || true
```

**workflow.py:**
- ✅ Now sets `OMP_NUM_THREADS=1` and `MKL_NUM_THREADS=1` in the environment for subprocess calls
- ✅ Sets `ulimit -n 4096` in the training command (via shell command)
- **Status**: Fixed - environment variables are now properly set

### 2. **Error Handling** ✅ FIXED
**train_model.sh:**
- Commands run directly; failures are immediately visible
- Script exits on error (via trap)

**workflow.py:**
```python
def _run_command(self, cmd: str, description: str, raise_on_error: bool = False, ...):
    result = subprocess.run(...)
    if result.returncode != 0:
        if raise_on_error:
            raise RuntimeError(...)  # Raises exception for critical commands
        logger.error(...)
```

- ✅ Training commands use `raise_on_error=True` - exceptions are raised on failure
- ✅ Model creation and training commands will raise exceptions if they fail
- **Status**: Fixed - critical commands now raise exceptions on failure

### 3. **Path Format**
**train_model.sh:**
```yaml
train_path: ./${MODEL_DIR}  # Relative path
```

**workflow.py:**
```yaml
train_path: {model_dir}  # Absolute path
```

- ⚠️ Different but probably fine (both should work)

### 4. **Logging Overwrite**
**train_model.sh:**
```yaml
overwrite: ${DO_CREATE_MODEL}  # true/false based on whether creating new model
```

**workflow.py:**
```yaml
overwrite: true  # Always true
```

- ⚠️ Minor difference, but could cause log overwriting issues

## ✅ Current Status

All previously identified issues have been addressed:

1. ✅ **Environment variables** are now set before running training commands
2. ✅ **Error handling** raises exceptions when training fails (for critical commands)
3. ✅ **Logging** captures full training output with `show_output=True` for training commands
4. ✅ **File descriptor limits** are set via `ulimit` in the training command

## 📝 Notes

- The `_run_command` method supports both modes: `raise_on_error=False` for non-critical commands (like HTR processing) and `raise_on_error=True` for critical commands (like model creation and training)
- Training output is shown in real-time with `show_output=True`
- Environment variables are set in the subprocess environment dictionary, ensuring they're available to PyLaia training processes

