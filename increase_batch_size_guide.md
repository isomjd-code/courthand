# Increasing Batch Size for ScrabbleGAN

## Current Status
- **Current batch size**: 8
- **GPU memory used**: 10GB / 24GB (14GB free)
- **Safe to increase**: Yes!

## Recommendations

### Conservative Increase (Recommended First)
Try **batch_size = 16** (doubles current):
- Should use ~18-20GB total (well within 24GB limit)
- 2x faster training
- Lower risk of OOM errors

### Moderate Increase
Try **batch_size = 24**:
- Should use ~20-22GB total
- 3x faster training
- Still safe with 2GB buffer

### Aggressive Increase
Try **batch_size = 32**:
- May use 22-24GB total
- 4x faster training
- Risk of OOM if memory spikes

## Important Notes

### ⚠️ Checkpoint Directory Name
The checkpoint directory name includes batch size: `_bs8`, `_bs16`, etc.

**If you change batch size:**
- It will create a NEW checkpoint directory
- Your existing checkpoints in `GANres32_bs8` won't be automatically loaded
- You'll need to either:
  1. **Start fresh** (new training run)
  2. **Manually copy weights** from old checkpoint to new directory
  3. **Keep same batch size** and just increase for new training

### Memory Considerations
- Batch size affects memory linearly (2x batch = ~2x memory)
- Variable-length text images can cause memory spikes
- Keep some buffer (2-4GB) for safety

## How to Apply

### Option 1: Continue with New Batch Size (New Checkpoint)
```bash
cd scrabblegan
python train.py \
    --dataname LatinBHOtrH32 \
    --name_prefix latin_bho \
    --dataset_mode text \
    --model ScrabbleGAN \
    --input_nc 1 \
    --resolution 32 \
    --batch_size 16 \
    --D_lr 0.0004 \
    --num_critic_train 2
```
This will create a new checkpoint directory: `latin_bho_LatinBHOtrH32_GANres32_bs16`

### Option 2: Use Training Script
```bash
python train_scrabblegan.py \
    --train \
    --continue-train \
    --resolution 32 \
    --batch-size 16 \
    --D-lr 0.0004 \
    --num-critic-train 2
```

## Testing Batch Size

To test if a batch size works without starting full training:

1. Start training with new batch size
2. Watch GPU memory usage for first few iterations
3. If it stays under 22GB, you're safe
4. If you get OOM error, reduce batch size

## Benefits of Larger Batch Size

- **Faster training**: More samples per iteration
- **More stable gradients**: Better gradient estimates
- **Better batch normalization**: More accurate statistics
- **Fewer iterations per epoch**: Faster epochs

## Trade-offs

- **Less frequent updates**: Generator updates less often (if num_critic_train stays same)
- **Less memory for other operations**: Less headroom
- **New checkpoint directory**: Can't directly continue from old checkpoint

## Recommended Approach

1. **Start with batch_size = 16** (safe, 2x speedup)
2. **Monitor memory** for first 100 iterations
3. **If stable, try 24** for even faster training
4. **Don't go above 24** unless you're sure about memory spikes

