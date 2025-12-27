# Fix for ScrabbleGAN Training Issues

## Critical Problem Identified

When using `--OCR_init ./checkpoints/latin_bho_LatinBHOtrH32_GANres32_bs16`, the code loads ONLY the OCR network from that directory. However, if G and D checkpoints exist in that directory and are somehow loaded, they would be the collapsed weights.

## Solution: Isolate OCR Checkpoint

### Step 1: Create OCR-only directory and copy checkpoint

```bash
cd /home/qj/projects/latin_bho/scrabblegan/checkpoints
mkdir -p ocr_only
cp latin_bho_LatinBHOtrH32_GANres32_bs16/latest_net_OCR.pth ocr_only/
```

### Step 2: Verify G and D are deleted (they should be)

```bash
# These should NOT exist:
ls latin_bho_LatinBHOtrH32_GANres32_bs16/latest_net_G.pth  # Should fail
ls latin_bho_LatinBHOtrH32_GANres32_bs16/latest_net_D.pth  # Should fail
```

### Step 3: Start fresh training with explicit initialization

```bash
cd /home/qj/projects/latin_bho/scrabblegan

python train.py \
    --dataname LatinBHOtrH32 \
    --name_prefix latin_bho \
    --dataset_mode text \
    --model ScrabbleGAN \
    --input_nc 1 \
    --resolution 32 \
    --batch_size 16 \
    --G_lr 0.0002 \
    --D_lr 0.0002 \
    --OCR_lr 0.0002 \
    --num_critic_train 2 \
    --num_accumulations 1 \
    --save_epoch_freq 5 \
    --print_freq 100 \
    --OCR_init ./checkpoints/ocr_only \
    --G_init N02 \
    --D_init N02
```

## What to Verify in Logs

1. **Generator initialization** (should see):
   ```
   initialize network with N02
   [Network G] Total number of parameters : ...
   ```
   NOT: `loading the model from ... Generator`

2. **Discriminator initialization** (should see):
   ```
   initialize network with N02
   [Network D] Total number of parameters : ...
   ```
   NOT: `loading the model from ... Discriminator`

3. **OCR loading** (should see):
   ```
   loading the model from ./checkpoints/ocr_only/latest_net_OCR.pth
   [Network OCR] Total number of parameters : ...
   ```

4. **Early training (iterations 100-500)**:
   - OCR_fake loss should be HIGH initially (10-50), not near zero
   - Fake OCR predictions should show random characters forming, not empty strings
   - Generator should produce fuzzy shapes that gradually form letters

## If Still Seeing Issues

If the generator still produces blanks after 1000+ iterations:

1. Check web images: `./checkpoints/.../web/images/epoch001_fake.png`
   - Should see fuzzy gray blobs turning into letter shapes
   - NOT solid black/white boxes or static noise

2. Try slightly different learning rates:
   ```bash
   --G_lr 0.0001 \
   --D_lr 0.0002 \
   ```

3. Increase discriminator training frequency:
   ```bash
   --num_critic_train 4 \
   ```

