#!/usr/bin/env python3
"""
Calculate log probability of a string given a line image using Pylaia model.

Usage:
    python calculate_string_probability.py <image_name> <line_id> <alternate_string>
    
Example:
    python calculate_string_probability.py "CP40-565_481a.jpg" "L02" "William Smith"
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import os
import sys
import argparse
import json
from pathlib import Path

# Import Pylaia components
try:
    # Try Pylaia first (newer package)
    try:
        import pylaia
        from pylaia.utils import SymbolsTable
        print("✓ Using pylaia package")
        PYLAIA_PACKAGE = pylaia
    except ImportError:
        # Fallback to laia (older package name)
        import laia
        from laia.utils import SymbolsTable
        print("✓ Using laia package (fallback)")
        PYLAIA_PACKAGE = laia
except ImportError as e:
    print("=" * 70)
    print("ERROR: Pylaia (laia) not found")
    print("=" * 70)
    print(f"Import error: {e}")
    print(f"\nCurrent Python: {sys.executable}")
    print(f"Python version: {sys.version}")
    
    # Try to diagnose the issue
    print("\nDiagnostics:")
    try:
        import site
        print(f"Site packages: {site.getsitepackages()}")
    except:
        pass
    
    print("\nTroubleshooting steps:")
    print("1. Run the diagnostic script first:")
    print("   python test_pylaia_import.py")
    print("\n2. Verify you're in the correct environment:")
    print("   which python  # Should show pylaia-env path")
    print("   echo $VIRTUAL_ENV  # Should show pylaia-env path")
    print("\n3. Check if pylaia is installed:")
    print("   python -c 'import pylaia; print(pylaia.__file__)'")
    print("   # OR (fallback):")
    print("   python -c 'import laia; print(laia.__file__)'")
    print("\n4. If not installed, install it:")
    print("   pip install pylaia")
    print("   # OR if using conda:")
    print("   conda install -c conda-forge pylaia")
    print("\n5. Note: Your workflow uses 'pylaia-htr-decode-ctc' CLI tool,")
    print("   which means Pylaia IS installed. The Python package might be")
    print("   named 'pylaia' or 'laia' depending on version.")
    print("=" * 70)
    sys.exit(1)

# Import project settings
from workflow_manager.settings import (
    PYLAIA_MODEL,
    PYLAIA_SYMS,
    PYLAIA_ARCH,
    WORK_DIR,
    OUTPUT_DIR,
    IMAGE_DIR,
    BASE_DIR
)


def load_model_via_pylaia_cli(checkpoint_path: str, model_arch_path: str, symbols_path: str):
    """
    Load model using Pylaia's CLI tool approach.
    The CLI tool (pylaia-htr-decode-ctc) uses laia.common.loader.ModelLoader.
    This matches the exact code from laia.scripts.htr.decode_ctc.run():
    
        loader = ModelLoader(
            common.train_path, filename=common.model_filename, device="cpu"
        )
        checkpoint = loader.prepare_checkpoint(
            common.checkpoint,
            common.experiment_dirpath,
            common.monitor,
        )
        model = loader.load_by(checkpoint)
    """
    try:
        from laia.common.loader import ModelLoader
        
        # Extract train_path from model_arch_path (it's in the same directory)
        train_path = os.path.dirname(model_arch_path)
        model_filename = os.path.basename(model_arch_path)
        
        print(f"Using CLI's ModelLoader approach:")
        print(f"  train_path: {train_path}")
        print(f"  model_filename: {model_filename}")
        print(f"  checkpoint: {checkpoint_path}")
        
        # Create loader exactly as CLI does
        loader = ModelLoader(
            train_path, 
            filename=model_filename, 
            device="cpu"
        )
        
        # Prepare checkpoint (this handles finding the checkpoint file)
        # Try different signatures based on laia version
        try:
            # Try with all 3 arguments (as in CLI code)
            checkpoint = loader.prepare_checkpoint(
                checkpoint_path,
                experiment_dirpath=None,
                monitor=None,
            )
        except TypeError:
            # Try with just checkpoint path (older/newer version)
            try:
                checkpoint = loader.prepare_checkpoint(checkpoint_path)
            except TypeError:
                # If prepare_checkpoint doesn't exist or has different signature,
                # just use the checkpoint path directly
                checkpoint = checkpoint_path
        
        # Load model using the checkpoint
        model = loader.load_by(checkpoint)
        
        if model is None:
            raise ValueError("ModelLoader.load_by returned None. Have you run pylaia-htr-create-model?")
        
        print("✓ Loaded model using CLI's ModelLoader (exact CLI method)")
        return model
        
    except ImportError as e:
        print(f"Could not import laia.common.loader.ModelLoader: {e}")
        return None
    except Exception as e:
        print(f"CLI ModelLoader method failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def load_pylaia_model(checkpoint_path: str, model_arch_path: str, symbols_path: str):
    """
    Load Pylaia model from checkpoint and architecture file.
    Uses the same method as pylaia-htr-decode-ctc CLI tool.
    
    Args:
        checkpoint_path: Path to .ckpt checkpoint file
        model_arch_path: Path to model architecture file
        symbols_path: Path to symbols.txt file
        
    Returns:
        Loaded model in eval mode
    """
    print(f"Loading symbols from: {symbols_path}")
    syms = SymbolsTable(symbols_path)
    
    print(f"Loading model using Pylaia's model loading (matching CLI approach)...")
    
    # Method 0: Try using CLI's model loading API
    model = load_model_via_pylaia_cli(checkpoint_path, model_arch_path, symbols_path)
    if model is not None:
        # The ModelLoader should return a LightningModule with the actual model nested
        # Check if it has the model attribute (checkpoint keys suggest 'model.' prefix)
        if hasattr(model, 'model'):
            actual_model = model.model
            print("✓ Extracted model from model.model (CLI loaded)")
            actual_model.eval()
            return actual_model, syms
        elif hasattr(model, 'net'):
            actual_model = model.net
            print("✓ Extracted model from model.net (CLI loaded)")
            actual_model.eval()
            return actual_model, syms
        else:
            # ModelLoader might return the actual model directly, or it's the LightningModule itself
            # Check if it has forward and looks like a CRNN
            if hasattr(model, 'forward'):
                # Try using it directly, but check state_dict structure
                state_dict = model.state_dict() if hasattr(model, 'state_dict') else {}
                if any('conv' in k or 'rnn' in k for k in list(state_dict.keys())[:5]):
                    print("✓ Using model directly (appears to be CRNN)")
                    model.eval()
                    return model, syms
                # If state_dict is empty but checkpoint has model. prefix, we need to load differently
                print("⚠ ModelLoader returned LightningModule without nested model")
                print("  Will try to extract from checkpoint structure...")
    
    try:
        # Method 1: Try using pylaia/laia's model loading utilities
        # This matches how pylaia-htr-decode-ctc loads models
        try:
            from laia.common import ModelParams
            from laia.models import load_model_from_checkpoint
            
            # Load model using laia's checkpoint loading
            # This should match the CLI tool's approach
            print(f"Loading checkpoint: {checkpoint_path}")
            print(f"Model architecture file: {model_arch_path}")
            
            # Try loading the model architecture first
            if os.path.exists(model_arch_path):
                # The model file contains the model class definition
                import pickle
                with open(model_arch_path, 'rb') as f:
                    model_class = pickle.load(f)
                print("✓ Loaded model class from architecture file")
                
                # Now load the checkpoint into this model class
                import pytorch_lightning as pl
                model = model_class.load_from_checkpoint(
                    checkpoint_path,
                    map_location='cpu',
                    strict=False
                )
                print("✓ Loaded model from checkpoint")
            else:
                # Try loading directly from checkpoint (Lightning should handle it)
                import pytorch_lightning as pl
                model = pl.LightningModule.load_from_checkpoint(
                    checkpoint_path,
                    map_location='cpu',
                    strict=False
                )
                print("✓ Loaded model from checkpoint (no arch file)")
                
        except (ImportError, AttributeError, FileNotFoundError) as e:
            print(f"Method 1 failed: {e}")
            print("Trying Method 2: Direct Lightning loading...")
            
            # Method 2: Use PyTorch Lightning directly (fallback)
            import pytorch_lightning as pl
            
            # Load the model architecture from file if it exists
            if os.path.exists(model_arch_path):
                try:
                    import pickle
                    with open(model_arch_path, 'rb') as f:
                        model_class = pickle.load(f)
                    print("✓ Loaded model class from architecture file")
                    
                    # Load checkpoint into the model class
                    model = model_class.load_from_checkpoint(
                        checkpoint_path,
                        map_location='cpu',
                        strict=False
                    )
                    print("✓ Loaded checkpoint into model class")
                except Exception as e2:
                    print(f"Could not load from arch file: {e2}")
                    # Fallback: try loading checkpoint directly
                    model = pl.LightningModule.load_from_checkpoint(
                        checkpoint_path,
                        map_location='cpu',
                        strict=False
                    )
                    print("✓ Loaded model directly from checkpoint")
            else:
                # No architecture file, load directly
                model = pl.LightningModule.load_from_checkpoint(
                    checkpoint_path,
                    map_location='cpu',
                    strict=False
                )
                print("✓ Loaded model from checkpoint")
        
        # Extract the underlying PyTorch model from Lightning wrapper
        # Pylaia models typically have the actual model in model.model or model.net
        print(f"Model type: {type(model)}")
        print(f"Model has 'model' attribute: {hasattr(model, 'model')}")
        print(f"Model has 'net' attribute: {hasattr(model, 'net')}")
        print(f"Model has 'forward' method: {hasattr(model, 'forward')}")
        
        # Inspect model attributes
        attrs = [attr for attr in dir(model) if not attr.startswith('_') and not callable(getattr(model, attr, None))]
        print(f"Model attributes (non-callable): {attrs[:20]}")
        
        # Try multiple ways to extract the actual model
        actual_model = None
        
        # Method 1: Check for model.model
        if hasattr(model, 'model'):
            candidate = getattr(model, 'model')
            if hasattr(candidate, 'forward'):
                actual_model = candidate
                print("✓ Extracted model from model.model")
        
        # Method 2: Check for model.net
        if actual_model is None and hasattr(model, 'net'):
            candidate = getattr(model, 'net')
            if hasattr(candidate, 'forward'):
                actual_model = candidate
                print("✓ Extracted model from model.net")
        
        # Method 3: Check for model.crnn
        if actual_model is None and hasattr(model, 'crnn'):
            candidate = getattr(model, 'crnn')
            if hasattr(candidate, 'forward'):
                actual_model = candidate
                print("✓ Extracted model from model.crnn")
        
        # Method 4: Check state_dict keys to infer model structure
        if actual_model is None:
            try:
                state_dict = model.state_dict()
                keys = list(state_dict.keys())[:10]
                print(f"Model state dict keys (first 10): {keys}")
                
                # If state_dict is empty, check the checkpoint directly
                if not keys:
                    checkpoint = torch.load(checkpoint_path, map_location='cpu')
                    checkpoint_state = checkpoint.get('state_dict', {})
                    checkpoint_keys = list(checkpoint_state.keys())[:10]
                    print(f"Checkpoint state dict keys (first 10): {checkpoint_keys}")
                    
                    # If checkpoint keys start with 'model.', we need to create the model
                    # and load the state_dict with prefix handling
                    if checkpoint_keys and checkpoint_keys[0].startswith('model.'):
                        print("⚠ Checkpoint has 'model.' prefix but model.state_dict() is empty")
                        print("   This suggests the model wasn't properly initialized.")
                        print("   Attempting to manually load state_dict...")
                        
                        # Try to access model.model if it exists
                        if hasattr(model, 'model'):
                            actual_model = model.model
                            # Load state_dict into it, removing 'model.' prefix
                            filtered_state = {k.replace('model.', ''): v 
                                            for k, v in checkpoint_state.items() 
                                            if k.startswith('model.')}
                            actual_model.load_state_dict(filtered_state, strict=False)
                            print("✓ Manually loaded state_dict into model.model")
                elif keys and keys[0].startswith('model.'):
                    if hasattr(model, 'model'):
                        actual_model = model.model
                        print("✓ Extracted model from model.model (based on state_dict keys)")
            except Exception as e:
                print(f"Error in Method 4: {e}")
                import traceback
                traceback.print_exc()
        
        # Method 5: Try to access via __dict__
        if actual_model is None:
            try:
                model_dict = model.__dict__
                for key in ['model', 'net', 'crnn', 'network', 'module']:
                    if key in model_dict:
                        candidate = model_dict[key]
                        if hasattr(candidate, 'forward'):
                            actual_model = candidate
                            print(f"✓ Extracted model from __dict__['{key}']")
                            break
            except:
                pass
        
        # Method 6: Try to manually construct model and load state_dict
        if actual_model is None:
            try:
                # Check checkpoint structure
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                checkpoint_state = checkpoint.get('state_dict', {})
                if checkpoint_state:
                    checkpoint_keys = list(checkpoint_state.keys())[:5]
                    print(f"Attempting manual construction. Checkpoint keys: {checkpoint_keys}")
                    
                    # First, try loading the model class from the architecture file
                    ModelClass = None
                    if os.path.exists(model_arch_path):
                        try:
                            import pickle
                            with open(model_arch_path, 'rb') as f:
                                ModelClass = pickle.load(f)
                            print("✓ Loaded model class from architecture file")
                        except Exception as e:
                            print(f"Could not load model class from arch file: {e}")
                    
                    # If we have the model class, use it
                    if ModelClass is not None:
                        try:
                            # Try to instantiate - we might need hparams from checkpoint
                            hparams = checkpoint.get('hyper_parameters', {})
                            if hparams:
                                # Try instantiating with hparams
                                try:
                                    temp_model = ModelClass(**hparams)
                                    print("✓ Instantiated model from class with hparams")
                                except:
                                    # Try with just num_output_labels
                                    num_classes = len(syms)
                                    temp_model = ModelClass(num_output_labels=num_classes)
                                    print("✓ Instantiated model from class with num_output_labels")
                            else:
                                num_classes = len(syms)
                                temp_model = ModelClass(num_output_labels=num_classes)
                                print("✓ Instantiated model from class")
                            
                            # Load state_dict, handling 'model.' prefix
                            filtered_state = {k.replace('model.', ''): v 
                                            for k, v in checkpoint_state.items() 
                                            if k.startswith('model.')}
                            
                            # The model might be nested
                            if hasattr(temp_model, 'model'):
                                temp_model.model.load_state_dict(filtered_state, strict=False)
                                actual_model = temp_model.model
                            else:
                                temp_model.load_state_dict(filtered_state, strict=False)
                                actual_model = temp_model
                            
                            print("✓ Manually constructed and loaded model from architecture file")
                        except Exception as e:
                            print(f"Error instantiating model class: {e}")
                            import traceback
                            traceback.print_exc()
                    
                    # Fallback: Try to import CRNN from laia
                    if actual_model is None:
                        # Try different import paths
                        CRNN = None
                        
                        # Method 1: Try pylaia first
                        try:
                            from pylaia.models import CRNN
                            print("✓ Imported CRNN from pylaia.models")
                        except ImportError:
                            pass
                        
                        # Method 2: Try pylaia.models.crnn
                        if CRNN is None:
                            try:
                                from pylaia.models.crnn import CRNN
                                print("✓ Imported CRNN from pylaia.models.crnn")
                            except ImportError:
                                pass
                        
                        # Method 3: Try laia (fallback for older installations)
                        if CRNN is None:
                            try:
                                from laia.models import CRNN
                                print("✓ Imported CRNN from laia.models (fallback)")
                            except ImportError:
                                pass
                        
                        # Method 4: Try laia.models.crnn
                        if CRNN is None:
                            try:
                                from laia.models.crnn import CRNN
                                print("✓ Imported CRNN from laia.models.crnn (fallback)")
                            except ImportError:
                                pass
                        
                        # Method 5: Search in pylaia.models module
                        if CRNN is None:
                            try:
                                import pylaia.models as pylaia_models
                                # Check what's in pylaia.models
                                model_attrs = [attr for attr in dir(pylaia_models) if not attr.startswith('_')]
                                print(f"Available in pylaia.models: {model_attrs[:10]}")
                                
                                # Try common names
                                for name in ['CRNN', 'CRNNModel', 'CRNNModule', 'Model']:
                                    if hasattr(pylaia_models, name):
                                        CRNN = getattr(pylaia_models, name)
                                        print(f"✓ Found {name} in pylaia.models")
                                        break
                            except Exception as e:
                                print(f"Error searching pylaia.models: {e}")
                        
                        # Method 6: Search in laia.models (fallback)
                        if CRNN is None:
                            try:
                                import laia.models as laia_models
                                model_attrs = [attr for attr in dir(laia_models) if not attr.startswith('_')]
                                print(f"Available in laia.models: {model_attrs[:10]}")
                                
                                for name in ['CRNN', 'CRNNModel', 'CRNNModule', 'Model']:
                                    if hasattr(laia_models, name):
                                        CRNN = getattr(laia_models, name)
                                        print(f"✓ Found {name} in laia.models")
                                        break
                            except Exception as e:
                                print(f"Error searching laia.models: {e}")
                        
                        # Method 4: Check if checkpoint has the model class
                        if CRNN is None:
                            try:
                                if 'hyper_parameters' in checkpoint:
                                    hparams = checkpoint['hyper_parameters']
                                    print(f"Checkpoint hyper_parameters keys: {list(hparams.keys())[:10]}")
                            except:
                                pass
                        
                        if CRNN is None:
                            available = []
                            if 'model_attrs' in locals():
                                available = model_attrs[:10]
                            elif 'pylaia_models' in locals():
                                available = [attr for attr in dir(pylaia_models) if not attr.startswith('_')][:10]
                            raise ImportError(f"CRNN not found in pylaia.models or laia.models. Available: {available}")
                        
                        try:
                            num_classes = len(syms)
                            actual_model = CRNN(
                                num_input_channels=1,
                                num_output_labels=num_classes,
                                cnn_num_features=[16, 32, 64, 128, 256],
                                cnn_kernel_size=[[3, 3], [3, 3], [3, 3], [3, 3], [3, 3]],
                                cnn_stride=[[1, 1], [1, 1], [1, 1], [1, 1], [1, 1]],
                                cnn_dilation=[[1, 1], [1, 1], [1, 1], [1, 1], [1, 1]],
                                cnn_activation=['LeakyReLU', 'LeakyReLU', 'LeakyReLU', 'LeakyReLU', 'LeakyReLU'],
                                cnn_poolsize=[[2, 2], [2, 2], [2, 2], [2, 2], [0, 0]],
                                cnn_dropout=[0.0, 0.0, 0.0, 0.0, 0.0],
                                cnn_batchnorm=[True, True, True, True, True],
                                rnn_layers=3,
                                rnn_units=512,
                                rnn_dropout=0.5,
                                rnn_type='LSTM',
                                lin_dropout=0.5,
                                fixed_input_height=128,
                                adaptive_pooling='avg',
                            )
                            
                            # Load state_dict, removing 'model.' prefix
                            filtered_state = {k.replace('model.', ''): v 
                                            for k, v in checkpoint_state.items() 
                                            if k.startswith('model.')}
                            missing, unexpected = actual_model.load_state_dict(filtered_state, strict=False)
                            if missing:
                                print(f"⚠ Missing keys: {len(missing)}")
                            if unexpected:
                                print(f"⚠ Unexpected keys: {len(unexpected)}")
                            print("✓ Manually constructed and loaded model")
                        except ImportError as ie:
                            print(f"Could not import or instantiate CRNN: {ie}")
                            raise
                        except Exception as e:
                            print(f"Error instantiating CRNN: {e}")
                            import traceback
                            traceback.print_exc()
                            raise
            except Exception as e:
                print(f"Error in Method 6: {e}")
                import traceback
                traceback.print_exc()
        
        # Method 7: If model itself has forward, use it
        if actual_model is None and hasattr(model, 'forward'):
            try:
                # Test if forward actually works
                test_input = torch.randn(1, 1, 128, 100)
                _ = model(test_input)
                actual_model = model
                print("✓ Using model directly (forward method works)")
            except Exception as e:
                print(f"Forward test failed: {e}")
        
        # If still no model found, raise error with helpful info
        if actual_model is None:
            print("\n❌ Could not extract underlying model from Lightning wrapper")
            print(f"Model type: {type(model)}")
            print(f"Model dir: {[x for x in dir(model) if not x.startswith('_')][:30]}")
            raise RuntimeError(
                f"Cannot extract underlying PyTorch model from Lightning module.\n"
                f"Model type: {type(model)}\n"
                f"Please check the model architecture file or checkpoint structure."
            )
        
        # Set model to eval mode
        actual_model.eval()
        
        # Move to CPU (we'll handle GPU later if needed)
        actual_model = actual_model.cpu()
        
        print("✓ Model loaded successfully!")
        return actual_model, syms
        
    except Exception as e:
        print(f"Error loading model: {e}")
        print("\nTrying alternative: Manual model construction...")
        import traceback
        traceback.print_exc()
        
        # Last resort: Try to construct model manually
        try:
            # Try pylaia first
            try:
                from pylaia.models import CRNN
            except ImportError:
                try:
                    from pylaia.models.crnn import CRNN
                except ImportError:
                    # Fallback to laia
                    from laia.models.crnn import CRNN
            num_classes = len(syms)
            model = CRNN(
                num_input_channels=1,
                num_output_labels=num_classes,
                cnn_num_features=[16, 32, 64, 128, 256],
                cnn_kernel_size=[[3, 3], [3, 3], [3, 3], [3, 3], [3, 3]],
                cnn_stride=[[1, 1], [1, 1], [1, 1], [1, 1], [1, 1]],
                cnn_dilation=[[1, 1], [1, 1], [1, 1], [1, 1], [1, 1]],
                cnn_activation=['LeakyReLU', 'LeakyReLU', 'LeakyReLU', 'LeakyReLU', 'LeakyReLU'],
                cnn_poolsize=[[2, 2], [2, 2], [2, 2], [2, 2], [0, 0]],
                cnn_dropout=[0.0, 0.0, 0.0, 0.0, 0.0],
                cnn_batchnorm=[True, True, True, True, True],
                rnn_layers=3,
                rnn_units=512,
                rnn_dropout=0.5,
                rnn_type='LSTM',
                lin_dropout=0.5,
                fixed_input_height=128,
                adaptive_pooling='avg',
            )
            
            # Load state dict from checkpoint
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                # Remove 'model.' prefix if present
                state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
            else:
                state_dict = checkpoint
            
            model.load_state_dict(state_dict, strict=False)
            model.eval()
            print("✓ Manually constructed and loaded model")
            return model, syms
            
        except Exception as e2:
            raise RuntimeError(
                f"Failed to load model using all methods.\n"
                f"Last error: {e2}\n"
                f"Please ensure:\n"
                f"1. Checkpoint file exists: {checkpoint_path}\n"
                f"2. Model architecture file exists: {model_arch_path}\n"
                f"3. Symbols file exists: {symbols_path}"
            )


def find_master_record(image_name: str) -> str:
    """
    Find the master_record.json file for a given image.
    
    Args:
        image_name: Name of the source image (e.g., "CP40-565_481a.jpg")
        
    Returns:
        Path to master_record.json file
        
    Raises:
        FileNotFoundError: If master_record.json cannot be found
    """
    base_name = os.path.splitext(image_name)[0]
    base_name_clean = base_name.replace(' ', '_')
    
    # Try to extract group_id from image name
    # Group ID is typically like "CP40-565_481" (without the "a" suffix)
    group_id_candidates = [
        base_name.rsplit('_', 1)[0] if '_' in base_name else base_name,  # Without suffix
        base_name.split('_')[0] if '_' in base_name else base_name,  # First part
        base_name,  # Full name
        base_name_clean,
    ]
    
    # Search for master_record.json in OUTPUT_DIR
    for group_id in group_id_candidates:
        # Try with and without spaces/underscores
        for gid in [group_id, group_id.replace(' ', '_'), group_id.replace('_', ' ')]:
            master_path = os.path.join(OUTPUT_DIR, gid, "master_record.json")
            if os.path.exists(master_path):
                return master_path
    
    # Search recursively
    for root, dirs, files in os.walk(OUTPUT_DIR):
        if "master_record.json" in files:
            # Check if this master_record contains our image
            try:
                with open(os.path.join(root, "master_record.json"), 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    source_material = data.get("source_material", [])
                    for item in source_material:
                        if item.get("filename") == image_name or item.get("filename") == base_name + ".jpg":
                            return os.path.join(root, "master_record.json")
            except:
                continue
    
    raise FileNotFoundError(
        f"master_record.json not found for {image_name}.\n"
        f"Searched in: {OUTPUT_DIR}\n"
        f"Tip: Make sure the image has been processed through the workflow."
    )


def get_original_file_id(image_name: str, line_id: str) -> str:
    """
    Look up the original_file_id for a line_id from master_record.json.
    
    Args:
        image_name: Name of the source image
        line_id: Line identifier (e.g., "L01", "L02")
        
    Returns:
        The original_file_id (UUID string)
        
    Raises:
        FileNotFoundError: If master_record.json or line_id not found
    """
    master_record_path = find_master_record(image_name)
    
    print(f"Loading master_record.json from: {master_record_path}")
    with open(master_record_path, 'r', encoding='utf-8') as f:
        master_data = json.load(f)
    
    # Search through source_material for the line
    source_material = master_data.get("source_material", [])
    
    for item in source_material:
        filename = item.get("filename", "")
        # Match image name (with or without extension)
        if filename == image_name or filename == os.path.splitext(image_name)[0] + ".jpg":
            lines = item.get("lines", [])
            for line in lines:
                if line.get("line_id") == line_id:
                    original_file_id = line.get("original_file_id")
                    if original_file_id:
                        print(f"Found line_id {line_id} -> original_file_id: {original_file_id}")
                        return original_file_id
                    else:
                        raise ValueError(f"Line {line_id} found but has no original_file_id")
    
    # If not found, list available line_ids for debugging
    available_line_ids = []
    for item in source_material:
        filename = item.get("filename", "")
        if filename == image_name or filename == os.path.splitext(image_name)[0] + ".jpg":
            lines = item.get("lines", [])
            available_line_ids = [line.get("line_id") for line in lines if line.get("line_id")]
            break
    
    error_msg = f"Line ID '{line_id}' not found in master_record.json for {image_name}"
    if available_line_ids:
        error_msg += f"\nAvailable line IDs: {', '.join(available_line_ids[:20])}"
        if len(available_line_ids) > 20:
            error_msg += f" ... and {len(available_line_ids) - 20} more"
    else:
        error_msg += f"\nNo lines found for this image in master_record.json"
    
    raise FileNotFoundError(error_msg)


def find_line_image(image_name: str, line_id: str) -> str:
    """
    Find the path to a processed line image using original_file_id from master_record.json.
    
    Args:
        image_name: Name of the source image (e.g., "CP40-565_481a.jpg")
        line_id: Line identifier (e.g., "L01", "L02")
        
    Returns:
        Path to the line image file
    """
    # First, get the original_file_id from master_record.json
    original_file_id = get_original_file_id(image_name, line_id)
    
    # Line images are saved with the original_file_id as the filename
    # Structure: OUTPUT_DIR/{group_id}/{basename}/lines/{original_file_id}.png
    base_name = os.path.splitext(image_name)[0]
    base_name_clean = base_name.replace(' ', '_')
    
    # Extract group_id from image name (typically without the "a" suffix)
    group_id_candidates = [
        base_name.rsplit('_', 1)[0] if '_' in base_name else base_name,  # Without suffix
        base_name.split('_')[0] if '_' in base_name else base_name,  # First part
        base_name,  # Full name
        base_name_clean,
    ]
    
    # Try multiple possible locations
    search_paths = []
    
    # Method 1: Direct paths in OUTPUT_DIR (most common)
    for group_id in group_id_candidates:
        for gid in [group_id, group_id.replace(' ', '_'), group_id.replace('_', ' ')]:
            search_paths.extend([
                os.path.join(OUTPUT_DIR, gid, base_name, "lines", f"{original_file_id}.png"),
                os.path.join(OUTPUT_DIR, gid, base_name_clean, "lines", f"{original_file_id}.png"),
            ])
    
    # Method 2: In WORK_DIR structure
    for group_id in group_id_candidates:
        for gid in [group_id, group_id.replace(' ', '_'), group_id.replace('_', ' ')]:
            search_paths.extend([
                os.path.join(WORK_DIR, gid, base_name, "lines", f"{original_file_id}.png"),
                os.path.join(WORK_DIR, gid, base_name_clean, "lines", f"{original_file_id}.png"),
                os.path.join(WORK_DIR, base_name, "lines", f"{original_file_id}.png"),
                os.path.join(WORK_DIR, base_name_clean, "lines", f"{original_file_id}.png"),
            ])
    
    # Remove duplicates while preserving order
    seen = set()
    unique_paths = []
    for path in search_paths:
        if path not in seen:
            seen.add(path)
            unique_paths.append(path)
    
    # First try direct paths
    for path in unique_paths:
        if os.path.exists(path):
            return path
    
    # If not found, search recursively in OUTPUT_DIR first
    print(f"Searching recursively for {original_file_id}.png in {OUTPUT_DIR}...")
    for root, dirs, files in os.walk(OUTPUT_DIR):
        filename = f"{original_file_id}.png"
        if filename in files:
            found_path = os.path.join(root, filename)
            print(f"Found line image at: {found_path}")
            return found_path
    
    # Then search in WORK_DIR
    print(f"Searching recursively for {original_file_id}.png in {WORK_DIR}...")
    for root, dirs, files in os.walk(WORK_DIR):
        filename = f"{original_file_id}.png"
        if filename in files:
            found_path = os.path.join(root, filename)
            print(f"Found line image at: {found_path}")
            return found_path
    
    # If still not found, raise error
    print(f"\nError: Line image not found for {image_name} line {line_id}")
    print(f"Original file ID: {original_file_id}")
    print(f"\nSearched in {len(unique_paths)} specific locations and recursively in:")
    print(f"  - {OUTPUT_DIR}")
    print(f"  - {WORK_DIR}")
    print(f"\nTip: The image must be processed through the workflow first.")
    print(f"     Run: python workflow_manager.py")
    raise FileNotFoundError(
        f"Line image {original_file_id}.png not found for {image_name} line {line_id}"
    )


def preprocess_image(image_path: str) -> torch.Tensor:
    """
    Preprocess image for Pylaia model.
    
    Args:
        image_path: Path to line image
        
    Returns:
        Preprocessed image tensor (1, C, H, W)
    """
    # Load image
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    
    # Pylaia expects:
    # - Grayscale (1 channel)
    # - Fixed height of 128 pixels
    # - Width can vary (maintain aspect ratio)
    
    # Resize to fixed height 128, maintain aspect ratio
    original_width, original_height = img.size
    target_height = 128
    aspect_ratio = original_width / original_height
    target_width = int(target_height * aspect_ratio)
    
    img = img.resize((target_width, target_height), Image.LANCZOS)
    
    # Convert to numpy array and normalize
    img_array = np.array(img, dtype=np.float32) / 255.0
    
    # Convert to tensor: (H, W) -> (1, 1, H, W) for batch and channel
    img_tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0)
    
    return img_tensor


def calculate_log_probability(
    model: torch.nn.Module,
    symbols,
    image_tensor: torch.Tensor,
    text: str
) -> float:
    """
    Calculate log probability of text given image using CTC loss.
    
    Args:
        model: Loaded Pylaia model
        symbols: Symbols table for character mapping
        image_tensor: Preprocessed image tensor (1, C, H, W)
        text: Text string to evaluate
        
    Returns:
        Log probability (log P(text | image))
    """
    # Convert text to indices
    # Handle spaces - Pylaia often uses '<space>' instead of ' '
    # Also handle case sensitivity
    target_indices = []
    missing_chars = []
    
    for c in text:
        idx = None
        char_to_try = c
        
        # Try the character as-is first
        try:
            idx = symbols[char_to_try]
            # Check if it's actually None (not found) vs 0 (CTC blank)
            if idx is None:
                idx = None  # Explicitly set to None if not found
        except (KeyError, TypeError):
            idx = None
        
        # If not found and it's a space, try '<space>'
        if idx is None and c == ' ':
            try:
                idx = symbols['<space>']
                char_to_try = '<space>'
            except (KeyError, TypeError):
                idx = None
        
        # If still not found, try lowercase
        if idx is None and c.isupper():
            try:
                idx = symbols[c.lower()]
                if idx is not None:
                    char_to_try = c.lower()
            except (KeyError, TypeError):
                pass
        
        # If still not found, try uppercase
        if idx is None and c.islower():
            try:
                idx = symbols[c.upper()]
                if idx is not None:
                    char_to_try = c.upper()
            except (KeyError, TypeError):
                pass
        
        if idx is None:
            missing_chars.append(c)
        else:
            target_indices.append(idx)
    
    if missing_chars:
        print(f"Warning: Characters not in symbol table: {set(missing_chars)}")
        # Try to show available characters
        try:
            if hasattr(symbols, 'syms'):
                # laia.utils.SymbolsTable has a 'syms' attribute
                available = list(symbols.syms.keys())[:50]
                print(f"Available characters (first 50): {''.join(available)}")
            elif hasattr(symbols, 'keys'):
                available = list(symbols.keys())[:50]
                print(f"Available characters (first 50): {''.join(available)}")
        except:
            pass
        
        if not target_indices:
            print("Error: No valid characters found in text")
            return float('-inf')
        else:
            print(f"Using {len(target_indices)} valid characters out of {len(text)} total")
    
    if not target_indices:
        print("Error: Empty text string after filtering")
        return float('-inf')
    
    # Forward pass through model
    # The model should already be extracted in load_pylaia_model, but double-check
    with torch.no_grad():
        # Ensure we have the actual PyTorch model (not Lightning wrapper)
        if not hasattr(model, 'forward'):
            # Try to extract again (shouldn't be needed, but just in case)
            if hasattr(model, 'model'):
                model = model.model
            elif hasattr(model, 'net'):
                model = model.net
        
        # Call the model
        output = model(image_tensor)
        
        # Output shape should be (Time, Batch, Classes) or (Batch, Time, Classes)
        # Pylaia typically outputs (Time, Batch, Classes)
        print(f"Model output shape: {output.shape}")
        if len(output.shape) == 3:
            if output.shape[0] == image_tensor.shape[0]:
                # If first dimension is batch, transpose to (Time, Batch, Classes)
                output = output.transpose(0, 1)
                print(f"Transposed output shape: {output.shape}")
        
        # Apply log softmax for CTC (dim=2 is the class dimension)
        log_probs = F.log_softmax(output, dim=2)
    
    # Get dimensions
    T, N, C = log_probs.shape
    
    # Prepare for CTC loss
    target_tensor = torch.tensor(target_indices, dtype=torch.long)
    input_lengths = torch.tensor([T], dtype=torch.long)
    target_lengths = torch.tensor([len(target_indices)], dtype=torch.long)
    
    # Debug: Print target information
    print(f"Target indices ({len(target_indices)}): {target_indices[:20]}..." if len(target_indices) > 20 else f"Target indices: {target_indices}")
    print(f"Target length: {len(target_indices)}, Input length (T): {T}")
    
    # Check if target is longer than input (CTC requires T >= S for feasible alignment)
    if T < len(target_indices):
        print(f"Warning: Input sequence length ({T}) is shorter than target length ({len(target_indices)})")
        print("This will result in infinite loss (probability = 0)")
        return float('-inf')
    
    # CTC loss (blank=0 is standard in Pylaia)
    # Use 'sum' reduction to get total negative log probability
    # Note: zero_infinity=True converts inf losses to 0, which we don't want for comparison
    # So we'll set it to False and handle inf manually
    ctc_loss_fn = torch.nn.CTCLoss(blank=0, reduction='sum', zero_infinity=False)
    
    # Calculate loss
    # CTC expects: log_probs (T, N, C), targets (S,), input_lengths (N,), target_lengths (N,)
    try:
        loss = ctc_loss_fn(log_probs, target_tensor, input_lengths, target_lengths)
        loss_value = loss.item()
        print(f"CTC Loss (raw): {loss_value:.6f}")
    except RuntimeError as e:
        print(f"Error calculating CTC loss: {e}")
        return float('-inf')
    
    # Convert loss to log probability
    # CTC loss with 'sum' reduction = -log(P(text|image))
    # So log(P) = -loss
    # Handle infinite loss (impossible alignment)
    if not torch.isfinite(torch.tensor(loss_value)) or loss_value == float('inf'):
        print("Warning: Infinite loss (impossible alignment)")
        return float('-inf')
    
    log_prob = -loss_value
    
    return log_prob


def main():
    parser = argparse.ArgumentParser(
        description='Calculate log probability of a string given a line image',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Calculate probability for a specific reading
  python calculate_string_probability.py "CP40-565_481a.jpg" "L02" "William Smith"
  
  # Compare multiple readings
  python calculate_string_probability.py "CP40-565_481a.jpg" "L02" "William Smyth"
        """
    )
    
    parser.add_argument(
        'image_name',
        help='Name of the source image (e.g., "CP40-565_481a.jpg")'
    )
    parser.add_argument(
        'line_id',
        help='Line identifier (e.g., "L02")'
    )
    parser.add_argument(
        'alternate_string',
        help='Alternate text reading to evaluate'
    )
    parser.add_argument(
        '--model',
        default=PYLAIA_MODEL,
        help=f'Path to model checkpoint (default: {PYLAIA_MODEL})'
    )
    parser.add_argument(
        '--arch',
        default=PYLAIA_ARCH,
        help=f'Path to model architecture file (default: {PYLAIA_ARCH})'
    )
    parser.add_argument(
        '--syms',
        default=PYLAIA_SYMS,
        help=f'Path to symbols file (default: {PYLAIA_SYMS})'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Pylaia String Probability Calculator")
    print("=" * 70)
    print(f"Image: {args.image_name}")
    print(f"Line ID: {args.line_id}")
    print(f"Alternate reading: {args.alternate_string}")
    print()
    
    try:
        # Load model
        print("Loading model...")
        model, symbols = load_pylaia_model(args.model, args.arch, args.syms)
        
        # Find line image
        print(f"\nFinding line image...")
        line_image_path = find_line_image(args.image_name, args.line_id)
        print(f"Found: {line_image_path}")
        
        # Preprocess image
        print(f"\nPreprocessing image...")
        image_tensor = preprocess_image(line_image_path)
        print(f"Image tensor shape: {image_tensor.shape}")
        
        # Calculate log probability
        print(f"\nCalculating log probability...")
        log_prob = calculate_log_probability(
            model, symbols, image_tensor, args.alternate_string
        )
        
        # Convert to probability
        probability = np.exp(log_prob) if log_prob != float('-inf') else 0.0
        
        # Print results
        print("\n" + "=" * 70)
        print("RESULTS")
        print("=" * 70)
        print(f"Text: {args.alternate_string}")
        print(f"Log Probability: {log_prob:.4f}")
        print(f"Probability: {probability:.6f}")
        print(f"Probability (%): {probability * 100:.4f}%")
        print("=" * 70)
        
        return 0
        
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        return 1
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())

