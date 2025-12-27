#!/usr/bin/env python3
"""Script to generate PDF report for a CP40 processing directory."""

import os
import sys
import subprocess
import json
import time
import traceback
from pathlib import Path

def install_dependencies():
    """Install required dependencies if missing."""
    required_packages = [
        'numpy',
        'scipy',
        'pillow',
        'opencv-python',
        'tqdm'
    ]
    
    print("Checking and installing dependencies...")
    for package in required_packages:
        try:
            __import__(package.replace('-', '_') if package == 'opencv-python' else package)
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])

def generate_report(master_record_path):
    """Generate LaTeX report from master_record.json."""
    try:
        # Add project root to path
        project_root = Path(__file__).parent
        sys.path.insert(0, str(project_root))
        
        # Import after path is set
        from report_generator.report import generate_latex_report
        
        print(f"Loading master record from: {master_record_path}")
        try:
            with open(master_record_path, 'r', encoding='utf-8') as f:
                master_data = json.load(f)
        except json.JSONDecodeError as e:
            error_msg = f"ERROR: Failed to parse JSON from {master_record_path}: {e}"
            print(f"\n{'='*80}", file=sys.stderr)
            print(error_msg, file=sys.stderr)
            print(f"{'='*80}\n", file=sys.stderr)
            print("Waiting 5 seconds for you to see this error...")
            time.sleep(5)
            return None
        except Exception as e:
            error_msg = f"ERROR: Failed to read master_record.json: {e}"
            print(f"\n{'='*80}", file=sys.stderr)
            print(error_msg, file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)
            print(f"{'='*80}\n", file=sys.stderr)
            print("Waiting 5 seconds for you to see this error...")
            time.sleep(5)
            return None
        
        # Determine output directory
        output_dir = os.path.dirname(os.path.abspath(master_record_path))
        
        # Get API key from environment
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key or not api_key.strip():
            error_msg = (
                "ERROR: GEMINI_API_KEY environment variable is REQUIRED for postea and pleadings matching.\n"
                "Please set GEMINI_API_KEY environment variable with a valid Gemini API key."
            )
            print(f"\n{'='*80}", file=sys.stderr)
            print(error_msg, file=sys.stderr)
            print(f"{'='*80}\n", file=sys.stderr)
            print("Waiting 5 seconds for you to see this error...")
            time.sleep(5)
            return None
        
        # Generate LaTeX file
        print("Generating LaTeX report...")
        try:
            metrics = generate_latex_report(master_data, filename=None, api_key=api_key)
        except Exception as e:
            error_msg = f"ERROR: Failed to generate LaTeX report: {e}"
            print(f"\n{'='*80}", file=sys.stderr)
            print(error_msg, file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)
            print(f"{'='*80}\n", file=sys.stderr)
            print("Waiting 5 seconds for you to see this error...")
            time.sleep(5)
            return None
        
        # Find the generated .tex file
        meta = master_data.get("case_metadata", {})
        roll_number = meta.get("roll_number", "unknown")
        rotulus_number = meta.get("rotulus_number", "unknown")
        tex_filename = f"comparison_report_CP40-{roll_number}_{rotulus_number}.tex"
        tex_path = os.path.join(output_dir, tex_filename)
        
        if not os.path.exists(tex_path):
            error_msg = f"ERROR: LaTeX file not found at {tex_path}"
            print(f"\n{'='*80}", file=sys.stderr)
            print(error_msg, file=sys.stderr)
            print(f"Expected filename: {tex_filename}", file=sys.stderr)
            print(f"Output directory: {output_dir}", file=sys.stderr)
            print(f"Directory contents:", file=sys.stderr)
            if os.path.exists(output_dir):
                for f in os.listdir(output_dir):
                    if f.endswith('.tex'):
                        print(f"  - {f}", file=sys.stderr)
            print(f"{'='*80}\n", file=sys.stderr)
            print("Waiting 5 seconds for you to see this error...")
            time.sleep(5)
            return None
        
        print(f"LaTeX file generated: {tex_path}")
        return tex_path
    except Exception as e:
        error_msg = f"ERROR: Unexpected error in generate_report: {e}"
        print(f"\n{'='*80}", file=sys.stderr)
        print(error_msg, file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        print(f"{'='*80}\n", file=sys.stderr)
        print("Waiting 5 seconds for you to see this error...")
        time.sleep(5)
        return None

def compile_pdf(tex_path):
    """Compile LaTeX file to PDF using xelatex."""
    output_dir = os.path.dirname(tex_path)
    tex_file = os.path.basename(tex_path)
    
    print(f"Compiling PDF from {tex_file}...")
    print("This requires xelatex to be installed.")
    
    # Change to output directory for compilation
    original_dir = os.getcwd()
    try:
        os.chdir(output_dir)
        
        # Check if xelatex is available first
        try:
            check_result = subprocess.run(
                ['xelatex', '--version'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if check_result.returncode != 0:
                error_msg = "ERROR: xelatex --version failed. xelatex may not be properly installed."
                print(f"\n{'='*80}", file=sys.stderr)
                print(error_msg, file=sys.stderr)
                print("Please install TeX Live or MiKTeX:", file=sys.stderr)
                print("  Ubuntu/Debian: sudo apt-get install texlive-xetex", file=sys.stderr)
                print(f"{'='*80}\n", file=sys.stderr)
                print("Waiting 5 seconds for you to see this error...")
                time.sleep(5)
                return None
        except FileNotFoundError:
            error_msg = "ERROR: 'xelatex' command not found. Please install TeX Live or MiKTeX."
            print(f"\n{'='*80}", file=sys.stderr)
            print(error_msg, file=sys.stderr)
            print("On Ubuntu/Debian: sudo apt-get install texlive-xetex", file=sys.stderr)
            print(f"{'='*80}\n", file=sys.stderr)
            print("Waiting 5 seconds for you to see this error...")
            time.sleep(5)
            return None
        except subprocess.TimeoutExpired:
            error_msg = "ERROR: xelatex --version timed out."
            print(f"\n{'='*80}", file=sys.stderr)
            print(error_msg, file=sys.stderr)
            print(f"{'='*80}\n", file=sys.stderr)
            print("Waiting 5 seconds for you to see this error...")
            time.sleep(5)
            return None
        except Exception as e:
            error_msg = f"ERROR: Could not verify xelatex installation: {e}"
            print(f"\n{'='*80}", file=sys.stderr)
            print(error_msg, file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)
            print(f"{'='*80}\n", file=sys.stderr)
            print("Waiting 5 seconds for you to see this error...")
            time.sleep(5)
            return None
        
        # Run xelatex (may need to run multiple times for TOC and references)
        # First pass: collect TOC entries and write .toc file
        # Second pass: read .toc file and generate TOC with page numbers
        # Third pass: resolve all cross-references and finalize TOC
        for i in range(3):
            print(f"Running xelatex pass {i+1}/3 (TOC generation requires multiple passes)...")
            try:
                result = subprocess.run(
                    ['xelatex', '-interaction=nonstopmode', '-halt-on-error', tex_file],
                    capture_output=True,
                    text=True,
                    timeout=120,
                    cwd=output_dir
                )
                # Small delay to ensure file system sync between passes
                time.sleep(0.5)
                
                # Check if .toc file exists and has content after each pass
                toc_file = os.path.splitext(tex_file)[0] + '.toc'
                toc_path = os.path.join(output_dir, toc_file)
                if os.path.exists(toc_path):
                    toc_size = os.path.getsize(toc_path)
                    if toc_size > 0:
                        print(f"  Pass {i+1}: TOC file exists ({toc_size} bytes)")
                        if i == 2:  # After final pass, show a preview
                            try:
                                with open(toc_path, 'r', encoding='utf-8') as f:
                                    toc_content = f.read()
                                    if 'contentsline' in toc_content.lower():
                                        print(f"  TOC file contains entries (found 'contentsline')")
                                    else:
                                        print(f"  WARNING: TOC file exists but may be empty or malformed")
                            except Exception as e:
                                print(f"  Could not read TOC file: {e}")
                    else:
                        print(f"  Pass {i+1}: TOC file exists but is empty (0 bytes)")
                else:
                    if i == 0:
                        print(f"  Pass {i+1}: TOC file not yet created (this is normal for pass 1)")
                    else:
                        print(f"  WARNING: Pass {i+1}: TOC file still not found")
                if result.returncode != 0:
                    error_msg = f"ERROR: xelatex run {i+1} failed with return code {result.returncode}"
                    print(f"\n{'='*80}", file=sys.stderr)
                    print(error_msg, file=sys.stderr)
                    if result.stderr:
                        print(f"\nStderr (first 1000 chars):", file=sys.stderr)
                        print(result.stderr[:1000], file=sys.stderr)
                    if result.stdout:
                        # Look for error messages in stdout
                        error_lines = [line for line in result.stdout.split('\n') if 'Error' in line or 'Fatal' in line or '!' in line]
                        if error_lines:
                            print(f"\nError lines from stdout:", file=sys.stderr)
                            print('\n'.join(error_lines[:50]), file=sys.stderr)
                    print(f"{'='*80}\n", file=sys.stderr)
            except subprocess.TimeoutExpired:
                error_msg = f"ERROR: xelatex run {i+1} timed out after 120 seconds"
                print(f"\n{'='*80}", file=sys.stderr)
                print(error_msg, file=sys.stderr)
                print(f"{'='*80}\n", file=sys.stderr)
                print("Waiting 5 seconds for you to see this error...")
                time.sleep(5)
                return None
            except Exception as e:
                error_msg = f"ERROR: Exception during xelatex run {i+1}: {e}"
                print(f"\n{'='*80}", file=sys.stderr)
                print(error_msg, file=sys.stderr)
                print(traceback.format_exc(), file=sys.stderr)
                print(f"{'='*80}\n", file=sys.stderr)
                print("Waiting 5 seconds for you to see this error...")
                time.sleep(5)
                return None
        
        pdf_path = tex_path.replace('.tex', '.pdf')
        if os.path.exists(pdf_path):
            print(f"\n{'='*60}")
            print(f"PDF REPORT GENERATED: {pdf_path}")
            print(f"{'='*60}\n")
            return pdf_path
        else:
            error_msg = "ERROR: PDF file was not generated after compilation."
            print(f"\n{'='*80}", file=sys.stderr)
            print(error_msg, file=sys.stderr)
            print(f"Expected PDF path: {pdf_path}", file=sys.stderr)
            print(f"Working directory: {output_dir}", file=sys.stderr)
            if os.path.exists(output_dir):
                print("Files in directory:", file=sys.stderr)
                for f in os.listdir(output_dir):
                    if f.endswith(('.pdf', '.tex', '.log', '.aux')):
                        print(f"  - {f}", file=sys.stderr)
            print("Make sure xelatex is installed:", file=sys.stderr)
            print("  Ubuntu/Debian: sudo apt-get install texlive-xetex", file=sys.stderr)
            print(f"{'='*80}\n", file=sys.stderr)
            print("Waiting 5 seconds for you to see this error...")
            time.sleep(5)
            return None
    except Exception as e:
        error_msg = f"ERROR: Unexpected error in compile_pdf: {e}"
        print(f"\n{'='*80}", file=sys.stderr)
        print(error_msg, file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        print(f"{'='*80}\n", file=sys.stderr)
        print("Waiting 5 seconds for you to see this error...")
        time.sleep(5)
        return None
    finally:
        os.chdir(original_dir)

def main():
    if len(sys.argv) < 2:
        print("Usage: python generate_pdf_report.py <directory_path>")
        print("Example: python generate_pdf_report.py cp40_processing/output/CP40-562_534d")
        sys.exit(1)
    
    directory = sys.argv[1]
    
    # Convert Windows path if needed
    if directory.startswith('\\\\'):
        # WSL path conversion
        directory = directory.replace('\\\\wsl.localhost\\Ubuntu', '').replace('\\', '/')
    
    # Find master_record.json
    master_record_path = os.path.join(directory, 'master_record.json')
    if not os.path.exists(master_record_path):
        print(f"Error: master_record.json not found in {directory}")
        sys.exit(1)
    
    # Install dependencies
    try:
        install_dependencies()
    except Exception as e:
        print(f"Warning: Could not install all dependencies: {e}")
        print("Continuing anyway...")
    
    # Generate LaTeX report
    try:
        tex_path = generate_report(master_record_path)
        if not tex_path:
            error_msg = "ERROR: Failed to generate LaTeX report"
            print(f"\n{'='*80}", file=sys.stderr)
            print(error_msg, file=sys.stderr)
            print(f"{'='*80}\n", file=sys.stderr)
            print("Waiting 5 seconds for you to see this error...")
            time.sleep(5)
            sys.exit(1)
        
        # Compile to PDF
        pdf_path = compile_pdf(tex_path)
        if pdf_path:
            print(f"Success! PDF report available at: {pdf_path}")
        else:
            print(f"\nLaTeX file generated at: {tex_path}")
            print("You can compile it manually with:")
            print(f"  cd {os.path.dirname(tex_path)}")
            print(f"  xelatex {os.path.basename(tex_path)}")
            print("Waiting 5 seconds for you to see this message...")
            time.sleep(5)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting...")
        sys.exit(1)
    except Exception as e:
        error_msg = f"ERROR: Unexpected error in main: {e}"
        print(f"\n{'='*80}", file=sys.stderr)
        print(error_msg, file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        print(f"{'='*80}\n", file=sys.stderr)
        print("Waiting 5 seconds for you to see this error...")
        time.sleep(5)
        sys.exit(1)

if __name__ == '__main__':
    main()

