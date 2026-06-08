#!/usr/bin/env python3
"""
Semantic Folding Pipeline Runner
Interactive TUI for executing the semantic folding pipeline with state management.

This script provides an interactive command-line interface for running a 7-step
semantic folding pipeline:
  1. Phrase Extraction
  2. Term-Context Matrix
  3. Semantic Space
  4. Phrase Fingerprints
  5. Document Fingerprints
  6. Customtext Fingerprints
  7. Query Processing

Features:
- State persistence across runs (config/exec_state.yml)
- Configuration defaults (config/semantic_folding.yml)
- Dynamic menu based on execution state
- Parameter collection with smart defaults from previous steps
- Run history management
- Extensible visualization system (phrase, document and customtext fingerprints)

Architecture:
- SemanticRunner: Main pipeline orchestrator
- VisualizationHandler: Abstract base for visualization operations
- PhraseVisualizationHandler: Phrase fingerprint visualization
- DocumentVisualizationHandler: Document fingerprint visualization
- CustomtextVisualizationHandler: Customtext fingerprint visualization
"""

import os, time
import shutil
import sys
import yaml
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from abc import ABC, abstractmethod
from lib import get_logger
logger = get_logger("semantic_folder")

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
    # level="INFO"
    level="DEBUG"
)
logger.add(
    "logs/semantic_runner.log",
    rotation="10 MB",
    retention="7 days",
    level="DEBUG",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}"
)


# ============================================================================
# TERMINAL COLORS
# ============================================================================

class Colors:
    """ANSI color codes for terminal output."""
    HEADER    = '\033[95m'
    BLUE      = '\033[94m'
    CYAN      = '\033[96m'
    GREEN     = '\033[92m'
    YELLOW    = '\033[93m'
    RED       = '\033[91m'
    ENDC      = '\033[0m'
    BOLD      = '\033[1m'
    UNDERLINE = '\033[4m'


# ============================================================================
# VISUALIZATION HANDLER BASE CLASS
# ============================================================================

class VisualizationHandler(ABC):
    """
    Abstract base class for visualization handlers.
    
    Each visualization type (phrase, document, customtext, etc.) should implement this
    interface to provide consistent parameter collection and execution.
    """
    
    def __init__(self, runner: 'SemanticRunner'):
        self.runner = runner
    
    @abstractmethod
    def get_step_definition(self) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def collect_parameters(self, run_id: str) -> Optional[Dict[str, str]]:
        pass
    
    @abstractmethod
    def build_command(self, params: Dict[str, str]) -> List[str]:
        pass
    
    def execute(self, run_id: str) -> bool:
        logger.info(f"Starting {self.__class__.__name__}")
        params = self.collect_parameters(run_id)
        if params is None:
            self.runner.print_warning("Visualization cancelled")
            return False
        cmd = self.build_command(params)
        logger.info(f"Executing visualization command")
        logger.debug(f"Full command: {' '.join(cmd)}")
        print(f"\n{Colors.CYAN}{'─' * 70}{Colors.ENDC}")
        print(f"{Colors.BOLD}Executing:{Colors.ENDC} {' '.join(cmd)}")
        print(f"{Colors.CYAN}{'─' * 70}{Colors.ENDC}\n")
        try:
            subprocess.run(cmd, check=True, text=True)
            logger.info("Visualization completed successfully")
            self.runner.print_success("Visualization completed successfully")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Visualization failed (code {e.returncode})")
            self.runner.print_error(f"Visualization failed (code {e.returncode})")
            return False
        except Exception as e:
            logger.exception(f"Unexpected error during visualization: {e}")
            self.runner.print_error(f"Unexpected error: {e}")
            return False


# ============================================================================
# PHRASE VISUALIZATION HANDLER
# ============================================================================

class PhraseVisualizationHandler(VisualizationHandler):
    """
    Handler for phrase fingerprint visualization.
    
    Supports two modes:
    - Single: Visualize one phrase fingerprint
    - Compare: Side-by-side comparison of two phrase fingerprints
    
    Auto-calculates figure height based on mode and width:
    - Single mode: height = width / 3 (horizontal layout)
    - Compare mode: height constrained between 2/3 and 3/3 of width
    """
    
    def get_step_definition(self) -> Dict[str, Any]:
        return {
            'id': 'viz',
            'name': 'Phrase Fingerprint Visualization',
            'script': 'semantic_folding/phrase_visualizer.py',
            'required_params': ['fingerprints', 'output'],
            'optional_params': [
                'grid_size', 'threshold', 'morton', 'grid_borders',
                'border_color', 'border_width', 'max_shapes', 'width', 'height',
                'colorscale', 'generate_html', 'generate_png', 'save_metadata'
            ],
            'default_output': 'phrase_viz'
        }
    
    def handle(self):
        self.runner.print_header("Phrase Fingerprint Visualization")
        run_id = self.runner.exec_state.get("last_run_id", "default")
        params = self.collect_parameters(run_id)
        if not params:
            return
        cmd = self.build_command(params)
        print(f"\n{Colors.CYAN}{'─' * 70}{Colors.ENDC}")
        print(f"{Colors.BOLD}Executing:{Colors.ENDC} {' '.join(cmd)}")
        print(f"{Colors.CYAN}{'─' * 70}{Colors.ENDC}\n")
        try:
            subprocess.run(cmd, check=True, text=True)
            self.runner.print_success("Visualization completed successfully")
        except subprocess.CalledProcessError as e:
            self.runner.print_error(f"Visualization failed (code {e.returncode})")
        except Exception as e:
            logger.exception(f"Unexpected error during visualization: {e}")
            self.runner.print_error(f"Unexpected error: {e}")

    def collect_parameters(self, run_id: str) -> Optional[Dict[str, str]]:
        logger.info("Collecting parameters for phrase visualization")
        params = {}
        
        print(f"\n{Colors.BOLD}Configure: Phrase Fingerprint Visualization{Colors.ENDC}")
        print(f"{Colors.CYAN}{'─' * 70}{Colors.ENDC}")
        
        # 1. Fingerprints directory
        default_fp = self._get_step4_output() or f'outputs/{run_id}/phrase_fingerprints'
        fingerprints = self.runner.get_input(
            f"{Colors.BOLD}fingerprints{Colors.ENDC} (required)", default_fp
        )
        while not fingerprints:
            self.runner.print_error("'fingerprints' is required")
            fingerprints = self.runner.get_input(
                f"{Colors.BOLD}fingerprints{Colors.ENDC} (required)", default_fp
            )
        params['fingerprints'] = fingerprints
        
        # 2. Output directory
        default_out = f'outputs/{run_id}/phrase_viz'
        output = self.runner.get_input(
            f"{Colors.BOLD}output{Colors.ENDC} (required)", default_out
        )
        while not output:
            self.runner.print_error("'output' is required")
            output = self.runner.get_input(
                f"{Colors.BOLD}output{Colors.ENDC} (required)", default_out
            )
        params['output'] = output
        
        # 3. Visualization mode
        mode_choice = self.runner.get_choice(
            "Select visualization mode:",
            ['Visualize single phrase', 'Compare two phrases', 'Cancel']
        )
        if mode_choice == 3:
            return None
        mode = 'single' if mode_choice == 1 else 'compare'
        
        # 4. Phrase(s)
        if mode == 'single':
            phrase = self.runner.get_input(f"{Colors.BOLD}phrase{Colors.ENDC} (required)", None)
            while not phrase:
                self.runner.print_error("'phrase' is required for single mode")
                phrase = self.runner.get_input(f"{Colors.BOLD}phrase{Colors.ENDC} (required)", None)
            params['phrase'] = phrase
        else:
            phrase1 = self.runner.get_input(f"{Colors.BOLD}phrase1{Colors.ENDC} (required)", None)
            while not phrase1:
                self.runner.print_error("'phrase1' is required for compare mode")
                phrase1 = self.runner.get_input(f"{Colors.BOLD}phrase1{Colors.ENDC} (required)", None)
            phrase2 = self.runner.get_input(f"{Colors.BOLD}phrase2{Colors.ENDC} (required)", None)
            while not phrase2:
                self.runner.print_error("'phrase2' is required for compare mode")
                phrase2 = self.runner.get_input(f"{Colors.BOLD}phrase2{Colors.ENDC} (required)", None)
            params['phrase1'] = phrase1
            params['phrase2'] = phrase2
        
        # 5. Optional parameters
        print(f"\n{Colors.CYAN}Optional parameters (Enter to skip):{Colors.ENDC}")
        
        # Width first
        width_default = self.runner.get_default_value("width", "viz")
        width_val = self.runner.get_input(f"  width", width_default)
        if width_val:
            params['width'] = width_val
            width = int(width_val)
            if mode == 'single':
                height = width // 3
                params['height'] = str(height)
                print(f"  {Colors.GREEN}✓ Auto-calculated height (1/3 width): {height}{Colors.ENDC}")
            else:
                min_height = int(width * 2 / 3)
                max_height = width
                height_default = str(width)
                while True:
                    height_val = self.runner.get_input(
                        f"  height (must be between {min_height} and {max_height})", height_default
                    )
                    if not height_val:
                        break
                    height = int(height_val)
                    if min_height <= height <= max_height:
                        params['height'] = str(height)
                        break
                    else:
                        self.runner.print_error(f"Height must be between {min_height} and {max_height}")
        
        # Scalar parameters
        optional_params = [
            ('grid_size', 'Grid size'),
            ('threshold', 'Activation threshold'),
            ('border_color', 'Border color'),
            ('border_width', 'Border width'),
            ('max_shapes', 'Maximum shapes to render'),
            ('colorscale', 'Plotly colorscale name'),
        ]
        for param_name, param_prompt in optional_params:
            default = self.runner.get_default_value(param_name, "viz")
            value = self.runner.get_input(f"  {param_prompt}", default)
            if value:
                params[param_name] = value
        
        # Boolean parameters (positive semantics)
        bool_params = [
            ('morton', 'Use Morton (Z-order) encoding (true/false)'),
            ('grid_borders', 'Show grid borders (true/false)'),
            ('generate_html', 'Generate HTML output (true/false)'),
            ('generate_png', 'Generate PNG output (true/false)'),
            ('save_metadata', 'Save metadata JSON (true/false)')
        ]
        for param_name, param_prompt in bool_params:
            default = self.runner.get_default_value(param_name, "viz")
            value = self.runner.get_input(f"  {param_prompt}", default)
            if value:
                params[param_name] = value
        
        return params
    
    def _get_step4_output(self) -> Optional[str]:
        run_id = self.runner.exec_state.get("last_run_id")
        if not run_id:
            return None
        run_data = self.runner.exec_state["runs"].get(run_id, {})
        step4_data = run_data.get("steps", {}).get(4)
        if step4_data and "output" in step4_data:
            output_path = step4_data["output"]
            if Path(output_path).exists():
                return output_path
        return None
    
    def build_command(self, params: Dict[str, str]) -> List[str]:
        script_path = str(Path(__file__).resolve().parent.parent / "semantic_folding/phrase_visualizer.py")
        cmd = [
            "D:\\darsi\\ms\\Thesis\\Dr.Banaie\\code050302\\SemanticFolding\\.venv\\scripts\\python",
            script_path
        ]
        cmd.extend(['--fingerprints', params['fingerprints']])
        cmd.extend(['--output', params['output']])
        
        if 'phrase' in params:
            cmd.extend(['--phrase', params['phrase']])
        elif 'phrase1' in params and 'phrase2' in params:
            cmd.extend(['--phrase1', params['phrase1']])
            cmd.extend(['--phrase2', params['phrase2']])
        
        for param, value in params.items():
            if param in ['fingerprints', 'output', 'phrase', 'phrase1', 'phrase2']:
                continue
            flag_name = self.runner.CLI_RENAME_MAP.get(param, param)
            flag = f"--{flag_name.replace('_', '-')}"
            if param in self.runner.NEGATE_FLAG_MAP:
                if str(value).lower() in ("false", "no", "0"):
                    cmd.append(f"--{self.runner.NEGATE_FLAG_MAP[param]}")
                    logger.debug(f"Added negation flag: --{self.runner.NEGATE_FLAG_MAP[param]}")
            elif str(value).lower() in ("true", "false"):
                if str(value).lower() == "true":
                    cmd.append(flag)
                    logger.debug(f"Added boolean flag: {flag}")
            else:
                cmd.extend([flag, value])
                logger.debug(f"Added param: {flag} {value}")
        return cmd


class DocumentVisualizationHandler(VisualizationHandler):
    """Handler for document fingerprint visualization."""
    def __init__(self, runner):
        super().__init__(runner)

    def get_step_definition(self) -> Dict[str, Any]:
        return {
            'id': 'doc_viz',
            'name': 'Document Fingerprint Visualization',
            'script': 'semantic_folding/doc_visualizer.py',
            'required_params': ['run_dir', 'doc_id', 'output'],
            'optional_params': ['no_grid_borders', 'border_color', 'border_width'],
            'default_output': 'doc_viz'
        }
    
    def collect_parameters(self, run_id: str) -> Optional[Dict[str, Any]]:
        logger.info("Collecting parameters for document visualization")
        params = {}
        
        print(f"\n{Colors.BOLD}Configure: Document Fingerprint Visualization{Colors.ENDC}")
        print(f"{Colors.CYAN}{'─' * 70}{Colors.ENDC}")
        
        # Select run directory
        runs = self._list_available_runs()
        if not runs:
            print(f"{Colors.RED}❌ No runs found in outputs/{Colors.ENDC}")
            input("\nPress Enter to continue...")
            return None
        
        print(f"\n{Colors.BOLD}Available runs:{Colors.ENDC}")
        for i, run in enumerate(runs, 1):
            print(f"  {i}. {run.name}")
        
        run_choice = input(f"\n{Colors.BOLD}Select run number:{Colors.ENDC} ").strip()
        try:
            run_idx = int(run_choice) - 1
            if 0 <= run_idx < len(runs):
                params['run_dir'] = str(runs[run_idx])
            else:
                print(f"{Colors.RED}❌ Invalid run selection{Colors.ENDC}")
                input("\nPress Enter to continue...")
                return None
        except ValueError:
            print(f"{Colors.RED}❌ Invalid input{Colors.ENDC}")
            input("\nPress Enter to continue...")
            return None
        
        # Document ID
        doc_id = self.runner.get_input(
            f"{Colors.BOLD}Enter document ID{Colors.ENDC} (e.g., doc_001)", None
        )
        while not doc_id:
            self.runner.print_error("'doc_id' is required")
            doc_id = self.runner.get_input(
                f"{Colors.BOLD}Enter document ID{Colors.ENDC} (e.g., doc_001)", None
            )
        params['doc_id'] = doc_id
        
        # Output directory (inside run directory)
        default_output = str(Path(params['run_dir']) / 'doc_viz')
        output = self.runner.get_input(
            f"{Colors.BOLD}Output directory{Colors.ENDC}", default_output
        )
        params['output'] = output if output else default_output
        
        # Optional parameters
        print(f"\n{Colors.CYAN}Optional parameters (Enter to skip):{Colors.ENDC}")
        show_borders = self.runner.get_input(
            f"  Show 4×4 grid borders? (y/n)", "y"
        )
        params['no_grid_borders'] = (show_borders.lower() == 'n')
        if show_borders.lower() != 'n':
            border_color = self.runner.get_input(f"  Border color", "lightgray")
            params['border_color'] = border_color if border_color else 'lightgray'
            border_width = self.runner.get_input(f"  Border width", "1.0")
            try:
                params['border_width'] = float(border_width) if border_width else 1.0
            except ValueError:
                print(f"  {Colors.YELLOW}⚠️  Invalid border width, using default 1.0{Colors.ENDC}")
                params['border_width'] = 1.0
        
        valid, error_msg = self._validate_params(params)
        if not valid:
            print(f"{Colors.RED}❌ Validation failed: {error_msg}{Colors.ENDC}")
            input("\nPress Enter to continue...")
            return None
        return params
    
    def handle(self):
        print(f"\n{Colors.BOLD}Document Fingerprint Visualization{Colors.ENDC}")
        params = self.collect_parameters(run_id="")
        if not params:
            return
        cmd = self.build_command(params)
        print(f"\n{Colors.CYAN}{'─' * 70}{Colors.ENDC}")
        print(f"{Colors.BOLD}Executing:{Colors.ENDC} {' '.join(str(c) for c in cmd)}")
        print(f"{Colors.CYAN}{'─' * 70}{Colors.ENDC}\n")
        try:
            subprocess.run(cmd, check=True, text=True)
            print(f"\n{Colors.GREEN}✅ Document visualization completed successfully!{Colors.ENDC}")
        except subprocess.CalledProcessError as e:
            print(f"\n{Colors.RED}❌ Document visualization failed (code {e.returncode}){Colors.ENDC}")
        except Exception as e:
            logger.exception(f"Unexpected error during visualization: {e}")
            print(f"\n{Colors.RED}❌ Unexpected error: {e}{Colors.ENDC}")
        input("\nPress Enter to continue...")
    
    def build_command(self, params: Dict[str, Any]) -> List[str]:
        script_path = str(Path(__file__).resolve().parent.parent / "semantic_folding/doc_visualizer.py")
        cmd = [
            "D:\\darsi\\ms\\Thesis\\Dr.Banaie\\code050302\\SemanticFolding\\.venv\\scripts\\python",
            script_path
        ]
        cmd.extend(['--run-dir', str(params['run_dir'])])
        cmd.extend(['--doc-id', params['doc_id']])
        cmd.extend(['--output', str(params['output'])])
        if params.get('no_grid_borders', False):
            cmd.append('--no-grid-borders')
        if 'border_color' in params and params['border_color'] != 'lightgray':
            cmd.extend(['--border-color', params['border_color']])
        if 'border_width' in params and params['border_width'] != 1.0:
            cmd.extend(['--border-width', str(params['border_width'])])
        return cmd
    
    def _validate_params(self, params: Dict[str, Any]) -> Tuple[bool, str]:
        run_dir = Path(params['run_dir'])
        if not run_dir.exists():
            return False, f"Run directory not found: {run_dir}"
        doc_fp_dir = run_dir / 'doc_fingerprints'   # aligned with Step 5 default
        if not doc_fp_dir.exists():
            return False, f"Document fingerprints directory not found: {doc_fp_dir}"
        if not params.get('doc_id'):
            return False, "Document ID is required"
        return True, ""
    
    def _list_available_runs(self) -> List[Path]:
        outputs_dir = Path('outputs')
        if not outputs_dir.exists():
            return []
        return sorted([d for d in outputs_dir.iterdir() if d.is_dir()], reverse=True)
    


"""
Custom Text Fingerprint Visualizer
"""

class CustomTextVisualizationHandler(VisualizationHandler):
    """
    Handler for custom text fingerprint visualization.

    This handler allows the user to enter any text manually,
    generate a semantic fingerprint for it,
    and visualize the resulting activation map.
    
    Supports two modes:
    - Single: Visualize one custom text fingerprint
    - Compare: Side-by-side comparison of two custom text fingerprints
    
    Auto-calculates figure height based on mode and width:
    - Single mode: height = width / 3 (horizontal layout)
    - Compare mode: height constrained between 2/3 and 3/3 of width
    """
    
    def __init__(self, runner):
        super().__init__(runner)
    
    def get_step_definition(self) -> Dict[str, Any]:
        return {
            'id': 'customtext_viz',
            'name': 'Customtext Fingerprint Visualization',
            'script': 'semantic_folding/customtext_visualizer.py',
            'required_params': ['run_dir', 'doc_id', 'output'],
            'optional_params': [
                'no_grid_borders', 'border_color', 'border_width',
                'grid_size', 'threshold', 'morton', 'grid_borders',
                'max_shapes', 'width', 'height',
                'colorscale', 'generate_html', 'generate_png', 'save_metadata'
                ],
            'default_output': 'customtext_viz'
        }
    
    def collect_parameters(self, run_id: str) -> Optional[Dict[str, str]]:
        logger.info("Collecting parameters for customtext visualization")
        params = {}
        
        print(f"\n{Colors.BOLD}Configure: Customtext Fingerprint Visualization{Colors.ENDC}")
        print(f"{Colors.CYAN}{'─' * 70}{Colors.ENDC}")
        
        # Select run directory
        runs = self._list_available_runs()
        if not runs:
            print(f"{Colors.RED}❌ No runs found in outputs/{Colors.ENDC}")
            input("\nPress Enter to continue...")
            return None
        
        print(f"\n{Colors.BOLD}Available runs:{Colors.ENDC}")
        for i, run in enumerate(runs, 1):
            print(f"  {i}. {run.name}")
        
        run_choice = input(f"\n{Colors.BOLD}Select run number:{Colors.ENDC} ").strip()
        try:
            run_idx = int(run_choice) - 1
            if 0 <= run_idx < len(runs):
                params['run_dir'] = str(runs[run_idx])
            else:
                print(f"{Colors.RED}❌ Invalid run selection{Colors.ENDC}")
                input("\nPress Enter to continue...")
                return None
        except ValueError:
            print(f"{Colors.RED}❌ Invalid input{Colors.ENDC}")
            input("\nPress Enter to continue...")
            return None
        
        mode_choice = self.runner.get_choice(
            "Select visualization mode:",
            ['Visualize single custom text', 'Compare two custom texts', 'Cancel']
        )
        if mode_choice == 3:
            return None
        mode = 'single' if mode_choice == 1 else 'compare'
        
        if mode == 'single':
            doc_id = self.runner.get_input(
                f"{Colors.BOLD}Enter customtext ID{Colors.ENDC} (e.g., doc_001)", None
            )
            while not doc_id:
                self.runner.print_error("'doc_id' is required")
                doc_id = self.runner.get_input(
                    f"{Colors.BOLD}Enter customtext ID{Colors.ENDC} (e.g., doc_001)", None
                )
            params['doc_id'] = doc_id

        else:
            doc_id1 = self.runner.get_input(f"{Colors.BOLD}Enter customtext ID1{Colors.ENDC} (e.g., doc_001)", None)
            while not doc_id1:
                self.runner.print_error("'doc_id1' is required")
                doc_id1 = self.runner.get_input(f"{Colors.BOLD}Enter customtext ID1{Colors.ENDC} (e.g., doc_001)", None)
            doc_id2 = self.runner.get_input(f"{Colors.BOLD}Enter customtext ID2{Colors.ENDC} (e.g., doc_001)", None)
            while not doc_id2:
                self.runner.print_error("'doc_id2' is required")
                doc_id2 = self.runner.get_input(f"{Colors.BOLD}Enter customtext ID2{Colors.ENDC} (e.g., doc_001)", None)
            
            params['doc_id1'] = doc_id1
            params['doc_id2'] = doc_id2
        
        
        # Output directory (inside run directory)
        # default_out = f'outputs/{run_id}/customtext_viz'
        default_output = str(Path(params['run_dir']) / 'customtext_viz')
        output = self.runner.get_input(
            f"{Colors.BOLD}Output directory{Colors.ENDC}", default_output
        )
        while not output:
            self.runner.print_error("'output' is required")
            output = self.runner.get_input(
                f"{Colors.BOLD}Output directory{Colors.ENDC} (required)", default_output
            )
        params['output'] = output
        # params['output'] = output if output else default_output

        # Optional parameters
        print(f"\n{Colors.CYAN}Optional parameters (Enter to skip):{Colors.ENDC}")
        
        show_borders = self.runner.get_input(
            f"  Show 4×4 grid borders? (y/n)", "y"
        )
        params['no_grid_borders'] = (show_borders.lower() == 'n')
        if show_borders.lower() != 'n':
            border_color = self.runner.get_input(f"  Border color", "lightgray")
            params['border_color'] = border_color if border_color else 'lightgray'
            border_width = self.runner.get_input(f"  Border width", "1.0")
            try:
                params['border_width'] = float(border_width) if border_width else 1.0
            except ValueError:
                print(f"  {Colors.YELLOW}⚠️  Invalid border width, using default 1.0{Colors.ENDC}")
                params['border_width'] = 1.0

        # Width first
        width_default = self.runner.get_default_value("width", "customtext_viz")
        width_val = self.runner.get_input(f"  width", width_default)
        if width_val:
            params['width'] = width_val
            width = int(width_val)
            if mode == 'single':
                height = width // 3
                params['height'] = str(height)
                print(f"  {Colors.GREEN}✓ Auto-calculated height (1/3 width): {height}{Colors.ENDC}")
            else:
                min_height = int(width * 2 / 3)
                max_height = width
                height_default = str(width)
                while True:
                    height_val = self.runner.get_input(
                        f"  height (must be between {min_height} and {max_height})", height_default
                    )
                    if not height_val:
                        break
                    height = int(height_val)
                    if min_height <= height <= max_height:
                        params['height'] = str(height)
                        break
                    else:
                        self.runner.print_error(f"Height must be between {min_height} and {max_height}")
        
        # Scalar parameters
        optional_params = [
            ('grid_size', 'Grid size'),
            ('threshold', 'Activation threshold'),
            ('border_color', 'Border color'),###
            ('border_width', 'Border width'),
            ('max_shapes', 'Maximum shapes to render'),
            ('colorscale', 'Plotly colorscale name'),
        ]
        for param_name, param_prompt in optional_params:
            default = self.runner.get_default_value(param_name, "customtext_viz")
            value = self.runner.get_input(f"  {param_prompt}", default)
            if value:
                params[param_name] = value
        
        # Boolean parameters (positive semantics)
        bool_params = [
            ('morton', 'Use Morton (Z-order) encoding (true/false)'),
            ('grid_borders', 'Show grid borders (true/false)'),
            ('generate_html', 'Generate HTML output (true/false)'),
            ('generate_png', 'Generate PNG output (true/false)'),
            ('save_metadata', 'Save metadata JSON (true/false)')
        ]
        for param_name, param_prompt in bool_params:
            default = self.runner.get_default_value(param_name, "customtext_viz")
            value = self.runner.get_input(f"  {param_prompt}", default)
            if value:
                params[param_name] = value
        
        valid, error_msg = self._validate_params(params)
        if not valid:
            print(f"{Colors.RED}❌ Validation failed: {error_msg}{Colors.ENDC}")
            input("\nPress Enter to continue...")
            return None        
        return params
    
    def handle(self):
        print(f"\n{Colors.BOLD}Customtext Fingerprint Visualization{Colors.ENDC}")
        params = self.collect_parameters(run_id="")
        if not params:
            return
        cmd = self.build_command(params)
        print(f"\n{Colors.CYAN}{'─' * 70}{Colors.ENDC}")
        print(f"{Colors.BOLD}Executing:{Colors.ENDC} {' '.join(str(c) for c in cmd)}")
        print(f"{Colors.CYAN}{'─' * 70}{Colors.ENDC}\n")
        try:
            subprocess.run(cmd, check=True, text=True)
            print(f"\n{Colors.GREEN}✅ Customtext visualization completed successfully!{Colors.ENDC}")
        except subprocess.CalledProcessError as e:
            print(f"\n{Colors.RED}❌ Customtext visualization failed (code {e.returncode}){Colors.ENDC}")
        except Exception as e:
            logger.exception(f"Unexpected error during visualization: {e}")
            print(f"\n{Colors.RED}❌ Unexpected error: {e}{Colors.ENDC}")
        input("\nPress Enter to continue...")

    
    def _get_step4_output(self) -> Optional[str]:
        run_id = self.runner.exec_state.get("last_run_id")
        if not run_id:
            return None
        run_data = self.runner.exec_state["runs"].get(run_id, {})
        step4_data = run_data.get("steps", {}).get(4)
        if step4_data and "output" in step4_data:
            output_path = step4_data["output"]
            if Path(output_path).exists():
                return output_path
        return None
    
    def build_command(self, params: Dict[str, str]) -> List[str]:
        script_path = str(Path(__file__).resolve().parent.parent / "semantic_folding/customtext_visualizer.py")
        cmd = [
            "D:\\darsi\\ms\\Thesis\\Dr.Banaie\\code050302\\SemanticFolding\\.venv\\scripts\\python",
            script_path
        ]
        cmd.extend(['--run-dir', str(params['run_dir'])])
        cmd.extend(['--output', str(params['output'])])
        
        if 'doc_id' in params:
            cmd.extend(['--doc-id', params['doc_id']])
        elif 'doc_id1' in params and 'doc_id2' in params:
            cmd.extend(['--doc-id1', params['doc_id1']])
            cmd.extend(['--doc-id2', params['doc_id2']])

        if params.get('no_grid_borders', False):
            cmd.append('--no-grid-borders')
        if 'border_color' in params and params['border_color'] != 'lightgray':
            cmd.extend(['--border-color', params['border_color']])
        if 'border_width' in params and params['border_width'] != 1.0:
            cmd.extend(['--border-width', str(params['border_width'])])
        
        
        for param, value in params.items():
            if param in ['run_dir', 'output', 'doc_id', 'doc_id1', 'doc_id2']:
                continue
            flag_name = self.runner.CLI_RENAME_MAP.get(param, param)
            flag = f"--{flag_name.replace('_', '-')}"
            if param in self.runner.NEGATE_FLAG_MAP:
                if str(value).lower() in ("false", "no", "0"):
                    cmd.append(f"--{self.runner.NEGATE_FLAG_MAP[param]}")
                    logger.debug(f"Added negation flag: --{self.runner.NEGATE_FLAG_MAP[param]}")
            elif str(value).lower() in ("true", "false"):
                if str(value).lower() == "true":
                    cmd.append(flag)
                    logger.debug(f"Added boolean flag: {flag}")
            else:
                cmd.extend([flag, value])
                logger.debug(f"Added param: {flag} {value}")
        return cmd
    
    def _validate_params(self, params: Dict[str, Any]) -> Tuple[bool, str]:
        run_dir = Path(params['run_dir'])
        if not run_dir.exists():
            return False, f"Run directory not found: {run_dir}"
        customtext_fp_dir = run_dir / 'customtext_fingerprints'   # aligned with Step 6 default
        if not customtext_fp_dir.exists():
            return False, f"Customtext fingerprints directory not found: {customtext_fp_dir}"
        ####
        if (not params.get('doc_id')) and (not (params.get('doc_id1') and params.get('doc_id2'))):
            return False, "Customtext ID is required"
        return True, ""
    
    def _list_available_runs(self) -> List[Path]:
        outputs_dir = Path('outputs')
        if not outputs_dir.exists():
            return []
        return sorted([d for d in outputs_dir.iterdir() if d.is_dir()], reverse=True)

# ============================================================================
# MAIN PIPELINE RUNNER
# ============================================================================

class SemanticRunner:
    """Interactive runner for the semantic folding pipeline."""

    CONFIG_PATH_IN_YAML = {
        # Global / shared
        "grid_size":            ["grid_size"],
        "min_freq":             ["min_freq"],
        "keep_verbs":           ["keep_verbs"],
        "smoothing_sigma":      ["smoothing_sigma"],

        # Phase 1: Phrase Extraction
        "min_word_length":      ["phrase_extraction", "min_word_length"],
        "no_spacy":             ["phrase_extraction", "no_spacy"],
        "max_ngram":            ["phrase_extraction", "max_ngram"],
        "no_filter_generic":    ["phrase_extraction", "no_filter_generic"],
        "stats":                ["phrase_extraction", "stats"],

        # Phase 2: Term-Context Matrix
        "no_tfidf":             ["term_context_matrix", "no_tfidf"],

        # Phase 3: Semantic Space
        "method":               ["semantic_space", "method"],
        "visualize":            ["semantic_space", "visualize"],
        "show_density":         ["semantic_space", "show_density"],
        "enable_grid":          ["semantic_space", "enable_grid"],
        "grid_padding":         ["semantic_space", "grid_padding"],
        "collision_resolution": ["semantic_space", "collision_resolution"],
        "n_jobs":               ["semantic_space", "n_jobs"],
        "use_sparse":           ["semantic_space", "use_sparse"],

        # Phase 4: Phrase Fingerprints
        "morton":               ["phrase_fingerprints", "morton"],
        "no_smooth":            ["phrase_fingerprints", "no_smooth"],
        "smooth":               ["phrase_fingerprints", "smooth"],
        "sigma":                ["phrase_fingerprints", "sigma"],

        # Phase 5: Document Fingerprints
        "top_percent":          ["document_fingerprints", "top_percent"],
        "no_normalize":         ["document_fingerprints", "no_normalize"],
        "normalize_method":     ["document_fingerprints", "normalize_method"],
        "use_spacy":            ["document_fingerprints", "use_spacy"],
        "filter_generic":       ["document_fingerprints", "filter_generic"],
        "min_word_length":      ["document_fingerprints", "min_word_length"],
        "compute_diversity":    ["document_fingerprints", "compute_diversity"],
        "diversity_sample":     ["document_fingerprints", "diversity_sample"],
        "min_peak_distance":    ["document_fingerprints", "min_peak_distance"],
        
        # Phase 6: Customtext Fingerprints
        "top_percent":          ["customtext_fingerprints", "top_percent"],
        "no_normalize":         ["customtext_fingerprints", "no_normalize"],
        "normalize_method":     ["customtext_fingerprints", "normalize_method"],
        "use_spacy":            ["customtext_fingerprints", "use_spacy"],
        "filter_generic":       ["customtext_fingerprints", "filter_generic"],
        "min_word_length":      ["customtext_fingerprints", "min_word_length"],
        "compute_diversity":    ["customtext_fingerprints", "compute_diversity"],
        "diversity_sample":     ["customtext_fingerprints", "diversity_sample"],
        "min_peak_distance":    ["customtext_fingerprints", "min_peak_distance"],

        # Phase 7: Query Processing
        "weighting":            ["query_processing", "weighting"],
        "idf_weights":          ["query_processing", "idf_weights"],
        "top_k":                ["query_processing", "top_k"],
        "spreading_steps":      ["query_processing", "spreading_steps"],

        # Phrase Visualization
        "threshold":            ["phrase_visualization", "threshold"],
        "grid_borders":         ["phrase_visualization", "grid_borders"],
        "border_color":         ["phrase_visualization", "border_color"],
        "border_width":         ["phrase_visualization", "border_width"],
        "max_shapes":           ["phrase_visualization", "max_shapes"],
        "width":                ["phrase_visualization", "figure_width"],
        "height":               ["phrase_visualization", "figure_height"],
        "colorscale":           ["phrase_visualization", "colorscale"],
        "generate_html":        ["phrase_visualization", "generate_html"],
        "generate_png":         ["phrase_visualization", "generate_png"],
        "save_metadata":        ["phrase_visualization", "save_metadata"],
    }

    PIPELINE_STEPS = [
        {
            "id": 1,
            "name": "Phrase Extraction",
            "script": "semantic_folding/phrase_extractor.py",
            "required_params": ["corpus", "output"],
            "optional_params": [
                "min_freq", "min_word_length", "no_spacy",
                "no_filter_generic", "keep_verbs", "stats"
            ],
            "default_output": "extracted_phrases",
            "extra_outputs": {
                "vocab":   lambda output: str(Path(output) / "vocabulary.csv"),
                "mapping": lambda output: str(Path(output) / "phrase_to_contexts.json"),
            },
            "depends_on": []
        },
        {
            "id": 2,
            "name": "Term-Context Matrix",
            "script": "semantic_folding/term_context.py",
            "required_params": ["corpus", "vocab", "mapping", "output"],
            "optional_params": ["no_tfidf"],
            "default_output": "term_context_matrix",
            "extra_outputs": {
                "matrix_npz" : lambda output: str(Path(output) / "term_context_matrix.npz"),
                "metadata"   : lambda output: str(Path(output) / "term_context_matrix.json"),
                "idf_weights": lambda output: str(Path(output) / "idf_weights.json"),
            },
            "depends_on": [1]
        },
        {
            "id": 3,
            "name": "Semantic Space",
            "script": "semantic_folding/semantic_space.py",
            "required_params": ["matrix_npz", "metadata", "output"],
            "optional_params": [
                "method", "grid_size", "visualize", "show_density"
            ],
            "default_output": "semantic_space",
            "extra_outputs": {
                "coordinates": lambda output: str(Path(output) / "context_coordinates.json"),
            },
            "depends_on": [2]
        },
        {
            "id": 4,
            "name": "Phrase Fingerprints",
            "script": "semantic_folding/phrase_fingerprints.py",
            "required_params": ["coordinates", "metadata", "output"],
            "optional_params": [
                "grid_size", "morton", "no_smooth", "smoothing_sigma"
            ],
            "default_output": "phrase_fingerprints",
            "depends_on": [3]
        },
        {
            "id": 5,
            "name": "Document Fingerprints",
            "script": "semantic_folding/doc_fingerprints.py",
            "required_params": ["corpus", "fingerprints", "output"],
            "optional_params": [
                "grid_size", "idf_weights", "top_percent",
                "no_normalize", "normalize_method", "min_word_length",
                "keep_verbs", "filter_generic", "min_peak_distance",
                "smoothing_sigma", "morton", "compute_diversity",
                "diversity_sample"
            ],
            "default_output": "doc_fingerprints",
            "depends_on": [4]
        },
        {
            "id": 6,
            "name": "Customtext Fingerprints",
            "script": "semantic_folding/customtext_fingerprints.py",
            "required_params": ["corpus", "fingerprints", "output"],
            "optional_params": [
                "grid_size", "idf_weights", "top_percent",
                "no_normalize", "normalize_method", "min_word_length",
                "keep_verbs", "filter_generic", "min_peak_distance",
                "smoothing_sigma", "morton", "compute_diversity",
                "diversity_sample"
            ],
            "default_output": "customtext_fingerprints",
            "depends_on": [4]
        },
        {
            "id": 7,
            "name": "Query Processing",
            "script": "semantic_folding/query_processor.py",
            "required_params": [
                "query", "fingerprints", "doc_fingerprints", "output"
            ],
            "optional_params": [
                "weighting", "idf_weights", "top_k", "spreading_steps", "grid_size"
            ],
            "default_output": "query_results",
            "depends_on": [2, 4, 5]
        }
    ]

    CLI_RENAME_MAP = {
        "matrix_npz":           "matrix",
        "metadata":             "metadata",
        "coordinates":          "coordinates",
        "fingerprints":         "fingerprints",
        "doc_fingerprints":     "doc-fingerprints",
        "idf_weights":          "idf-weights",
        "max_ngram":            "max-ngram",
        "min_word_length":      "min-word-length",
        "no_spacy":             "no-spacy",
        "no_filter_generic":    "no-filter-generic",
        "keep_verbs":           "keep-verbs",
        "no_tfidf":             "no-tfidf",
        "show_density":         "show-density",
        "enable_grid":          "enable-grid",
        "grid_padding":         "grid-padding",
        "collision_resolution": "collision-resolution",
        "n_jobs":               "n-jobs",
        "use_sparse":           "use-sparse",
        "morton":               "morton",
        "no_smooth":            "no-smooth",
        "smoothing_sigma":      "smoothing-sigma",
        "top_percent":          "top-percent",
        "normalize_method":     "normalize-method",
        "no_normalize":         "no-normalize",
        "filter_generic":       "filter-generic",
        "compute_diversity":    "compute-diversity",
        "diversity_sample":     "diversity-sample",
        "min_peak_distance":    "min-peak-distance",
        "top_k":                "top-k",
        "spreading_steps":      "spreading-steps",
        "grid_borders":         "grid-borders",
        "border_color":         "border-color",
        "border_width":         "border-width",
        "max_shapes":           "max-shapes",
        "generate_html":        "generate-html",
        "generate_png":         "generate-png",
        "save_metadata":        "save-metadata",
    }

    NEGATE_FLAG_MAP = {
        "morton":           "no-morton",
        "filter_generic":   "no-filter-generic",
        "grid_borders":     "no-grid-borders",
        "generate_html":    "no-html",
        "generate_png":     "no-png",
        "save_metadata":    "no-metadata",
    }

    # ------------------------------------------------------------------
    def __init__(self):
        self.state_file = Path("config/exec_state.yml")
        self.config_file = Path("config/semantic_folding.yml")
        self.exec_state = self.load_state()
        self.config = self.load_config()
        self.viz_handlers = {
            'phrase': PhraseVisualizationHandler(self),
            'document': DocumentVisualizationHandler(self),
            'customtext': CustomTextVisualizationHandler(self)
        }
        logger.info("SemanticRunner initialized")

    # ------------------------------------------------------------------
    # STATE & CONFIG LOADING
    # ------------------------------------------------------------------
    def load_state(self) -> Dict[str, Any]:
        if not self.state_file.exists():
            empty = {"last_run_id": None, "last_step": None, "runs": {}}
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            ##### add , encoding="utf-8" to below line
            with open(self.state_file, 'w', encoding="utf-8") as f:
                yaml.dump(empty, f)
            return empty
        ##### add , encoding="utf-8" to below line
        with open(self.state_file, 'r', encoding="utf-8") as f:
            state = yaml.safe_load(f) or {}
        state.setdefault("last_run_id", None)
        state.setdefault("last_step", None)
        state.setdefault("runs", {})
        return state

    def save_state(self):
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, 'w', encoding='utf-8') as f:
            yaml.dump(self.exec_state, f, default_flow_style=False, allow_unicode=True)

    def load_config(self) -> Dict[str, Any]:
        if self.config_file.exists():
            with open(self.config_file, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        return {}

    def get_default_value(self, param_name: str, step_id: Any) -> Optional[str]:
        if param_name not in self.CONFIG_PATH_IN_YAML:
            return None
        path = self.CONFIG_PATH_IN_YAML[param_name]
        value = self.config
        try:
            for key in path:
                value = value[key]
            return "true" if isinstance(value, bool) and value else (
                "false" if isinstance(value, bool) else str(value)
            )
        except (KeyError, TypeError):
            return None

    # ------------------------------------------------------------------
    # RUN MANAGEMENT
    # ------------------------------------------------------------------
    def create_new_run(self, corpus: str) -> str:
        run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.exec_state["runs"][run_id] = {
            "corpus": corpus,
            "created_at": datetime.now().isoformat(),
            "steps": {}
        }
        self.exec_state["last_run_id"] = run_id
        self.save_state()
        return run_id

    def get_run_info(self, run_id: str) -> Dict[str, Any]:
        return self.exec_state["runs"].get(run_id, {})

    def update_step_completion(self, run_id, step_id, output, params):
        if run_id not in self.exec_state["runs"]:
            return
        self.exec_state["runs"][run_id]["steps"][step_id] = {
            "completed_at": datetime.now().isoformat(),
            "output": output,
            "params": params
        }
        step_config = self.get_step_definition(step_id)
        if step_config and "extra_outputs" in step_config:
            for key, path_func in step_config["extra_outputs"].items():
                file_path = path_func(output)
                if Path(file_path).exists():
                    self.exec_state["runs"][run_id]["steps"][step_id][key] = file_path
        self.exec_state["last_run_id"] = run_id
        self.exec_state["last_step"] = step_id
        self.save_state()

    def resolve_parameter_from_previous_step(self, run_id, param_name, step_id):
        run_data = self.get_run_info(run_id)
        if not run_data:
            return None
        completed_steps = run_data.get("steps", {})
        for prev_step_id in range(step_id - 1, 0, -1):
            if prev_step_id not in completed_steps:
                continue
            prev_step_data = completed_steps[prev_step_id]
            prev_step_def = self.get_step_definition(prev_step_id)
            if not prev_step_def:
                continue
            if param_name == "output" and "output" in prev_step_data:
                continue
            if param_name in prev_step_data:
                resolved = prev_step_data[param_name]
                if isinstance(resolved, str) and Path(resolved).exists():
                    return resolved
            extra_outputs = prev_step_def.get("extra_outputs", {})
            if param_name in extra_outputs:
                output_path = prev_step_data.get("output")
                if output_path:
                    resolved = extra_outputs[param_name](output_path)
                    if Path(resolved).exists():
                        return resolved
            if "output" in prev_step_data:
                output_path = prev_step_data["output"]
                if param_name == "corpus" and prev_step_id == 1:
                    return run_data.get("corpus")
                if param_name == "fingerprints" and prev_step_id == 4:
                    return output_path
                if param_name == "doc_fingerprints" and prev_step_id == 5:
                    return output_path
        return None

    def get_step_definition(self, step_id):
        for step in self.PIPELINE_STEPS:
            if step["id"] == step_id:
                return step
        return None

    # ------------------------------------------------------------------
    # PARAMETER COLLECTION & COMMAND BUILDING
    # ------------------------------------------------------------------
    def collect_step_parameters(self, step, run_id):
        logger.info(f"Collecting parameters for step {step['id']}: {step['name']}")
        params = {}
        print(f"\n{Colors.BOLD}Configure: {step['name']}{Colors.ENDC}")
        print(f"{Colors.CYAN}{'─' * 70}{Colors.ENDC}")
        for param in step["required_params"]:
            resolved = self.resolve_parameter_from_previous_step(run_id, param, step["id"])
            if param == "output":
                default = f"outputs/{run_id}/{step['default_output']}"
            elif param == "corpus":
                run_data = self.get_run_info(run_id)
                if step["id"] == 6:
                    default = "data\\customtexts.txt"
                else:
                    default = run_data.get("corpus") or resolved
            else:
                default = resolved
            value = self.get_input(f"{Colors.BOLD}{param}{Colors.ENDC} (required)", default)
            while not value:
                self.print_error(f"'{param}' is required")
                value = self.get_input(f"{Colors.BOLD}{param}{Colors.ENDC} (required)", default)
            params[param] = value
        if step["optional_params"]:
            print(f"\n{Colors.CYAN}Optional parameters (Enter to skip):{Colors.ENDC}")
            for param in step["optional_params"]:
                resolved = self.resolve_parameter_from_previous_step(run_id, param, step["id"])
                config_default = self.get_default_value(param, step["id"])
                default = resolved if resolved else config_default
                value = self.get_input(f"  {param}", default)
                if value:
                    params[param] = value
        return params

    def build_command(self, step, params):
        script_path = str(Path(__file__).resolve().parent.parent / step["script"])
        cmd = [
            "D:\\darsi\\ms\\Thesis\\Dr.Banaie\\code050302\\SemanticFolding\\.venv\\scripts\\python",
            script_path
        ]
        for param, value in params.items():
            flag_name = self.CLI_RENAME_MAP.get(param, param)
            flag = f"--{flag_name.replace('_', '-')}"
            if param in self.NEGATE_FLAG_MAP:
                if str(value).lower() in ("false", "no", "0"):
                    cmd.append(f"--{self.NEGATE_FLAG_MAP[param]}")
                    logger.debug(f"Added negation flag: --{self.NEGATE_FLAG_MAP[param]}")
            elif str(value).lower() in ("true", "false"):
                if str(value).lower() == "true":
                    cmd.append(flag)
                    logger.debug(f"Added boolean flag: {flag}")
            else:
                cmd.extend([flag, value])
                logger.debug(f"Added param: {flag} {value}")
        return cmd

    def execute_step(self, step, run_id):
        logger.info(f"Executing step {step['id']}: {step['name']}")
        params = self.collect_step_parameters(step, run_id)
        if params is None:
            self.print_warning("Step cancelled")
            return False
        cmd = self.build_command(step, params)
        print(f"\n{Colors.CYAN}{'─' * 70}{Colors.ENDC}")
        print(f"{Colors.BOLD}Executing:{Colors.ENDC} {' '.join(cmd)}")
        print(f"{Colors.CYAN}{'─' * 70}{Colors.ENDC}\n")
        try:
            subprocess.run(cmd, check=True, text=True)
            self.update_step_completion(run_id, step["id"], params.get("output", ""), params)
            self.print_success(f"Step {step['id']} completed successfully")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Step {step['id']} failed (code {e.returncode})")
            self.print_error(f"Step {step['id']} failed (code {e.returncode})")
            return False
        except Exception as e:
            logger.exception(f"Unexpected error during step {step['id']}: {e}")
            self.print_error(f"Unexpected error: {e}")
            return False

    # ------------------------------------------------------------------
    # UI HELPERS
    # ------------------------------------------------------------------
    def print_header(self, text: str):
        """Print formatted header."""
        print(f"\n{Colors.HEADER}{Colors.BOLD}{'=' * 70}{Colors.ENDC}")
        print(f"{Colors.HEADER}{Colors.BOLD}{text.center(70)}{Colors.ENDC}")
        print(f"{Colors.HEADER}{Colors.BOLD}{'=' * 70}{Colors.ENDC}\n")

    def print_success(self, text: str):
        print(f"{Colors.GREEN}✓ {text}{Colors.ENDC}")

    def print_error(self, text: str):
        print(f"{Colors.RED}✗ {text}{Colors.ENDC}")

    def print_warning(self, text: str):
        print(f"{Colors.YELLOW}⚠ {text}{Colors.ENDC}")

    def get_input(self, prompt: str, default: Any = None) -> str:
        if default is not None:
            full_prompt = f"{prompt} [{Colors.YELLOW}{default}{Colors.ENDC}]: "
        else:
            full_prompt = f"{prompt}: "
        value = input(full_prompt).strip()
        return value if value else (str(default) if default is not None else "")

    def get_choice(self, prompt: str, options: List[str]) -> int:
        print(f"\n{Colors.BOLD}{prompt}{Colors.ENDC}")
        for i, option in enumerate(options, 1):
            print(f"  {i}. {option}")
        while True:
            try:
                choice = int(input(f"\n{Colors.BOLD}Enter choice (1-{len(options)}): {Colors.ENDC}"))
                if 1 <= choice <= len(options):
                    return choice
                else:
                    self.print_error(f"Please enter a number between 1 and {len(options)}")
            except ValueError:
                self.print_error("Please enter a valid number")

    # ------------------------------------------------------------------
    # MENU & RUN CONTROL
    # ------------------------------------------------------------------
    def show_main_menu(self):
        self.print_header("Semantic Folding Pipeline Runner")
        options = []
        options.append("Start new run")
        if self.exec_state["runs"]:
            options.append("Change current run/step")
        if self.exec_state.get("last_run_id"):
            last_run = self.exec_state["last_run_id"]
            last_step = self.exec_state.get("last_step")
            if last_step is not None and last_step < 7:
                options.append(f"Continue from step {last_step + 1} (run: {last_run})")
            elif last_step is None:
                options.append(f"Start step 1 (run: {last_run})")
        if self.exec_state["runs"]:
            options.append("Manage runs")
        options.append("Visualize results")
        options.append("Exit")
        choice = self.get_choice("Main Menu:", options)
        selected = options[choice - 1]
        if selected == "Start new run":
            self.start_new_run()
        elif selected == "Change current run/step":
            self.select_run()
        elif "Continue from step" in selected or "Start step 1" in selected:
            self.continue_run()
        elif selected == "Manage runs":
            self.view_manage_runs()
        elif selected == "Visualize results":
            self.visualize_menu()
        elif selected == "Exit":
            print(f"\n{Colors.GREEN}Goodbye!{Colors.ENDC}\n")
            sys.exit(0)

    def start_new_run(self):
        self.print_header("Start New Run")
        default_corpus = self.config.get("paths", {}).get("corpus_path", None)
        corpus = self.get_input(f"{Colors.BOLD}Corpus path{Colors.ENDC} (required)", default_corpus)
        while not corpus or not Path(corpus).exists():
            if not corpus:
                self.print_error("Corpus path is required")
            else:
                self.print_error(f"File not found: {corpus}")
            corpus = self.get_input(f"{Colors.BOLD}Corpus path{Colors.ENDC} (required)", default_corpus)
        run_id = self.create_new_run(corpus)
        self.print_success(f"Created run: {run_id}")
        self.run_pipeline(run_id, start_step=1)

    def select_run(self):
        self.print_header("Select Run")
        runs = list(self.exec_state["runs"].keys())
        if not runs:
            self.print_warning("No existing runs found")
            return
        print(f"\n{Colors.BOLD}Available runs:{Colors.ENDC}")
        for i, run_id in enumerate(runs, 1):
            run_data = self.get_run_info(run_id)
            corpus = run_data.get("corpus", "N/A")
            created = run_data.get("created_at", "N/A")
            completed_steps = len(run_data.get("steps", {}))
            print(f"  {i}. {run_id}")
            print(f"     Corpus: {corpus}")
            print(f"     Created: {created}")
            print(f"     Completed steps: {completed_steps}/7")
        choice = self.get_choice("Select run:", [f"{run_id}" for run_id in runs])
        selected_run = runs[choice - 1]
        self.exec_state["last_run_id"] = selected_run
        self.save_state()
        self.print_success(f"Selected run: {selected_run}")
        run_data = self.get_run_info(selected_run)
        completed_steps = run_data.get("steps", {})
        if completed_steps:
            last_completed = max(completed_steps.keys())
            next_step = last_completed + 1
            options = []
            if next_step <= 7:
                options.append(f"Continue from step {next_step}")
            options.append("Re-run a specific step")
            options.append("Back to main menu")
            choice = self.get_choice("What would you like to do?", options)
            if options[choice - 1].startswith("Continue"):
                self.run_pipeline(selected_run, start_step=next_step)
            elif options[choice - 1] == "Re-run a specific step":
                self.select_step(selected_run)
            else:
                return
        else:
            self.run_pipeline(selected_run, start_step=1)

    def continue_run(self):
        run_id = self.exec_state.get("last_run_id")
        if not run_id:
            self.print_error("No run to continue")
            return
        last_step = self.exec_state.get("last_step")
        if last_step is None:
            start_step = 1
        elif last_step >= 7:
            start_step = 7
        else:
            start_step = last_step + 1
        if start_step > 7:
            self.print_warning("All steps completed")
            return
        self.run_pipeline(run_id, start_step=start_step)

    def select_step(self, run_id: str):
        self.print_header("Select Step")
        options = [f"Step {step['id']}: {step['name']}" for step in self.PIPELINE_STEPS]
        choice = self.get_choice("Select step to run:", options)
        step = self.PIPELINE_STEPS[choice - 1]
        if not self.check_dependencies(run_id, step):
            self.print_error("Dependencies not met. Please complete previous steps first.")
            return
        success = self.execute_step(step, run_id)
        if success and step["id"] < 7:
            cont = self.get_input(f"\n{Colors.BOLD}Continue to next step?{Colors.ENDC} (y/n)", "y")
            if cont.lower() == "y":
                self.run_pipeline(run_id, start_step=step["id"] + 1)

    def check_dependencies(self, run_id: str, step: Dict[str, Any]) -> bool:
        run_data = self.get_run_info(run_id)
        completed_steps = set(run_data.get("steps", {}).keys())
        depends_on = step.get("depends_on", [])
        for dep_step_id in depends_on:
            if dep_step_id not in completed_steps:
                logger.warning(f"Step {step['id']} depends on step {dep_step_id} which is not completed")
                return False
        return True

    def run_pipeline(self, run_id: str, start_step: int = 1):
        self.print_header(f"Running Pipeline (Run: {run_id})")
        for step in self.PIPELINE_STEPS:
            if step["id"] < start_step:
                continue
            if not self.check_dependencies(run_id, step):
                self.print_error(f"Cannot run step {step['id']}: dependencies not met")
                break
            success = self.execute_step(step, run_id)
            if not success:
                self.print_error(f"Pipeline stopped at step {step['id']}")
                break
            if step["id"] < 7:
                cont = self.get_input(f"\n{Colors.BOLD}Continue to next step?{Colors.ENDC} (y/n)", "y")
                if cont.lower() != "y":
                    self.print_warning("Pipeline paused")
                    break
        else:
            self.print_success("Pipeline completed successfully!")

    def visualize_menu(self):
        self.print_header("Visualization")
        options = [
            "Phrase Extraction Fingerprint Visualization",
            "Doc Fingerprint Visualization",
            "Custom Text Fingerprint Visualization",
            "Back to main menu"
        ]
        choice = self.get_choice("Select visualization type:", options)
        if choice == 1:
            self.viz_handlers['phrase'].handle()
        elif choice == 2:
            self.viz_handlers['document'].handle()
        elif choice == 3:
            self.viz_handlers['customtext'].handle()
        elif choice == 4:
            return

    # ------------------------------------------------------------------
    # RUN MANAGEMENT (view/delete)
    # ------------------------------------------------------------------
    def view_manage_runs(self):
        while True:
            self.print_header("Run Management")
            self.exec_state = self.load_state()
            runs = self.exec_state.get("runs", {})
            if not runs:
                print(f"{Colors.YELLOW}No runs found in history.{Colors.ENDC}\n")
                input(f"{Colors.GREEN}Press Enter to return to main menu...{Colors.ENDC}")
                return
            print(f"{Colors.GREEN}Historical Runs:{Colors.ENDC}\n")
            for idx, (run_id, run_data) in enumerate(sorted(runs.items()), 1):
                corpus = run_data.get("corpus", "N/A")
                timestamp = run_data.get("timestamp", "N/A")
                completed = run_data.get("completed_steps", [])
                print(f"  {idx}. {Colors.CYAN}{run_id}{Colors.ENDC}")
                print(f"     Corpus: {corpus}")
                print(f"     Created: {timestamp}")
                print(f"     Completed steps: {completed}")
                print()
            orphaned = self._find_orphaned_directories(runs)
            if orphaned:
                print(f"{Colors.YELLOW}Orphaned Directories (not in run history):{Colors.ENDC}\n")
                for idx, orphan_path in enumerate(orphaned, 1):
                    size = self._get_directory_size(orphan_path)
                    print(f"  {idx}. {orphan_path} ({size})")
                print()
            options = ["Delete specific run", "Delete all runs"]
            if orphaned:
                options.extend(["Delete specific orphaned directory", "Delete all orphaned directories"])
            options.append("Back to main menu")
            choice = self.get_choice("Run Management:", options)
            selected = options[choice - 1]
            if selected == "Back to main menu":
                return
            elif selected == "Delete specific run":
                self._delete_specific_run(runs)
            elif selected == "Delete all runs":
                self._delete_all_runs(runs)
            elif selected == "Delete specific orphaned directory":
                self._delete_specific_orphaned(orphaned)
            elif selected == "Delete all orphaned directories":
                self._delete_all_orphaned(orphaned)

    def _find_orphaned_directories(self, runs: dict) -> list:
        output_base = Path("output")
        if not output_base.exists():
            return []
        known_runs = set(runs.keys())
        existing_dirs = [d for d in output_base.iterdir() if d.is_dir()]
        orphaned = []
        for dir_path in existing_dirs:
            is_orphaned = True
            for run_id in known_runs:
                if dir_path.name.startswith(run_id) or run_id in dir_path.name:
                    is_orphaned = False
                    break
            if is_orphaned:
                orphaned.append(dir_path)
        return sorted(orphaned)

    def _get_directory_size(self, path: Path) -> str:
        try:
            total_size = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
            for unit in ['B', 'KB', 'MB', 'GB']:
                if total_size < 1024.0:
                    return f"{total_size:.1f} {unit}"
                total_size /= 1024.0
            return f"{total_size:.1f} TB"
        except Exception as e:
            logger.warning(f"Could not calculate size for {path}: {e}")
            return "unknown size"

    def _delete_specific_run(self, runs: dict):
        run_list = sorted(runs.keys())
        options = [f"{run_id} ({runs[run_id].get('corpus', 'N/A')})" for run_id in run_list]
        options.append("Cancel")
        choice = self.get_choice("Select run to delete:", options)
        if options[choice - 1] == "Cancel":
            return
        run_id = run_list[choice - 1]
        print(f"\n{Colors.RED}Are you sure you want to delete run '{run_id}'?{Colors.ENDC}")
        print(f"{Colors.RED}This will remove all associated output directories.{Colors.ENDC}")
        confirm = input(f"{Colors.YELLOW}Type 'yes' to confirm: {Colors.ENDC}").strip().lower()
        if confirm != "yes":
            print(f"{Colors.YELLOW}Deletion cancelled.{Colors.ENDC}")
            time.sleep(1)
            return
        deleted_dirs = self._delete_run_directories(run_id)
        del self.exec_state["runs"][run_id]
        if self.exec_state.get("last_run_id") == run_id:
            self.exec_state["last_run_id"] = None
            self.exec_state["last_step"] = None
        self.save_state()
        print(f"\n{Colors.GREEN}Run '{run_id}' deleted successfully.{Colors.ENDC}")
        if deleted_dirs:
            print(f"{Colors.GREEN}Deleted directories:{Colors.ENDC}")
            for dir_path in deleted_dirs:
                print(f"  - {dir_path}")
        time.sleep(2)

    def _delete_all_runs(self, runs: dict):
        print(f"\n{Colors.RED}WARNING: This will delete ALL runs and their output directories!{Colors.ENDC}")
        confirm = input(f"{Colors.YELLOW}Type 'DELETE ALL' to confirm: {Colors.ENDC}").strip()
        if confirm != "DELETE ALL":
            print(f"{Colors.YELLOW}Deletion cancelled.{Colors.ENDC}")
            time.sleep(1)
            return
        all_deleted = []
        for run_id in runs.keys():
            deleted_dirs = self._delete_run_directories(run_id)
            all_deleted.extend(deleted_dirs)
        self.exec_state["runs"] = {}
        self.exec_state["last_run_id"] = None
        self.exec_state["last_step"] = None
        self.save_state()
        print(f"\n{Colors.GREEN}All runs deleted successfully.{Colors.ENDC}")
        if all_deleted:
            print(f"{Colors.GREEN}Deleted {len(all_deleted)} directories.{Colors.ENDC}")
        time.sleep(2)

    def _delete_run_directories(self, run_id: str) -> list:
        output_base = Path("output")
        if not output_base.exists():
            return []
        deleted = []
        for dir_path in output_base.iterdir():
            if dir_path.is_dir() and (dir_path.name.startswith(run_id) or run_id in dir_path.name):
                try:
                    shutil.rmtree(dir_path)
                    deleted.append(dir_path)
                    logger.info(f"Deleted directory: {dir_path}")
                except Exception as e:
                    logger.error(f"Failed to delete {dir_path}: {e}")
                    print(f"{Colors.RED}Failed to delete {dir_path}: {e}{Colors.ENDC}")
        return deleted

    def _delete_specific_orphaned(self, orphaned: list):
        options = [str(path) for path in orphaned]
        options.append("Cancel")
        choice = self.get_choice("Select orphaned directory to delete:", options)
        if options[choice - 1] == "Cancel":
            return
        dir_path = orphaned[choice - 1]
        print(f"\n{Colors.RED}Are you sure you want to delete '{dir_path}'?{Colors.ENDC}")
        confirm = input(f"{Colors.YELLOW}Type 'yes' to confirm: {Colors.ENDC}").strip().lower()
        if confirm != "yes":
            print(f"{Colors.YELLOW}Deletion cancelled.{Colors.ENDC}")
            time.sleep(1)
            return
        try:
            shutil.rmtree(dir_path)
            print(f"\n{Colors.GREEN}Directory '{dir_path}' deleted successfully.{Colors.ENDC}")
            logger.info(f"Deleted orphaned directory: {dir_path}")
        except Exception as e:
            print(f"{Colors.RED}Failed to delete directory: {e}{Colors.ENDC}")
            logger.error(f"Failed to delete {dir_path}: {e}")
        time.sleep(2)

    def _delete_all_orphaned(self, orphaned: list):
        print(f"\n{Colors.RED}This will delete {len(orphaned)} orphaned directories.{Colors.ENDC}")
        confirm = input(f"{Colors.YELLOW}Type 'yes' to confirm: {Colors.ENDC}").strip().lower()
        if confirm != "yes":
            print(f"{Colors.YELLOW}Deletion cancelled.{Colors.ENDC}")
            time.sleep(1)
            return
        deleted_count = 0
        failed_count = 0
        for dir_path in orphaned:
            try:
                shutil.rmtree(dir_path)
                deleted_count += 1
                logger.info(f"Deleted orphaned directory: {dir_path}")
            except Exception as e:
                failed_count += 1
                logger.error(f"Failed to delete {dir_path}: {e}")
                print(f"{Colors.RED}Failed to delete {dir_path}: {e}{Colors.ENDC}")
        print(f"\n{Colors.GREEN}Deleted {deleted_count} orphaned directories.{Colors.ENDC}")
        if failed_count > 0:
            print(f"{Colors.YELLOW}Failed to delete {failed_count} directories.{Colors.ENDC}")
        time.sleep(2)

    # ------------------------------------------------------------------
    # MAIN ENTRY POINT
    # ------------------------------------------------------------------
    def run(self):
        try:
            while True:
                self.show_main_menu()
        except KeyboardInterrupt:
            print(f"\n\n{Colors.YELLOW}Interrupted by user{Colors.ENDC}")
            sys.exit(0)
        except Exception as e:
            logger.exception(f"Unexpected error: {e}")
            self.print_error(f"Unexpected error: {e}")
            sys.exit(1)


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    runner = SemanticRunner()
    runner.run()

if __name__ == "__main__":
    main()