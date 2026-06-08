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