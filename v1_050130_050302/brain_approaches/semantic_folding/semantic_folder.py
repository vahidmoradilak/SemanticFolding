#!/usr/bin/env python3
"""
Semantic Folding Pipeline Runner
Interactive TUI for executing the semantic folding pipeline with state management.
"""

import os
import sys
import yaml
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from loguru import logger


# Configure loguru
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
    level="INFO"
)
logger.add(
    "logs/semantic_runner.log",
    rotation="10 MB",
    retention="7 days",
    level="DEBUG",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}"
)


class Colors:
    HEADER    = '\033[95m'
    BLUE      = '\033[94m'
    CYAN      = '\033[96m'
    GREEN     = '\033[92m'
    YELLOW    = '\033[93m'
    RED       = '\033[91m'
    ENDC      = '\033[0m'
    BOLD      = '\033[1m'
    UNDERLINE = '\033[4m'


class SemanticRunner:
    """Interactive runner for the semantic folding pipeline."""

    # ------------------------------------------------------------------
    # CONFIG_PATH_IN_YAML
    # Maps parameter names → nested YAML key path for default resolution.
    # ------------------------------------------------------------------
    CONFIG_PATH_IN_YAML = {
        # Global
        "grid_size":            ["grid_size"],
        "min_freq" :            ["min_freq"],
        "keep_verbs":           ["keep_verbs"],

        # Phase 1
        "min_word_length":      ["phrase_extraction", "min_word_length"],
        "use_spacy":            ["phrase_extraction", "use_spacy"],
        "max_ngram":            ["phrase_extraction", "max_ngram"],
        "filter_generic":       ["phrase_extraction", "filter_generic"],
        "stats":                ["phrase_extraction", "stats"],

        # Phase 2
        "use_tfidf":            ["term_context_matrix", "use_tfidf"],

        # Phase 3
        "method":               ["semantic_space", "method"],
        "visualize":            ["semantic_space", "visualize"],
        "show_density":         ["semantic_space", "show_density"],
        "enable_grid":          ["semantic_space", "enable_grid"],
        "grid_padding":         ["semantic_space", "grid_padding"],
        "collision_resolution": ["semantic_space", "collision_resolution"],
        "n_jobs":               ["semantic_space", "n_jobs"],
        "use_sparse":           ["semantic_space", "use_sparse"],

        # Phase 5
        "top_percent":          ["document_fingerprints", "top_percent"],
        "normalize":            ["document_fingerprints", "normalize"],
        "normalize_method":     ["document_fingerprints", "normalize_method"],
        "compute_diversity":    ["document_fingerprints", "compute_diversity"],
        "diversity_sample":     ["document_fingerprints", "diversity_sample"],

        # Phase 6
        "weighting":            ["query_processing", "weighting"],
        "idf":                  ["query_processing", "idf"],
        "top_k":                ["query_processing", "top_k"],
        "spreading_steps":      ["query_processing", "spreading_steps"],
    }

    # ------------------------------------------------------------------
    # PIPELINE_STEPS
    # ------------------------------------------------------------------
    PIPELINE_STEPS = [
        {
            "id": 1,
            "name": "Phrase Extraction",
            "script": "brain_approaches/semantic_folding/phrase_extractor.py",
            "required_params": ["corpus", "output"],
            "optional_params": [
                "min_freq", "min_word_length", "use_spacy",
                "filter_generic", "keep_verbs", "stats"
            ],
            "default_output": "extracted_phrases", # NOW A DIRECTORY
            "extra_outputs": {
                "vocab":   lambda output: str(Path(output) / "vocabulary.csv"),
                "mapping": lambda output: str(Path(output) / "phrase_to_contexts.json"),
            },
            "depends_on": []
        },
        {
            "id": 2,
            "name": "Term-Context Matrix",
            "script": "brain_approaches/semantic_folding/term_context.py",
            "required_params": ["corpus", "vocab", "mapping", "output"], # UPDATED PARAMS
            "optional_params": ["use_tfidf"], # REMOVED text-processing args
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
            "script": "brain_approaches/semantic_folding/semantic_space.py",
            "required_params": ["matrix", "metadata", "output"],
            "optional_params": [
                "method", "grid_size", "visualize", "show_density"
            ],
            "default_output": "semantic_space",
            "depends_on": [2]
        },
        {
            "id": 4,
            "name": "Phrase Fingerprints",
            "script": "brain_approaches/semantic_folding/phrase_fingerprints.py",
            "required_params": ["coordinates", "metadata", "output"],
            "optional_params": ["grid_size"],
            "default_output": "phrase_fingerprints",
            "depends_on": [3]
        },
        {
            "id": 5,
            "name": "Document Fingerprints",
            "script": "brain_approaches/semantic_folding/doc_fingerprints.py",
            "required_params": ["corpus", "fingerprints", "output"], 
            "optional_params": [
                "idf_weights", "grid_size", "top_percent", "normalize", 
                "normalize_method", "use_spacy", "keep_verbs", "filter_generic", 
                "min_word_length"
            ],
            "default_output": "doc_fingerprints",
            "depends_on": [3, 4]
        },
        {
            "id": 6,
            "name": "Query Processing",
            "script": "brain_approaches/semantic_folding/query_processing.py",
            "required_params": [
                "query",
                "phrase_fp_dir",
                "doc_fp_dir",
                "weighting",
            ],
            "optional_params": [
                "idf",
                "top_k",
                "spreading_steps",
                "keep_verbs",
                "output",
                "grid_size"
            ],
            "default_output": "query_results.json",
            "depends_on": [5]
        },

    ]

    # ------------------------------------------------------------------
    # CLI flag renaming
    # ------------------------------------------------------------------
    CLI_RENAME_MAP = {
        "phrase_fp_dir":   "phrase-fp-dir",
        "doc_fp_dir":      "doc-fp-dir",
        "spreading_steps": "spreading-steps",
        "min_phrase_freq": "min-freq"
    }

    # Flags that are boolean and use --no-X negation pattern
    NEGATE_FLAG_MAP = {
        "use_spacy":           "no-spacy",
        "filter_generic":      "no-filter-generic",
        "use_tfidf":           "no-tfidf",
        "use_word_boundaries": "no-word-boundaries",
        "enable_grid":         "no-grid",
        "keep_verbs":          "no-verbs",       # ADDED for Step 1 & 5
        "normalize":           "no-normalize",   # ADDED for Step 5
    }

    def __init__(self):
        self.config_dir       = Path("config")
        self.exec_state_path  = self.config_dir / "exec_state.yml"
        self.config_path      = self.config_dir / "semantic_folding.yml"

        self.config_dir.mkdir(exist_ok=True)
        Path("logs").mkdir(exist_ok=True)

        logger.info("Initializing SemanticRunner")
        self.config     = self.load_config()
        self.exec_state = self.load_exec_state()
        logger.debug(f"Config loaded from: {self.config_path}")
        logger.debug(f"Exec state loaded from: {self.exec_state_path}")

    # ------------------------------------------------------------------
    # Config / state I/O
    # ------------------------------------------------------------------

    def load_config(self) -> Dict[str, Any]:
        if not self.config_path.exists():
            logger.warning(f"Config file not found at {self.config_path}, using empty config")
            return {}
        with open(self.config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
        logger.info(f"Loaded config with {len(config)} top-level keys")
        return config

    def load_exec_state(self) -> Dict[str, Any]:
        if not self.exec_state_path.exists():
            logger.info("No existing exec state found, starting fresh")
            return {"last_run_id": None, "last_step": None, "runs": {}}
        with open(self.exec_state_path, "r", encoding="utf-8") as f:
            state = yaml.safe_load(f) or {
                "last_run_id": None, "last_step": None, "runs": {}
            }
        logger.info(f"Loaded exec state: {len(state.get('runs', {}))} previous run(s)")
        return state

    def save_exec_state(self):
        with open(self.exec_state_path, "w", encoding="utf-8") as f:
            yaml.dump(self.exec_state, f, default_flow_style=False,
                    sort_keys=False, allow_unicode=True)
        logger.debug(f"Exec state saved to {self.exec_state_path}")

    # ------------------------------------------------------------------
    # TUI helpers
    # ------------------------------------------------------------------

    def print_header(self, text: str):
        print(f"\n{Colors.BOLD}{Colors.CYAN}{'=' * 70}{Colors.ENDC}")
        print(f"{Colors.BOLD}{Colors.CYAN}{text.center(70)}{Colors.ENDC}")
        print(f"{Colors.BOLD}{Colors.CYAN}{'=' * 70}{Colors.ENDC}\n")

    def print_success(self, text: str):
        print(f"{Colors.GREEN}✓ {text}{Colors.ENDC}")

    def print_error(self, text: str):
        print(f"{Colors.RED}✗ {text}{Colors.ENDC}")

    def print_warning(self, text: str):
        print(f"{Colors.YELLOW}⚠ {text}{Colors.ENDC}")

    def get_input(self, prompt: str, default: Optional[str] = None) -> str:
        if default is not None:
            prompt = f"{prompt} [{Colors.YELLOW}{default}{Colors.ENDC}]: "
        else:
            prompt = f"{prompt}: "
        value = input(prompt).strip()
        return value if value else (default if default is not None else "")

    def get_choice(self, prompt: str, options: List[str]) -> int:
        print(f"\n{prompt}")
        for i, option in enumerate(options, 1):
            print(f"  {Colors.BOLD}{i}.{Colors.ENDC} {option}")
        while True:
            try:
                choice = int(
                    input(f"\n{Colors.BOLD}Enter choice (1-{len(options)}): {Colors.ENDC}")
                )
                if 1 <= choice <= len(options):
                    return choice
                print(f"{Colors.RED}Invalid choice.{Colors.ENDC}")
            except ValueError:
                print(f"{Colors.RED}Please enter a number.{Colors.ENDC}")

    def generate_run_id(self) -> str:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        logger.debug(f"Generated run ID: {run_id}")
        return run_id

    def get_step_by_id(self, step_id: int) -> Optional[Dict[str, Any]]:
        for step in self.PIPELINE_STEPS:
            if step["id"] == step_id:
                return step
        logger.warning(f"Step ID {step_id} not found")
        return None

    # ------------------------------------------------------------------
    # Default value resolution
    # ------------------------------------------------------------------

    def get_default_value(self, param: str, step_id: int) -> Optional[str]:
        _MISSING = object()

        # ── Priority 1, 2, 3: previous run state ──────────────────────────
        if self.exec_state.get("last_run_id"):
            run_id   = self.exec_state["last_run_id"]
            run_data = self.exec_state["runs"].get(run_id)

            if run_data:
                # UPDATED mapping to support new artefacts
                param_mapping = {
                    "vocab":       "vocabulary.csv",
                    "mapping":     "phrase_to_contexts.json",
                    "phrases":     "vocabulary.csv", # Fallback for step 5
                    "matrix":      "term_context_matrix.npz",
                    "metadata":    "term_context_matrix.json",
                    "coordinates": "context_coordinates.json",
                    "idf":         "idf_weights.json",
                    "idf_weights": "idf_weights.json", # ADDED for Step 5
                    "som":         "som_model.pkl"     # ADDED for Step 5
                }

                # Priority 1a & 1b loop (already reversed)
                for prev_step_id in range(step_id - 1, 0, -1):
                    step_data = run_data["steps"].get(prev_step_id)
                    if not step_data:
                        continue

                    extra = step_data.get("extra_outputs", {})
                    if param in extra:
                        logger.debug(f"Default '{param}' from step {prev_step_id} extra_outputs")
                        return extra[param]

                    if "output" in step_data and param in param_mapping:
                        output_path = step_data["output"]
                        if Path(output_path).is_file() and param_mapping[param] in output_path:
                            return output_path
                        candidate = str(Path(output_path) / param_mapping[param])
                        if Path(candidate).exists():
                            logger.debug(f"Default '{param}' from step {prev_step_id} output dir")
                            return candidate

                # Priority 2: same-step parameters
                step_params = (
                    run_data["steps"].get(step_id, {}).get("parameters", {})
                )
                if param in step_params:
                    logger.debug(
                        f"Default '{param}' from previous run step {step_id} params"
                    )
                    return str(step_params[param])

                # Priority 3a: coordinates from Step 3 output dir
                if param == "coordinates":
                    step3_out = run_data["steps"].get(3, {}).get("output", "")
                    if step3_out:
                        candidate = str(Path(step3_out) / "context_coordinates.json")
                        if Path(candidate).exists():
                            return candidate

                # Priority 3b: phrase_fp_dir and fingerprints → Step 4 output
                if param in ("fingerprints", "phrase_fp_dir", "phrase-fp-dir"):
                    step4_out = run_data["steps"].get(4, {}).get("output", "")
                    if step4_out and Path(step4_out).exists():
                        logger.debug(
                            f"Default '{param}' from Step 4 output: {step4_out}"
                        )
                        return step4_out

                # Priority 3c: doc_fp_dir → Step 5 output
                if param in("doc_fp_dir","doc-fp-dir") :
                    step5_out = run_data["steps"].get(5, {}).get("output", "")
                    if step5_out and Path(step5_out).exists():
                        logger.debug(
                            f"Default 'doc_fp_dir' from Step 5 output: {step5_out}"
                        )
                        return step5_out

                # Priority 3d: idf / idf_weights → Step 2 output dir / idf_weights.json
                if param in ("idf", "idf_weights"):
                    # FIXED: Step 2 generates the IDF weights in your pipeline, not Step 3
                    step2_out = run_data["steps"].get(2, {}).get("output", "")
                    if step2_out:
                        candidate = str(Path(step2_out) / "idf_weights.json")
                        if Path(candidate).exists():
                            logger.debug(
                                f"Default '{param}' resolved from Step 2 dir: {candidate}"
                            )
                            return candidate

        # ── Priority 4: YAML config fallback ──────────────────────────────
        if param not in self.CONFIG_PATH_IN_YAML or not self.config:
            return None

        path  = self.CONFIG_PATH_IN_YAML[param]
        value = self.config

        for key in path:
            if not isinstance(value, dict):
                return None
            value = value.get(key, _MISSING)
            if value is _MISSING:
                return None

        if value is None:
            return None

        resolved = str(value)
        logger.debug(f"Default '{param}' from YAML config: {resolved}")
        return resolved

    # ------------------------------------------------------------------
    # Output path helper
    # ------------------------------------------------------------------

    def get_output_path(self, step: Dict[str, Any], run_id: str) -> str:
        output_base = self.config.get("paths", {}).get("output_base", "outputs")
        output_dir  = Path(output_base) / run_id
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(output_dir / step["default_output"])
        logger.debug(f"Output path for step {step['id']}: {output_path}")
        return output_path

    # ------------------------------------------------------------------
    # Parameter collection
    # ------------------------------------------------------------------

    def collect_parameters(
        self, step: Dict[str, Any], run_id: str
    ) -> Dict[str, str]:
        logger.info(f"Collecting parameters for step {step['id']}: {step['name']}")
        params = {}

        print(f"\n{Colors.BOLD}Configure: {step['name']}{Colors.ENDC}")
        print(f"{Colors.CYAN}{'─' * 70}{Colors.ENDC}")

        for param in step["required_params"]:
            if param == "output":
                default = self.get_output_path(step, run_id)
            elif param == "corpus":
                default = (
                    self.get_default_value(param, step["id"])
                    or self.config.get("paths", {}).get("corpus_path", "data/corpus.txt")
                )
            elif param == "query":
                # query is always entered fresh — no sensible default
                default = None
            else:
                default = self.get_default_value(param, step["id"])

            value = self.get_input(
                f"{Colors.BOLD}{param}{Colors.ENDC} (required)", default
            )
            while not value:
                self.print_error(f"'{param}' is required")
                value = self.get_input(
                    f"{Colors.BOLD}{param}{Colors.ENDC} (required)", default
                )

            params[param] = value
            logger.debug(f"Required param — {param}: {value}")

        print(f"\n{Colors.CYAN}Optional parameters (Enter to skip):{Colors.ENDC}")

        for param in step["optional_params"]:
            if param == "output":
                default = self.get_output_path(step, run_id)
            else:
                default = self.get_default_value(param, step["id"])

            value = self.get_input(f"  {param}", default)

            if value:
                if isinstance(value, str) and value.lower() in ("true", "false"):
                    value = value.lower() == "true"
                params[param] = value
                logger.debug(f"Optional param — {param}: {value}")
            elif default is not None and value == default:
                if isinstance(default, str) and default.lower() in ("true", "false"):
                    params[param] = default.lower() == "true"
                else:
                    params[param] = default
                logger.debug(f"Optional param (default kept) — {param}: {params[param]}")

        logger.info(
            f"Parameter collection done for step {step['id']}: {len(params)} params"
        )
        return params

    # ------------------------------------------------------------------
    # Command builder
    # ------------------------------------------------------------------

    def build_command(self, step: Dict, params: Dict) -> List[str]:
        # Keeps your original python path
        cmd = [
            # "E:\\PHD\\GraphRag-Implementations\\YaALI\\"
            # "knowledge-graph-builder\\.venv\\scripts\\python",
            # "C:\\Users\\vahid\\AppData\\Local\\Programs\\Python",
            "D:\\darsi\\ارشد\\Thesis\\Dr.Banaie\\code050302\\"
            "SemanticFolding\\.venv\\Scripts\\python",
            step["script"],
        ]

        for key, value in params.items():
            if isinstance(value, bool):
                if key in self.NEGATE_FLAG_MAP:
                    if not value:
                        cmd.append(f"--{self.NEGATE_FLAG_MAP[key]}")
                else:
                    if value:
                        cli_key = self.CLI_RENAME_MAP.get(key, key.replace("_", "-"))
                        cmd.append(f"--{cli_key}")
            else:
                # SPECIAL HANDLING FOR STEP 1, 2, & 5 DIRECTORY FLAGS
                if key == "output" and step["id"] in [1, 2, 5]:
                    cmd.extend(["--output-dir", str(value)])
                else:
                    cli_key = self.CLI_RENAME_MAP.get(key, key.replace("_", "-"))
                    cmd.extend([f"--{cli_key}", str(value)])

        logger.debug(f"Built command: {' '.join(cmd)}")
        return cmd

    # ------------------------------------------------------------------
    # Step execution
    # ------------------------------------------------------------------

    def execute_step(
        self, step: Dict[str, Any], params: Dict[str, str], run_id: str
    ) -> bool:
        cmd = self.build_command(step, params)

        logger.info(f"Executing step {step['id']}: {step['name']} | run_id={run_id}")
        logger.debug(f"Full command: {' '.join(cmd)}")

        print(f"\n{Colors.CYAN}{'─' * 70}{Colors.ENDC}")
        print(f"{Colors.BOLD}Executing:{Colors.ENDC} {' '.join(cmd)}")
        print(f"{Colors.CYAN}{'─' * 70}{Colors.ENDC}\n")

        try:
            subprocess.run(cmd, check=True, text=True)

            if run_id not in self.exec_state["runs"]:
                self.exec_state["runs"][run_id] = {
                    "created_at": datetime.now().isoformat(),
                    "steps": {},
                }

            output_path = params.get("output", self.get_output_path(step, run_id))

            extra_outputs = {}
            for label, path_fn in step.get("extra_outputs", {}).items():
                extra_outputs[label] = path_fn(output_path)
                logger.debug(f"Extra output — {label}: {extra_outputs[label]}")

            step_record = {
                "name":         step["name"],
                "completed_at": datetime.now().isoformat(),
                "parameters":   params,
                "output":       output_path,
            }
            if extra_outputs:
                step_record["extra_outputs"] = extra_outputs

            self.exec_state["runs"][run_id]["steps"][step["id"]] = step_record
            self.exec_state["last_run_id"] = run_id
            self.exec_state["last_step"]   = step["id"]
            self.save_exec_state()

            logger.info(f"Step {step['id']} completed — output: {output_path}")
            self.print_success(
                f"Step {step['id']}: {step['name']} completed successfully"
            )
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"Step {step['id']} failed (code {e.returncode})")
            self.print_error(
                f"Step {step['id']}: {step['name']} failed (code {e.returncode})"
            )
            return False
        except Exception as e:
            logger.exception(f"Unexpected error in step {step['id']}: {e}")
            self.print_error(f"Unexpected error: {e}")
            return False

    # ------------------------------------------------------------------
    # Menu actions
    # ------------------------------------------------------------------

    def show_last_run_info(self):
        if not self.exec_state["last_run_id"]:
            return
        run_id      = self.exec_state["last_run_id"]
        last_step_id = self.exec_state["last_step"]
        if run_id not in self.exec_state["runs"]:
            return
        run_data  = self.exec_state["runs"][run_id]
        last_step = self.get_step_by_id(last_step_id)
        if not last_step:
            return
        print(f"\n{Colors.BOLD}Last Run:{Colors.ENDC}")
        print(f"  Run ID    : {Colors.YELLOW}{run_id}{Colors.ENDC}")
        print(
            f"  Last Step : {Colors.YELLOW}{last_step['name']} "
            f"(Step {last_step_id}){Colors.ENDC}"
        )
        if last_step_id in run_data["steps"]:
            sd = run_data["steps"][last_step_id]
            print(f"  Completed : {Colors.GREEN}{sd['completed_at']}{Colors.ENDC}")
            print(f"  Output    : {Colors.CYAN}{sd['output']}{Colors.ENDC}")

    def rerun_last_step(self):
        if not self.exec_state["last_run_id"]:
            self.print_error("No previous run found")
            return
        run_id   = self.exec_state["last_run_id"]
        run_data = self.exec_state["runs"][run_id]
        if not run_data["steps"]:
            self.print_error("No completed steps found in last run")
            return
        last_step_id = max(run_data["steps"].keys())
        step_data    = run_data["steps"][last_step_id]
        step_def     = self.get_step_by_id(last_step_id)
        if step_def is None:
            self.print_error(f"Step definition for id={last_step_id} not found")
            return
        print(f"\n{Colors.HEADER}Rerunning Step {last_step_id}: {step_data['name']}{Colors.ENDC}")
        prev_params = step_data.get("parameters", {})
        print(f"{Colors.CYAN}Previous parameters:{Colors.ENDC}")
        for k, v in prev_params.items():
            print(f"  {k}: {v}")
        if self.get_input("\nModify parameters? (y/n)", "n").lower() == "y":
            params = self.collect_parameters(step_def, run_id)
        else:
            params = prev_params
        if self.get_input("Proceed? (y/n)", "y").lower() != "y":
            print(f"{Colors.YELLOW}Rerun cancelled{Colors.ENDC}")
            return
        self.execute_step(step_def, params, run_id)

    def continue_run(self):
        if not self.exec_state["last_run_id"] or not self.exec_state["last_step"]:
            self.print_error("No previous run found.")
            return self.start_new_run()
        run_id       = self.exec_state["last_run_id"]
        last_step_id = self.exec_state["last_step"]
        if last_step_id >= len(self.PIPELINE_STEPS):
            self.print_success("Pipeline already completed.")
            return True
        next_step = self.PIPELINE_STEPS[last_step_id]
        print(f"\n{Colors.BOLD}Continuing run: {Colors.YELLOW}{run_id}{Colors.ENDC}")
        print(f"Next: Step {next_step['id']}: {next_step['name']}")
        params = self.collect_parameters(next_step, run_id)
        return self.execute_step(next_step, params, run_id)

    def show_run_history(self):
        if not self.exec_state["runs"]:
            print(f"\n{Colors.YELLOW}No run history.{Colors.ENDC}")
            return
        print(f"\n{Colors.BOLD}Run History:{Colors.ENDC}")
        print(f"{Colors.CYAN}{'─' * 70}{Colors.ENDC}")
        for run_id, run_data in sorted(
            self.exec_state["runs"].items(), reverse=True
        ):
            print(f"\n{Colors.BOLD}Run: {Colors.YELLOW}{run_id}{Colors.ENDC}")
            print(f"  Created : {run_data['created_at']}")
            print(f"  Steps   : {len(run_data['steps'])}")
            for sid in sorted(run_data["steps"].keys()):
                sd = run_data["steps"][sid]
                print(f"  {Colors.GREEN}✓{Colors.ENDC} Step {sid}: {sd['name']}")
                print(f"    Output: {Colors.CYAN}{sd['output']}{Colors.ENDC}")
        print(f"\n{Colors.CYAN}{'─' * 70}{Colors.ENDC}")
        input(f"\n{Colors.BOLD}Press Enter to continue...{Colors.ENDC}")

    def start_new_run(self) -> bool:
        run_id = self.generate_run_id()
        self.print_header(f"New Run: {run_id}")
        step_options  = [f"Step {s['id']}: {s['name']}" for s in self.PIPELINE_STEPS]
        start_choice  = self.get_choice("Start from which step?", step_options)

        self.exec_state["runs"][run_id] = {
            "created_at": datetime.now().isoformat(),
            "steps": {},
        }
        self.save_exec_state()

        current_index = start_choice - 1
        while current_index < len(self.PIPELINE_STEPS):
            step = self.PIPELINE_STEPS[current_index]
            print(
                f"\n{Colors.BOLD}{Colors.HEADER}"
                f"Step {step['id']}: {step['name']}{Colors.ENDC}"
            )
            params = self.collect_parameters(step, run_id)
            print(f"\n{Colors.CYAN}Parameters:{Colors.ENDC}")
            for k, v in params.items():
                print(f"  {k}: {v}")
            if self.get_input(f"Execute step {step['id']}? (y/n)", "y").lower() != "y":
                self.print_warning(f"Step {step['id']} skipped")
                break
            if not self.execute_step(step, params, run_id):
                self.print_error(f"Run stopped at step {step['id']}")
                return False
            current_index += 1
            if current_index < len(self.PIPELINE_STEPS):
                nxt = self.PIPELINE_STEPS[current_index]
                if self.get_input(
                    f"Continue to Step {nxt['id']}: {nxt['name']}? (y/n)", "y"
                ).lower() != "y":
                    self.print_warning("Run paused. Continue from main menu.")
                    return True
            else:
                self.print_success("All pipeline steps completed!")
        return True

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self):
        logger.info("SemanticRunner started")
        self.print_header("Semantic Folding Pipeline Runner")

        while True:
            self.show_last_run_info()
            options = []
            if self.exec_state["last_run_id"]:
                last_step = self.get_step_by_id(self.exec_state["last_step"])
                options.append(
                    f"Rerun last step (Step {self.exec_state['last_step']}: "
                    f"{last_step['name']})"
                )
                if self.exec_state["last_step"] < len(self.PIPELINE_STEPS):
                    options.append("Continue to next step")
            options.extend(["Start new run", "View run history", "Exit"])
            choice = self.get_choice("What would you like to do?", options)

            offset = 0
            if self.exec_state["last_run_id"]:
                if choice == 1:
                    self.rerun_last_step()
                    continue
                if self.exec_state["last_step"] < len(self.PIPELINE_STEPS):
                    if choice == 2:
                        self.continue_run()
                        continue
                    offset = 1

            adjusted = choice - offset - (1 if self.exec_state["last_run_id"] else 0)
            if adjusted == 1:
                self.start_new_run()
            elif adjusted == 2:
                self.show_run_history()
            elif adjusted == 3:
                logger.info("User exited")
                print(f"\n{Colors.GREEN}Goodbye!{Colors.ENDC}\n")
                break


def main():
    try:
        runner = SemanticRunner()
        runner.run()
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        print(f"\n\n{Colors.YELLOW}Interrupted{Colors.ENDC}\n")
        sys.exit(0)
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        print(f"\n{Colors.RED}Fatal error: {e}{Colors.ENDC}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
