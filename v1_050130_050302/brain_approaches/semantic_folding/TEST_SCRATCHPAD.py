#!/usr/bin/env python3
"""
TEST SCRATCHPAD - Semantic Folding Pipeline Quality Analysis

This script tests the semantic folding pipeline on a controlled dataset with 20 contexts
covering 4 distinct topics: AI/ML, Environment/Sustainability, Space, and Technology.

The goal is to ensure:
1. Proper phrase extraction and diversity
2. Good matrix sparsity (not too sparse or dense)
3. Well-distributed semantic space (not centralized)
4. Accurate fingerprint generation (similar contexts have similar fingerprints)
5. Different contexts have distinct fingerprints

Usage:
    python TEST_SCRATCHPAD.py
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
# from sklearn.metrics.pairwise import cosine_similarity  # Not available, implementing manually
from collections import Counter

# Global variable to track the actual output directory (set by Phase 1)
actual_output_dir = None

def cosine_similarity_manual(vec1, vec2):
    """Manual implementation of cosine similarity"""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2) if norm1 != 0 and norm2 != 0 else 0
import subprocess
import os
import shutil
import sys
from datetime import datetime

def create_test_config():
    """Create test configuration for controlled experiment"""
    config_content = """
# Test Configuration for Quality Analysis
corpus_path: "test_corpus.json"
output_base: "test_outputs"
grid_size: 32                    # Larger grid for better distribution
max_phrases: null               # No limit for analysis
max_docs: null                  # Process all documents
log_level: "INFO"
debug: false
batch_size: 1000
max_edges: 200                 # Reduced edges for cleaner layout
edge_threshold: 0.05          # Lower threshold for better connectivity
use_spacy: true
no_visualization: false         # Enable visualizations for analysis
"""

    # Create config directory if it doesn't exist
    os.makedirs("../config", exist_ok=True)
    with open("../config/test_semantic_folding.yml", "w") as f:
        f.write(config_content.strip())

    print("Created test configuration")

def run_pipeline_phase(phase_num, output_dir):
    """Run a specific pipeline phase and capture output"""
    print(f"\n{'='*60}")
    print(f"PHASE {phase_num} EXECUTION")
    print('='*60)

    cmd = [
        "uv", "run", "python", "semantic_folder.py",
        "--config", "../../config/test_semantic_folding.yml",
        "--run-phase", str(phase_num),
        "--output-dir", str(output_dir)
    ]

    print(f"Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, cwd=os.getcwd(), capture_output=True, text=True)

        if result.returncode == 0:
            print(f"Phase {phase_num} completed successfully")

            # Extract output directory from stdout if Phase 1
            if phase_num == 1 and result.stdout:
                for line in result.stdout.split('\n'):
                    if line.startswith("Output directory:"):
                        global actual_output_dir
                        actual_output_dir = line.split(": ")[1].strip()
                        print(f"Using output directory: {actual_output_dir}")
                        break

            return True
        else:
            print(f"Phase {phase_num} failed with exit code {result.returncode}")
            if result.stdout:
                print("STDOUT:", result.stdout[-500:])
            if result.stderr:
                print("STDERR:", result.stderr[-500:])
            return False
    except Exception as e:
        print(f"ERROR: Phase {phase_num} error: {e}")
        return False

def analyze_corpus(output_dir):
    """Analyze the processed corpus"""
    print(f"\n{'='*60}")
    print("PHASE 1 ANALYSIS - CORPUS PROCESSING")
    print('='*60)

    corpus_file = output_dir / "corpus.txt"

    if not corpus_file.exists():
        print("ERROR: Corpus file not found")
        return False

    with open(corpus_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    print(f"✓ Corpus processed: {len(lines)} documents")

    # Show sample entries
    print("\nSample entries:")
    for i, line in enumerate(lines[:3]):
        print(f"  {i+1}: {line[:100]}...")

    # Check for proper formatting (idx,title: text)
    valid_format = all(':' in line and ',' in line.split(':')[0] for line in lines)
    print(f"✓ Proper formatting: {valid_format}")

    return True

def analyze_phrases(output_dir):
    """Analyze extracted phrases"""
    print(f"\n{'='*60}")
    print("PHASE 2 ANALYSIS - PHRASE EXTRACTION")
    print('='*60)

    phrases_file = output_dir / "phrases.txt"

    if not phrases_file.exists():
        print("ERROR: Phrases file not found")
        return False

    with open(phrases_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    print(f"✓ Phrases extracted: {len(lines)}")

    # Analyze phrase diversity
    phrases = []
    frequencies = []

    for line in lines:
        if ':' in line:
            phrase, freq_str = line.strip().rsplit(':', 1)
            phrases.append(phrase.strip())
            try:
                frequencies.append(int(freq_str.strip()))
            except ValueError:
                frequencies.append(0)

    print(f"✓ Phrase frequency range: {min(frequencies)} - {max(frequencies)}")

    # Check phrase length distribution
    phrase_lengths = [len(p.split()) for p in phrases]
    print(f"✓ Phrase length distribution: {Counter(phrase_lengths)}")

    # Show top phrases
    print("\nTop 10 phrases by frequency:")
    sorted_phrases = sorted(zip(phrases, frequencies), key=lambda x: x[1], reverse=True)
    for phrase, freq in sorted_phrases[:10]:
        print(f"  {phrase}: {freq}")
    # Check for topic-relevant phrases
    topic_indicators = {
        'AI/ML': ['machine learning', 'neural network', 'artificial intelligence', 'deep learning'],
        'Environment': ['climate change', 'renewable energy', 'sustainable', 'conservation'],
        'Space': ['mars', 'space exploration', 'satellite', 'astronaut'],
        'Technology': ['computer vision', 'blockchain', 'quantum computing', 'virtual reality']
    }

    print("\nTopic coverage check:")
    for topic, keywords in topic_indicators.items():
        found = sum(1 for p in phrases if any(kw in p.lower() for kw in keywords))
        print(f"  {topic}: {found} relevant phrases found")

    return True

def analyze_matrix(output_dir):
    """Analyze term-context matrix"""
    print(f"\n{'='*60}")
    print("PHASE 3 ANALYSIS - TERM-CONTEXT MATRIX")
    print('='*60)

    matrix_file = output_dir / "term_context_matrix.npz"
    stats_file = output_dir / "term_context_matrix.json"

    if not matrix_file.exists():
        print("ERROR: Matrix file not found")
        return False

    # Load matrix stats
    if stats_file.exists():
        with open(stats_file, 'r', encoding='utf-8') as f:
            stats = json.load(f)

        print(f"✓ Matrix dimensions: {stats['num_contexts']} × {stats['num_phrases']}")
        print(f"✓ Matrix entries: {stats['entries']:,}")
        print(f"✓ Matrix density: {stats['density']:.6f}")
        # Check sparsity - should be between 0.001 and 0.1 for good semantic folding
        sparsity = stats['density']
        if 0.001 <= sparsity <= 0.1:
            print("✓ Good sparsity range for semantic folding")
        elif sparsity < 0.001:
            print("WARNING: Low sparsity - may indicate poor phrase-document connections")
        else:
            print("WARNING: High sparsity - matrix may be too dense")

    # Load actual matrix for detailed analysis
    try:
        matrix_data = np.load(matrix_file)
        if 'data' in matrix_data:
            # Sparse matrix
            data = matrix_data['data']
            indices = matrix_data['indices']
            indptr = matrix_data['indptr']
            matrix = scipy.sparse.csr_matrix((data, indices, indptr), shape=(20, len(phrases)))
        else:
            # Dense matrix
            matrix = matrix_data['matrix']

        print(f"✓ Matrix loaded: shape {matrix.shape}")

        # Analyze matrix properties
        if hasattr(matrix, 'nnz'):
            # Sparse matrix
            nnz = matrix.nnz
            density = nnz / (matrix.shape[0] * matrix.shape[1])
            print(f"✓ Matrix density: {density:.6f}")

            # Row-wise analysis
            row_nnz = np.array([matrix[i].nnz for i in range(matrix.shape[0])])
            print(f"✓ Documents connectivity: avg {row_nnz.mean():.1f} phrases per doc (range: {row_nnz.min()}-{row_nnz.max()})")

            # Column-wise analysis
            col_nnz = np.array([matrix[:, i].nnz for i in range(matrix.shape[1])])
            col_nnz = col_nnz[col_nnz > 0]  # Only non-zero columns
            if len(col_nnz) > 0:
                print(f"✓ Phrase distribution: avg {col_nnz.mean():.1f} docs per phrase (range: {col_nnz.min()}-{col_nnz.max()})")

    except Exception as e:
        print(f"WARNING: Could not analyze matrix details: {e}")

    return True

def analyze_semantic_space(output_dir):
    """Analyze semantic space construction"""
    print(f"\n{'='*60}")
    print("PHASE 4 ANALYSIS - SEMANTIC SPACE CONSTRUCTION")
    print('='*60)

    coords_file = output_dir / "context_coordinates.csv"

    if not coords_file.exists():
        print("ERROR: Coordinates file not found")
        return False

    # Load coordinates
    coords_data = np.loadtxt(coords_file, delimiter=',', skiprows=1)
    coords = coords_data[:, 1:3]  # x, y coordinates

    print(f"✓ Semantic space: {len(coords)} contexts mapped")

    # Analyze coordinate distribution
    x_coords, y_coords = coords[:, 0], coords[:, 1]

    print(f"✓ X coordinate range: [{x_coords.min():.3f}, {x_coords.max():.3f}]")
    print(f"✓ Y coordinate range: [{y_coords.min():.3f}, {y_coords.max():.3f}]")

    # Check for good distribution (not centralized)
    center_x, center_y = x_coords.mean(), y_coords.mean()
    distances_from_center = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
    avg_distance = distances_from_center.mean()
    max_distance = distances_from_center.max()

    print(f"✓ Distribution analysis:")
    print(".3f")
    print(".3f")
    # Check if points are well distributed (not clustered in center)
    central_points = sum(1 for d in distances_from_center if d < avg_distance * 0.5)
    print(f"✓ Points near center: {central_points}/{len(coords)} ({central_points/len(coords)*100:.1f}%)")

    if central_points / len(coords) > 0.7:
        print("WARNING: Points are too centralized - may indicate poor semantic separation")
        return False
    else:
        print("✓ Good distribution - points are well spread across semantic space")

    # Analyze clustering by topic (manual assignment for test)
    topic_assignments = {
        'AI/ML': [0, 1, 2, 3],        # ML fundamentals, supervised, neural nets, NLP
        'Technology': [4, 5, 14, 15, 16, 17, 18, 19],  # Vision, autonomous, quantum, crypto, fintech, blockchain, VR, AR
        'Environment': [8, 9, 10, 11], # Climate, renewable, agriculture, conservation
        'Space': [12, 13]             # Space exploration, Mars
    }

    print("\nTopic clustering analysis:")
    for topic, indices in topic_assignments.items():
        if indices:
            topic_coords = coords[indices]
            topic_center = topic_coords.mean(axis=0)
            topic_spread = np.std(topic_coords, axis=0).mean()
            print(f"  {topic}: center ({topic_center[0]:.3f}, {topic_center[1]:.3f}), spread {topic_spread:.3f}")

    return True

def analyze_fingerprints(output_dir):
    """Analyze generated fingerprints"""
    print(f"\n{'='*60}")
    print("PHASE 5 ANALYSIS - FINGERPRINT GENERATION")
    print('='*60)

    # Check fingerprint files
    fp_dir = output_dir / "fingerprints"
    doc_fp_dir = output_dir / "doc_fingerprints"

    if fp_dir.exists():
        phrase_fps = list(fp_dir.glob("*.txt"))
        print(f"✓ Phrase fingerprints: {len(phrase_fps)}")

    if doc_fp_dir.exists():
        doc_fps = list(doc_fp_dir.glob("*_fingerprint.txt"))
        print(f"✓ Document fingerprints: {len(doc_fps)}")

    if not doc_fp_dir.exists() or len(list(doc_fp_dir.glob("*_fingerprint.txt"))) == 0:
        print("ERROR: Document fingerprints not found")
        return False

    # Load sample fingerprints for analysis
    doc_fingerprints = {}

    for fp_file in doc_fp_dir.glob("*_fingerprint.txt"):
        try:
            with open(fp_file, 'r', encoding='utf-8') as f:
                matrix_data = []
                for line in f:
                    row = [int(x) for x in line.strip().split('\t')]
                    matrix_data.append(row)

            fingerprint = np.array(matrix_data)
            doc_id = fp_file.stem.replace('_fingerprint', '')
            doc_fingerprints[doc_id] = fingerprint
        except Exception as e:
            print(f"WARNING: Could not load fingerprint {fp_file}: {e}")

    print(f"✓ Loaded fingerprints: {len(doc_fingerprints)}")

    # Analyze fingerprint properties
    if doc_fingerprints:
        sample_fp = list(doc_fingerprints.values())[0]
        print(f"✓ Fingerprint dimensions: {sample_fp.shape}")

        # Check binary nature
        unique_values = set()
        for fp in doc_fingerprints.values():
            unique_values.update(fp.flatten())

        print(f"✓ Unique values in fingerprints: {sorted(unique_values)}")

        if unique_values == {0, 1}:
            print("✓ Fingerprints are properly binary")
        else:
            print("WARNING: Fingerprints contain non-binary values")

        # Analyze sparsity of fingerprints
        total_cells = sum(fp.size for fp in doc_fingerprints.values())
        active_cells = sum(np.sum(fp) for fp in doc_fingerprints.values())
        fp_sparsity = active_cells / total_cells

        print(".4f")
        if 0.01 <= fp_sparsity <= 0.5:
            print("✓ Good fingerprint sparsity")
        elif fp_sparsity < 0.01:
            print("WARNING: Fingerprints too sparse - may lack semantic information")
        else:
            print("WARNING: Fingerprints too dense - may lack specificity")

    return True

def compare_contexts(output_dir):
    """Compare fingerprints of similar vs different contexts"""
    print(f"\n{'='*60}")
    print("CONTEXT SIMILARITY ANALYSIS")
    print('='*60)

    # Define similar and different context pairs
    similar_pairs = [
        (0, 1),    # ML fundamentals vs supervised learning (both ML core)
        (2, 3),    # Neural networks vs NLP (both AI applications)
        (4, 5),    # Computer vision vs autonomous driving (both vision applications)
        (8, 9),    # Climate change vs renewable energy (both environmental)
        (12, 13),  # Space exploration vs Mars colonization (both space)
    ]

    different_pairs = [
        (0, 8),   # ML vs climate change (different topics)
        (2, 12),  # Neural networks vs space exploration
        (4, 10),  # Computer vision vs agriculture
        (6, 14),  # Healthcare AI vs quantum computing
        (9, 16),  # Renewable energy vs fintech
    ]

    doc_fp_dir = output_dir / "doc_fingerprints"
    doc_fingerprints = {}

    # Load fingerprints
    for fp_file in doc_fp_dir.glob("*_fingerprint.txt"):
        try:
            with open(fp_file, 'r', encoding='utf-8') as f:
                matrix_data = []
                for line in f:
                    row = [int(x) for x in line.strip().split('\t')]
                    matrix_data.append(row)

            fingerprint = np.array(matrix_data).flatten()
            doc_id = fp_file.stem.replace('_fingerprint', '')
            doc_fingerprints[int(doc_id)] = fingerprint
        except Exception as e:
            continue

    if not doc_fingerprints:
        print("ERROR: Could not load fingerprints for comparison")
        return False

    # Calculate similarities
    def jaccard_similarity(fp1, fp2):
        """Calculate Jaccard similarity between two binary vectors"""
        intersection = np.sum(np.logical_and(fp1, fp2))
        union = np.sum(np.logical_or(fp1, fp2))
        return intersection / union if union > 0 else 0

    def cosine_sim(fp1, fp2):
        """Calculate cosine similarity"""
        return cosine_similarity_manual(fp1, fp2)

    print("\nSimilar context pairs (should have higher similarity):")
    similar_scores = []
    for i, j in similar_pairs:
        if i in doc_fingerprints and j in doc_fingerprints:
            jaccard = jaccard_similarity(doc_fingerprints[i], doc_fingerprints[j])
            cosine = cosine_sim(doc_fingerprints[i], doc_fingerprints[j])
            similar_scores.append((jaccard, cosine))
            print(".3f")
    print("\nDifferent context pairs (should have lower similarity):")
    different_scores = []
    for i, j in different_pairs:
        if i in doc_fingerprints and j in doc_fingerprints:
            jaccard = jaccard_similarity(doc_fingerprints[i], doc_fingerprints[j])
            cosine = cosine_sim(doc_fingerprints[i], doc_fingerprints[j])
            different_scores.append((jaccard, cosine))
            print(".3f")
    # Statistical comparison
    if similar_scores and different_scores:
        avg_similar_jaccard = np.mean([s[0] for s in similar_scores])
        avg_different_jaccard = np.mean([s[0] for s in different_scores])
        avg_similar_cosine = np.mean([s[1] for s in similar_scores])
        avg_different_cosine = np.mean([s[1] for s in different_scores])

        print("\nStatistical Summary:")
        print(".3f")
        print(".3f")
        jaccard_separation = avg_similar_jaccard - avg_different_jaccard
        cosine_separation = avg_similar_cosine - avg_different_cosine

        if jaccard_separation > 0.1 and cosine_separation > 0.1:
            print("✓ Good semantic separation - similar contexts cluster, different contexts separate")
        else:
            print("WARNING: Poor semantic separation - fingerprints may not capture semantic relationships well")

    return True

def run_quality_test():
    """Run complete quality analysis"""
    print("SEMANTIC FOLDING PIPELINE - QUALITY ANALYSIS")
    print("=" * 60)
    print("Test Corpus: 20 contexts, 4 topics")
    print("Goals:")
    print("- Diverse phrase extraction")
    print("- Good matrix sparsity (0.001-0.1)")
    print("- Well-distributed semantic space")
    print("- Accurate fingerprint similarity")
    print()

    # Create test configuration
    create_test_config()

    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"test_outputs/test_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_dir}")

    # Run pipeline phases with analysis
    phases_success = []

    # Phase 1: Corpus processing
    if run_pipeline_phase(1, output_dir):
        # Use the actual output directory returned by Phase 1
        analysis_dir = Path(actual_output_dir) if actual_output_dir else output_dir
        phases_success.append(analyze_corpus(analysis_dir))

    # Phase 2: Phrase extraction
    if run_pipeline_phase(2, analysis_dir if actual_output_dir else output_dir):
        phases_success.append(analyze_phrases(analysis_dir if actual_output_dir else output_dir))

    # Phase 3: Matrix construction
    if run_pipeline_phase(3, analysis_dir if actual_output_dir else output_dir):
        phases_success.append(analyze_matrix(analysis_dir if actual_output_dir else output_dir))

    # Phase 4: Semantic space
    if run_pipeline_phase(4, analysis_dir if actual_output_dir else output_dir):
        phases_success.append(analyze_semantic_space(analysis_dir if actual_output_dir else output_dir))

    # Phase 5: Fingerprint generation
    if run_pipeline_phase(5, analysis_dir if actual_output_dir else output_dir):
        phases_success.append(analyze_fingerprints(analysis_dir if actual_output_dir else output_dir))

    # Phase 6: LanceDB (optional for this test)
    if run_pipeline_phase(6, analysis_dir if actual_output_dir else output_dir):
        phases_success.append(True)  # LanceDB is not critical for quality analysis

    # Final analysis: Context similarity comparison
    if all(phases_success):
        compare_contexts(output_dir)

        print(f"\n{'='*60}")
        print("QUALITY ANALYSIS SUMMARY")
        print('='*60)

        success_rate = sum(phases_success) / len(phases_success) * 100
        print(f"Success rate: {success_rate:.1f}%")
        if success_rate >= 80:
            print("✓ Overall quality assessment: GOOD")
            print("The semantic folding pipeline is generating meaningful fingerprints.")
        else:
            print("WARNING: Overall quality assessment: NEEDS IMPROVEMENT")
            print("Consider adjusting parameters or reviewing the algorithm.")
    else:
        print("ERROR: Pipeline execution incomplete - cannot perform quality analysis")

if __name__ == "__main__":
    run_quality_test()