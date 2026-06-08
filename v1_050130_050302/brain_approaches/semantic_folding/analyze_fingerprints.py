#!/usr/bin/env python3
"""
Fingerprint Quality Analysis for Semantic Folding Test

Analyzes the generated fingerprints to assess quality:
- Semantic similarity between related contexts
- Distribution quality (avoiding centralization)
- Fingerprint sparsity and information content
- Topic clustering validation
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter, defaultdict
import pandas as pd
from typing import Dict, List, Tuple

def load_fingerprints(fp_dir: Path, fp_type: str = "document") -> Dict[str, np.ndarray]:
    """Load fingerprints from directory"""
    fingerprints = {}

    if fp_type == "document":
        pattern = "*_fingerprint.txt"
        subdir = fp_dir / "doc_fingerprints"
    else:  # phrase
        pattern = "*.txt"
        subdir = fp_dir / "fingerprints"

    if not subdir.exists():
        print(f"Directory {subdir} not found")
        return fingerprints

    for fp_file in subdir.glob(pattern):
        try:
            with open(fp_file, 'r', encoding='utf-8') as f:
                matrix_data = []
                for line in f:
                    row = [float(x) for x in line.strip().split('\t')]
                    matrix_data.append(row)

            fingerprint = np.array(matrix_data)
            if fp_type == "document":
                doc_id = fp_file.stem.replace('_fingerprint', '')
            else:
                doc_id = fp_file.stem.replace('_fingerprint', '')

            fingerprints[doc_id] = fingerprint
        except Exception as e:
            print(f"Error loading {fp_file}: {e}")

    return fingerprints

def analyze_fingerprint_properties(fingerprints: Dict[str, np.ndarray], title: str) -> Dict:
    """Analyze basic properties of fingerprints"""
    if not fingerprints:
        return {}

    sample_fp = list(fingerprints.values())[0]
    grid_size = sample_fp.shape[0]

    properties = {
        'count': len(fingerprints),
        'grid_size': grid_size,
        'total_cells': grid_size * grid_size,
        'sparsities': [],
        'max_values': [],
        'active_positions': []
    }

    for fp in fingerprints.values():
        flattened = fp.flatten()

        # Sparsity (percentage of non-zero cells)
        sparsity = np.count_nonzero(fp) / properties['total_cells']
        properties['sparsities'].append(sparsity)

        # Maximum value
        properties['max_values'].append(np.max(fp))

        # Number of active positions
        properties['active_positions'].append(np.count_nonzero(fp))

    # Summary statistics
    properties.update({
        'avg_sparsity': np.mean(properties['sparsities']),
        'std_sparsity': np.std(properties['sparsities']),
        'avg_max_value': np.mean(properties['max_values']),
        'avg_active_positions': np.mean(properties['active_positions'])
    })

    print(f"\n{title} Analysis:")
    print(f"  Count: {properties['count']}")
    print(f"  Grid size: {properties['grid_size']}×{properties['grid_size']}")
    print(".4f")
    print(".2f")
    print(".1f")

    return properties

def analyze_semantic_space_distribution(coords_file: Path) -> Dict:
    """Analyze the distribution of contexts in semantic space"""
    coords_data = {}

    with open(coords_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()[1:]  # Skip header

        for line in lines:
            line = line.strip()
            if ',' in line:
                # Split on the first comma only
                comma_idx = line.find(',')
                context_id = line[:comma_idx]
                coords_part = line[comma_idx+1:].strip()

                # Remove quotes and parse coordinates
                coords_str = coords_part.strip('"')
                try:
                    x, y = map(int, coords_str.split(','))
                    coords_data[context_id] = (x, y)
                except ValueError:
                    continue
                coords_data[context_id] = (x, y)

    # Analyze distribution
    positions = list(coords_data.values())
    x_coords = [p[0] for p in positions]
    y_coords = [p[1] for p in positions]

    # Count positions
    position_counts = Counter(positions)

    # Find most common positions
    most_common = position_counts.most_common(5)

    # Calculate spread
    x_range = max(x_coords) - min(x_coords)
    y_range = max(y_coords) - min(y_coords)
    total_positions = len(set(positions))
    unique_positions = len(position_counts)

    distribution = {
        'total_contexts': len(coords_data),
        'unique_positions': unique_positions,
        'position_utilization': unique_positions / (x_range + 1) / (y_range + 1),
        'most_crowded_positions': most_common,
        'x_range': x_range,
        'y_range': y_range
    }

    print("\nSemantic Space Distribution:")
    print(f"  Total contexts: {distribution['total_contexts']}")
    print(f"  Grid range: {x_range+1}×{y_range+1} ({(x_range+1)*(y_range+1)} positions)")
    print(f"  Unique positions used: {distribution['unique_positions']}")
    print(".1%")

    if most_common:
        print("  Most crowded positions:")
        for pos, count in most_common[:3]:
            print(f"    {pos}: {count} contexts")

    # Check for centralization
    center_x, center_y = x_range // 2, y_range // 2
    center_count = position_counts.get((center_x, center_y), 0)
    center_percentage = center_count / len(coords_data) * 100

    print(".1f")

    if center_percentage > 30:
        print("  WARNING: High centralization detected!")
        distribution['centralized'] = True
    else:
        print("  OK: Well-distributed across semantic space")
        distribution['centralized'] = False

    return distribution

def compare_context_similarities(doc_fingerprints: Dict[str, np.ndarray]) -> Dict:
    """Compare similarities between similar and different contexts"""
    if not doc_fingerprints:
        return {}

    # Define similar pairs (contexts about the same topic)
    similar_pairs = [
        ('0', '1'),    # ML fundamentals vs supervised learning
        ('2', '3'),    # Neural networks vs NLP (both AI)
        ('4', '5'),    # Computer vision vs autonomous driving
        ('8', '9'),    # Climate change vs renewable energy
        ('12', '13'),  # Space exploration vs Mars colonization
    ]

    # Define different pairs (contexts about different topics)
    different_pairs = [
        ('0', '8'),   # ML vs climate change
        ('2', '12'),  # Neural networks vs space
        ('4', '10'),  # Computer vision vs agriculture
        ('6', '14'),  # Healthcare AI vs quantum computing
        ('9', '16'),  # Renewable energy vs fintech
    ]

    def jaccard_similarity(fp1: np.ndarray, fp2: np.ndarray) -> float:
        """Calculate Jaccard similarity between two fingerprints"""
        fp1_flat = fp1.flatten()
        fp2_flat = fp2.flatten()

        intersection = np.sum(np.logical_and(fp1_flat > 0, fp2_flat > 0))
        union = np.sum(np.logical_or(fp1_flat > 0, fp2_flat > 0))

        return intersection / union if union > 0 else 0

    def weighted_cosine_similarity(fp1: np.ndarray, fp2: np.ndarray) -> float:
        """Calculate cosine similarity using weighted values"""
        fp1_flat = fp1.flatten()
        fp2_flat = fp2.flatten()
        return cosine_similarity([fp1_flat], [fp2_flat])[0][0]

    # Calculate similarities
    similar_scores = {'jaccard': [], 'cosine': []}
    different_scores = {'jaccard': [], 'cosine': []}

    for pair in similar_pairs:
        if pair[0] in doc_fingerprints and pair[1] in doc_fingerprints:
            fp1, fp2 = doc_fingerprints[pair[0]], doc_fingerprints[pair[1]]
            similar_scores['jaccard'].append(jaccard_similarity(fp1, fp2))
            similar_scores['cosine'].append(weighted_cosine_similarity(fp1, fp2))

    for pair in different_pairs:
        if pair[0] in doc_fingerprints and pair[1] in doc_fingerprints:
            fp1, fp2 = doc_fingerprints[pair[0]], doc_fingerprints[pair[1]]
            different_scores['jaccard'].append(jaccard_similarity(fp1, fp2))
            different_scores['cosine'].append(weighted_cosine_similarity(fp1, fp2))

    # Calculate statistics
    results = {
        'similar_jaccard_avg': np.mean(similar_scores['jaccard']) if similar_scores['jaccard'] else 0,
        'similar_cosine_avg': np.mean(similar_scores['cosine']) if similar_scores['cosine'] else 0,
        'different_jaccard_avg': np.mean(different_scores['jaccard']) if different_scores['jaccard'] else 0,
        'different_cosine_avg': np.mean(different_scores['cosine']) if different_scores['cosine'] else 0,
        'similar_pairs_count': len(similar_scores['jaccard']),
        'different_pairs_count': len(different_scores['jaccard'])
    }

    results['jaccard_separation'] = results['similar_jaccard_avg'] - results['different_jaccard_avg']
    results['cosine_separation'] = results['similar_cosine_avg'] - results['different_cosine_avg']

    print("\nContext Similarity Analysis:")
    print(f"  Similar pairs analyzed: {results['similar_pairs_count']}")
    print(f"  Different pairs analyzed: {results['different_pairs_count']}")

    if results['similar_pairs_count'] > 0 and results['different_pairs_count'] > 0:
        print("\nSimilarity Scores:")
        print(".4f")
        print(".4f")
        print(".4f")
        print(".4f")
        print("\nSeparation Quality:")
        print(".4f")
        print(".4f")
        if results['jaccard_separation'] > 0.1 and results['cosine_separation'] > 0.1:
            print("  RESULT: Good semantic separation - similar contexts cluster together!")
            results['good_separation'] = True
        else:
            print("  RESULT: Poor semantic separation - fingerprints may not capture semantic relationships")
            results['good_separation'] = False

    return results

def analyze_topic_clustering(doc_fingerprints: Dict[str, np.ndarray], coords_data: Dict[str, Tuple[int, int]]) -> Dict:
    """Analyze how well topics cluster in semantic space"""
    # Define topic assignments
    topic_assignments = {
        'AI/ML': ['0', '1', '2', '3'],        # ML fundamentals, supervised, neural nets, NLP
        'Technology': ['4', '5', '14', '15', '16', '17', '18', '19'],  # Vision, autonomous, quantum, crypto, fintech, blockchain, VR, AR
        'Environment': ['8', '9', '10', '11'], # Climate, renewable, agriculture, conservation
        'Space': ['12', '13']                # Space exploration, Mars
    }

    topic_analysis = {}

    for topic_name, doc_ids in topic_assignments.items():
        topic_docs = [doc_id for doc_id in doc_ids if doc_id in doc_fingerprints]

        if len(topic_docs) < 2:
            continue

        # Get coordinates for topic documents
        topic_coords = [coords_data[doc_id] for doc_id in topic_docs if doc_id in coords_data]

        if topic_coords:
            # Calculate centroid
            centroid_x = sum(x for x, y in topic_coords) / len(topic_coords)
            centroid_y = sum(y for x, y in topic_coords) / len(topic_coords)

            # Calculate spread (average distance from centroid)
            distances = [np.sqrt((x - centroid_x)**2 + (y - centroid_y)**2) for x, y in topic_coords]
            avg_spread = np.mean(distances)

            topic_analysis[topic_name] = {
                'doc_count': len(topic_docs),
                'centroid': (centroid_x, centroid_y),
                'avg_spread': avg_spread,
                'coord_spread': (max(x for x, y in topic_coords) - min(x for x, y in topic_coords),
                               max(y for x, y in topic_coords) - min(y for x, y in topic_coords))
            }

    print("\nTopic Clustering Analysis:")
    for topic, data in topic_analysis.items():
        print(f"  {topic}:")
        print(f"    Documents: {data['doc_count']}")
        print(".2f")
        print(f"    Coordinate spread: {data['coord_spread'][0]}×{data['coord_spread'][1]}")

    return topic_analysis

def create_visualizations(output_dir: Path, doc_fingerprints: Dict[str, np.ndarray], coords_file: Path):
    """Create visualizations of the results"""
    try:
        # 1. Semantic space distribution heatmap
        coords_data = {}
        with open(coords_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()[1:]
            for line in lines:
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    context_id = parts[0].split('_')[1]
                    coords = parts[1].strip('"')
                    x, y = map(int, coords.split(','))
                    coords_data[context_id] = (x, y)

        # Create heatmap of context distribution
        grid_size = 8  # From config
        heatmap = np.zeros((grid_size, grid_size))

        for coord in coords_data.values():
            if coord[0] < grid_size and coord[1] < grid_size:
                heatmap[coord[1], coord[0]] += 1  # Note: y,x indexing for imshow

        plt.figure(figsize=(8, 6))
        sns.heatmap(heatmap, annot=True, fmt='d', cmap='YlOrRd')
        plt.title('Context Distribution in Semantic Space\n(8×8 grid)')
        plt.xlabel('X coordinate')
        plt.ylabel('Y coordinate')
        plt.tight_layout()
        plt.savefig(output_dir / 'context_distribution_heatmap.png', dpi=150, bbox_inches='tight')
        plt.close()

        # 2. Sample fingerprint visualization
        if doc_fingerprints:
            sample_ids = list(doc_fingerprints.keys())[:4]  # First 4 documents

            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            axes = axes.flatten()

            for i, doc_id in enumerate(sample_ids):
                fp = doc_fingerprints[doc_id]
                sns.heatmap(fp, ax=axes[i], cmap='viridis', cbar=True)
                axes[i].set_title(f'Document {doc_id} Fingerprint')
                axes[i].set_xlabel('X')
                axes[i].set_ylabel('Y')

            plt.tight_layout()
            plt.savefig(output_dir / 'sample_fingerprints.png', dpi=150, bbox_inches='tight')
            plt.close()

        print("\nVisualizations created:")
        print("  - context_distribution_heatmap.png")
        print("  - sample_fingerprints.png")

    except Exception as e:
        print(f"Error creating visualizations: {e}")

def main():
    """Main analysis function"""
    print("SEMANTIC FOLDING FINGERPRINT QUALITY ANALYSIS")
    print("=" * 60)

    # Use the test_phase1 directory which has complete results
    output_dir = Path("test_phase1/test_20260215_031744")

    if not output_dir.exists():
        # Fallback to finding the most recent test output
        test_dirs = []
        for path in Path(".").glob("test_outputs/test_*/test_*"):
            if path.is_dir():
                test_dirs.append(path)

        if not test_dirs:
            print("No test output directories found. Run TEST_SCRATCHPAD.py first.")
            return

        # Use the most recent
        output_dir = max(test_dirs, key=lambda x: x.stat().st_mtime)
    print(f"Analyzing results from: {output_dir}")

    # Load fingerprints
    doc_fingerprints = load_fingerprints(output_dir, "document")
    phrase_fingerprints = load_fingerprints(output_dir, "phrase")

    if not doc_fingerprints:
        print("No document fingerprints found to analyze.")
        return

    # Run analyses
    doc_props = analyze_fingerprint_properties(doc_fingerprints, "Document Fingerprints")
    phrase_props = analyze_fingerprint_properties(phrase_fingerprints, "Phrase Fingerprints")

    coords_file = output_dir / "context_coordinates.csv"
    if coords_file.exists():
        distribution = analyze_semantic_space_distribution(coords_file)
        topic_clustering = analyze_topic_clustering(doc_fingerprints, load_coords_as_dict(coords_file))
    else:
        distribution = {}
        topic_clustering = {}

    similarity_results = compare_context_similarities(doc_fingerprints)

    # Create visualizations
    create_visualizations(output_dir, doc_fingerprints, coords_file)

    # Overall assessment
    print("\n" + "="*60)
    print("OVERALL QUALITY ASSESSMENT")
    print("="*60)

    score = 0
    max_score = 5

    # Check sparsity
    if doc_props and 0.01 <= doc_props['avg_sparsity'] <= 0.5:
        print("Good fingerprint sparsity")
        score += 1
    else:
        print("Suboptimal fingerprint sparsity")

    # Check semantic separation
    if similarity_results.get('good_separation', False):
        print("Good semantic separation between similar/different contexts")
        score += 1
    else:
        print("Poor semantic separation")

    # Check distribution (not centralized)
    if distribution and not distribution.get('centralized', True):
        print("Well-distributed semantic space")
        score += 1
    else:
        print("Centralized semantic space (needs improvement)")

    # Check topic clustering
    if topic_clustering:
        print("Topic clustering analysis available")
        score += 1
    else:
        print("Topic clustering not properly analyzed")

    # Check visualization generation
    viz_files = [output_dir / f for f in ['context_distribution_heatmap.png', 'sample_fingerprints.png']]
    if all(f.exists() for f in viz_files):
        print("Visualizations generated successfully")
        score += 1
    else:
        print("Some visualizations missing")

    print(f"\nQuality Score: {score}/{max_score} ({score/max_score*100:.1f}%)")

    if score >= 4:
        print("RESULT: Excellent quality - semantic folding is working well!")
    elif score >= 3:
        print("RESULT: Good quality - minor improvements needed")
    else:
        print("RESULT: Needs improvement - review algorithm parameters")

    print("\nAnalysis complete. Check the output directory for visualizations.")

def load_coords_as_dict(coords_file: Path) -> Dict[str, Tuple[int, int]]:
    """Load coordinates as dictionary"""
    coords_data = {}
    with open(coords_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()[1:]
        for line in lines:
            line = line.strip()
            if ',' in line:
                # Split on the first comma only
                comma_idx = line.find(',')
                context_id = line[:comma_idx]
                coords_part = line[comma_idx+1:].strip()

                # Remove quotes and parse coordinates
                coords_str = coords_part.strip('"')
                try:
                    x, y = map(int, coords_str.split(','))
                    coords_data[context_id] = (x, y)
                except ValueError:
                    continue
                coords_data[context_id] = (x, y)
    return coords_data

if __name__ == "__main__":
    main()