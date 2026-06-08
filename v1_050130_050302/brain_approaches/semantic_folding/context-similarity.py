import numpy as np
import scipy.sparse
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import norm as sparse_norm
import json
import argparse
from pathlib import Path
from typing import Tuple, Dict, Any, List
import sys

def load_sparse_matrix(matrix_path: Path) -> Tuple[csr_matrix, Dict[str, Any]]:
    """Load sparse term-context matrix from NPZ format"""
    try:
        npz_data = np.load(matrix_path)
        matrix = csr_matrix(
            (npz_data['data'], npz_data['indices'], npz_data['indptr']),
            shape=npz_data['shape']
        )
        
        # Load metadata
        metadata_path = matrix_path.with_suffix('.json')
        if not metadata_path.exists():
            print(f"Warning: Metadata file not found at {metadata_path}")
            metadata = {'num_contexts': matrix.shape[0], 'num_terms': matrix.shape[1]}
        else:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        
        return matrix, metadata
    except Exception as e:
        print(f"Error loading matrix: {e}")
        sys.exit(1)

def load_phrases(phrases_path: Path) -> List[str]:
    """Load phrases from file"""
    phrases = []
    try:
        with open(phrases_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if ':' in line:
                    phrase = line.split(':', 1)[0].strip()
                else:
                    phrase = line
                if phrase:
                    phrases.append(phrase)
        return phrases
    except Exception as e:
        print(f"Error loading phrases: {e}")
        sys.exit(1)

def load_corpus(corpus_path: Path) -> List[str]:
    """Load corpus contexts from file"""
    try:
        with open(corpus_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
    except Exception as e:
        print(f"Warning: Could not load corpus file: {e}")
        return []

def get_context_similarity_efficient(matrix: csr_matrix, context1: int, context2: int) -> float:
    """Get cosine similarity between two contexts using sparse operations"""
    vec1 = matrix[context1]
    vec2 = matrix[context2]
    
    # Compute dot product efficiently for sparse vectors
    dot_product = vec1.dot(vec2.T).toarray()[0, 0]
    
    # Compute norms efficiently
    norm1 = sparse_norm(vec1)
    norm2 = sparse_norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)

def get_shared_terms(matrix: csr_matrix, phrases: List[str], context1: int, context2: int, 
                     top_n: int = None) -> List[Tuple[str, float, float]]:
    """Get shared terms between two contexts with their weights"""
    vec1 = matrix[context1].toarray().flatten()
    vec2 = matrix[context2].toarray().flatten()
    
    shared_terms = []
    for i in range(len(phrases)):
        if vec1[i] > 0 and vec2[i] > 0:
            # Store phrase with weights from both contexts
            shared_terms.append((phrases[i], vec1[i], vec2[i]))
    
    # Sort by minimum weight (most significant shared terms)
    shared_terms.sort(key=lambda x: min(x[1], x[2]), reverse=True)
    
    if top_n:
        return shared_terms[:top_n]
    return shared_terms

def get_top_terms(matrix: csr_matrix, phrases: List[str], context_idx: int, top_n: int = 10) -> List[Tuple[str, float]]:
    """Get top N terms for a given context"""
    vec = matrix[context_idx].toarray().flatten()
    
    # Get indices of non-zero elements
    nonzero_indices = np.nonzero(vec)[0]
    
    if len(nonzero_indices) == 0:
        return []
    
    # Get weights for non-zero elements
    weights = vec[nonzero_indices]
    
    # Sort by weight
    sorted_indices = nonzero_indices[np.argsort(weights)[::-1]]
    
    top_terms = []
    for idx in sorted_indices[:top_n]:
        if idx < len(phrases):
            top_terms.append((phrases[idx], vec[idx]))
    
    return top_terms

def find_similar_contexts(matrix: csr_matrix, context_idx: int, top_k: int = 5) -> List[Tuple[int, float]]:
    """Find top-k most similar contexts to the given context"""
    target_vec = matrix[context_idx]
    
    # Normalize target vector
    target_norm = sparse_norm(target_vec)
    if target_norm == 0:
        return []
    
    # Compute similarities with all contexts
    similarities = []
    for i in range(matrix.shape[0]):
        if i == context_idx:
            continue
        sim = get_context_similarity_efficient(matrix, context_idx, i)
        similarities.append((i, sim))
    
    # Sort by similarity
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    return similarities[:top_k]

def interactive_mode(matrix: csr_matrix, metadata: Dict[str, Any], phrases: List[str], corpus: List[str]):
    """Interactive mode for exploring context similarities"""
    num_contexts = metadata['num_contexts']
    
    print("\n" + "="*80)
    print("Context Similarity Explorer")
    print("="*80)
    print(f"Total contexts: {num_contexts}")
    print(f"Total terms: {len(phrases)}")
    print("\nCommands:")
    print("  compare <ctx1> <ctx2>  - Compare two contexts")
    print("  show <ctx>             - Show details for a context")
    print("  similar <ctx> [k]      - Find k most similar contexts (default: 5)")
    print("  random                 - Compare two random contexts")
    print("  quit                   - Exit")
    print("="*80 + "\n")
    
    while True:
        try:
            command = input("\n> ").strip().lower()
            
            if not command:
                continue
            
            if command == "quit" or command == "exit":
                print("Goodbye!")
                break
            
            parts = command.split()
            cmd = parts[0]
            
            if cmd == "compare" and len(parts) >= 3:
                try:
                    ctx1 = int(parts[1])
                    ctx2 = int(parts[2])
                    
                    if ctx1 < 0 or ctx1 >= num_contexts or ctx2 < 0 or ctx2 >= num_contexts:
                        print(f"Error: Context indices must be between 0 and {num_contexts-1}")
                        continue
                    
                    compare_contexts(matrix, phrases, corpus, ctx1, ctx2)
                    
                except ValueError:
                    print("Error: Context indices must be integers")
            
            elif cmd == "show" and len(parts) >= 2:
                try:
                    ctx = int(parts[1])
                    
                    if ctx < 0 or ctx >= num_contexts:
                        print(f"Error: Context index must be between 0 and {num_contexts-1}")
                        continue
                    
                    show_context_details(matrix, phrases, corpus, ctx)
                    
                except ValueError:
                    print("Error: Context index must be an integer")
            
            elif cmd == "similar" and len(parts) >= 2:
                try:
                    ctx = int(parts[1])
                    k = int(parts[2]) if len(parts) >= 3 else 5
                    
                    if ctx < 0 or ctx >= num_contexts:
                        print(f"Error: Context index must be between 0 and {num_contexts-1}")
                        continue
                    
                    show_similar_contexts(matrix, phrases, corpus, ctx, k)
                    
                except ValueError:
                    print("Error: Invalid input")
            
            elif cmd == "random":
                ctx1 = np.random.randint(0, num_contexts)
                ctx2 = np.random.randint(0, num_contexts)
                while ctx2 == ctx1:
                    ctx2 = np.random.randint(0, num_contexts)
                
                print(f"\nRandomly selected: Context {ctx1} and Context {ctx2}")
                compare_contexts(matrix, phrases, corpus, ctx1, ctx2)
            
            else:
                print("Unknown command. Type 'quit' to exit.")
        
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")

def compare_contexts(matrix: csr_matrix, phrases: List[str], corpus: List[str], ctx1: int, ctx2: int):
    """Compare two contexts in detail"""
    print("\n" + "-"*80)
    print(f"Comparing Context {ctx1} and Context {ctx2}")
    print("-"*80)
    
    # Show context text if available
    if corpus:
        if ctx1 < len(corpus):
            print(f"\nContext {ctx1}: {corpus[ctx1][:200]}{'...' if len(corpus[ctx1]) > 200 else ''}")
        if ctx2 < len(corpus):
            print(f"Context {ctx2}: {corpus[ctx2][:200]}{'...' if len(corpus[ctx2]) > 200 else ''}")
    
    # Compute similarity
    similarity = get_context_similarity_efficient(matrix, ctx1, ctx2)
    print(f"\nCosine Similarity: {similarity:.4f}")
    
    # Get shared terms
    shared_terms = get_shared_terms(matrix, phrases, ctx1, ctx2, top_n=15)
    
    if shared_terms:
        print(f"\nTop Shared Terms ({len(shared_terms)} total):")
        print(f"{'Term':<40} {'Weight (ctx1)':<15} {'Weight (ctx2)':<15}")
        print("-"*70)
        for term, w1, w2 in shared_terms[:15]:
            print(f"{term:<40} {w1:<15.4f} {w2:<15.4f}")
    else:
        print("\nNo shared terms found.")
    
    # Show unique top terms for each context
    print(f"\nTop Terms in Context {ctx1}:")
    top_terms_1 = get_top_terms(matrix, phrases, ctx1, top_n=10)
    for term, weight in top_terms_1:
        print(f"  {term:<40} {weight:.4f}")
    
    print(f"\nTop Terms in Context {ctx2}:")
    top_terms_2 = get_top_terms(matrix, phrases, ctx2, top_n=10)
    for term, weight in top_terms_2:
        print(f"  {term:<40} {weight:.4f}")

def show_context_details(matrix: csr_matrix, phrases: List[str], corpus: List[str], ctx: int):
    """Show detailed information about a single context"""
    print("\n" + "-"*80)
    print(f"Context {ctx} Details")
    print("-"*80)
    
    # Show context text if available
    if corpus and ctx < len(corpus):
        print(f"\nText: {corpus[ctx]}")
    
    # Get top terms
    top_terms = get_top_terms(matrix, phrases, ctx, top_n=20)
    
    print(f"\nTop Terms ({len(top_terms)}):")
    print(f"{'Term':<40} {'Weight':<10}")
    print("-"*50)
    for term, weight in top_terms:
        print(f"{term:<40} {weight:.4f}")
    
    # Get vector statistics
    vec = matrix[ctx].toarray().flatten()
    nonzero_count = np.count_nonzero(vec)
    total_weight = np.sum(vec)
    
    print(f"\nStatistics:")
    print(f"  Non-zero terms: {nonzero_count}")
    print(f"  Total weight: {total_weight:.4f}")
    print(f"  Average weight: {total_weight/nonzero_count if nonzero_count > 0 else 0:.4f}")

def show_similar_contexts(matrix: csr_matrix, phrases: List[str], corpus: List[str], ctx: int, k: int):
    """Show k most similar contexts"""
    print("\n" + "-"*80)
    print(f"Top {k} Most Similar Contexts to Context {ctx}")
    print("-"*80)
    
    # Show source context
    if corpus and ctx < len(corpus):
        print(f"\nSource Context: {corpus[ctx][:200]}{'...' if len(corpus[ctx]) > 200 else ''}")
    
    # Find similar contexts
    similar = find_similar_contexts(matrix, ctx, top_k=k)
    
    if not similar:
        print("\nNo similar contexts found.")
        return
    
    print(f"\n{'Rank':<6} {'Context':<10} {'Similarity':<12} {'Preview'}")
    print("-"*80)
    
    for rank, (similar_ctx, sim) in enumerate(similar, 1):
        preview = ""
        if corpus and similar_ctx < len(corpus):
            preview = corpus[similar_ctx][:60] + "..." if len(corpus[similar_ctx]) > 60 else corpus[similar_ctx]
        
        print(f"{rank:<6} {similar_ctx:<10} {sim:<12.4f} {preview}")

def main():
    parser = argparse.ArgumentParser(
        description="Explore context similarities in term-context matrix",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python context_similarity.py --matrix_path outputs/pipeline/term_context_matrix.npz --phrases_path outputs/pipeline/phrases.txt
  
  # Compare specific contexts
  python context_similarity.py --matrix_path outputs/pipeline/term_context_matrix.npz --phrases_path outputs/pipeline/phrases.txt --context1 10 --context2 25
  
  # Find similar contexts
  python context_similarity.py --matrix_path outputs/pipeline/term_context_matrix.npz --phrases_path outputs/pipeline/phrases.txt --context 10 --top_k 10
        """
    )
    parser.add_argument("--matrix_path", required=True, help="Path to term_context_matrix.npz")
    parser.add_argument("--phrases_path", required=True, help="Path to phrases.txt")
    parser.add_argument("--corpus_path", help="Path to corpus.txt (optional, for showing context text)")
    parser.add_argument("--context1", type=int, help="First context index for comparison")
    parser.add_argument("--context2", type=int, help="Second context index for comparison")
    parser.add_argument("--context", type=int, help="Context index to analyze")
    parser.add_argument("--top_k", type=int, default=5, help="Number of similar contexts to find (default: 5)")
    
    args = parser.parse_args()
    
    matrix_path = Path(args.matrix_path)
    phrases_path = Path(args.phrases_path)
    
    # Load data
    print("Loading data...")
    matrix, metadata = load_sparse_matrix(matrix_path)
    phrases = load_phrases(phrases_path)
    
    # Load corpus if provided
    corpus = []
    if args.corpus_path:
        corpus_path = Path(args.corpus_path)
        corpus = load_corpus(corpus_path)
    elif matrix_path.parent.joinpath('corpus.txt').exists():
        corpus = load_corpus(matrix_path.parent / 'corpus.txt')
    
    print(f"Loaded matrix: {matrix.shape[0]} contexts × {matrix.shape[1]} terms")
    print(f"Loaded {len(phrases)} phrases")
    if corpus:
        print(f"Loaded {len(corpus)} corpus contexts")
    
    # Handle different modes
    if args.context1 is not None and args.context2 is not None:
        # Direct comparison mode
        num_contexts = metadata['num_contexts']
        if args.context1 < 0 or args.context1 >= num_contexts or args.context2 < 0 or args.context2 >= num_contexts:
            print(f"Error: Context indices must be between 0 and {num_contexts-1}")
            sys.exit(1)
        
        compare_contexts(matrix, phrases, corpus, args.context1, args.context2)
    
    elif args.context is not None:
        # Single context analysis mode
        num_contexts = metadata['num_contexts']
        if args.context < 0 or args.context >= num_contexts:
            print(f"Error: Context index must be between 0 and {num_contexts-1}")
            sys.exit(1)
        
        show_context_details(matrix, phrases, corpus, args.context)
        print("\n")
        show_similar_contexts(matrix, phrases, corpus, args.context, args.top_k)
    
    else:
        # Interactive mode
        interactive_mode(matrix, metadata, phrases, corpus)

if __name__ == "__main__":
    main()
