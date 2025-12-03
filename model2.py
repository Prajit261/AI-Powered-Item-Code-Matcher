"""
Item Code Pattern Matcher - CSV Dataset Version
Uses Sentence Transformers for embedding-based semantic search 
to match invoice attributes to item codes from a CSV file.

"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
from pathlib import Path

# =============================================================================
# EMBEDDING MODEL SETUP
# =============================================================================

print("Loading embedding model...")
MODEL = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded!\n")


def get_embedding(text: str) -> np.ndarray:
    """Convert text to embedding vector using Sentence Transformers."""
    return MODEL.encode(text, convert_to_numpy=True)


def get_embeddings_batch(texts: list[str]) -> np.ndarray:
    """Batch encode multiple texts (more efficient for large datasets)."""
    return MODEL.encode(texts, convert_to_numpy=True, show_progress_bar=True)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class ItemRecord:
    code: str
    name: str
    attributes: dict
    category: str
    
    def to_text(self) -> str:
        """Convert item to searchable text representation."""
        # Create a rich text representation combining all attributes
        attr_parts = []
        for k, v in self.attributes.items():
            if pd.notna(v) and str(v).strip():  # Skip empty values
                attr_parts.append(f"{k}: {v}")
        
        attr_str = " ".join(attr_parts)
        return f"{self.name} {self.category} {attr_str}"


@dataclass
class MatchResult:
    item_code: str
    item_name: str
    similarity: float
    confidence: str
    item_attributes: dict

# =============================================================================
# CSV LOADER
# =============================================================================

def load_items_from_csv(csv_path: str = "styles.csv") -> list[ItemRecord]:
    """
    Load items from CSV file.
    
    Expected CSV format:
    id, gender, masterCategory, subCategory, articleType, 
    baseColour, season, year, usage, productDisplayName
    """
    if not Path(csv_path).exists():
        raise FileNotFoundError(
            f"CSV file not found: {csv_path}\n"
            f"Please place 'styles.csv' in the same directory as this script."
        )
    
    print(f"Loading data from {csv_path}...")
    
    # Read CSV with error handling for malformed rows
    try:
        # Try with default settings first
        df = pd.read_csv(csv_path)
    except pd.errors.ParserError:
        print("⚠️  CSV has formatting issues, trying with error handling...")
        # Skip bad lines and warn user
        df = pd.read_csv(
            csv_path,
            on_bad_lines='skip',  # Skip problematic rows
            engine='python',       # Use more forgiving parser
            encoding='utf-8',
            quoting=1              # Handle quotes properly
        )
        print(f"⚠️  Skipped some malformed rows. Loaded {len(df)} valid rows.")
    
    # Validate required columns
    required_cols = ['id', 'productDisplayName']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Convert to ItemRecord objects
    items = []
    for _, row in df.iterrows():
        # Use ID as the item code
        item_code = str(row['id'])
        
        # Product name
        item_name = row.get('productDisplayName', f"Item {item_code}")
        
        # Build attributes dictionary (exclude id and productDisplayName)
        attributes = {}
        for col in df.columns:
            if col not in ['id', 'productDisplayName']:
                val = row[col]
                # Only include non-null values
                if pd.notna(val):
                    attributes[col] = str(val)
        
        # Use masterCategory as the main category, fallback to 'Product'
        category = attributes.get('masterCategory', 'Product')
        
        items.append(ItemRecord(
            code=item_code,
            name=item_name,
            attributes=attributes,
            category=category
        ))
    
    print(f"Loaded {len(items)} items from CSV")
    return items

# =============================================================================
# ITEM DATABASE
# =============================================================================

class ItemDatabase:
    def __init__(self):
        self.items: list[ItemRecord] = []
        self.embeddings: np.ndarray = None
    
    def bulk_load(self, items: list[ItemRecord]):
        """Efficiently load multiple items using batch encoding."""
        self.items = items
        print("\nGenerating embeddings for all items...")
        texts = [item.to_text() for item in items]
        self.embeddings = get_embeddings_batch(texts)
        print(f"Database ready with {len(items)} items\n")
    
    def load_from_csv(self, csv_path: str = "styles.csv"):
        """Load items directly from CSV file."""
        items = load_items_from_csv(csv_path)
        self.bulk_load(items)
    
    def search(self, query: str, top_k: int = 3, min_confidence: str = None) -> list[MatchResult]:
        """
        Find best matching items for a query.
        
        Args:
            query: Search query text
            top_k: Number of results to return
            min_confidence: Filter results by confidence ('high', 'medium', 'low')
        """
        query_embedding = get_embedding(query)
        
        # Calculate all similarities at once (vectorized)
        similarities = np.dot(self.embeddings, query_embedding) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding) + 1e-8
        )
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k * 2]  # Get extra for filtering
        
        results = []
        for idx in top_indices:
            sim = similarities[idx]
            item = self.items[idx]
            
            # Determine confidence based on similarity threshold
            if sim > 0.65:
                conf = "high"
            elif sim > 0.50:
                conf = "medium"
            else:
                conf = "low"
            
            # Filter by minimum confidence if specified
            if min_confidence:
                conf_levels = {"low": 0, "medium": 1, "high": 2}
                if conf_levels[conf] < conf_levels[min_confidence]:
                    continue
            
            results.append(MatchResult(
                item_code=item.code,
                item_name=item.name,
                similarity=round(float(sim), 3),
                confidence=conf,
                item_attributes=item.attributes
            ))
            
            if len(results) >= top_k:
                break
        
        return results
    
    def search_batch(self, queries: list[str], top_k: int = 1) -> list[list[MatchResult]]:
        """Search multiple queries efficiently."""
        print(f"\nProcessing {len(queries)} queries in batch...")
        query_embeddings = get_embeddings_batch(queries)
        
        all_results = []
        for q_emb in query_embeddings:
            similarities = np.dot(self.embeddings, q_emb) / (
                np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(q_emb) + 1e-8
            )
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            results = []
            for idx in top_indices:
                item = self.items[idx]
                sim = similarities[idx]
                
                if sim > 0.65:
                    conf = "high"
                elif sim > 0.50:
                    conf = "medium"
                else:
                    conf = "low"
                
                results.append(MatchResult(
                    item_code=item.code,
                    item_name=item.name,
                    similarity=round(float(sim), 3),
                    confidence=conf,
                    item_attributes=item.attributes
                ))
            all_results.append(results)
        
        return all_results
    
    def get_stats(self):
        """Print database statistics."""
        if not self.items:
            print("Database is empty")
            return
        
        print(f"\n{'='*60}")
        print("DATABASE STATISTICS")
        print(f"{'='*60}")
        print(f"Total items: {len(self.items)}")
        
        # Category distribution
        categories = {}
        for item in self.items:
            cat = item.category
            categories[cat] = categories.get(cat, 0) + 1
        
        print(f"\nCategories:")
        for cat, count in sorted(categories.items(), key=lambda x: -x[1])[:10]:
            print(f"  {cat}: {count}")
        
        # Sample items
        print(f"\nSample items:")
        for item in self.items[:3]:
            print(f"  [{item.code}] {item.name}")
        print()

# =============================================================================
# MAIN DEMO
# =============================================================================

def main():
    # Initialize database and load CSV
    db = ItemDatabase()
    
    try:
        db.load_from_csv("styles.csv")
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease ensure 'styles.csv' is in the same directory as this script.")
        return
    except Exception as e:
        print(f"\n❌ Error loading CSV: {e}")
        return
    
    # Show database stats
    db.get_stats()
    
    print("="*60)
    print("ITEM CODE PATTERN MATCHER - CSV DATASET")
    print("="*60)
    
    # Test queries (simulating invoice descriptions)
    test_queries = [
        "navy blue shirt casual men",
        "blue jeans men casual",
        "casual blue sweatshirt",
    ]
    
    for query in test_queries:
        print(f"\n📋 Invoice Text: \"{query}\"")
        print("-" * 50)
        
        results = db.search(query, top_k=3)
        
        if not results:
            print("  ❌ No matches found")
            continue
        
        for i, result in enumerate(results, 1):
            conf_emoji = {"high": "✅", "medium": "⚠️", "low": "❓"}[result.confidence]
            print(f"  {i}. [ID: {result.item_code}] {result.item_name}")
            print(f"     Similarity: {result.similarity:.1%} {conf_emoji} ({result.confidence})")
            
            # Show key attributes
            key_attrs = ['gender', 'articleType', 'baseColour', 'usage']
            attr_display = []
            for key in key_attrs:
                if key in result.item_attributes:
                    attr_display.append(f"{key}: {result.item_attributes[key]}")
            if attr_display:
                print(f"     {', '.join(attr_display)}")
        
        # Show recommendation
        best = results[0]
        if best.confidence == "high":
            print(f"\n  ➡️  AUTO-MATCH: ID {best.item_code}")
        elif best.confidence == "medium":
            print(f"\n  ➡️  SUGGESTED: ID {best.item_code} (review recommended)")
        else:
            print(f"\n  ➡️  MANUAL REVIEW NEEDED")

    # Batch processing example
    print("\n" + "="*60)
    print("BATCH PROCESSING EXAMPLE")
    print("="*60)
    
    invoice_items = [
        "blue shirt men casual",
        "black formal pants",
        "white sneakers casual",
    ]
    
    batch_results = db.search_batch(invoice_items, top_k=1)
    
    print("\nResults:")
    for query, results in zip(invoice_items, batch_results):
        if results:
            best = results[0]
            status = "✅" if best.confidence == "high" else "⚠️"
            print(f"  {status} \"{query}\" → ID {best.item_code} ({best.similarity:.1%})")
            print(f"      {best.item_name}")
        else:
            print(f"  ❌ \"{query}\" → No match found")


if __name__ == "__main__":
    main()