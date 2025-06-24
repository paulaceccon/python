import pandas as pd
import numpy as np
import pickle
from pathlib import Path

def generate_test_data():
    """Generate realistic test data that mimics production scenarios"""
    print("Generating test data...")

    # Generate larger dataset to demonstrate performance issues more clearly
    # Use 500K products for more realistic production scenario
    np.random.seed(42)
    n_products = 500_000  # Increased from 100K to show clearer performance differences
    embedding_dim = 384  # Common embedding dimension

    print(f"Generating {n_products} products with {embedding_dim}-dim embeddings...")
    print("This may take a few minutes for the larger dataset...")

    products_data = {
        'product_id': [f'prod_{i:06d}' for i in range(n_products)],
        'name': [f'Product {i}' for i in range(n_products)],
        'category': np.random.choice(['electronics', 'books', 'clothing', 'home'], n_products),
        'price': np.random.uniform(10, 1000, n_products),
        'embedding': [np.random.normal(0, 1, embedding_dim) for _ in range(n_products)]
    }

    products_df = pd.DataFrame(products_data)

    # Save data
    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True)

    print("Saving products data...")
    products_df.to_parquet('data/products.parquet')

    # Pre-normalize embeddings for cosine similarity
    print("Processing embeddings...")
    embeddings = np.array(products_df['embedding'].tolist())
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    with open('data/embeddings.pkl', 'wb') as f:
        pickle.dump(embeddings, f)

    print(f"Generated {n_products} products with {embedding_dim}-dim embeddings")
    print(f"Data size: {products_df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    print(f"Embeddings size: {embeddings.nbytes / 1024**2:.1f} MB")

if __name__ == "__main__":
    generate_test_data()