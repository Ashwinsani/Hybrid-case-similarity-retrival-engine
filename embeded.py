#embeded
import numpy as np
import pandas as pd
import json
import os
from sentence_transformers import SentenceTransformer

class EmbeddingManager:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.metadata = None
        self.id_to_index = None
    
    def load_preprocessed_data(self, data_path):
        """Load preprocessed data from your existing folder"""
        print(f"📂 Loading preprocessed data from {data_path}...")
        
        df = pd.read_excel(data_path)  # Use the parameter
        print(f"✅ Loaded {len(df)} preprocessed cases")
        return df  # RETURN THE DATAFRAME
    
    def generate_embeddings(self, preprocessed_texts, case_ids, metadata_df):
        """Generate embeddings from preprocessed text"""
        print(f"🔄 Generating embeddings for {len(preprocessed_texts)} cases...")
        
        # Generate embeddings
        embeddings = self.model.encode(
            preprocessed_texts,
            normalize_embeddings=True,
            show_progress_bar=True,
            batch_size=32
        )
        
        self.embeddings = embeddings
        self.metadata = metadata_df
        self.id_to_index = {case_id: idx for idx, case_id in enumerate(case_ids)}
        
        print(f"✅ Generated {len(embeddings)} embeddings (dim: {embeddings.shape[1]})")
        return embeddings
    
    def save_embeddings(self, save_dir='embeddings'):
        """Save embeddings with metadata"""
        os.makedirs(save_dir, exist_ok=True)
        
        # Save embeddings
        np.save(f'{save_dir}/embeddings.npy', self.embeddings)
        
        # Save ID mapping
        with open(f'{save_dir}/id_mapping.json', 'w') as f:
            json.dump(self.id_to_index, f, indent=2)
        
        # Save metadata
        self.metadata.to_csv(f'{save_dir}/metadata.csv', index=False)
        
        # Save info
        info = {
            'total_cases': len(self.embeddings),
            'embedding_dim': self.embeddings.shape[1],
            'model': 'all-MiniLM-L6-v2',
            'saved_at': pd.Timestamp.now().isoformat()
        }
        with open(f'{save_dir}/info.json', 'w') as f:
            json.dump(info, f, indent=2)
        
        print(f"💾 Embeddings saved to '{save_dir}/'")
        print(f"📊 Files: embeddings.npy, id_mapping.json, metadata.csv, info.json")
    
    def load_embeddings(self, load_dir='embeddings'):
        """Load saved embeddings"""
        self.embeddings = np.load(f'{load_dir}/embeddings.npy')
        
        with open(f'{load_dir}/id_mapping.json', 'r') as f:
            self.id_to_index = json.load(f)
        
        self.metadata = pd.read_csv(f'{load_dir}/metadata.csv')
        
        with open(f'{load_dir}/info.json', 'r') as f:
            info = json.load(f)
        
        print(f"📥 Loaded {len(self.embeddings)} embeddings from '{load_dir}/'")
        print(f"📈 Dimension: {self.embeddings.shape[1]}, Model: {info['model']}")
        
        return self.embeddings, self.metadata, self.id_to_index
    
    def get_embedding_by_id(self, case_id):
        """Get embedding for specific case ID"""
        if self.id_to_index and case_id in self.id_to_index:
            return self.embeddings[self.id_to_index[case_id]]
        return None

# ============================================================================
# USAGE WITH YOUR PREPROCESSED DATA
# ============================================================================

if __name__ == "__main__":
    # Initialize manager
    manager = EmbeddingManager()
    
    # Load your preprocessed data from another folder
    preprocessed_df = manager.load_preprocessed_data('processed_data/similarity_optimized_idea.xlsx')  # Fixed path
    
    # Extract the necessary columns (adjust column names as needed)
    preprocessed_texts = preprocessed_df['similarity_text'].tolist()  # Use actual column name from your Excel
    case_ids = preprocessed_df['Idea Id'].tolist()
    metadata_df = preprocessed_df[['Idea Id', 'Idea Name', 'Domain', 'technologies']]
    
    # Generate embeddings
    embeddings = manager.generate_embeddings(
        preprocessed_texts=preprocessed_texts,
        case_ids=case_ids,
        metadata_df=metadata_df
    )
    
    # Save everything
    manager.save_embeddings('case_embeddings')
    
    print("🎉 Embedding generation complete!")