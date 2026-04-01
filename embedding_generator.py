# embedding_generator.py (FIXED VERSION)
import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import traceback

class EmbeddingGenerator:
    def __init__(self):
        # Use the SAME model as your corpus embeddings
        try:
            print("🔄 Loading embedding model...")
            # Use the 384-dimensional model to match your corpus
            self.model = SentenceTransformer('all-MiniLM-L6-v2')  # ← 384 dimensions
            print(f"✅ Model loaded: all-MiniLM-L6-v2")
            print(f"📊 Embedding dimensions: {self.model.get_sentence_embedding_dimension()}")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            raise

    def load_user_input(self, user_input_file):
        """Load user input data from JSON file"""
        try:
            if not os.path.exists(user_input_file):
                raise FileNotFoundError(f"User input file not found: {user_input_file}")
            
            with open(user_input_file, 'r', encoding='utf-8') as f:
                user_data = json.load(f)
            
            # Validate required fields
            required_fields = ['user_id', 'timestamp', 'similarity_text', 'form_data']
            for field in required_fields:
                if field not in user_data:
                    raise ValueError(f"Missing required field in user data: {field}")
            
            print(f"📥 Loaded user: {user_data['user_id']}")
            print(f"📝 Similarity text length: {len(user_data['similarity_text'])} chars")
            
            return user_data
            
        except Exception as e:
            print(f"❌ Error loading user input: {e}")
            raise

    def generate_embedding(self, similarity_text):
        """Generate embedding from similarity text"""
        try:
            if not similarity_text or not similarity_text.strip():
                raise ValueError("Similarity text is empty")
            
            print(f"🔢 Generating embedding for text: '{similarity_text[:50]}...'")
            
            # Generate embedding
            embedding = self.model.encode([similarity_text], convert_to_numpy=True)
            
            print(f"✅ Embedding generated - Shape: {embedding.shape}")
            return embedding[0]  # Return single embedding
            
        except Exception as e:
            print(f"❌ Error generating embedding: {e}")
            raise

    def save_embeddings(self, user_data, embedding, output_dir="user_embeddings"):
        """Save embeddings in the dynamic folder structure"""
        try:
            user_id = user_data['user_id']
            timestamp = user_data['timestamp']
            
            # Create user-specific folder
            folder = os.path.join(output_dir, f"{user_id}_{timestamp}")
            os.makedirs(folder, exist_ok=True)
            
            print(f"📁 Creating embedding folder: {folder}")
            
            # ✅ FIXED: Save embedding as numpy array with correct filename 'embeddings.npy'
            embedding_path = os.path.join(folder, "embeddings.npy")
            np.save(embedding_path, np.array([embedding]))
            print(f"💾 Embedding saved: {embedding_path}")
            
            # Save metadata
            metadata_path = os.path.join(folder, "metadata.json")
            with open(metadata_path, "w", encoding='utf-8') as f:
                json.dump(user_data['form_data'], f, indent=4, ensure_ascii=False)
            print(f"📋 Metadata saved: {metadata_path}")
            
            # Save processing info
            info = {
                "user_id": user_id,
                "created_at": timestamp,
                "model": "all-MiniLM-L6-v2",
                "embedding_dim": int(embedding.shape[0]),
                "similarity_text": user_data['similarity_text'],
                "similarity_text_length": len(user_data['similarity_text']),
                "form_fields": list(user_data['form_data'].keys())
            }
            
            info_path = os.path.join(folder, "info.json")
            with open(info_path, "w", encoding='utf-8') as f:
                json.dump(info, f, indent=4, ensure_ascii=False)
            print(f"📄 Info saved: {info_path}")
            
            print("✅ All embeddings and metadata saved successfully.")
            return folder
            
        except Exception as e:
            print(f"❌ Error saving embeddings: {e}")
            raise

    def process_user_file(self, user_input_file):
        """Complete pipeline: Load user input → Generate embedding → Save"""
        print(f"\n🎯 Starting embedding generation pipeline...")
        print(f"📥 Input file: {user_input_file}")
        
        try:
            # Load user data
            user_data = self.load_user_input(user_input_file)
            
            # Generate embedding
            embedding = self.generate_embedding(user_data['similarity_text'])
            
            # Save embeddings
            folder = self.save_embeddings(user_data, embedding)
            
            return {
                'folder': folder,
                'user_id': user_data['user_id'],
                'embedding': embedding,
                'user_data': user_data,
                'status': 'success'
            }
            
        except Exception as e:
            print(f"❌ Pipeline failed: {e}")
            traceback.print_exc()
            return {
                'folder': None,
                'user_id': 'ERROR',
                'embedding': None,
                'user_data': None,
                'status': f'error: {str(e)}'
            }

# Main function for easy usage
def generate_embeddings_from_user_input(user_input_file):
    """
    Main function to generate embeddings from user input file
    """
    print("🚀 EMBEDDING GENERATOR")
    print("=" * 40)
    
    generator = EmbeddingGenerator()
    result = generator.process_user_file(user_input_file)
    
    if result['status'] == 'success':
        print(f"\n🎉 Embedding generation completed!")
        print(f"📁 Embeddings saved to: {result['folder']}")
        print(f"🆔 User ID: {result['user_id']}")
        print(f"🔢 Embedding dimension: {result['embedding'].shape[0]}")
    else:
        print(f"\n💥 Embedding generation failed!")
    
    return result

# Function to process latest user input automatically
def process_latest_user_input(user_inputs_dir="user_inputs"):
    """Automatically process the most recent user input"""
    print("\n🔍 Looking for latest user input...")
    
    if not os.path.exists(user_inputs_dir):
        raise FileNotFoundError(f"User inputs directory not found: {user_inputs_dir}")
    
    json_files = [f for f in os.listdir(user_inputs_dir) if f.endswith('.json') and f.startswith('USER_')]
    
    if not json_files:
        raise FileNotFoundError(f"No user input files found in {user_inputs_dir}")
    
    # Get the most recent file (sorted by name which includes timestamp)
    json_files.sort(reverse=True)
    latest_file = os.path.join(user_inputs_dir, json_files[0])
    
    print(f"✅ Found latest user input: {latest_file}")
    
    # Generate embeddings
    return generate_embeddings_from_user_input(latest_file)

# Test function
def test_embedding_generator():
    """Test the embedding generator with sample data"""
    print("🧪 TESTING EMBEDDING GENERATOR")
    print("=" * 40)
    
    # First, create a test user input
    from user_input_processor import process_user_input
    
    test_data = {
        'Idea Name': 'Test AI System',
        'Domain': 'Healthcare',
        'fundingSource': 'Test Funding',
        'Expected benefits': 'Test benefits',
        'Idea Description': 'Test description for embedding generation',
        'potentail Challenges': 'Test challenges'
    }
    
    try:
        # Process user input first
        input_result = process_user_input(test_data)
        
        # Then generate embeddings
        embedding_result = generate_embeddings_from_user_input(input_result['filepath'])
        
        return embedding_result
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return None

if __name__ == "__main__":
    # Run test
    test_embedding_generator()