# main.py
from user_input_processor import process_user_input
from embedding_generator import process_latest_user_input

def main():
    print("🚀 COMPLETE USER INPUT PROCESSING PIPELINE")
    print("=" * 60)
    
    # Step 1: User submits form data
    user_form_data = {
        'idea_name': 'AI Tax Fraud Detection System',
        'domain': 'tax_revenue',
        'technologies': 'ai_ml, nlp',
        'processes': 'detection, monitoring',
        'enhanced_description': 'ai machine learning detect tax fraud pattern anomaly detection',
        'enhanced_challenges': 'data privacy compliance integration legacy systems'
    }
    
    # Step 2: Process user input (saves to user_inputs/ folder)
    print("📝 Step 1: Processing user input...")
    input_result = process_user_input(user_form_data)
    print(f"✅ User input saved to: {input_result['filepath']}")
    
    # Step 3: Generate embeddings from the processed input
    print("\n🔢 Step 2: Generating embeddings...")
    embedding_result = process_latest_user_input()
    print(f"✅ Embeddings saved to: {embedding_result['folder']}")
    
    # Final results
    print(f"\n🎉 PIPELINE COMPLETED!")
    print(f"📊 User ID: {embedding_result['user_id']}")
    print(f"🔢 Embedding dimension: {embedding_result['embedding'].shape[0]}")
    print(f"📝 Similarity text preview: {embedding_result['user_data']['similarity_text'][:100]}...")

if __name__ == "__main__":
    main()