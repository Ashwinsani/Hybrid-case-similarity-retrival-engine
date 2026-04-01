# user_input_processor.py (FIXED VERSION)
import pandas as pd
import json
import os
import re
from datetime import datetime
from preprocessing import SimilarityOptimizedPreprocessor

# user_input_processor.py - FIXED VERSION
class UserInputProcessor:
    def __init__(self):
        self.required_fields = [
            'Domain', 'fundingSource', 'Expected benefits', 
            'Idea Description', 'potential Challenges', 'Idea Name'  # FIXED: 'potential' not 'potentail'
        ]
        self.preprocessor = SimilarityOptimizedPreprocessor()
    
    def validate_form_data(self, form_data):
        """Validate that all required fields are present"""
        missing_fields = []
        for field in self.required_fields:
            if field not in form_data or not form_data[field]:
                missing_fields.append(field)
        
        if missing_fields:
            raise ValueError(f"Missing required fields: {', '.join(missing_fields)}")
        
        return True
    
    def create_similarity_text(self, form_data):
        """
        Create similarity text using the EXACT SAME preprocessing as corpus
        """
        print(f"🔍 DEBUG - Creating similarity text EXACTLY like corpus preprocessing")
        
        # Create a DataFrame-like row to mimic corpus processing
        row_data = {
            'Domain': form_data.get('Domain', ''),
            'Idea Name': form_data.get('Idea Name', ''),
            'Idea Description': form_data.get('Idea Description', ''),
            'potential Challenges': form_data.get('potential Challenges', ''),  # FIXED
            'fundingSource': form_data.get('fundingSource', ''),
            'Expected benefits': form_data.get('Expected benefits', '')
        }
        
        # STEP 1: Preprocess core text fields (EXACTLY like corpus)
        enhanced_description = self.preprocessor.preprocess_description(row_data['Idea Description'])
        enhanced_challenges = self.preprocessor.preprocess_challenges(row_data['potential Challenges'])  # FIXED
        cleaned_name = self.preprocessor.preprocess_description(row_data['Idea Name'])
        
        # STEP 2: Extract categories for similarity dimensions (EXACTLY like corpus)
        # Combine text for category extraction (same as corpus)
        combined_text = f"{row_data['Domain']} {enhanced_description} {enhanced_challenges}"
        categories = self.preprocessor.extract_categories(combined_text)
        
        # STEP 3: Create optimized similarity text (EXACTLY like corpus preprocessing.py)
        parts = []
        
        # Add domain from existing column (same as corpus)
        if row_data['Domain'] and pd.notna(row_data['Domain']) and str(row_data['Domain']).strip():
            domain_clean = re.sub(r'[^\w\s]', '', str(row_data['Domain'])).lower().replace(' ', '_')
            parts.append(f"domain_{domain_clean}")
        
        # Add extracted categories (same as corpus)
        parts.extend(categories['technologies'])
        parts.extend(categories['domains'])
        parts.extend(categories['processes'])
        
        # Add enhanced text content (same as corpus)
        if enhanced_description:
            parts.append(enhanced_description)
        
        if enhanced_challenges:
            parts.append(enhanced_challenges)
        
        similarity_text = " ".join(parts)
        
        print(f"🔍 DEBUG - Similarity text created (EXACTLY like corpus):")
        print(f"   Domain: {row_data['Domain']}")
        print(f"   Technologies: {categories['technologies']}")
        print(f"   Domains: {categories['domains']}")
        print(f"   Processes: {categories['processes']}")
        print(f"   Enhanced description length: {len(enhanced_description)}")
        print(f"   Enhanced challenges length: {len(enhanced_challenges)}")
        print(f"   Final similarity text preview: {similarity_text[:100]}...")
        print(f"   Final similarity text length: {len(similarity_text)}")
        
        return similarity_text
    
    def generate_user_id(self, form_data):
        """Generate unique user ID based on form content and timestamp"""
        import hashlib
        raw = json.dumps(form_data, sort_keys=True) + str(datetime.now().timestamp())
        hashed = hashlib.md5(raw.encode()).hexdigest()
        return f"USER_{hashed[:8]}"
    
    def save_user_input(self, form_data, output_dir="user_inputs"):
        """
        Save user input in a format ready for embedding_generator.py
        """
        # Validate input
        self.validate_form_data(form_data)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate user ID and timestamp
        user_id = self.generate_user_id(form_data)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create similarity text USING EXACT SAME PREPROCESSING AS CORPUS
        similarity_text = self.create_similarity_text(form_data)
        
        # Prepare data for saving
        user_data = {
            'user_id': user_id,
            'timestamp': timestamp,
            'form_data': form_data,
            'similarity_text': similarity_text,
            'processed_at': datetime.now().isoformat()
        }
        
        # Save as JSON file
        filename = f"{user_id}_{timestamp}.json"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(user_data, f, indent=4, ensure_ascii=False)
        
        print(f"✅ User input saved to: {filepath}")
        print(f"📊 Similarity text length: {len(similarity_text)} chars")
        
        return filepath, user_data
    
    def process_multiple_users(self, form_data_list, output_dir="user_inputs"):
        """Process multiple user inputs at once"""
        results = []
        for i, form_data in enumerate(form_data_list):
            print(f"🔄 Processing user input {i+1}/{len(form_data_list)}...")
            try:
                filepath, user_data = self.save_user_input(form_data, output_dir)
                results.append({
                    'filepath': filepath,
                    'user_id': user_data['user_id'],
                    'status': 'success'
                })
            except Exception as e:
                results.append({
                    'filepath': None,
                    'user_id': f"USER_ERROR_{i+1}",
                    'status': f'error: {str(e)}'
                })
        
        return results
    
    def get_latest_user_input(self, input_dir="user_inputs"):
        """Get the most recent user input file"""
        if not os.path.exists(input_dir):
            return None
        
        json_files = [f for f in os.listdir(input_dir) if f.endswith('.json') and f.startswith('USER_')]
        
        if not json_files:
            return None
        
        # Sort by timestamp in filename (newest first)
        json_files.sort(reverse=True)
        latest_file = os.path.join(input_dir, json_files[0])
        
        # Load the file
        with open(latest_file, 'r', encoding='utf-8') as f:
            user_data = json.load(f)
        
        return user_data

# Main function for easy usage
def process_user_input(form_data):
    """
    Main function to process user form input
    Returns filepath and processed data for embedding_generator.py
    """
    processor = UserInputProcessor()
    filepath, user_data = processor.save_user_input(form_data)
    
    return {
        'filepath': filepath,
        'user_data': user_data,
        'status': 'success'
    }

# Example usage and testing
if __name__ == "__main__":
    # Example user input with new column structure
    sample_form_data = {
        'Idea Name': 'AI Tax Fraud Detection System',
        'Domain': 'Tax and Revenue',
        'fundingSource': 'Government Grants, Private Investment',
        'Expected benefits': 'Increased revenue, Reduced fraud, Better compliance',
        'Idea Description': 'An AI-powered system that uses machine learning to detect tax fraud patterns and anomalies in real-time',
        'potentail Challenges': 'Data privacy concerns, Integration with legacy systems, Regulatory compliance'
    }
    
    print("🚀 USER INPUT PROCESSOR")
    print("=" * 50)
    
    # Process single user input
    result = process_user_input(sample_form_data)
    
    print(f"\n📁 Output file: {result['filepath']}")
    print(f"🆔 User ID: {result['user_data']['user_id']}")
    print(f"📝 Similarity text length: {len(result['user_data']['similarity_text'])} characters")
    
    # Show what embedding_generator.py will receive
    print(f"\n📋 Data ready for embedding_generator.py:")
    print(f"  - User ID: {result['user_data']['user_id']}")
    print(f"  - Similarity text: {result['user_data']['similarity_text'][:100]}...")
    print(f"  - All form data available for metadata")