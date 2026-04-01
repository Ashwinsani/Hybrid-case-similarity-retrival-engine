#preprocessing.py
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import os
import sys

# Force UTF-8 encoding for standard output to support emojis
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Download NLTK data
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('wordnet_ic', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    print("✅ NLTK data downloaded successfully")
except Exception as e:
    print(f"❌ NLTK data download failed: {e}")

class SimilarityOptimizedPreprocessor:
    def __init__(self):
        try:
            self.stop_words = set(stopwords.words('english'))
            self.stop_words.update(['na', 'none', 'null', 'nil', '●', '•', 'o', '§'])
            # Initialize lemmatizer with proper error handling
            try:
                self.lemmatizer = WordNetLemmatizer()
                # Test the lemmatizer to ensure it works
                test = self.lemmatizer.lemmatize('testing')
            except AttributeError as ae:
                # WordNet corpus corruption - use simple version without lemmatization
                print(f"⚠️ WordNet initialization failed: {ae}")
                self.lemmatizer = None
        except Exception as e:
            print(f"❌ NLTK initialization failed: {e}")
            self.stop_words = set()
            self.lemmatizer = None
        
        # Protected phrases for similarity preservation
        self.protected_phrases = {
            'revenue forecasting': 'revenue_forecasting',
            'fraud detection': 'fraud_detection',
            'machine learning': 'machine_learning',
            'natural language processing': 'natural_language_processing', 
            'computer vision': 'computer_vision',
            'document management': 'document_management',
            'risk assessment': 'risk_assessment',
            'anomaly detection': 'anomaly_detection',
            'government orders': 'government_orders',
            'tax fraud': 'tax_fraud',
            'goods movement': 'goods_movement',
            'route deviation': 'route_deviation',
            'ghost shipments': 'ghost_shipments',
            'fake invoices': 'fake_invoices',
            'itc fraud': 'itc_fraud',
            'circular trading': 'circular_trading',
            'predictive analytics': 'predictive_analytics',
            'facial recognition': 'facial_recognition',
            'vehicle identification': 'vehicle_identification',
            'complaint management': 'complaint_management',
            'data validation': 'data_validation',
            'beneficiary selection': 'beneficiary_selection',
            'behaviour analysis': 'behaviour_analysis',
            'socio-economic trend': 'socio_economic_trend',
            'environmental data': 'environmental_data',
            'legal opinion': 'legal_opinion',
            'decision support': 'decision_support',
            'local development': 'local_development',
            'landslide risk': 'landslide_risk',
            'damage assessment': 'damage_assessment',
            'health screening': 'health_screening',
            'compliance monitoring': 'compliance_monitoring',
            'financial health': 'financial_health',
            'document digitization': 'document_digitization',
            'gst': 'goods_services_tax',
            'itc': 'input_tax_credit',
            'e-way bill': 'eway_bill',
            'fir': 'first_information_report',
            'cctv': 'closed_circuit_television',
            'iot': 'internet_of_things',
            'nlp': 'natural_language_processing',
            'ocr': 'optical_character_recognition',
            'gps': 'global_positioning_system',
            'ai': 'artificial_intelligence',
            'ml': 'machine_learning'
        }
        
        # Category keywords for similarity dimensions
        self.technology_keywords = {
            'ai_ml': ['ai', 'machine learning', 'ml', 'artificial intelligence', 'neural network'],
            'nlp': ['nlp', 'natural language processing', 'text analysis', 'language model'],
            'computer_vision': ['computer vision', 'image recognition', 'video analysis', 'facial recognition'],
            'iot': ['iot', 'internet of things', 'sensor', 'smart device'],
            'blockchain': ['blockchain', 'distributed ledger', 'smart contract'],
            'big_data': ['big data', 'data analytics', 'data mining'],
            'robotics': ['robotics', 'automation', 'robot']
        }
        
        self.domain_keywords = {
            'tax_revenue': ['tax', 'revenue', 'gst', 'income tax', 'taxation'],
            'healthcare': ['health', 'medical', 'hospital', 'patient', 'disease'],
            'law_enforcement': ['police', 'crime', 'investigation', 'fir', 'law enforcement'],
            'agriculture': ['agriculture', 'crop', 'farmer', 'livestock', 'dairy'],
            'education': ['education', 'student', 'teacher', 'school', 'learning'],
            'environment': ['environment', 'pollution', 'conservation', 'climate'],
            'governance': ['governance', 'administration', 'public service', 'government']
        }
        
        self.process_keywords = {
            'detection': ['detect', 'identification', 'recognition', 'find'],
            'prediction': ['predict', 'forecast', 'prediction', 'future'],
            'monitoring': ['monitor', 'track', 'surveillance', 'observe'],
            'optimization': ['optimize', 'efficient', 'improve', 'enhance'],
            'automation': ['automate', 'automatic', 'auto', 'streamline'],
            'analysis': ['analyze', 'analysis', 'examine', 'study']
        }

    def protect_key_phrases(self, text):
        """Protect important phrases before general cleaning"""
        if not isinstance(text, str) or pd.isna(text):
            return ""
        
        text = text.lower()
        
        # Replace important phrases with protected versions
        for phrase, protected in self.protected_phrases.items():
            text = text.replace(phrase, protected)
        
        return text

    def clean_text(self, text):
        """Clean text while preserving protected phrases"""
        if not isinstance(text, str) or pd.isna(text):
            return ""
        
        # Remove bullet points and special markers
        text = re.sub(r'[●•\-–—]\s*', ' ', text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove extra whitespace and newlines
        text = re.sub(r'\s+', ' ', text)
        
        # Remove text within parentheses that are codes/abbreviations
        text = re.sub(r'\([^)]*[A-Z]{2,}[^)]*\)', '', text)
        
        # Keep only alphanumeric and basic punctuation (including underscores)
        text = re.sub(r'[^a-zA-Z0-9\s_\.!?,;:]', ' ', text)
        
        # Remove standalone numbers
        text = re.sub(r'\b\d+\b', ' ', text)
        
        # Final whitespace cleanup
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def restore_protected_phrases(self, text):
        """Restore protected phrases after cleaning"""
        for phrase, protected in self.protected_phrases.items():
            text = text.replace(protected, phrase.replace(' ', '_'))
        return text

    def extract_categories(self, text):
        """Extract technology, domain, and process categories"""
        if not isinstance(text, str):
            return {'technologies': [], 'domains': [], 'processes': []}
        
        text_lower = text.lower()
        categories = {'technologies': [], 'domains': [], 'processes': []}
        
        # Extract technologies
        for tech_type, keywords in self.technology_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                categories['technologies'].append(tech_type)
        
        # Extract domains
        for domain_type, keywords in self.domain_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                categories['domains'].append(domain_type)
        
        # Extract processes
        for process_type, keywords in self.process_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                categories['processes'].append(process_type)
        
        return categories

    def preprocess_description(self, text):
        """Enhanced preprocessing for idea descriptions"""
        if not isinstance(text, str) or pd.isna(text):
            return ""
        
        # Step 1: Protect key phrases
        text = self.protect_key_phrases(text)
        
        # Step 2: Clean text
        text = self.clean_text(text)
        
        # Step 3: Restore protected phrases
        text = self.restore_protected_phrases(text)
        
        # Step 4: Tokenize and remove stopwords
        tokens = word_tokenize(text)
        tokens = [token for token in tokens if token not in self.stop_words and len(token) > 2]
        
        # Step 5: Lemmatize (with fallback if lemmatizer failed)
        if tokens and self.lemmatizer:
            try:
                tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
            except AttributeError:
                # Fallback: skip lemmatization if WordNet fails
                pass
        
        return ' '.join(tokens)

    def preprocess_challenges(self, text):
        """Preprocess challenges text"""
        if not isinstance(text, str) or pd.isna(text):
            return ""
        
        # Protect and clean
        text = self.protect_key_phrases(text)
        text = self.clean_text(text)
        text = self.restore_protected_phrases(text)
        
        # Simple tokenization for challenges
        tokens = text.split()
        tokens = [token for token in tokens if token not in self.stop_words and len(token) > 2]
        
        return ' '.join(tokens)

def preprocess_ideas_for_similarity(input_file_path, output_folder="processed_data"):
    """
    Main preprocessing function that saves the processed file
    
    Parameters:
    - input_file_path: path to your Excel file
    - output_folder: folder to save processed data (default: "processed_data")
    """
    
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"📁 Created output folder: {output_folder}")
    
    # Load data
    try:
        df = pd.read_excel(input_file_path)
        print(f"✅ Dataset loaded: {len(df)} ideas from {input_file_path}")
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return None
    
    # Initialize preprocessor
    preprocessor = SimilarityOptimizedPreprocessor()
    
    print("🔄 Starting preprocessing steps...")
    
    # STEP 1: Preprocess core text fields
    print("📝 Step 1: Preprocessing text fields...")
    df['enhanced_description'] = df['Idea Description'].apply(preprocessor.preprocess_description)
    df['enhanced_challenges'] = df['potentail Challenges'].apply(preprocessor.preprocess_challenges)
    df['cleaned_name'] = df['Idea Name'].apply(lambda x: preprocessor.preprocess_description(x) if pd.notna(x) else "")
    
    # STEP 2: Extract categories for similarity dimensions
    print("🏷️ Step 2: Extracting categories...")
    category_data = df['enhanced_description'].apply(preprocessor.extract_categories)
    df['technologies'] = category_data.apply(lambda x: x['technologies'])
    df['domains'] = category_data.apply(lambda x: x['domains'])
    df['processes'] = category_data.apply(lambda x: x['processes'])
    
    # STEP 3: Create optimized similarity text
    print("🔗 Step 3: Creating similarity text...")
    def create_similarity_text(row):
        parts = []
        
        # Add domain from existing column
        if pd.notna(row.get('Domain')) and row['Domain']:
            domain_clean = re.sub(r'[^\w\s]', '', str(row['Domain'])).lower().replace(' ', '_')
            parts.append(f"domain_{domain_clean}")
        
        # Add extracted categories
        parts.extend(row['technologies'])
        parts.extend(row['domains'])
        parts.extend(row['processes'])
        
        # Add enhanced text content
        if pd.notna(row.get('enhanced_description')) and row['enhanced_description']:
            parts.append(row['enhanced_description'])
        
        if pd.notna(row.get('enhanced_challenges')) and row['enhanced_challenges']:
            parts.append(row['enhanced_challenges'])
        
        return " ".join(parts)
    
    df['similarity_text'] = df.apply(create_similarity_text, axis=1)
    
    # STEP 4: Create quick match text for fast filtering
    df['quick_match_text'] = df.apply(
        lambda row: f"{row.get('Domain', '')} {' '.join(row['technologies'])} {' '.join(row['processes'])}", 
        axis=1
    )
    
    # STEP 5: Validate and save
    print("💾 Step 4: Saving processed file...")
    
    # Generate output filename
    input_filename = os.path.basename(input_file_path)
    output_filename = f"similarity_optimized_{input_filename}"
    output_path = os.path.join(output_folder, output_filename)
    
    # Save the processed data
    df.to_excel(output_path, index=False)
    
    # Print summary
    print("\n✨ PREPROCESSING COMPLETED SUCCESSFULLY!")
    print("=" * 50)
    print(f"📊 Input file: {input_file_path}")
    print(f"💾 Output file: {output_path}")
    print(f"📁 Total ideas processed: {len(df)}")
    
    # Show statistics
    non_empty_similarity = df['similarity_text'].str.strip().ne('').sum()
    total_tech = df['technologies'].apply(len).sum()
    total_domains = df['domains'].apply(len).sum()
    total_processes = df['processes'].apply(len).sum()
    
    print(f"📈 Processing Statistics:")
    print(f"   - Ideas with similarity text: {non_empty_similarity}/{len(df)}")
    print(f"   - Technologies extracted: {total_tech}")
    print(f"   - Domains extracted: {total_domains}")
    print(f"   - Processes extracted: {total_processes}")
    
    # Show sample of processed data
    print(f"\n🔍 Sample of processed data:")
    print("-" * 60)
    for i in range(min(2, len(df))):
        row = df.iloc[i]
        print(f"📌 {row['Idea Id']}: {row['Idea Name']}")
        print(f"   Tech: {row['technologies']} | Domain: {row['domains']} | Processes: {row['processes']}")
        print(f"   Similarity text preview: {row['similarity_text'][:100]}...")
        print()
    
    return df, output_path

# =============================================================================
# MAIN EXECUTION - 
# =============================================================================
if __name__ == "__main__":
    # Configure your input file path here
    INPUT_FILE = "idea.xlsx"  # Change this to your actual file path
    
    print("🚀 STARTING IDEA PREPROCESSING FOR SIMILARITY DETECTION")
    print("=" * 60)
    
    # Run the preprocessing
    processed_df, output_path = preprocess_ideas_for_similarity(INPUT_FILE)
    
    if processed_df is not None:
        print(f"\n✅ Preprocessing completed! File saved to: {output_path}")
        print(f"🎯 Use the 'similarity_text' column for best similarity results")
    else:
        print("\n❌ Preprocessing failed. Please check the error messages above.")