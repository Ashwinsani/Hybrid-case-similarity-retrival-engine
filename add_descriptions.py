# add_descriptions.py
import pandas as pd
import os
from datetime import datetime
import ast

def add_case_descriptions():
    print("📝 ENHANCING CASE DATABASE WITH DESCRIPTIONS")
    print("=" * 60)
    
    case_dir = "case_embeddings"
    metadata_path = os.path.join(case_dir, "metadata.csv")
    backup_path = os.path.join(case_dir, f"metadata_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    
    if not os.path.exists(metadata_path):
        print("❌ metadata.csv not found!")
        return False
    
    try:
        # Load current metadata
        df = pd.read_csv(metadata_path)
        print(f"✅ Loaded metadata with {len(df)} cases")
        
        # Create backup with timestamp
        df.to_csv(backup_path, index=False)
        print(f"✅ Backup created: {backup_path}")
        
        # Check if description column exists, if not create it
        if 'Idea Description' not in df.columns:
            df['Idea Description'] = ''
            print("✅ Added 'Idea Description' column")
        else:
            print("✅ 'Idea Description' column already exists - updating descriptions")
        
        # Create enhanced descriptions based on case data
        descriptions_updated = 0
        
        for idx, row in df.iterrows():
            case_name = row['Idea Name']
            technologies = row.get('technologies', '[]')
            domain = row.get('Domain', 'General')
            
            # Generate meaningful descriptions based on case content
            description = generate_description(case_name, technologies, domain)
            df.at[idx, 'Idea Description'] = description
            descriptions_updated += 1
        
        # Save enhanced metadata
        df.to_csv(metadata_path, index=False)
        print(f"✅ Updated descriptions for {descriptions_updated} cases")
        print(f"💾 Enhanced metadata saved: {metadata_path}")
        
        # Show samples
        print(f"\n🔍 SAMPLE DESCRIPTIONS:")
        print("-" * 50)
        sample_cases = df.head(5)
        for idx, row in sample_cases.iterrows():
            print(f"📌 {row['Idea Name']}")
            print(f"   Domain: {row.get('Domain', 'N/A')}")
            print(f"   Description: {row['Idea Description']}")
            print()
        
        return True
        
    except Exception as e:
        print(f"❌ Error enhancing descriptions: {e}")
        return False

def generate_description(case_name, technologies, domain):
    """Generate meaningful description based on case data"""
    
    # Enhanced description templates
    templates = [
        "An AI-powered system designed to {function} through advanced {technologies} implementation in the {domain} sector.",
        "This innovative solution leverages {technologies} to {function}, specifically tailored for {domain} applications.",
        "A comprehensive platform that uses {technologies} for {function}, enhancing efficiency and outcomes in {domain} operations.",
        "An automated system that employs {technologies} to {function}, delivering improved performance in {domain} management.",
        "This advanced solution utilizes {technologies} to {function}, addressing key challenges in the {domain} domain."
    ]
    
    # Extract function from case name
    function = extract_function_from_name(case_name)
    
    # Process technologies safely
    tech_list = parse_technologies(technologies)
    tech_text = format_technologies(tech_list)
    
    # Choose template and fill it
    import random
    template = random.choice(templates)
    
    description = template.format(
        function=function,
        technologies=tech_text,
        domain=domain.lower() if domain and isinstance(domain, str) else "various sectors"
    )
    
    # Ensure proper capitalization and punctuation
    description = description.strip()
    if not description.endswith('.'):
        description += '.'
    
    return description

def parse_technologies(technologies):
    """Parse technologies from various formats safely"""
    if isinstance(technologies, list):
        return technologies
    elif isinstance(technologies, str):
        if technologies.startswith('[') and technologies.endswith(']'):
            try:
                return ast.literal_eval(technologies)
            except:
                # Safe fallback for malformed lists
                tech_str = technologies.strip('[]').replace("'", "").replace('"', '')
                return [tech.strip() for tech in tech_str.split(',') if tech.strip()]
        else:
            return [tech.strip() for tech in technologies.split(',') if tech.strip()]
    else:
        return []

def format_technologies(tech_list):
    """Format technologies list into readable text"""
    if not tech_list:
        return "advanced analytics and machine learning"
    
    # Clean and filter technologies
    tech_list = [str(tech).strip() for tech in tech_list if tech and str(tech).strip()]
    
    if len(tech_list) == 1:
        return tech_list[0]
    elif len(tech_list) == 2:
        return f"{tech_list[0]} and {tech_list[1]}"
    else:
        return f"{', '.join(tech_list[:-1])}, and {tech_list[-1]}"

def extract_function_from_name(case_name):
    """Extract the main function from case name with enhanced granularity"""
    if not isinstance(case_name, str):
        return "enhance operational capabilities and service delivery"
    
    name_lower = case_name.lower()
    
    # Enhanced function mapping with more specific descriptions
    function_mapping = [
        (['fraud', 'detection', 'anomaly'], "detect and prevent fraudulent activities through advanced pattern recognition"),
        (['forecast', 'prediction', 'predictive'], "provide accurate predictions and future trend analysis"),
        (['research', 'analysis', 'analytics'], "conduct comprehensive data analysis and generate actionable insights"),
        (['optimization', 'efficiency', 'automation'], "optimize processes and improve operational efficiency"),
        (['monitoring', 'surveillance', 'tracking'], "monitor activities and track performance metrics in real-time"),
        (['management', 'administration', 'coordination'], "streamline management and coordination tasks"),
        (['registration', 'document', 'filing'], "manage document registration and filing processes efficiently"),
        (['complaint', 'grievance', 'feedback'], "handle complaints and feedback efficiently with automated workflows"),
        (['validation', 'verification'], "validate data accuracy and ensure information integrity"),
        (['assessment', 'evaluation'], "conduct comprehensive risk and performance assessments"),
        (['selection', 'identification'], "facilitate intelligent selection and identification processes"),
        (['planning', 'scheduling'], "optimize planning and scheduling operations"),
        (['reporting', 'dashboard'], "generate comprehensive reports and interactive dashboards"),
        (['compliance', 'regulation'], "ensure regulatory compliance and adherence to standards"),
        (['legal', 'assistant', 'advisor'], "provide legal assistance and advisory services"),
        (['finance', 'revenue', 'tax'], "manage financial operations and revenue optimization"),
        (['agriculture', 'farming', 'crop'], "support agricultural operations and crop management"),
        (['health', 'medical', 'patient'], "enhance healthcare services and patient care")
    ]
    
    for keywords, function in function_mapping:
        if any(keyword in name_lower for keyword in keywords):
            return function
    
    # Default function based on common patterns
    if any(word in name_lower for word in ['ai', 'machine learning', 'ml']):
        return "leverage artificial intelligence to enhance decision-making processes"
    elif any(word in name_lower for word in ['data', 'analytics']):
        return "analyze data and generate valuable business intelligence"
    elif any(word in name_lower for word in ['system', 'platform', 'tool']):
        return "provide comprehensive solutions for operational challenges"
    else:
        return "enhance operational capabilities and service delivery"

def verify_descriptions():
    """Comprehensive verification of descriptions"""
    print("\n🔍 COMPREHENSIVE DESCRIPTION VERIFICATION")
    print("=" * 50)
    
    metadata_path = os.path.join("case_embeddings", "metadata.csv")
    
    try:
        df = pd.read_csv(metadata_path)
        
        # Check if descriptions exist
        if 'Idea Description' in df.columns:
            desc_stats = df['Idea Description'].apply(lambda x: len(str(x)) if pd.notna(x) else 0)
            empty_descriptions = df['Idea Description'].isna().sum() + (df['Idea Description'] == '').sum()
            
            print(f"📊 Description Statistics:")
            print(f"   - Total cases: {len(df)}")
            print(f"   - Cases with descriptions: {len(df) - empty_descriptions}")
            print(f"   - Empty descriptions: {empty_descriptions}")
            print(f"   - Average length: {desc_stats.mean():.1f} characters")
            print(f"   - Min length: {desc_stats.min()} characters")
            print(f"   - Max length: {desc_stats.max()} characters")
            
            # Show samples
            print(f"\n🔍 SAMPLE DESCRIPTIONS:")
            print("-" * 60)
            
            # Show diverse samples
            samples = df.head(3)
            for idx, row in samples.iterrows():
                print(f"📌 {row['Idea Name']}")
                print(f"   Domain: {row.get('Domain', 'N/A')}")
                print(f"   Description: {row['Idea Description']}")
                print()
                    
        else:
            print("❌ Idea Description column not found!")
            
    except Exception as e:
        print(f"❌ Error during verification: {e}")

if __name__ == "__main__":
    print("🚀 CASE DESCRIPTION ENHANCEMENT TOOL")
    print("=" * 50)
    
    success = add_case_descriptions()
    
    if success:
        verify_descriptions()
        print(f"\n🎉 Description enhancement completed successfully!")
    else:
        print(f"\n❌ Description enhancement failed!")
