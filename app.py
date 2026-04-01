# app.py (COMPLETELY FIXED - TRUE PER-CASE OPTIMIZATION)
from flask import Flask, render_template, request, jsonify, flash, redirect, url_for, session
import os
import pandas as pd
import numpy as np
import json
from datetime import datetime
import traceback
from user_input_processor import process_user_input
from embedding_generator import generate_embeddings_from_user_input
import google_search
from google_search import rank_google_results


app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'a-very-secret-key')
app.config['SESSION_TYPE'] = 'filesystem'

# Cached matcher singleton
_matcher_singleton = None

def get_matcher(corpus_dir='case_embeddings'):
    global _matcher_singleton
    if _matcher_singleton is None:
        try:
            # Lazy import to avoid circular imports at module load time
            from multi_similarity_engine import EnhancedCosineSimilarityMatcher
            _matcher_singleton = EnhancedCosineSimilarityMatcher(corpus_dir)
            print(f"✅ Similarity matcher initialized successfully")
        except ImportError as ie:
            print(f"❌ Failed to import EnhancedCosineSimilarityMatcher: {ie}")
            print("   Check that multi_similarity_engine.py exists and has no syntax errors")
            _matcher_singleton = None
        except Exception as e:
            print(f"❌ Failed to initialize EnhancedCosineSimilarityMatcher: {e}")
            _matcher_singleton = None
    return _matcher_singleton

def fix_numpy_types(obj):
    """Convert numpy types to Python native types for JSON/template serialization"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, dict):
        return {k: fix_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [fix_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(fix_numpy_types(item) for item in obj)
    else:
        return obj

@app.route('/')
def index():
    return render_template('index.html')

def _process_and_find_similar(form_data, top_k=10, similarity_threshold=0.3):
    # 1. Process user input
    input_result = process_user_input(form_data)
    user_file_path = input_result.get('filepath') or input_result.get('file_path')
    user_id = (input_result.get('user_data') or {}).get('user_id') or input_result.get('user_id')

    if not user_file_path:
        raise ValueError("User input processing did not return a filepath")

    print(f"✅ User input saved (source file): {user_file_path}")

    # 2. Generate embeddings
    embedding_result = generate_embeddings_from_user_input(user_file_path)
    if not embedding_result or 'folder' not in embedding_result:
        raise ValueError("Embedding generator must return a dict containing key 'folder'")

    user_folder = embedding_result['folder']
    user_id = embedding_result.get('user_id', user_id)

    print(f"✅ Embeddings generated. user_id={user_id}, folder={user_folder}")

    # 3. Run similarity with TRUE per-case optimization
    matcher = get_matcher('case_embeddings')
    if not matcher:
        raise RuntimeError("Could not initialize the similarity matcher.")
    
    similarity_results = matcher.enhanced_find_similar_cases(
        user_folder=user_folder,
        top_k=top_k,
        similarity_threshold=similarity_threshold
    )

    # 4. Apply Final Blended Threshold (Post-Reranking Filter)
    final_threshold = 0.5
    if 'similar_cases' in similarity_results:
        original_count = len(similarity_results['similar_cases'])
        
        # Save all pre-filtered scores so the dashboard can generate accurate sensitivity analysis mappings
        all_scores = [float(case.get('reranked_score', case.get('final_score', 0))) for case in similarity_results['similar_cases']]
        if 'match_statistics' not in similarity_results:
            similarity_results['match_statistics'] = {}
        similarity_results['match_statistics']['all_scores'] = all_scores

        filtered_cases = [
            case for case in similarity_results['similar_cases']
            if float(case.get('reranked_score', case.get('final_score', 0))) >= final_threshold
        ]
        similarity_results['similar_cases'] = filtered_cases
        
        # Update match statistics for UI display
        if 'match_statistics' in similarity_results:
            similarity_results['match_statistics']['total_matches_found'] = len(filtered_cases)
            similarity_results['match_statistics']['final_threshold_used'] = final_threshold
            # Store retrieval threshold separately if needed, but UI will use final_threshold_used
            similarity_results['match_statistics']['retrieval_threshold'] = similarity_threshold

        print(f"🎯 Final Threshold Filter: {original_count} -> {len(filtered_cases)} cases (Threshold: {final_threshold})")
    
    # Add folder to form_data so it's persisted in search_results for manual_search usage
    form_data['folder'] = user_folder
    
    return user_id, similarity_results, form_data

@app.route('/submit', methods=['POST'])
def submit_form():
    try:
        # 1. Collect form data
        form_data = {
            'Idea Name': request.form.get('idea_name', '').strip(),
            'Domain': request.form.get('domain', '').strip(),
            'fundingSource': request.form.get('fundingSource', '').strip(),
            'Expected benefits': request.form.get('expected_benefits', '').strip(),
            'Idea Description': request.form.get('idea_description', '').strip(),
            'potential Challenges': request.form.get('potential_challenges', '').strip(),
        }

        # 2. Validate required fields
        required_fields = ['Idea Name', 'Domain', 'fundingSource', 'Expected benefits', 'Idea Description', 'potential Challenges']
        missing_fields = [f for f in required_fields if not form_data.get(f)]
        if missing_fields:
            flash(f'Please fill in required fields: {", ".join(missing_fields)}', 'error')
            return redirect(url_for('index'))

        print(f"🔄 PROCESSING USER INPUT: {form_data['Idea Name']}")
        
        user_id, similarity_results, form_data = _process_and_find_similar(form_data)

        # Convert numpy types to python native for rendering/json
        similarity_results = fix_numpy_types(similarity_results)

        # If results exist, store last_search_results in session
        similar_cases = similarity_results.get('similar_cases', [])
        if similar_cases:
            print(f"✅ Found {len(similar_cases)} similar cases")
            
            # Save the results to a file
            search_results_dir = 'search_results'
            os.makedirs(search_results_dir, exist_ok=True)
            
            # Save the full results object for the GET request
            search_data = {
                'user_id': user_id,
                'user_input': form_data,
                'results': similarity_results,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(os.path.join(search_results_dir, f'{user_id}.json'), 'w') as f:
                json.dump(search_data, f)
            
            session['last_user_id'] = user_id
            
            return redirect(url_for('display_results', user_id=user_id))

        # No matches found
        flash('No similar cases found. Try adjusting your input or lowering the similarity threshold.', 'warning')
        return redirect(url_for('index'))

    except Exception as e:
        print(f"❌ Error in submit_form: {e}")
        traceback.print_exc()
        flash(f'Error processing your request: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.route('/results/<user_id>')
def display_results(user_id):
    """GET route to display similarity results to avoid form resubmission error"""
    search_results_file = os.path.join('search_results', f'{user_id}.json')
    
    if not os.path.exists(search_results_file):
        flash('Results not found. Please try your search again.', 'error')
        return redirect(url_for('index'))
        
    try:
        with open(search_results_file, 'r') as f:
            search_data = json.load(f)
            
        return render_template('results.html', 
                             results=search_data.get('results'), 
                             user_input=search_data.get('user_input'),
                             user_id=user_id)
    except Exception as e:
        print(f"❌ Error displaying results: {e}")
        flash('Error loading search results.', 'error')
        return redirect(url_for('index'))

@app.route('/global_search/<user_id>')
def handle_global_search(user_id):
    """Fetch and rank similar ideas from the Google web search"""
    search_results_file = os.path.join('search_results', f'{user_id}.json')
    
    if not os.path.exists(search_results_file):
        flash('Search data not found. Please submit your idea again.', 'error')
        return redirect(url_for('index'))
        
    try:
        with open(search_results_file, 'r') as f:
            search_data = json.load(f)
            
        form_data = search_data.get('user_input', {})
        idea_name = form_data.get('Idea Name', '')
        idea_description = form_data.get('Idea Description', '')
        
        # Construct a search query for Google
        search_query = f"{idea_name} {idea_description}"[:500] # Limit query length
        
        print(f"🌐 TRIGGERING GLOBAL SEARCH: {idea_name}")
        
        # 1. Fetch results from Google
        raw_web_results = google_search.google_search(search_query, num_results=10)
        
        # 2. Rank results semantically against the idea description
        ranked_web_results = rank_google_results(idea_description, raw_web_results)
        
        return render_template('global_search_results.html', 
                             web_results=ranked_web_results, 
                             user_input=form_data,
                             user_id=user_id)
                             
    except Exception as e:
        print(f"❌ Error in global_search: {e}")
        traceback.print_exc()
        flash(f'Web search failed: {str(e)}', 'error')
        return redirect(url_for('display_results', user_id=user_id))

@app.route('/manual_search/<user_id>')
def manual_search(user_id):
    """Render the manual weight adjustment page using full dataset results"""
    search_results_file = os.path.join('search_results', f'{user_id}.json')
    
    if not os.path.exists(search_results_file):
        flash('Search data not found. Please submit your idea again.', 'error')
        return redirect(url_for('index'))
        
    try:
        with open(search_results_file, 'r') as f:
            search_data = json.load(f)
            
        form_data = search_data.get('user_input', {})
        idea_name = form_data.get('Idea Name', '')
        idea_description = form_data.get('Idea Description', '')
        
        # Use the folder from the ORIGINAL search (this folder already has the embeddings)
        user_folder = form_data.get('folder')
        if not user_folder:
            # Fallback: check in results if not found in top-level form_data
            user_folder = search_data.get('results', {}).get('user_input', {}).get('folder')
        
        # 1. Initialize matcher
        matcher = get_matcher()
        if not matcher:
            raise RuntimeError("Similarity matcher not initialized")
            
        # 3. Perform a fresh search across the ENTIRE dataset (threshold=0.0)
        # We MUST pass skip_rerank=True to avoid Cross-Encoder as requested
        print(f"📊 MANUAL SEARCH: Searching entire dataset for '{idea_name}' using folder '{user_folder}'")
        raw_results = matcher.enhanced_find_similar_cases(
            user_folder=user_folder,
            top_k=2000, 
            similarity_threshold=0.0,
            skip_rerank=True
        )
        
        # Convert numpy types
        full_results = fix_numpy_types(raw_results)
        similar_cases = full_results.get('similar_cases', [])
        
        return render_template('manual_search.html', 
                             similar_cases=similar_cases, 
                             user_input=form_data,
                             user_id=user_id)
                             
    except Exception as e:
        print(f"❌ Error in manual_search: {e}")
        traceback.print_exc()
        flash(f'Manual search failed: {str(e)}', 'error')
        return redirect(url_for('display_results', user_id=user_id))

@app.route('/api/submit', methods=['POST'])
def api_submit():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        api_form_data = {
            'Idea Name': data.get('idea_name', ''),
            'Domain': data.get('domain', ''),
            'fundingSource': data.get('fundingSource', ''),
            'Expected benefits': data.get('expected_benefits', ''),
            'Idea Description': data.get('idea_description', ''),
            'potential Challenges': data.get('potential_challenges', ''),
        }
        
        user_id, similarity_results, _ = _process_and_find_similar(api_form_data, top_k=10, similarity_threshold=0.2)
        
        similarity_results = fix_numpy_types(similarity_results)
        return jsonify({'status': 'success', 'user_id': user_id, 'results': similarity_results})

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/case_analysis', methods=['GET'])
def get_case_analysis():
    """Return analysis for a given user_id"""
    user_id = request.args.get('user_id')
    if not user_id:
        return jsonify({'error': 'No user_id provided'}), 400
        
    search_results_file = os.path.join('search_results', f'{user_id}.json')
    if not os.path.exists(search_results_file):
        return jsonify({'error': 'No search results found for this user_id'}), 404
        
    try:
        with open(search_results_file, 'r') as f:
            last_search = json.load(f)

        case_analysis = last_search.get('case_analysis') or {}
        if not case_analysis:
            return jsonify({'error': 'No analysis available'}), 400

        analysis = {
            'optimization_mode': 'true_per_case',
            'case_features': case_analysis.get('case_features', {}),
            'method_evaluation_scores': case_analysis.get('method_evaluation_scores', {}),
            'computed_weights': case_analysis.get('computed_weights', {}),
            'weight_explanation': case_analysis.get('explanation', []),
            'analysis_timestamp': case_analysis.get('timestamp', '')
        }

        sim_cases = last_search.get('similar_cases', [])
        if sim_cases:
            scores = [c.get('final_score', 0) for c in sim_cases]
            analysis['case_statistics'] = {
                'total_cases_found': len(sim_cases),
                'top_score': float(max(scores)) if scores else 0,
                'average_score': float(np.mean(scores)) if scores else 0
            }

        return jsonify({'status': 'success', 'analysis': fix_numpy_types(analysis)})

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/system_stats', methods=['GET'])
def get_system_stats():
    """Return lightweight system statistics"""
    try:
        matcher = get_matcher('case_embeddings')
        total_corpus_cases = len(matcher.corpus_embeddings) if matcher and hasattr(matcher, 'corpus_embeddings') else 0

        user_embeddings_dir = 'user_embeddings'
        user_cases_processed = 0
        if os.path.exists(user_embeddings_dir):
            user_folders = [f for f in os.listdir(user_embeddings_dir)
                            if os.path.isdir(os.path.join(user_embeddings_dir, f))]
            user_cases_processed = len(user_folders)

        return jsonify({
            'status': 'success',
            'system_statistics': {
                'optimization_mode': 'true_per_case_heuristic',
                'description': 'Pure heuristic-based per-case weight optimization',
                'total_corpus_cases': total_corpus_cases,
                'user_cases_processed': user_cases_processed,
                'similarity_methods': ['semantic', 'field_score', 'keyword_matching', 'bm25'],
                'weight_constraints': {
                    'minimum_weight': 0.05,
                    'maximum_weight': 0.50,
                    'normalized_sum': 1.0
                },
                'system_status': 'operational' if matcher else 'degraded',
                'last_updated': datetime.now().isoformat()
            }
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/weights_dashboard')
def weights_dashboard():
    """Render the weights dashboard using a user_id"""
    user_id = request.args.get('user_id')
    if not user_id:
        user_id = session.get('last_user_id')
    
    if not user_id:
        flash('No user ID provided. Please submit a search first.', 'warning')
        return redirect(url_for('index'))

    search_results_file = os.path.join('search_results', f'{user_id}.json')

    if not os.path.exists(search_results_file):
        flash('No search results found for this user. Please submit a search first.', 'warning')
        return redirect(url_for('index'))

    try:
        with open(search_results_file, 'r') as f:
            search_data = json.load(f)
        
        results_obj = search_data.get('results', {})
        similar_cases = results_obj.get('similar_cases', [])
        match_stats = results_obj.get('match_statistics', {})
        
        if not similar_cases:
            flash('No similar cases found for this user.', 'warning')
            return redirect(url_for('index'))
        
        matcher = get_matcher()
        if not matcher:
            raise RuntimeError("Could not initialize the similarity matcher.")
        
        dashboard_data = {
            'optimization_mode': 'true_per_case_heuristic',
            'total_corpus_cases': len(matcher.corpus_embeddings),
            'last_case_analyzed': search_data.get('timestamp', 'Never'),
            'recent_weights': {},
            'recent_case': {},
            'actual_threshold': match_stats.get('final_threshold_used', 0.5),
            'threshold_sensitivity': []
        }

        first_case = similar_cases[0]
        # Use reranked_score if available as the top_score for dashboard
        top_dashboard_score = float(first_case.get('reranked_score', first_case.get('final_score', 0)))
        
        dashboard_data['recent_weights'] = match_stats.get('case_analysis', {}).get('computed_weights', {})
        
        dashboard_data['recent_case'] = {
            'user_id': user_id,
            'idea_name': search_data.get('user_input', {}).get('Idea Name', 'Unknown'),
            'domain': search_data.get('user_input', {}).get('Domain', 'Unknown'),
            'matches_found': len(similar_cases),
            'top_score': top_dashboard_score,
            'current_threshold': dashboard_data['actual_threshold'],
        }
        
        # --- Threshold Sensitivity Analysis ---
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        actual_thresh = dashboard_data['actual_threshold']
        
        # Use unfiltered scores if available, otherwise fallback to truncated array
        all_scores = match_stats.get('all_scores')
        if not all_scores:
            all_scores = [float(c.get('reranked_score', c.get('final_score', 0))) for c in similar_cases]
        
        sensitivity_list = []
        prev_matches = len(all_scores)
        
        for t in thresholds:
            matches_count = sum(1 for s in all_scores if s >= t)
            
            if prev_matches == 0:
                drop_pct = 0.0
            else:
                drop_pct = ((prev_matches - matches_count) / prev_matches) * 100.0
                
            if drop_pct <= 10.0:
                stability = "Stable"
            elif drop_pct <= 25.0:
                stability = "Moderate"
            else:
                stability = "Unstable"
                
            is_current = abs(t - actual_thresh) < 0.01
                
            sensitivity_list.append({
                "threshold": t,
                "matches": matches_count,
                "drop_pct": drop_pct,
                "stability": stability,
                "is_current": is_current,
                "is_real_data": True,
                "is_recommended": False
            })
            prev_matches = matches_count
            
        # Recommendation logic: Find the FIRST threshold where stability transitions from Stable to Moderate with enough matches
        max_matches = max((item['matches'] for item in sensitivity_list), default=0)
        min_matches = max(3, int(0.2 * max_matches))
        
        recommended_set = False
        prev_stability = "Stable"
        
        for item in sensitivity_list:
            current_stability = item['stability']
            current_matches = item['matches']
            
            if (
                prev_stability == "Stable"
                and current_stability == "Moderate"
                and current_matches >= min_matches
            ):
                item['is_recommended'] = True
                recommended_set = True
                break
                
            prev_stability = current_stability
                
        # Fallback if no transition found
        if not recommended_set and sensitivity_list:
            for item in sensitivity_list:
                if item['threshold'] == 0.3:
                    item['is_recommended'] = True
            
        dashboard_data['threshold_sensitivity'] = sensitivity_list
        # ----------------------------------------
        
        return render_template('weights_dashboard.html', dashboard_data=fix_numpy_types(dashboard_data))

    except Exception as e:
        traceback.print_exc()
        flash(f'Error loading dashboard: {str(e)}', 'error')
        return redirect(url_for('index'))


@app.route('/api/demo_comparison', methods=['GET'])
def demo_comparison():
    """Return simulated demo comparison showing how different case types produce different weights"""
    try:
        technical_weights = {
            'semantic': 0.40,
            'field_score': 0.08,
            'keyword_matching': 0.32,
            'bm25': 0.20
        }

        descriptive_weights = {
            'semantic': 0.12,
            'field_score': 0.45,
            'keyword_matching': 0.28,
            'bm25': 0.15
        }

        return jsonify({
            'status': 'success',
            'demo_comparison': {
                'technical_case': {
                    'name': 'AI Medical Diagnosis System',
                    'description': 'Deep learning system using convolutional neural networks for medical image analysis with 98% accuracy',
                    'domain': 'Healthcare',
                    'computed_weights': technical_weights,
                    'explanation': [
                        "High technical term density → Semantic/Keyword Matching weights increased",
                        "Complex description with specific algorithms → Keyword matching detects technical terms",
                        "Less descriptive text → Lexical weight decreased"
                    ]
                },
                'descriptive_case': {
                    'name': 'Community Recycling Initiative',
                    'description': 'Local community program to encourage recycling through education and incentives',
                    'domain': 'Environmental',
                    'computed_weights': descriptive_weights,
                    'explanation': [
                        "Descriptive text with concrete examples → Lexical weight increased",
                        "Community-focused content → Keyword matching effective",
                        "Less technical content → Semantic weight decreased"
                    ]
                },
                'comparison_note': 'These are example weights showing how different case types receive different weight distributions in true per-case optimization.'
            }
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    os.makedirs('user_inputs', exist_ok=True)
    os.makedirs('user_embeddings', exist_ok=True)
    os.makedirs('search_results', exist_ok=True)
    print("🚀 Starting Flask application with TRUE PER-CASE OPTIMIZATION...")
    app.run(debug=True, host='0.0.0.0', port=5000)
