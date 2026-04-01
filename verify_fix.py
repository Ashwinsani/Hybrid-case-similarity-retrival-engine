
import json
import os

def test_manual_search_logic_mock():
    # Mock search_data structure
    search_data = {
        "user_input": {
            "Idea Name": "Test Idea",
            "folder": "correct_folder_path"
        },
        "results": {
            "user_input": {
                "folder": "fallback_folder_path"
            }
        }
    }
    
    # Logic from app.py manual_search
    form_data = search_data.get('user_input', {})
    user_folder = form_data.get('folder')
    if not user_folder:
        user_folder = search_data.get('results', {}).get('user_input', {}).get('folder')
    
    print(f"Test 1 (Folder in user_input): {user_folder}")
    assert user_folder == "correct_folder_path"

    # Mock search_data structure (missing in top level, present in results)
    search_data_fallback = {
        "user_input": {
            "Idea Name": "Test Idea"
        },
        "results": {
            "user_input": {
                "folder": "fallback_folder_path"
            }
        }
    }
    
    form_data = search_data_fallback.get('user_input', {})
    user_folder = form_data.get('folder')
    if not user_folder:
        user_folder = search_data_fallback.get('results', {}).get('user_input', {}).get('folder')
    
    print(f"Test 2 (Folder in results): {user_folder}")
    assert user_folder == "fallback_folder_path"

    print("✅ All tests passed!")

if __name__ == "__main__":
    test_manual_search_logic_mock()
