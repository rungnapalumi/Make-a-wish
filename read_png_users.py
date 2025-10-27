#!/usr/bin/env python3
"""
Script to read users from PNG file using OCR and add them to users.json
"""

import json
import os
import re
from PIL import Image
import pytesseract

def read_png_users(png_file_path):
    """Read users from PNG file using OCR"""
    try:
        # Open and process the image
        image = Image.open(png_file_path)
        
        # Use OCR to extract text
        text = pytesseract.image_to_string(image)
        
        print(f"OCR extracted text from {png_file_path}:")
        print("=" * 50)
        print(text)
        print("=" * 50)
        
        # Parse the text to extract usernames and passwords
        users = {}
        
        # Split text into lines and process each line
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Look for patterns like "username: password" or "username password"
            # Try different patterns
            patterns = [
                r'(\w+)\s*[:=]\s*(\w+)',  # username: password or username=password
                r'(\w+)\s+(\w+)',         # username password
                r'^(\w+)\s*-\s*(\w+)$',   # username - password
            ]
            
            for pattern in patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    username = match.group(1).strip()
                    password = match.group(2).strip()
                    
                    # Basic validation
                    if len(username) > 0 and len(password) > 0:
                        users[username] = password
                        print(f"Found user: {username} -> {password}")
                        break
        
        print(f"\nExtracted {len(users)} users:")
        for username, password in users.items():
            print(f"  {username}: {password}")
        
        return users
        
    except Exception as e:
        print(f"Error reading PNG file: {e}")
        return None

def update_users_json(users, json_file_path="users.json"):
    """Update users.json with new users"""
    try:
        # Load existing users
        if os.path.exists(json_file_path):
            with open(json_file_path, 'r', encoding='utf-8') as f:
                existing_users = json.load(f)
        else:
            existing_users = {}
        
        # Add new users (don't overwrite existing ones)
        updated = False
        for username, password in users.items():
            if username not in existing_users:
                existing_users[username] = password
                updated = True
                print(f"Added new user: {username}")
            else:
                print(f"User {username} already exists, skipping...")
        
        # Save updated users
        with open(json_file_path, 'w', encoding='utf-8') as f:
            json.dump(existing_users, f, ensure_ascii=False, indent=2)
        
        print(f"\nUpdated {json_file_path} with {len(existing_users)} total users")
        return True
        
    except Exception as e:
        print(f"Error updating users.json: {e}")
        return False

if __name__ == "__main__":
    png_file = "users and password.png"
    
    if not os.path.exists(png_file):
        print(f"PNG file '{png_file}' not found!")
        exit(1)
    
    print(f"Reading users from {png_file}...")
    users = read_png_users(png_file)
    
    if users:
        print(f"\nUpdating users.json...")
        success = update_users_json(users)
        
        if success:
            print("\n✅ Successfully added users from PNG to users.json!")
        else:
            print("\n❌ Failed to update users.json")
    else:
        print("\n❌ No users found in PNG file")
