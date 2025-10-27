#!/usr/bin/env python3
"""
Script to read users from Excel file and add them to users.json
"""

import pandas as pd
import json
import os

def read_excel_users(excel_file_path):
    """Read users from Excel file"""
    try:
        # Read Excel file
        df = pd.read_excel(excel_file_path)
        print(f"Excel file loaded successfully!")
        print(f"Columns found: {list(df.columns)}")
        print(f"Number of rows: {len(df)}")
        print("\nFirst few rows:")
        print(df.head())
        
        # Try to identify username and password columns
        username_col = None
        password_col = None
        
        # Common variations of column names
        username_variations = ['username', 'user', 'name', 'user_name', 'Username', 'User', 'Name', 'User_Name']
        password_variations = ['password', 'pass', 'pwd', 'Password', 'Pass', 'Pwd']
        
        for col in df.columns:
            if col in username_variations:
                username_col = col
            elif col in password_variations:
                password_col = col
        
        if username_col and password_col:
            print(f"\nFound columns: Username='{username_col}', Password='{password_col}'")
            
            # Extract users
            users = {}
            for _, row in df.iterrows():
                username = str(row[username_col]).strip()
                password = str(row[password_col]).strip()
                
                if username and password and username != 'nan' and password != 'nan':
                    users[username] = password
            
            print(f"\nExtracted {len(users)} users:")
            for username, password in users.items():
                print(f"  {username}: {password}")
            
            return users
        else:
            print(f"\nCould not identify username/password columns.")
            print(f"Available columns: {list(df.columns)}")
            return None
            
    except Exception as e:
        print(f"Error reading Excel file: {e}")
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
    excel_file = "users and password.xlsx"
    
    if not os.path.exists(excel_file):
        print(f"Excel file '{excel_file}' not found!")
        exit(1)
    
    print(f"Reading users from {excel_file}...")
    users = read_excel_users(excel_file)
    
    if users:
        print(f"\nUpdating users.json...")
        success = update_users_json(users)
        
        if success:
            print("\n✅ Successfully added users from Excel to users.json!")
        else:
            print("\n❌ Failed to update users.json")
    else:
        print("\n❌ No users found in Excel file")
