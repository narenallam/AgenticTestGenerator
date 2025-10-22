#!/usr/bin/env python3
"""
Example of generating API tests for FastAPI endpoints.

This demonstrates generating comprehensive API tests including
status codes, response validation, and error handling.
"""

from src.prompts import TestType
from src.test_agent import TestGenerationAgent


def main():
    """Run API test generation example."""
    # Sample FastAPI endpoint
    api_code = '''
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI()

class User(BaseModel):
    id: int
    name: str
    email: str
    active: bool = True

# In-memory storage
users_db: List[User] = []

@app.post("/users/", status_code=201)
async def create_user(user: User):
    """Create a new user."""
    # Check if user already exists
    if any(u.id == user.id for u in users_db):
        raise HTTPException(status_code=400, detail="User already exists")
    
    users_db.append(user)
    return user

@app.get("/users/{user_id}")
async def get_user(user_id: int):
    """Get user by ID."""
    user = next((u for u in users_db if u.id == user_id), None)
    
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    
    return user

@app.delete("/users/{user_id}", status_code=204)
async def delete_user(user_id: int):
    """Delete user by ID."""
    global users_db
    
    user = next((u for u in users_db if u.id == user_id), None)
    
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    
    users_db = [u for u in users_db if u.id != user_id]
    return None
'''
    
    print("=" * 70)
    print("API Test Generation Example")
    print("=" * 70)
    print("\nGenerating comprehensive API tests for FastAPI endpoints...\n")
    
    # Initialize agent with API test type
    agent = TestGenerationAgent(max_iterations=3)
    
    # Generate tests
    tests = agent.generate_tests(
        target_code=api_code,
        test_type=TestType.API
    )
    
    # Display results
    print("\n" + "=" * 70)
    print("Generated API Tests:")
    print("=" * 70)
    print(tests)
    
    # Save to file
    output_file = "test_user_api.py"
    with open(output_file, 'w') as f:
        f.write(tests)
    
    print(f"\n✓ API tests saved to: {output_file}")
    print("\nTo run the tests:")
    print(f"  pytest {output_file} -v")


if __name__ == "__main__":
    main()

