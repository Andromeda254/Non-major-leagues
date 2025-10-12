#!/usr/bin/env python3
"""
Secret Key Generator
===================

This script generates secure random keys for various purposes:
- SECRET_KEY: For Flask/Django application security
- JWT_SECRET: For JWT token signing
- API_SECRET: For API authentication
- ENCRYPTION_KEY: For data encryption

Author: AI Assistant
Date: 2025-10-11
"""

import secrets
import string
import argparse
from typing import Dict, List

def generate_secret_key(length: int = 32) -> str:
    """Generate a secure random secret key"""
    return secrets.token_urlsafe(length)

def generate_jwt_secret(length: int = 32) -> str:
    """Generate a secure JWT secret"""
    return secrets.token_urlsafe(length)

def generate_api_secret(length: int = 24) -> str:
    """Generate an API secret"""
    return secrets.token_urlsafe(length)

def generate_encryption_key(length: int = 32) -> str:
    """Generate an encryption key"""
    return secrets.token_hex(length)

def generate_password(length: int = 16) -> str:
    """Generate a secure password"""
    alphabet = string.ascii_letters + string.digits + "!@#$%^&*"
    return ''.join(secrets.choice(alphabet) for _ in range(length))

def generate_all_secrets() -> Dict[str, str]:
    """Generate all types of secrets"""
    return {
        'SECRET_KEY': generate_secret_key(),
        'JWT_SECRET': generate_jwt_secret(),
        'API_SECRET': generate_api_secret(),
        'ENCRYPTION_KEY': generate_encryption_key(),
        'DATABASE_PASSWORD': generate_password(),
        'REDIS_PASSWORD': generate_password(),
        'SMTP_PASSWORD': generate_password()
    }

def format_for_env_file(secrets_dict: Dict[str, str]) -> str:
    """Format secrets for .env file"""
    lines = ["# Generated Secrets - Keep these secure!"]
    for key, value in secrets_dict.items():
        lines.append(f"{key}={value}")
    return '\n'.join(lines)

def format_for_yaml(secrets_dict: Dict[str, str]) -> str:
    """Format secrets for YAML file"""
    lines = ["# Generated Secrets - Keep these secure!"]
    for key, value in secrets_dict.items():
        lines.append(f"{key}: \"{value}\"")
    return '\n'.join(lines)

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Generate secure random secrets')
    parser.add_argument('--type', choices=['all', 'secret', 'jwt', 'api', 'encryption', 'password'], 
                       default='all', help='Type of secret to generate')
    parser.add_argument('--length', type=int, default=32, help='Length of the secret')
    parser.add_argument('--format', choices=['env', 'yaml', 'plain'], default='env', 
                       help='Output format')
    parser.add_argument('--output', help='Output file (optional)')
    
    args = parser.parse_args()
    
    # Generate secrets based on type
    if args.type == 'all':
        secrets_dict = generate_all_secrets()
    elif args.type == 'secret':
        secrets_dict = {'SECRET_KEY': generate_secret_key(args.length)}
    elif args.type == 'jwt':
        secrets_dict = {'JWT_SECRET': generate_jwt_secret(args.length)}
    elif args.type == 'api':
        secrets_dict = {'API_SECRET': generate_api_secret(args.length)}
    elif args.type == 'encryption':
        secrets_dict = {'ENCRYPTION_KEY': generate_encryption_key(args.length)}
    elif args.type == 'password':
        secrets_dict = {'PASSWORD': generate_password(args.length)}
    
    # Format output
    if args.format == 'env':
        output = format_for_env_file(secrets_dict)
    elif args.format == 'yaml':
        output = format_for_yaml(secrets_dict)
    else:  # plain
        output = '\n'.join([f"{key}: {value}" for key, value in secrets_dict.items()])
    
    # Output
    if args.output:
        with open(args.output, 'w') as f:
            f.write(output)
        print(f"✅ Secrets written to {args.output}")
    else:
        print(output)
    
    # Security reminder
    print("\n🔒 Security Reminders:")
    print("   - Never commit secrets to version control")
    print("   - Store secrets in environment variables")
    print("   - Use different secrets for different environments")
    print("   - Rotate secrets regularly")
    print("   - Use a secrets management service in production")

if __name__ == "__main__":
    main()
