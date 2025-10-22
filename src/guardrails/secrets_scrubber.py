"""
Secrets scrubber for environment and code protection.

Prevents secrets from leaking into generated tests and execution environment.
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Set

from pydantic import BaseModel, Field


class SecretsConfig(BaseModel):
    """Configuration for secrets scrubbing."""
    
    # Patterns to detect secrets
    secret_patterns: List[str] = Field(
        default=[
            r'[A-Za-z0-9_]+_API_KEY',
            r'[A-Za-z0-9_]+_TOKEN',
            r'[A-Za-z0-9_]+_SECRET',
            r'[A-Za-z0-9_]+_PASSWORD',
            r'GITHUB_TOKEN',
            r'AWS_ACCESS_KEY',
            r'OPENAI_API_KEY',
            r'GOOGLE_API_KEY',
        ],
        description="Regex patterns for detecting secret env vars"
    )
    
    # File patterns to protect
    protected_files: List[str] = Field(
        default=[
            '.env',
            '.env.local',
            '.env.production',
            '**/.aws/credentials',
            '**/.ssh/id_*',
            '**/token',
            '**/*.key',
            '**/*.pem',
        ],
        description="File patterns that should never be read"
    )
    
    # Replacement value
    redacted_value: str = Field(
        default="***REDACTED***",
        description="Value to replace secrets with"
    )


class SecretsScrubber:
    """
    Scrubs secrets from environment and code.
    
    Prevents API keys, tokens, and other secrets from leaking into:
    - Generated test code
    - Test execution environment
    - Logs and outputs
    
    Example:
        >>> scrubber = SecretsScrubber()
        >>> safe_env = scrubber.scrub_environment()
        >>> safe_code = scrubber.scrub_code("os.getenv('OPENAI_API_KEY')")
    """
    
    def __init__(self, config: Optional[SecretsConfig] = None):
        """
        Initialize secrets scrubber.
        
        Args:
            config: Scrubbing configuration
        """
        self.config = config or SecretsConfig()
        self._compiled_patterns = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in self.config.secret_patterns
        ]
        
    def scrub_environment(self) -> Dict[str, str]:
        """
        Create scrubbed environment dict safe for test execution.
        
        Returns:
            Environment dict with secrets removed
        """
        safe_env = {}
        
        for key, value in os.environ.items():
            # Check if key matches any secret pattern
            if self._is_secret_key(key):
                continue  # Skip secret keys entirely
            
            # Allow safe environment variables
            if self._is_safe_key(key):
                safe_env[key] = value
        
        # Add safe minimal environment
        safe_env.update({
            'PATH': os.environ.get('PATH', ''),
            'HOME': '/tmp/test_home',  # Isolated home
            'PYTHONPATH': os.environ.get('PYTHONPATH', ''),
            'LANG': 'en_US.UTF-8',
            'LC_ALL': 'en_US.UTF-8',
        })
        
        return safe_env
    
    def scrub_code(self, code: str) -> str:
        """
        Remove secret references from code.
        
        Replaces:
        - Direct secret values
        - os.getenv('SECRET_KEY')
        - References to secret files
        
        Args:
            code: Code to scrub
            
        Returns:
            Scrubbed code with secrets removed
        """
        scrubbed = code
        
        # Replace environment variable access to secrets
        for pattern in self._compiled_patterns:
            # Match os.getenv('SECRET'), os.environ['SECRET'], etc.
            scrubbed = re.sub(
                rf'os\.(?:getenv|environ\.get)\(["\']({pattern.pattern})["\'].*?\)',
                f'os.getenv("SAFE_VAR", "{self.config.redacted_value}")',
                scrubbed,
                flags=re.IGNORECASE
            )
        
        # Replace file reads of protected files
        for file_pattern in self.config.protected_files:
            # Match open('.env'), Path('.env').read_text(), etc.
            clean_pattern = file_pattern.replace('**/', '').replace('*', '\\w+')
            scrubbed = re.sub(
                rf'(?:open|Path)\(["\'].*{clean_pattern}.*["\']\)',
                f'# SECURITY: Access to {file_pattern} blocked',
                scrubbed,
                flags=re.IGNORECASE
            )
        
        return scrubbed
    
    def detect_secrets_in_code(self, code: str) -> List[str]:
        """
        Detect potential secrets in code.
        
        Args:
            code: Code to analyze
            
        Returns:
            List of detected secret patterns
        """
        detected = []
        
        for pattern in self._compiled_patterns:
            matches = pattern.findall(code)
            detected.extend(matches)
        
        # Check for hardcoded tokens (rough heuristic)
        token_pattern = r'["\'][A-Za-z0-9_\-]{32,}["\']'
        if re.search(token_pattern, code):
            detected.append("POSSIBLE_HARDCODED_TOKEN")
        
        return list(set(detected))
    
    def is_file_protected(self, file_path: str) -> bool:
        """
        Check if file should be protected from access.
        
        Args:
            file_path: Path to check
            
        Returns:
            True if file is protected
        """
        path = Path(file_path)
        
        for pattern in self.config.protected_files:
            if path.match(pattern):
                return True
        
        return False
    
    def _is_secret_key(self, key: str) -> bool:
        """Check if environment key is a secret."""
        for pattern in self._compiled_patterns:
            if pattern.match(key):
                return True
        return False
    
    def _is_safe_key(self, key: str) -> bool:
        """Check if environment key is safe to include."""
        safe_prefixes = (
            'PYTHON', 'PATH', 'HOME', 'USER', 'SHELL',
            'LANG', 'LC_', 'TERM', 'DISPLAY',
            'XDG_', 'TMPDIR', 'TMP', 'TEMP',
        )
        
        dangerous_keywords = (
            'KEY', 'TOKEN', 'SECRET', 'PASSWORD', 'AUTH',
            'CREDENTIAL', 'PRIVATE', 'GITHUB', 'AWS',
        )
        
        # Allow if starts with safe prefix
        if any(key.startswith(prefix) for prefix in safe_prefixes):
            return True
        
        # Deny if contains dangerous keyword
        if any(keyword in key.upper() for keyword in dangerous_keywords):
            return False
        
        return True


def create_secrets_scrubber(
    additional_patterns: Optional[List[str]] = None
) -> SecretsScrubber:
    """
    Factory function to create secrets scrubber.
    
    Args:
        additional_patterns: Additional regex patterns for secrets
        
    Returns:
        Configured SecretsScrubber
    """
    config = SecretsConfig()
    
    if additional_patterns:
        config.secret_patterns.extend(additional_patterns)
    
    return SecretsScrubber(config)

