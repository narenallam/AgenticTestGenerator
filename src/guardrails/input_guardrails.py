"""
Input Guardrails for LLM requests.

Protects against:
- PII (Personally Identifiable Information)
- Prompt injection attacks
- Toxic/harmful content
- Jailbreak attempts

This module scans and sanitizes all inputs before they reach the LLM.
"""

import re
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field
from rich.console import Console

console = Console()


class PIIType(str, Enum):
    """Types of PII that can be detected."""
    
    EMAIL = "email"
    PHONE = "phone"
    SSN = "ssn"
    CREDIT_CARD = "credit_card"
    IP_ADDRESS = "ip_address"
    API_KEY = "api_key"
    PASSWORD = "password"
    NAME = "name"
    ADDRESS = "address"


class ThreatLevel(str, Enum):
    """Threat levels for input violations."""
    
    SAFE = "safe"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class PIIDetection(BaseModel):
    """A detected PII instance."""
    
    pii_type: PIIType = Field(..., description="Type of PII")
    value: str = Field(..., description="Detected value")
    start: int = Field(..., description="Start position")
    end: int = Field(..., description="End position")
    confidence: float = Field(..., description="Detection confidence (0-1)")


class InputViolation(BaseModel):
    """An input guardrail violation."""
    
    violation_type: str = Field(..., description="Type of violation")
    threat_level: ThreatLevel = Field(..., description="Threat level")
    description: str = Field(..., description="Description of violation")
    evidence: str = Field(..., description="Evidence text")
    recommendation: str = Field(..., description="How to fix")


class InputScanResult(BaseModel):
    """Result of input scanning."""
    
    safe: bool = Field(..., description="Input is safe")
    pii_detected: List[PIIDetection] = Field(default_factory=list, description="Detected PII")
    violations: List[InputViolation] = Field(default_factory=list, description="Violations found")
    sanitized_input: Optional[str] = Field(default=None, description="Sanitized input")
    threat_level: ThreatLevel = Field(default=ThreatLevel.SAFE, description="Overall threat level")


class InputGuardrails:
    """
    Comprehensive input guardrails for LLM requests.
    
    Features:
    - PII detection and redaction
    - Prompt injection prevention
    - Toxic content detection
    - Jailbreak attempt detection
    - Input sanitization
    
    Example:
        >>> guardrails = InputGuardrails()
        >>> result = guardrails.scan_input(
        ...     "My email is john@example.com and SSN is 123-45-6789"
        ... )
        >>> if not result.safe:
        ...     raise SecurityError("Unsafe input detected")
        >>> safe_input = result.sanitized_input
    """
    
    def __init__(self, enable_pii_redaction: bool = True, enable_prompt_injection: bool = True):
        """
        Initialize input guardrails.
        
        Args:
            enable_pii_redaction: Enable PII detection and redaction
            enable_prompt_injection: Enable prompt injection detection
        """
        self.enable_pii_redaction = enable_pii_redaction
        self.enable_prompt_injection = enable_prompt_injection
        
        # PII regex patterns
        self.pii_patterns = {
            PIIType.EMAIL: r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            PIIType.PHONE: r'\b(?:\+?1[-.]?)?\(?\d{3}\)?[-.]?\d{3}[-.]?\d{4}\b',
            PIIType.SSN: r'\b\d{3}-\d{2}-\d{4}\b',
            PIIType.CREDIT_CARD: r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
            PIIType.IP_ADDRESS: r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',
            PIIType.API_KEY: r'\b[A-Za-z0-9]{32,}\b',
        }
        
        # Prompt injection patterns
        self.injection_patterns = [
            # Ignore previous instructions
            r'ignore\s+(previous|all|above|prior)\s+instructions?',
            r'disregard\s+(previous|all|above)\s+instructions?',
            
            # System prompt override
            r'you\s+are\s+now',
            r'new\s+instructions?:',
            r'system\s+prompt:',
            r'override\s+instructions?',
            
            # Role play attacks
            r'pretend\s+you\s+are',
            r'act\s+as\s+if',
            r'roleplay\s+as',
            
            # Jailbreak attempts
            r'DAN\s+mode',
            r'developer\s+mode',
            r'god\s+mode',
            r'simulate',
            
            # Output manipulation
            r'print\s+your\s+(instructions?|prompt|system)',
            r'reveal\s+your\s+(instructions?|prompt)',
            r'show\s+me\s+your\s+(rules|guidelines)',
        ]
        
        # Toxic patterns (simplified - production would use ML model)
        self.toxic_patterns = [
            r'\b(fuck|shit|damn|bitch|asshole)\b',
            r'\b(kill|murder|destroy|harm)\s+(yourself|myself)',
            r'\b(hate|despise)\s+you\b',
        ]
        
        console.print("[bold green]✅ Input Guardrails Initialized[/bold green]")
    
    def scan_input(self, text: str, context: Optional[Dict[str, Any]] = None) -> InputScanResult:
        """
        Scan input text for violations.
        
        Args:
            text: Input text to scan
            context: Optional context (user_id, session_id, etc.)
            
        Returns:
            InputScanResult with violations and sanitized text
        """
        violations = []
        pii_detected = []
        sanitized = text
        max_threat_level = ThreatLevel.SAFE
        
        # 1. PII Detection
        if self.enable_pii_redaction:
            pii_results, sanitized = self._detect_and_redact_pii(text)
            pii_detected.extend(pii_results)
            
            if pii_results:
                threat = ThreatLevel.MEDIUM if len(pii_results) > 2 else ThreatLevel.LOW
                max_threat_level = self._max_threat(max_threat_level, threat)
                
                violations.append(InputViolation(
                    violation_type="pii_detected",
                    threat_level=threat,
                    description=f"Detected {len(pii_results)} PII instance(s)",
                    evidence=", ".join([p.pii_type.value for p in pii_results]),
                    recommendation="PII has been redacted automatically"
                ))
        
        # 2. Prompt Injection Detection
        if self.enable_prompt_injection:
            injection_violations = self._detect_prompt_injection(text)
            if injection_violations:
                violations.extend(injection_violations)
                max_threat_level = self._max_threat(max_threat_level, ThreatLevel.CRITICAL)
        
        # 3. Toxic Content Detection
        toxic_violations = self._detect_toxic_content(text)
        if toxic_violations:
            violations.extend(toxic_violations)
            max_threat_level = self._max_threat(max_threat_level, ThreatLevel.HIGH)
        
        # 4. Length Check (prevent token bombs)
        if len(text) > 10000:
            violations.append(InputViolation(
                violation_type="excessive_length",
                threat_level=ThreatLevel.MEDIUM,
                description=f"Input too long: {len(text)} chars (max 10,000)",
                evidence=f"Length: {len(text)}",
                recommendation="Truncate input or split into smaller requests"
            ))
            max_threat_level = self._max_threat(max_threat_level, ThreatLevel.MEDIUM)
        
        # Determine if safe
        critical_violations = [v for v in violations if v.threat_level in (ThreatLevel.HIGH, ThreatLevel.CRITICAL)]
        safe = len(critical_violations) == 0
        
        return InputScanResult(
            safe=safe,
            pii_detected=pii_detected,
            violations=violations,
            sanitized_input=sanitized if self.enable_pii_redaction else None,
            threat_level=max_threat_level
        )
    
    def _detect_and_redact_pii(self, text: str) -> Tuple[List[PIIDetection], str]:
        """Detect and redact PII from text."""
        detections = []
        sanitized = text
        
        for pii_type, pattern in self.pii_patterns.items():
            for match in re.finditer(pattern, text, re.IGNORECASE):
                detections.append(PIIDetection(
                    pii_type=pii_type,
                    value=match.group(0),
                    start=match.start(),
                    end=match.end(),
                    confidence=0.9  # High confidence for regex
                ))
        
        # Sort by position (reverse) to maintain indices during replacement
        detections.sort(key=lambda x: x.start, reverse=True)
        
        # Redact PII
        for detection in detections:
            redaction = f"[REDACTED_{detection.pii_type.value.upper()}]"
            sanitized = sanitized[:detection.start] + redaction + sanitized[detection.end:]
        
        return detections, sanitized
    
    def _detect_prompt_injection(self, text: str) -> List[InputViolation]:
        """Detect prompt injection attempts."""
        violations = []
        
        for pattern in self.injection_patterns:
            matches = list(re.finditer(pattern, text, re.IGNORECASE))
            if matches:
                for match in matches:
                    violations.append(InputViolation(
                        violation_type="prompt_injection",
                        threat_level=ThreatLevel.CRITICAL,
                        description="Potential prompt injection detected",
                        evidence=match.group(0),
                        recommendation="Remove injection attempt or rephrase naturally"
                    ))
        
        return violations
    
    def _detect_toxic_content(self, text: str) -> List[InputViolation]:
        """Detect toxic/harmful content."""
        violations = []
        
        for pattern in self.toxic_patterns:
            matches = list(re.finditer(pattern, text, re.IGNORECASE))
            if matches:
                for match in matches:
                    violations.append(InputViolation(
                        violation_type="toxic_content",
                        threat_level=ThreatLevel.HIGH,
                        description="Toxic or harmful content detected",
                        evidence=match.group(0),
                        recommendation="Remove offensive language or rephrase professionally"
                    ))
        
        return violations
    
    def _max_threat(self, current: ThreatLevel, new: ThreatLevel) -> ThreatLevel:
        """Return the higher threat level."""
        levels = {
            ThreatLevel.SAFE: 0,
            ThreatLevel.LOW: 1,
            ThreatLevel.MEDIUM: 2,
            ThreatLevel.HIGH: 3,
            ThreatLevel.CRITICAL: 4
        }
        
        return current if levels[current] > levels[new] else new


def create_input_guardrails(
    enable_pii_redaction: bool = True,
    enable_prompt_injection: bool = True
) -> InputGuardrails:
    """Factory function to create input guardrails."""
    return InputGuardrails(enable_pii_redaction, enable_prompt_injection)

