"""
Constitutional AI for self-verification and safety.

Implements Constitutional AI principles:
- Self-verification loops
- Chain-of-thought safety checks
- Harm reduction prompts
- Principle-based evaluation

This module adds a self-critique layer where the LLM evaluates its own outputs
against constitutional principles before finalizing them.
"""

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from rich.console import Console

from config.settings import settings
from src.llm_providers import get_default_provider

console = Console()


class ConstitutionalPrinciple(str, Enum):
    """Constitutional principles for AI behavior."""
    
    HELPFUL = "helpful"  # Be helpful and informative
    HARMLESS = "harmless"  # Avoid harmful content
    HONEST = "honest"  # Be truthful, don't hallucinate
    SAFE = "safe"  # Generate safe, secure code
    RESPECTFUL = "respectful"  # Be respectful and professional
    LEGAL = "legal"  # Follow laws and regulations
    DETERMINISTIC = "deterministic"  # Tests must be deterministic
    ISOLATED = "isolated"  # Tests must be isolated


class ViolationSeverity(str, Enum):
    """Severity of principle violations."""
    
    MINOR = "minor"
    MODERATE = "moderate"
    MAJOR = "major"
    CRITICAL = "critical"


class PrincipleViolation(BaseModel):
    """A violation of a constitutional principle."""
    
    principle: ConstitutionalPrinciple = Field(..., description="Violated principle")
    severity: ViolationSeverity = Field(..., description="Violation severity")
    description: str = Field(..., description="What was violated")
    evidence: str = Field(..., description="Evidence of violation")
    suggestion: str = Field(..., description="How to fix")


class SelfVerificationResult(BaseModel):
    """Result of self-verification check."""
    
    passes: bool = Field(..., description="Passes all checks")
    score: float = Field(..., description="Overall score (0-1)")
    violations: List[PrincipleViolation] = Field(
        default_factory=list,
        description="Principle violations"
    )
    reasoning: str = Field(..., description="LLM's reasoning")
    revised_output: Optional[str] = Field(default=None, description="Revised output if needed")


class ConstitutionalAI:
    """
    Constitutional AI for self-verification and safety.
    
    This system implements Constitutional AI principles where the LLM
    evaluates its own outputs against a set of principles before finalizing.
    
    Process:
    1. LLM generates initial output
    2. Self-critique: LLM evaluates output against principles
    3. If violations found, LLM revises output
    4. Repeat until passes or max iterations
    
    Example:
        >>> constitution = ConstitutionalAI()
        >>> result = constitution.verify_output(
        ...     output="import os; os.system('rm -rf /')",
        ...     output_type="code",
        ...     principles=[ConstitutionalPrinciple.SAFE]
        ... )
        >>> if not result.passes:
        ...     output = result.revised_output or output
    """
    
    def __init__(
        self,
        llm_provider=None,
        max_revisions: int = 3,
        min_score: float = 0.8
    ):
        """
        Initialize Constitutional AI.
        
        Args:
            llm_provider: LLM provider (defaults to configured provider)
            max_revisions: Maximum revision attempts
            min_score: Minimum score to pass (0-1)
        """
        self.llm_provider = llm_provider or get_default_provider()
        self.max_revisions = max_revisions
        self.min_score = min_score
        
        # Principle descriptions
        self.principle_descriptions = {
            ConstitutionalPrinciple.HELPFUL: "Output should be helpful, informative, and directly address the task",
            ConstitutionalPrinciple.HARMLESS: "Output should not contain harmful, toxic, or offensive content",
            ConstitutionalPrinciple.HONEST: "Output should be truthful and not contain hallucinations or false claims",
            ConstitutionalPrinciple.SAFE: "Generated code should be safe, secure, and not perform dangerous operations",
            ConstitutionalPrinciple.RESPECTFUL: "Output should be respectful, professional, and inclusive",
            ConstitutionalPrinciple.LEGAL: "Output should comply with laws, regulations, and licensing requirements",
            ConstitutionalPrinciple.DETERMINISTIC: "Tests must be deterministic (no random, time-based, or network operations)",
            ConstitutionalPrinciple.ISOLATED: "Tests must be isolated (use mocks, no real databases or file system)",
        }
        
        console.print("[bold green]✅ Constitutional AI Initialized[/bold green]")
    
    def verify_output(
        self,
        output: str,
        output_type: str = "text",
        principles: Optional[List[ConstitutionalPrinciple]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> SelfVerificationResult:
        """
        Verify output against constitutional principles.
        
        Args:
            output: Output to verify
            output_type: Type of output (text, code, etc.)
            principles: Principles to check (default: all)
            context: Optional context
            
        Returns:
            SelfVerificationResult with violations and revisions
        """
        if principles is None:
            # Default principles based on output type
            if output_type == "code":
                principles = [
                    ConstitutionalPrinciple.SAFE,
                    ConstitutionalPrinciple.DETERMINISTIC,
                    ConstitutionalPrinciple.ISOLATED,
                    ConstitutionalPrinciple.LEGAL
                ]
            else:
                principles = [
                    ConstitutionalPrinciple.HELPFUL,
                    ConstitutionalPrinciple.HARMLESS,
                    ConstitutionalPrinciple.HONEST,
                    ConstitutionalPrinciple.RESPECTFUL
                ]
        
        current_output = output
        
        for iteration in range(self.max_revisions):
            # Run self-critique
            result = self._critique_output(current_output, principles, output_type)
            
            if result.passes:
                console.print(f"[green]✓ Passed constitutional check (score: {result.score:.2f})[/green]")
                return result
            
            # If not passed and we have a revision, try again
            if result.revised_output:
                console.print(f"[yellow]↻ Revision {iteration + 1}: {len(result.violations)} violations[/yellow]")
                current_output = result.revised_output
            else:
                # No revision provided, return failure
                break
        
        # Failed after max iterations
        console.print(f"[red]✗ Failed constitutional check after {self.max_revisions} attempts[/red]")
        result.passes = False
        return result
    
    def _critique_output(
        self,
        output: str,
        principles: List[ConstitutionalPrinciple],
        output_type: str
    ) -> SelfVerificationResult:
        """Run self-critique on output."""
        
        # Build critique prompt
        principles_text = "\n".join([
            f"- {p.value.upper()}: {self.principle_descriptions[p]}"
            for p in principles
        ])
        
        critique_prompt = f"""
You are a Constitutional AI critic evaluating the following {output_type} output against constitutional principles.

CONSTITUTIONAL PRINCIPLES:
{principles_text}

OUTPUT TO EVALUATE:
```
{output[:2000]}  
```

TASK:
1. Evaluate the output against EACH principle above
2. For each violation, provide:
   - Principle violated
   - Severity (MINOR/MODERATE/MAJOR/CRITICAL)
   - Description of the violation
   - Evidence (specific part of output)
   - Suggestion for improvement

3. Provide an overall score (0-1) where 1 is perfect compliance
4. If score < 0.8, provide a REVISED output that fixes all violations

RESPOND IN JSON FORMAT:
{{
  "score": 0.95,
  "violations": [
    {{
      "principle": "safe",
      "severity": "major",
      "description": "Uses eval() which is dangerous",
      "evidence": "eval(user_input)",
      "suggestion": "Remove eval() and use ast.literal_eval() or safer alternatives"
    }}
  ],
  "reasoning": "The output mostly complies but has one major safety issue...",
  "revised_output": "... corrected version if needed ..."
}}

Be thorough and honest in your critique.
"""
        
        try:
            # Call LLM for self-critique
            response = self.llm_provider.generate(
                prompt=critique_prompt,
                system="You are a rigorous Constitutional AI critic. Be thorough and honest.",
                temperature=0.1,  # Very low for consistency
                max_tokens=2048
            )
            
            # Parse response (simplified - production would use structured output)
            import json
            
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                critique_data = json.loads(json_match.group(0))
            else:
                # Fallback: manual parsing
                critique_data = {
                    "score": 0.5,
                    "violations": [],
                    "reasoning": response,
                    "revised_output": None
                }
            
            # Convert to our format
            violations = []
            for v in critique_data.get("violations", []):
                try:
                    violations.append(PrincipleViolation(
                        principle=ConstitutionalPrinciple(v.get("principle", "safe")),
                        severity=ViolationSeverity(v.get("severity", "moderate")),
                        description=v.get("description", ""),
                        evidence=v.get("evidence", ""),
                        suggestion=v.get("suggestion", "")
                    ))
                except Exception:
                    # Skip invalid violations
                    pass
            
            score = float(critique_data.get("score", 0.5))
            passes = score >= self.min_score and len(violations) == 0
            
            return SelfVerificationResult(
                passes=passes,
                score=score,
                violations=violations,
                reasoning=critique_data.get("reasoning", ""),
                revised_output=critique_data.get("revised_output")
            )
        
        except Exception as e:
            console.print(f"[red]Error in constitutional critique: {e}[/red]")
            
            # Return conservative failure
            return SelfVerificationResult(
                passes=False,
                score=0.0,
                violations=[],
                reasoning=f"Error during critique: {str(e)}"
            )


def create_constitutional_ai(
    llm_provider=None,
    max_revisions: int = 3,
    min_score: float = 0.8
) -> ConstitutionalAI:
    """Factory function to create Constitutional AI."""
    return ConstitutionalAI(llm_provider, max_revisions, min_score)


# Import re for JSON extraction
import re

