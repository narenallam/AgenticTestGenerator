"""
Critic module for LLM-based test quality review.

This module provides:
- Style and quality review
- Test completeness assessment
- Coverage risk analysis
- PR description generation
- Test anti-pattern detection
"""

from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field
from rich.console import Console

from config.settings import settings
from src.llm_providers import BaseLLMProvider, get_default_provider

console = Console()


class ReviewSeverity(str, Enum):
    """Severity levels for review findings."""
    
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class ReviewCategory(str, Enum):
    """Categories of review findings."""
    
    STYLE = "style"
    COMPLETENESS = "completeness"
    CORRECTNESS = "correctness"
    MAINTAINABILITY = "maintainability"
    PERFORMANCE = "performance"
    ANTI_PATTERN = "anti_pattern"


class ReviewFinding(BaseModel):
    """A single review finding."""
    
    category: ReviewCategory = Field(..., description="Finding category")
    severity: ReviewSeverity = Field(..., description="Severity level")
    line: Optional[int] = Field(default=None, description="Line number if applicable")
    message: str = Field(..., description="Finding description")
    suggestion: Optional[str] = Field(default=None, description="Suggested fix")


class StyleReview(BaseModel):
    """Style review result."""
    
    score: float = Field(..., description="Style score (0-100)")
    findings: List[ReviewFinding] = Field(default_factory=list)
    compliant: bool = Field(..., description="Meets style guidelines")
    summary: str = Field(..., description="Review summary")


class QualityScore(BaseModel):
    """Test quality assessment."""
    
    overall_score: float = Field(..., description="Overall quality score (0-100)")
    coverage_score: float = Field(..., description="Coverage adequacy (0-100)")
    assertion_quality: float = Field(..., description="Assertion quality (0-100)")
    maintainability: float = Field(..., description="Maintainability (0-100)")
    determinism: float = Field(..., description="Determinism (0-100)")
    findings: List[ReviewFinding] = Field(default_factory=list)
    summary: str = Field(..., description="Quality summary")


class CoverageRisk(BaseModel):
    """Coverage risk assessment."""
    
    risk_level: str = Field(..., description="Risk level (low, medium, high)")
    untested_branches: List[str] = Field(default_factory=list)
    missing_edge_cases: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)


class PRDescription(BaseModel):
    """Generated PR description."""
    
    title: str = Field(..., description="PR title")
    summary: str = Field(..., description="Summary of changes")
    tests_added: List[str] = Field(default_factory=list)
    coverage_info: str = Field(..., description="Coverage information")
    notable_changes: List[str] = Field(default_factory=list)
    body: str = Field(..., description="Full PR body in markdown")


class TestCritic:
    """
    LLM-based test quality reviewer.
    
    Uses LLM to assess test quality, completeness, and provide
    actionable feedback for improvement.
    """
    
    def __init__(self, llm_provider: Optional[BaseLLMProvider] = None):
        """
        Initialize test critic.
        
        Args:
            llm_provider: LLM provider (uses default if None)
        """
        self.llm_provider = llm_provider or get_default_provider()
        console.print(f"[green]✓[/green] Critic initialized with {self.llm_provider.provider_name}")
    
    def review_style(self, test_code: str) -> StyleReview:
        """
        Review test code style and formatting.
        
        Args:
            test_code: Test code to review
            
        Returns:
            StyleReview with findings
            
        Example:
            >>> critic = TestCritic()
            >>> review = critic.review_style(test_code)
            >>> print(f"Score: {review.score}, Compliant: {review.compliant}")
        """
        prompt = f"""Review the following test code for style and formatting issues.
Assess compliance with:
- PEP 8 guidelines
- Test naming conventions (test_<function>_<scenario>_<expected>)
- Docstring completeness
- Type hints usage
- Import organization

Test Code:
```python
{test_code}
```

Provide:
1. Style score (0-100)
2. List of specific issues with line numbers
3. Whether code is compliant (score >= 80)
4. Brief summary

Format your response as:
SCORE: <number>
COMPLIANT: <yes/no>
ISSUES:
- Line <num>: <issue> | Suggestion: <fix>
...
SUMMARY: <summary>"""

        try:
            response = self.llm_provider.generate(
                prompt=prompt,
                system="You are an expert code reviewer specializing in Python test quality.",
                temperature=settings.critic_temperature,
                max_tokens=settings.critic_max_tokens
            )
            
            # Parse response
            return self._parse_style_review(response.content)
        
        except Exception as e:
            console.print(f"[yellow]Warning: Style review failed: {e}[/yellow]")
            return StyleReview(
                score=50.0,
                findings=[],
                compliant=False,
                summary="Review failed"
            )
    
    def assess_quality(
        self,
        test_code: str,
        target_code: str
    ) -> QualityScore:
        """
        Assess overall test quality.
        
        Args:
            test_code: Test code
            target_code: Code being tested
            
        Returns:
            QualityScore with detailed assessment
            
        Example:
            >>> critic = TestCritic()
            >>> score = critic.assess_quality(test_code, source_code)
            >>> print(f"Quality: {score.overall_score}/100")
        """
        prompt = f"""Assess the quality of these tests for the target code.

Target Code:
```python
{target_code}
```

Test Code:
```python
{test_code}
```

Evaluate:
1. **Coverage**: Do tests cover all code paths? (0-100)
2. **Assertions**: Are assertions specific and meaningful? (0-100)
3. **Maintainability**: Is code clear and well-organized? (0-100)
4. **Determinism**: Are tests deterministic (no random values, time dependencies)? (0-100)

Identify issues:
- Missing test cases
- Weak assertions (e.g., assert True)
- Hard-coded values
- Lack of mocking for external dependencies
- Non-deterministic elements

Format:
COVERAGE: <score>
ASSERTIONS: <score>
MAINTAINABILITY: <score>
DETERMINISM: <score>
ISSUES:
- <category> | <severity> | Line <num>: <issue> | Suggestion: <fix>
...
SUMMARY: <overall assessment>"""

        try:
            response = self.llm_provider.generate(
                prompt=prompt,
                system="You are an expert test quality reviewer with deep knowledge of best practices.",
                temperature=settings.critic_temperature,
                max_tokens=settings.critic_max_tokens
            )
            
            return self._parse_quality_assessment(response.content)
        
        except Exception as e:
            console.print(f"[yellow]Warning: Quality assessment failed: {e}[/yellow]")
            return QualityScore(
                overall_score=50.0,
                coverage_score=50.0,
                assertion_quality=50.0,
                maintainability=50.0,
                determinism=50.0,
                findings=[],
                summary="Assessment failed"
            )
    
    def analyze_coverage_risk(
        self,
        test_code: str,
        target_code: str,
        cfg_info: Optional[Dict] = None
    ) -> CoverageRisk:
        """
        Analyze coverage risks and gaps.
        
        Args:
            test_code: Test code
            target_code: Target code
            cfg_info: Optional CFG information
            
        Returns:
            CoverageRisk assessment
        """
        cfg_context = ""
        if cfg_info:
            cfg_context = f"\nCFG Info: {cfg_info.get('branches', 'N/A')} branches, {cfg_info.get('paths', 'N/A')} paths"
        
        prompt = f"""Analyze coverage risks for these tests.

Target Code:
```python
{target_code}
```

Test Code:
```python
{test_code}
```
{cfg_context}

Identify:
1. Untested code branches (if/else, loops)
2. Missing edge cases (boundary values, None, empty inputs)
3. Exception paths not tested
4. Coverage risks (areas likely to have bugs)

Provide:
- Risk level: LOW, MEDIUM, or HIGH
- List of untested branches
- List of missing edge cases
- Recommendations for additional tests

Format:
RISK: <level>
UNTESTED_BRANCHES:
- <branch description>
...
MISSING_EDGE_CASES:
- <edge case>
...
RECOMMENDATIONS:
- <recommendation>
..."""

        try:
            response = self.llm_provider.generate(
                prompt=prompt,
                system="You are an expert at identifying test coverage gaps and risks.",
                temperature=settings.critic_temperature,
                max_tokens=settings.critic_max_tokens
            )
            
            return self._parse_coverage_risk(response.content)
        
        except Exception as e:
            console.print(f"[yellow]Warning: Coverage risk analysis failed: {e}[/yellow]")
            return CoverageRisk(
                risk_level="MEDIUM",
                untested_branches=[],
                missing_edge_cases=[],
                recommendations=[]
            )
    
    def generate_pr_body(
        self,
        tests: List[str],
        changes: List[Dict],
        coverage: Optional[float] = None
    ) -> PRDescription:
        """
        Generate PR description for test changes.
        
        Args:
            tests: List of test code strings
            changes: List of code changes
            coverage: Coverage percentage if available
            
        Returns:
            PRDescription with title and body
        """
        changes_summary = "\n".join([
            f"- {change.get('file')}: {change.get('type', 'modified')}"
            for change in changes[:5]
        ])
        
        coverage_str = f"{coverage:.1f}%" if coverage else "N/A"
        
        prompt = f"""Generate a professional PR description for these test additions.

Changes:
{changes_summary}

Number of tests: {len(tests)}
Coverage: {coverage_str}

Generate:
1. Concise PR title (max 60 chars)
2. Summary paragraph
3. List of tests added (brief descriptions)
4. Coverage information
5. Notable changes or improvements

Format as GitHub-flavored Markdown:
# Title

## Summary
<summary paragraph>

## Tests Added
- test_name_1: description
- test_name_2: description
...

## Coverage
<coverage info>

## Notable Changes
- change 1
- change 2
..."""

        try:
            response = self.llm_provider.generate(
                prompt=prompt,
                system="You are an expert at writing clear, professional PR descriptions.",
                temperature=settings.critic_temperature,
                max_tokens=settings.critic_max_tokens
            )
            
            return self._parse_pr_description(response.content)
        
        except Exception as e:
            console.print(f"[yellow]Warning: PR generation failed: {e}[/yellow]")
            return PRDescription(
                title="Add automated tests",
                summary="Automated test generation",
                tests_added=[],
                coverage_info=coverage_str,
                notable_changes=[],
                body="# Add automated tests\n\nAutomated test generation."
            )
    
    def detect_anti_patterns(self, test_code: str) -> List[ReviewFinding]:
        """
        Detect test anti-patterns.
        
        Args:
            test_code: Test code to analyze
            
        Returns:
            List of anti-pattern findings
        """
        anti_patterns = [
            ("assert True", "Meaningless assertion"),
            ("time.sleep", "Time-dependent test (flaky)"),
            ("random.", "Non-deterministic test"),
            ("print(", "Debug print statement left in code"),
            ("# TODO", "Incomplete test implementation"),
            ("pass  # ", "Empty test body"),
        ]
        
        findings = []
        lines = test_code.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            for pattern, message in anti_patterns:
                if pattern in line:
                    findings.append(ReviewFinding(
                        category=ReviewCategory.ANTI_PATTERN,
                        severity=ReviewSeverity.HIGH,
                        line=line_num,
                        message=f"Anti-pattern detected: {message}",
                        suggestion=f"Remove or fix: {line.strip()}"
                    ))
        
        return findings
    
    def _parse_style_review(self, content: str) -> StyleReview:
        """Parse LLM response into StyleReview."""
        import re
        
        # Extract score
        score_match = re.search(r'SCORE:\s*(\d+)', content)
        score = float(score_match.group(1)) if score_match else 50.0
        
        # Extract compliant
        compliant_match = re.search(r'COMPLIANT:\s*(yes|no)', content, re.IGNORECASE)
        compliant = compliant_match.group(1).lower() == 'yes' if compliant_match else score >= 80
        
        # Extract issues
        findings = []
        issue_pattern = r'Line (\d+):\s*(.+?)\s*\|\s*Suggestion:\s*(.+)'
        for match in re.finditer(issue_pattern, content):
            findings.append(ReviewFinding(
                category=ReviewCategory.STYLE,
                severity=ReviewSeverity.MEDIUM,
                line=int(match.group(1)),
                message=match.group(2).strip(),
                suggestion=match.group(3).strip()
            ))
        
        # Extract summary
        summary_match = re.search(r'SUMMARY:\s*(.+?)(?:\n|$)', content, re.DOTALL)
        summary = summary_match.group(1).strip() if summary_match else "Style review completed"
        
        return StyleReview(
            score=score,
            findings=findings,
            compliant=compliant,
            summary=summary
        )
    
    def _parse_quality_assessment(self, content: str) -> QualityScore:
        """Parse LLM response into QualityScore."""
        import re
        
        # Extract scores
        coverage = self._extract_score(content, "COVERAGE")
        assertions = self._extract_score(content, "ASSERTIONS")
        maintainability = self._extract_score(content, "MAINTAINABILITY")
        determinism = self._extract_score(content, "DETERMINISM")
        
        # Calculate overall
        overall = (coverage + assertions + maintainability + determinism) / 4.0
        
        # Extract findings
        findings = []
        # Simplified parsing - in production, use more robust parsing
        
        # Extract summary
        summary_match = re.search(r'SUMMARY:\s*(.+?)(?:\n|$)', content, re.DOTALL)
        summary = summary_match.group(1).strip() if summary_match else "Quality assessment completed"
        
        return QualityScore(
            overall_score=overall,
            coverage_score=coverage,
            assertion_quality=assertions,
            maintainability=maintainability,
            determinism=determinism,
            findings=findings,
            summary=summary
        )
    
    def _extract_score(self, content: str, key: str) -> float:
        """Extract a numeric score from content."""
        import re
        
        pattern = f"{key}:\\s*(\\d+)"
        match = re.search(pattern, content)
        return float(match.group(1)) if match else 50.0
    
    def _parse_coverage_risk(self, content: str) -> CoverageRisk:
        """Parse LLM response into CoverageRisk."""
        import re
        
        # Extract risk level
        risk_match = re.search(r'RISK:\s*(\w+)', content)
        risk_level = risk_match.group(1).upper() if risk_match else "MEDIUM"
        
        # Extract lists (simplified)
        untested = []
        missing = []
        recommendations = []
        
        # Simple line-by-line extraction
        current_section = None
        for line in content.split('\n'):
            line = line.strip()
            if 'UNTESTED_BRANCHES:' in line:
                current_section = 'untested'
            elif 'MISSING_EDGE_CASES:' in line:
                current_section = 'missing'
            elif 'RECOMMENDATIONS:' in line:
                current_section = 'recommendations'
            elif line.startswith('- '):
                item = line[2:].strip()
                if current_section == 'untested':
                    untested.append(item)
                elif current_section == 'missing':
                    missing.append(item)
                elif current_section == 'recommendations':
                    recommendations.append(item)
        
        return CoverageRisk(
            risk_level=risk_level,
            untested_branches=untested,
            missing_edge_cases=missing,
            recommendations=recommendations
        )
    
    def _parse_pr_description(self, content: str) -> PRDescription:
        """Parse LLM response into PRDescription."""
        lines = content.split('\n')
        
        # Extract title (first # heading)
        title = "Add automated tests"
        for line in lines:
            if line.startswith('# '):
                title = line[2:].strip()
                break
        
        # Simplified extraction
        return PRDescription(
            title=title[:60],  # Max 60 chars
            summary="Automated test generation for improved coverage",
            tests_added=[],
            coverage_info="See test execution results",
            notable_changes=[],
            body=content
        )


def create_test_critic(llm_provider: Optional[BaseLLMProvider] = None) -> TestCritic:
    """
    Factory function to create a test critic.
    
    Args:
        llm_provider: Optional LLM provider
        
    Returns:
        Configured TestCritic instance
    """
    return TestCritic(llm_provider=llm_provider)

