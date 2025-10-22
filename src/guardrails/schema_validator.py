"""
Schema Validator for tool parameters.

Validates all tool parameters against JSON schemas with auto-correction.
"""

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, ValidationError as PydanticValidationError
from rich.console import Console

console = Console()


class ValidationError(Exception):
    """Schema validation error."""
    pass


class ValidationResult(BaseModel):
    """Result of schema validation."""
    
    valid: bool = Field(..., description="Validation passed")
    errors: List[str] = Field(default_factory=list, description="Validation errors")
    corrected_params: Optional[Dict[str, Any]] = Field(default=None, description="Auto-corrected parameters")


class ToolSchema(BaseModel):
    """Schema definition for a tool."""
    
    tool_name: str = Field(..., description="Tool name")
    parameters: Dict[str, Dict[str, Any]] = Field(..., description="Parameter schemas")
    required: List[str] = Field(default_factory=list, description="Required parameters")


class SchemaValidator:
    """
    Validates tool parameters against JSON schemas.
    
    Features:
    - Type checking
    - Range validation (min/max)
    - Length validation (strings, arrays)
    - Enum validation
    - Auto-correction where possible
    - Required field checking
    
    Example:
        >>> validator = SchemaValidator()
        >>> result = validator.validate(
        ...     tool="generate_tests",
        ...     params={"max_iterations": 5, "target_coverage": 90}
        ... )
        >>> if not result.valid:
        ...     raise ValidationError(result.errors[0])
    """
    
    def __init__(self):
        """Initialize schema validator with tool schemas."""
        self.schemas = self._define_tool_schemas()
        console.print("[bold green]✅ Schema Validator Initialized[/bold green]")
    
    def validate(
        self,
        tool: str,
        params: Dict[str, Any],
        auto_correct: bool = True
    ) -> ValidationResult:
        """
        Validate tool parameters.
        
        Args:
            tool: Tool name
            params: Parameters to validate
            auto_correct: Attempt to auto-correct invalid params
            
        Returns:
            ValidationResult with errors and corrections
        """
        if tool not in self.schemas:
            return ValidationResult(
                valid=False,
                errors=[f"Unknown tool: {tool}"]
            )
        
        schema = self.schemas[tool]
        errors = []
        corrected = params.copy() if auto_correct else None
        
        # Check required parameters
        for required_param in schema.required:
            if required_param not in params:
                errors.append(f"Missing required parameter: {required_param}")
        
        # Validate each parameter
        for param_name, param_value in params.items():
            if param_name not in schema.parameters:
                errors.append(f"Unknown parameter: {param_name}")
                continue
            
            param_schema = schema.parameters[param_name]
            param_errors = self._validate_parameter(
                param_name,
                param_value,
                param_schema,
                corrected
            )
            errors.extend(param_errors)
        
        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            corrected_params=corrected if auto_correct and len(errors) == 0 else None
        )
    
    def _validate_parameter(
        self,
        name: str,
        value: Any,
        schema: Dict[str, Any],
        corrected: Optional[Dict[str, Any]]
    ) -> List[str]:
        """Validate a single parameter."""
        errors = []
        
        # Type validation
        expected_type = schema.get("type")
        if expected_type:
            type_error = self._check_type(name, value, expected_type)
            if type_error:
                errors.append(type_error)
                return errors  # Can't validate further if type is wrong
        
        # Numeric bounds
        if isinstance(value, (int, float)):
            if "minimum" in schema and value < schema["minimum"]:
                errors.append(f"{name} must be >= {schema['minimum']}, got {value}")
                if corrected is not None:
                    corrected[name] = schema["minimum"]
            
            if "maximum" in schema and value > schema["maximum"]:
                errors.append(f"{name} must be <= {schema['maximum']}, got {value}")
                if corrected is not None:
                    corrected[name] = schema["maximum"]
        
        # String length
        if isinstance(value, str):
            if "minLength" in schema and len(value) < schema["minLength"]:
                errors.append(f"{name} must be at least {schema['minLength']} characters")
            
            if "maxLength" in schema and len(value) > schema["maxLength"]:
                errors.append(f"{name} exceeds max length {schema['maxLength']}")
                if corrected is not None:
                    corrected[name] = value[:schema["maxLength"]]
        
        # Enum validation
        if "enum" in schema and value not in schema["enum"]:
            errors.append(f"{name} must be one of {schema['enum']}, got {value}")
        
        # Array validation
        if isinstance(value, list):
            if "minItems" in schema and len(value) < schema["minItems"]:
                errors.append(f"{name} must have at least {schema['minItems']} items")
            
            if "maxItems" in schema and len(value) > schema["maxItems"]:
                errors.append(f"{name} must have at most {schema['maxItems']} items")
        
        return errors
    
    def _check_type(self, name: str, value: Any, expected_type: str) -> Optional[str]:
        """Check if value matches expected type."""
        type_map = {
            "string": str,
            "integer": int,
            "number": (int, float),
            "boolean": bool,
            "array": list,
            "object": dict,
        }
        
        expected_python_type = type_map.get(expected_type)
        if expected_python_type is None:
            return f"Unknown type: {expected_type}"
        
        if not isinstance(value, expected_python_type):
            return f"{name} must be {expected_type}, got {type(value).__name__}"
        
        return None
    
    def _define_tool_schemas(self) -> Dict[str, ToolSchema]:
        """Define schemas for all tools."""
        return {
            "search_code": ToolSchema(
                tool_name="search_code",
                parameters={
                    "query": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 500,
                    },
                    "max_results": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 50,
                    },
                },
                required=["query"]
            ),
            
            "generate_tests": ToolSchema(
                tool_name="generate_tests",
                parameters={
                    "max_iterations": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 10,
                    },
                    "target_coverage": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 100,
                    },
                    "test_types": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 5,
                    },
                },
                required=[]
            ),
            
            "execute_tests": ToolSchema(
                tool_name="execute_tests",
                parameters={
                    "timeout": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 60,
                    },
                    "with_coverage": {
                        "type": "boolean",
                    },
                },
                required=[]
            ),
            
            "get_code_context": ToolSchema(
                tool_name="get_code_context",
                parameters={
                    "function_name": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 200,
                    },
                    "file_path": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 500,
                    },
                },
                required=["function_name"]
            ),
            
            "check_git_changes": ToolSchema(
                tool_name="check_git_changes",
                parameters={},
                required=[]
            ),
        }


def create_schema_validator() -> SchemaValidator:
    """Factory function to create schema validator."""
    return SchemaValidator()

