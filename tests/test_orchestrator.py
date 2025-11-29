"""
Tests for LangGraph orchestrator.

These tests verify the core LangGraph 1.0 orchestrator functionality.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from src.orchestrator import (
    create_test_generation_orchestrator,
    TestGenerationConfig,
    TestGenerationOrchestratorV2
)


class TestOrchestrator:
    """Test suite for LangGraph orchestrator."""
    
    def test_config_defaults(self):
        """Test TestGenerationConfig default values."""
        config = TestGenerationConfig()
        
        assert config.max_iterations == 10
        assert config.enable_hitl is False
        assert config.enable_summarization is False
        assert config.enable_pii_redaction is False
    
    def test_config_custom_values(self):
        """Test TestGenerationConfig with custom values."""
        config = TestGenerationConfig(
            max_iterations=5,
            enable_hitl=True,
            enable_summarization=True,
            enable_pii_redaction=True
        )
        
        assert config.max_iterations == 5
        assert config.enable_hitl is True
        assert config.enable_summarization is True
        assert config.enable_pii_redaction is True
    
    @patch('src.orchestrator.get_llm_provider')
    @patch('src.orchestrator.get_all_tools')
    @patch('src.orchestrator.create_guard_manager')
    def test_orchestrator_initialization(self, mock_guard, mock_tools, mock_llm):
        """Test orchestrator initializes correctly."""
        # Mock dependencies
        mock_llm_instance = Mock()
        mock_llm_instance.get_langchain_model.return_value = Mock()
        mock_llm.return_value = mock_llm_instance
        mock_tools.return_value = []
        mock_guard.return_value = Mock()
        
        # Create orchestrator
        config = TestGenerationConfig(max_iterations=5)
        orchestrator = TestGenerationOrchestratorV2(config=config)
        
        # Verify initialization
        assert orchestrator.config.max_iterations == 5
        assert orchestrator.agent is not None
        assert orchestrator.guard_manager is not None
    
    @patch('src.orchestrator.get_llm_provider')
    @patch('src.orchestrator.get_all_tools')
    @patch('src.orchestrator.create_guard_manager')
    def test_factory_function(self, mock_guard, mock_tools, mock_llm):
        """Test factory function creates orchestrator."""
        # Mock dependencies
        mock_llm_instance = Mock()
        mock_llm_instance.get_langchain_model.return_value = Mock()
        mock_llm.return_value = mock_llm_instance
        mock_tools.return_value = []
        mock_guard.return_value = Mock()
        
        # Create using factory
        orchestrator = create_test_generation_orchestrator()
        
        # Verify type
        assert isinstance(orchestrator, TestGenerationOrchestratorV2)
        assert orchestrator.agent is not None
    
    def test_post_process_extracts_code_blocks(self):
        """Test post-processing extracts Python code from markdown."""
        from src.orchestrator import TestGenerationOrchestratorV2
        
        # Create mock orchestrator just for post-processing
        with patch('src.orchestrator.get_llm_provider'), \
             patch('src.orchestrator.get_all_tools'), \
             patch('src.orchestrator.create_guard_manager'):
            
            mock_llm = Mock()
            mock_llm.get_langchain_model.return_value = Mock()
            
            with patch('src.orchestrator.get_llm_provider', return_value=mock_llm):
                orchestrator = TestGenerationOrchestratorV2()
                
                # Test markdown extraction
                markdown_tests = """
```python
def test_example():
    assert True
```
                """
                
                result = orchestrator._post_process_tests(markdown_tests, "test.py")
                
                # Should have extracted the code
                assert "def test_example():" in result
                assert "```python" not in result
                assert "```" not in result
    
    def test_post_process_adds_imports(self):
        """Test post-processing adds pytest import."""
        from src.orchestrator import TestGenerationOrchestratorV2
        
        with patch('src.orchestrator.get_llm_provider'), \
             patch('src.orchestrator.get_all_tools'), \
             patch('src.orchestrator.create_guard_manager'):
            
            mock_llm = Mock()
            mock_llm.get_langchain_model.return_value = Mock()
            
            with patch('src.orchestrator.get_llm_provider', return_value=mock_llm):
                orchestrator = TestGenerationOrchestratorV2()
                
                # Test without pytest import
                tests_without_import = """
def test_example():
    assert True
                """
                
                result = orchestrator._post_process_tests(tests_without_import, "test.py")
                
                # Should have added pytest import
                assert "import pytest" in result
    
    def test_post_process_adds_header(self):
        """Test post-processing adds file header."""
        from src.orchestrator import TestGenerationOrchestratorV2
        
        with patch('src.orchestrator.get_llm_provider'), \
             patch('src.orchestrator.get_all_tools'), \
             patch('src.orchestrator.create_guard_manager'):
            
            mock_llm = Mock()
            mock_llm.get_langchain_model.return_value = Mock()
            
            with patch('src.orchestrator.get_llm_provider', return_value=mock_llm):
                orchestrator = TestGenerationOrchestratorV2()
                
                # Test header addition
                tests = "def test_example():\n    assert True"
                result = orchestrator._post_process_tests(tests, "module.py")
                
                # Should have added header
                assert '"""' in result
                assert "Generated tests for module.py" in result
                assert "AgenticTestGenerator" in result


class TestParseResponse:
    """Test suite for LLM response parsing."""
    
    @patch('src.orchestrator.get_llm_provider')
    @patch('src.orchestrator.get_all_tools')
    @patch('src.orchestrator.create_guard_manager')
    def test_parse_json_response(self, mock_guard, mock_tools, mock_llm):
        """Test parsing structured JSON response."""
        mock_llm_instance = Mock()
        mock_llm_instance.get_langchain_model.return_value = Mock()
        mock_llm.return_value = mock_llm_instance
        mock_tools.return_value = []
        mock_guard.return_value = Mock()
        
        orchestrator = TestGenerationOrchestratorV2()
        
        json_response = """
{
    "reasoning": "Testing the function",
    "test_code": {
        "code": "def test_add():\\n    assert add(1, 2) == 3",
        "imports": ["pytest"],
        "test_functions": ["test_add"]
    }
}
        """
        
        result = orchestrator._parse_llm_response(json_response, set())
        
        assert "def test_add():" in result
        assert "assert add(1, 2) == 3" in result
    
    @patch('src.orchestrator.get_llm_provider')
    @patch('src.orchestrator.get_all_tools')
    @patch('src.orchestrator.create_guard_manager')
    def test_parse_json_in_markdown(self, mock_guard, mock_tools, mock_llm):
        """Test parsing JSON wrapped in markdown."""
        mock_llm_instance = Mock()
        mock_llm_instance.get_langchain_model.return_value = Mock()
        mock_llm.return_value = mock_llm_instance
        mock_tools.return_value = []
        mock_guard.return_value = Mock()
        
        orchestrator = TestGenerationOrchestratorV2()
        
        markdown_json = """
Here's the response:

```json
{
    "test_code": {
        "code": "def test_multiply():\\n    assert multiply(2, 3) == 6"
    }
}
```
        """
        
        result = orchestrator._parse_llm_response(markdown_json, set())
        
        assert "def test_multiply():" in result
    
    @patch('src.orchestrator.get_llm_provider')
    @patch('src.orchestrator.get_all_tools')
    @patch('src.orchestrator.create_guard_manager')
    def test_parse_fallback_to_raw(self, mock_guard, mock_tools, mock_llm):
        """Test fallback to raw content when JSON parsing fails."""
        mock_llm_instance = Mock()
        mock_llm_instance.get_langchain_model.return_value = Mock()
        mock_llm.return_value = mock_llm_instance
        mock_tools.return_value = []
        mock_guard.return_value = Mock()
        
        orchestrator = TestGenerationOrchestratorV2()
        
        raw_python = """
```python
def test_divide():
    assert divide(10, 2) == 5
```
        """
        
        result = orchestrator._parse_llm_response(raw_python, set())
        
        assert "def test_divide():" in result
        assert "```python" not in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

