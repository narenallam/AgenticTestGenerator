"""
Tests for TestGenerationAgent (compatibility wrapper).

These tests verify that the compatibility wrapper correctly delegates
to the LangGraph orchestrator.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from src.test_agent import TestGenerationAgent, create_test_agent


class TestAgentCompatibility:
    """Test suite for TestGenerationAgent compatibility wrapper."""
    
    @patch('src.test_agent.create_test_generation_orchestrator')
    def test_agent_initialization(self, mock_create_orch):
        """Test agent initializes with orchestrator."""
        mock_orchestrator = Mock()
        mock_create_orch.return_value = mock_orchestrator
        
        agent = TestGenerationAgent(max_iterations=3)
        
        # Verify orchestrator was created
        assert agent.orchestrator is mock_orchestrator
        assert agent.max_iterations == 3
        mock_create_orch.assert_called_once()
    
    @patch('src.test_agent.create_test_generation_orchestrator')
    def test_agent_delegates_to_orchestrator(self, mock_create_orch):
        """Test agent delegates generate_tests to orchestrator."""
        mock_orchestrator = Mock()
        mock_orchestrator.generate_tests.return_value = "def test_example(): pass"
        mock_create_orch.return_value = mock_orchestrator
        
        agent = TestGenerationAgent(max_iterations=5)
        
        # Call generate_tests
        result = agent.generate_tests(
            target_code="def add(a, b): return a + b",
            file_path="calculator.py"
        )
        
        # Verify delegation
        mock_orchestrator.generate_tests.assert_called_once_with(
            target_code="def add(a, b): return a + b",
            file_path="calculator.py",
            function_name=None,
            context=""
        )
        assert result == "def test_example(): pass"
    
    @patch('src.test_agent.create_test_generation_orchestrator')
    def test_factory_function(self, mock_create_orch):
        """Test factory function creates agent."""
        mock_orchestrator = Mock()
        mock_create_orch.return_value = mock_orchestrator
        
        agent = create_test_agent(max_iterations=7)
        
        # Verify creation
        assert isinstance(agent, TestGenerationAgent)
        assert agent.max_iterations == 7
    
    @patch('src.test_agent.create_test_generation_orchestrator')
    def test_batch_generation(self, mock_create_orch):
        """Test batch test generation."""
        mock_orchestrator = Mock()
        mock_orchestrator.generate_tests.side_effect = [
            "def test_func1(): pass",
            "def test_func2(): pass"
        ]
        mock_create_orch.return_value = mock_orchestrator
        
        agent = TestGenerationAgent()
        
        # Create mock contexts
        from src.rag_retrieval import CodeContext
        contexts = [
            CodeContext(
                target_code="def func1(): pass",
                related_code=[],
                dependencies=[],
                metadata={'function_name': 'func1', 'file_path': 'mod1.py'}
            ),
            CodeContext(
                target_code="def func2(): pass",
                related_code=[],
                dependencies=[],
                metadata={'function_name': 'func2', 'file_path': 'mod2.py'}
            )
        ]
        
        # Generate batch
        results = agent.generate_batch_tests(contexts)
        
        # Verify results
        assert len(results) == 2
        assert 'func1' in results
        assert 'func2' in results
        assert results['func1'] == "def test_func1(): pass"
        assert results['func2'] == "def test_func2(): pass"


class TestBackwardCompatibility:
    """Test backward compatibility with old API."""
    
    @patch('src.test_agent.create_test_generation_orchestrator')
    def test_deprecated_parameters_ignored(self, mock_create_orch):
        """Test deprecated parameters are accepted but ignored."""
        mock_orchestrator = Mock()
        mock_create_orch.return_value = mock_orchestrator
        
        # Old API included retriever and llm_provider
        agent = TestGenerationAgent(
            retriever=Mock(),  # Deprecated
            max_iterations=5,
            llm_provider=Mock()  # Deprecated
        )
        
        # Should still work
        assert agent.orchestrator is mock_orchestrator
        assert agent.max_iterations == 5
    
    @patch('src.test_agent.create_test_generation_orchestrator')
    def test_test_type_parameter_deprecated(self, mock_create_orch):
        """Test test_type parameter is accepted but ignored."""
        mock_orchestrator = Mock()
        mock_orchestrator.generate_tests.return_value = "def test_x(): pass"
        mock_create_orch.return_value = mock_orchestrator
        
        agent = TestGenerationAgent()
        
        # Old API included test_type parameter
        from src.prompts import TestType
        result = agent.generate_tests(
            target_code="def x(): pass",
            file_path="x.py",
            test_type=TestType.UNIT  # Deprecated but accepted
        )
        
        # Should still work
        assert result == "def test_x(): pass"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

