"""
Main evaluation runner.

Orchestrates the entire evaluation process:
1. Load datasets
2. Run evaluations
3. Store results
4. Check for regressions
5. Generate reports
"""

from pathlib import Path
from typing import Dict, List, Optional

from .base import EvalSuite, generate_eval_id
from .agents.agent_evals import CoderEvaluator, CriticEvaluator, PlannerEvaluator
from .datasets.dataset_manager import DatasetManager
from .metrics.multi_language import GoalAchievementCalculator, evaluate_multi_language
from .metrics.safety_evals import RedTeamEvaluator, SafetyGuardrailsEvaluator
from .metrics.test_quality import TestQualityEvaluator
from .reporters.report_generator import MultiReporter
from .reporters.result_storage import RegressionDetector, ResultStorage, TrendAnalyzer


# ═══════════════════════════════════════════════════════════════════════════
# Evaluation Runner
# ═══════════════════════════════════════════════════════════════════════════


class EvalRunner:
    """Main evaluation runner."""
    
    def __init__(
        self,
        workspace_dir: Path,
        db_path: Optional[Path] = None,
    ):
        """
        Initialize evaluation runner.
        
        Args:
            workspace_dir: Workspace directory for datasets and results
            db_path: Path to SQLite database (if None, uses workspace_dir/evals.db)
        """
        self.workspace_dir = Path(workspace_dir)
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        
        self.dataset_dir = self.workspace_dir / "datasets"
        self.results_dir = self.workspace_dir / "results"
        
        # Initialize components
        if db_path is None:
            db_path = self.workspace_dir / "evals.db"
        
        self.storage = ResultStorage(db_path)
        self.regression_detector = RegressionDetector(self.storage)
        self.trend_analyzer = TrendAnalyzer(self.storage)
        self.dataset_manager = DatasetManager(self.dataset_dir)
        self.reporter = MultiReporter()
        
        # Initialize evaluators
        self.evaluators = {
            "test_quality": TestQualityEvaluator(),
            "planner": PlannerEvaluator(),
            "coder": CoderEvaluator(),
            "critic": CriticEvaluator(),
            "safety": SafetyGuardrailsEvaluator(),
            "red_team": RedTeamEvaluator(),
        }
    
    def run_full_evaluation(
        self,
        suite_name: str = "full_evaluation",
        dataset_name: str = "mixed",
        check_regression: bool = True,
        save_reports: bool = True,
    ) -> EvalSuite:
        """
        Run full evaluation suite.
        
        Args:
            suite_name: Name for this evaluation suite
            dataset_name: Dataset to use for evaluation
            check_regression: Whether to check for regressions
            save_reports: Whether to save reports to disk
        
        Returns:
            EvalSuite with all results
        """
        suite_id = generate_eval_id("suite")
        suite = EvalSuite(
            suite_id=suite_id,
            name=suite_name,
            description="Comprehensive evaluation of the agentic test generation system",
        )
        
        print(f"\n🚀 Starting full evaluation suite: {suite_name}")
        print(f"   Suite ID: {suite_id}")
        print(f"   Dataset: {dataset_name}\n")
        
        # 1. Safety evaluations
        print("🔒 Running safety evaluations...")
        safety_result = self.evaluators["safety"].evaluate()
        suite.add_result(safety_result)
        print(f"   Safety score: {(safety_result.score or 0.0) * 100:.1f}%")
        
        # 2. Red team testing
        print("🔴 Running red team tests...")
        red_team_result = self.evaluators["red_team"].evaluate()
        suite.add_result(red_team_result)
        print(f"   Red team score: {(red_team_result.score or 0.0) * 100:.1f}%")
        
        # 3. Test quality evaluations (sample from dataset)
        print(f"✅ Running test quality evaluations on dataset '{dataset_name}'...")
        try:
            dataset = self.dataset_manager.load_dataset(dataset_name)
            
            # Evaluate a sample (first 3 entries)
            for i, entry in enumerate(dataset[:3]):
                print(f"   Evaluating: {entry.name}...")
                
                # For demonstration, assume we have generated tests
                # In practice, this would call the actual test generation system
                test_code = self._generate_sample_tests(entry.source_code, entry.language)
                
                test_quality_result = self.evaluators["test_quality"].evaluate(
                    test_code=test_code,
                    source_code=entry.source_code,
                )
                suite.add_result(test_quality_result)
                
                print(f"      Quality score: {(test_quality_result.score or 0.0) * 100:.1f}%")
        
        except FileNotFoundError:
            print(f"   ⚠️ Dataset '{dataset_name}' not found, skipping test quality evals")
        
        # Calculate overall suite score
        suite.calculate_overall_score()
        suite.mark_completed()
        
        print(f"\n📊 Evaluation complete!")
        print(f"   Overall score: {(suite.overall_score or 0.0) * 100:.1f}%")
        print(f"   Quality level: {suite.quality_level.value if suite.quality_level else 'N/A'}")
        print(f"   Duration: {suite.duration_seconds:.2f}s")
        
        # Save to database
        self.storage.save_suite(suite)
        print(f"\n💾 Results saved to database")
        
        # Check for regressions
        if check_regression:
            print(f"\n🔍 Checking for regressions...")
            self._check_regressions(suite)
        
        # Generate and save reports
        if save_reports:
            print(f"\n📄 Generating reports...")
            self._generate_reports(suite)
        
        # Print console report
        self.reporter.print_console(suite)
        
        return suite
    
    def evaluate_generated_tests(
        self,
        test_code: str,
        source_code: str,
        language: str = "python",
        check_goals: bool = True,
    ) -> Dict[str, any]:
        """
        Evaluate generated tests against our goals (90% coverage, 90% pass rate).
        
        Args:
            test_code: Generated test code
            source_code: Source code being tested
            language: Programming language
            check_goals: Whether to check 90/90 goals
        
        Returns:
            Dictionary with evaluation results
        """
        print(f"\n🔬 Evaluating generated tests ({language})...")
        
        # Run test quality evaluation
        test_quality_result = self.evaluators["test_quality"].evaluate(
            test_code=test_code,
            source_code=source_code,
        )
        
        # Run multi-language evaluation (coverage + pass rate)
        multi_lang_results = evaluate_multi_language(test_code, source_code, language)
        
        # Check goal achievement
        goal_results = GoalAchievementCalculator.calculate_goal_score(
            coverage=multi_lang_results["coverage"],
            pass_rate=multi_lang_results["pass_rate"],
        )
        
        # Combine results
        results = {
            "test_quality": {
                "score": test_quality_result.score,
                "metrics": {name: metric.value for name, metric in test_quality_result.metrics.items()},
            },
            "coverage": multi_lang_results["coverage"],
            "pass_rate": multi_lang_results["pass_rate"],
            "goal_achievement": goal_results,
            "language": language,
        }
        
        # Print summary
        print(f"   Test Quality: {(test_quality_result.score or 0.0) * 100:.1f}%")
        print(f"   Coverage: {multi_lang_results['coverage'] * 100:.1f}% (Goal: 90%)")
        print(f"   Pass Rate: {multi_lang_results['pass_rate'] * 100:.1f}% (Goal: 90%)")
        
        if check_goals:
            if goal_results["both_goals_met"]:
                print(f"   ✅ Both goals achieved!")
            else:
                if not goal_results["coverage_met"]:
                    print(f"   ⚠️ Coverage gap: {goal_results['coverage_gap'] * 100:.1f}%")
                if not goal_results["pass_rate_met"]:
                    print(f"   ⚠️ Pass rate gap: {goal_results['pass_rate_gap'] * 100:.1f}%")
        
        return results
    
    def set_baseline(self, eval_name: str) -> None:
        """
        Set baseline for an evaluation using the latest result.
        
        Args:
            eval_name: Evaluation name
        """
        latest = self.storage.get_latest_result(eval_name)
        
        if not latest:
            print(f"❌ No results found for '{eval_name}'")
            return
        
        self.storage.set_baseline(eval_name, latest)
        print(f"✅ Baseline set for '{eval_name}': {(latest.score or 0.0) * 100:.1f}%")
    
    def analyze_trend(self, eval_name: str, window: int = 10) -> Dict:
        """
        Analyze trend for an evaluation.
        
        Args:
            eval_name: Evaluation name
            window: Number of recent results to analyze
        
        Returns:
            Trend analysis results
        """
        trend = self.trend_analyzer.analyze_trend(eval_name, window=window)
        
        print(f"\n📈 Trend Analysis: {eval_name}")
        print(f"   Trend: {trend['trend']}")
        print(f"   Direction: {trend['direction']}")
        if 'recent_score' in trend and trend['recent_score']:
            print(f"   Recent Score: {trend['recent_score'] * 100:.1f}%")
        if 'oldest_score' in trend and trend['oldest_score']:
            print(f"   Oldest Score: {trend['oldest_score'] * 100:.1f}%")
            print(f"   Change: {trend['change'] * 100:+.1f}%")
        
        return trend
    
    def _check_regressions(self, suite: EvalSuite) -> None:
        """Check for regressions in suite results."""
        regressions_found = False
        
        for result in suite.eval_results:
            regression = self.regression_detector.check_regression(result)
            
            if regression["has_regression"]:
                regressions_found = True
                print(f"   ⚠️ REGRESSION in {result.eval_name}:")
                for reg in regression["regressions"]:
                    print(f"      {reg['metric']}: {reg['current']:.3f} → {reg['baseline']:.3f} (Δ {reg['delta_percent']:.1f}%)")
        
        if not regressions_found:
            print(f"   ✅ No regressions detected")
    
    def _generate_reports(self, suite: EvalSuite) -> None:
        """Generate and save reports in all formats."""
        self.reporter.generate_all(suite, output_dir=self.results_dir)
        
        print(f"   ✅ Console report: {self.results_dir / f'{suite.suite_id}.txt'}")
        print(f"   ✅ Markdown report: {self.results_dir / f'{suite.suite_id}.md'}")
        print(f"   ✅ JSON report: {self.results_dir / f'{suite.suite_id}.json'}")
    
    def _generate_sample_tests(self, source_code: str, language: str) -> str:
        """
        Generate sample tests for demonstration.
        
        In practice, this would call the actual test generation system.
        """
        # Placeholder: return simple test structure
        if language == "python":
            return f"""import pytest

def test_basic():
    # Sample test
    assert True

def test_edge_case():
    # Edge case test
    assert True
"""
        elif language == "java":
            return """
import org.junit.Test;
import static org.junit.Assert.*;

public class SampleTest {
    @Test
    public void testBasic() {
        assertTrue(true);
    }
}
"""
        else:
            return """
test('basic test', () => {
    expect(true).toBe(true);
});
"""


# ═══════════════════════════════════════════════════════════════════════════
# CLI Entry Point
# ═══════════════════════════════════════════════════════════════════════════


def main():
    """CLI entry point for evaluations."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run evaluations for the test generation system")
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path("evals"),
        help="Workspace directory for evaluations",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="mixed",
        help="Dataset to use for evaluation",
    )
    parser.add_argument(
        "--no-regression-check",
        action="store_true",
        help="Skip regression checking",
    )
    parser.add_argument(
        "--no-reports",
        action="store_true",
        help="Skip report generation",
    )
    parser.add_argument(
        "--setup",
        action="store_true",
        help="Setup: create default datasets",
    )
    
    args = parser.parse_args()
    
    runner = EvalRunner(workspace_dir=args.workspace)
    
    if args.setup:
        print("🔨 Setting up evaluation system...")
        from .datasets.dataset_manager import create_default_datasets
        create_default_datasets(runner.dataset_dir)
        print("✅ Setup complete!")
        return
    
    # Run full evaluation
    suite = runner.run_full_evaluation(
        dataset_name=args.dataset,
        check_regression=not args.no_regression_check,
        save_reports=not args.no_reports,
    )
    
    # Exit with appropriate code
    if suite.overall_score and suite.overall_score >= 0.70:
        exit(0)
    else:
        exit(1)


if __name__ == "__main__":
    main()

