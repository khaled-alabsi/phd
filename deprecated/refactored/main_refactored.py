#!/usr/bin/env python3
"""
Tennessee Eastman Process (TEP) Anomaly Detection - Refactored Main Script

This script demonstrates the refactored TEP analysis workflow using the modular
architecture. It provides a clean, maintainable way to perform the complete
analysis pipeline.

Usage:
    python main_refactored.py [--config CONFIG_FILE] [--skip-training] [--skip-visualization]

Author: TEP Analysis Team
Date: 2024
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from te_utils import check_python_version, VERSION
from te_data_loader import TEPDataLoader
from te_models import TEPModelTrainer
from te_visualization import TEPVisualizer


def setup_logging(log_level: str = "INFO") -> None:
    """
    Set up logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'output/{VERSION}/tep_analysis.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )


def load_config(config_file: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from file or use defaults.
    
    Args:
        config_file: Path to configuration file
        
    Returns:
        Dictionary containing configuration
    """
    # Default configuration
    default_config = {
        'data_dir': 'data/',
        'output_dir': f'output/{VERSION}/',
        'models_to_train': ['random_forest', 'xgboost', 'lightgbm', 'neural_network'],
        'binary_classification': True,
        'apply_pca': False,
        'pca_components': None,
        'normalize_features': True,
        'neural_network_architecture': 'simple',
        'save_models': True,
        'save_plots': True,
        'create_dashboard': True
    }
    
    # TODO: Add configuration file loading logic here
    # For now, return default configuration
    return default_config


def main_workflow(config: Dict[str, Any]) -> None:
    """
    Execute the main TEP analysis workflow.
    
    Args:
        config: Configuration dictionary
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting TEP anomaly detection analysis workflow")
    
    try:
        # Step 1: Data Loading and Exploration
        logger.info("Step 1: Loading and exploring data")
        data_loader = TEPDataLoader(data_dir=config['data_dir'])
        
        # Load data
        train_data, test_data = data_loader.load_data()
        
        # Explore data
        exploration_results = data_loader.explore_data()
        logger.info(f"Data exploration completed. Shape: {exploration_results['shape']}")
        
        # Display data summary
        print("\n" + "="*60)
        print(data_loader.get_data_summary())
        print("="*60 + "\n")
        
        # Create data visualizations
        logger.info("Creating data visualizations")
        data_loader.visualize_data_distribution(save_plots=config['save_plots'])
        data_loader.apply_tsne_visualization(save_plot=config['save_plots'])
        
        # Step 2: Data Preprocessing
        logger.info("Step 2: Preprocessing data")
        processed_train, processed_test = data_loader.preprocess_data(
            normalize=config['normalize_features'],
            apply_pca=config['apply_pca'],
            n_components=config['pca_components']
        )
        
        # Step 3: Model Training
        logger.info("Step 3: Training models")
        model_trainer = TEPModelTrainer(output_dir=config['output_dir'])
        
        # Prepare data for training
        X_train, X_test, y_train, y_test = model_trainer.prepare_data(
            processed_train, 
            processed_test,
            binary_classification=config['binary_classification']
        )
        
        # Store true labels for evaluation
        model_trainer.model_results = {
            'true_labels': y_test
        }
        
        # Train traditional models
        traditional_models = model_trainer.train_traditional_models(
            X_train, y_train, 
            models_to_train=[m for m in config['models_to_train'] if m != 'neural_network']
        )
        
        # Train neural network if requested
        if 'neural_network' in config['models_to_train']:
            n_classes = len(np.unique(y_train))
            model_trainer.train_neural_network(
                X_train, y_train, 
                n_classes=n_classes,
                architecture=config['neural_network_architecture']
            )
        
        # Step 4: Model Evaluation
        logger.info("Step 4: Evaluating models")
        evaluation_results = model_trainer.evaluate_models(X_test, y_test)
        
        # Step 5: Visualization and Reporting
        logger.info("Step 5: Creating visualizations and reports")
        visualizer = TEPVisualizer(output_dir=config['output_dir'])
        
        # Create performance comparison plots
        visualizer.plot_model_performance_comparison(evaluation_results, save_plot=config['save_plots'])
        
        # Create confusion matrices
        visualizer.plot_confusion_matrices_grid(model_trainer.model_results, y_test, save_plots=config['save_plots'])
        
        # Create ROC curves
        visualizer.plot_roc_curves(model_trainer.model_results, y_test, save_plot=config['save_plots'])
        
        # Create fault performance analysis
        visualizer.plot_fault_performance_analysis(model_trainer.model_results, y_test, save_plots=config['save_plots'])
        
        # Create performance summary table
        summary_table = visualizer.create_performance_summary_table(evaluation_results, save_table=config['save_plots'])
        
        # Create interactive dashboard if requested
        if config['create_dashboard']:
            visualizer.create_interactive_dashboard(model_trainer.model_results, y_test)
        
        # Generate comprehensive report
        report = visualizer.generate_comprehensive_report(model_trainer.model_results, y_test, save_report=config['save_plots'])
        
        # Step 6: Save Models
        if config['save_models']:
            logger.info("Step 6: Saving trained models")
            model_trainer.save_models()
        
        # Step 7: Final Summary
        logger.info("Step 7: Analysis completed successfully")
        best_model_name, best_model = model_trainer.get_best_model()
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETED SUCCESSFULLY")
        print("="*60)
        print(f"Best performing model: {best_model_name}")
        if best_model_name:
            best_score = model_trainer.best_score
            print(f"Best F1-Score: {best_score:.4f}")
        print(f"Results saved to: {config['output_dir']}")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Error in main workflow: {e}", exc_info=True)
        raise


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="TEP Anomaly Detection Analysis - Refactored Version",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main_refactored.py                           # Run full analysis
    python main_refactored.py --skip-training           # Skip model training
    python main_refactored.py --skip-visualization      # Skip visualization
    python main_refactored.py --config config.yaml      # Use custom config
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration file (YAML/JSON)'
    )
    
    parser.add_argument(
        '--skip-training',
        action='store_true',
        help='Skip model training (load existing models)'
    )
    
    parser.add_argument(
        '--skip-visualization',
        action='store_true',
        help='Skip visualization and reporting'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO',
        help='Set logging level'
    )
    
    return parser.parse_args()


def main() -> None:
    """
    Main entry point for the TEP analysis script.
    """
    # Check Python version
    check_python_version()
    
    # Parse arguments
    args = parse_arguments()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Load configuration
    config = load_config(args.config)
    
    # Modify config based on arguments
    if args.skip_training:
        config['models_to_train'] = []
        config['save_models'] = False
    
    if args.skip_visualization:
        config['save_plots'] = False
        config['create_dashboard'] = False
    
    # Create output directory
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # Execute main workflow
    main_workflow(config)


if __name__ == "__main__":
    # Import numpy here to avoid import issues
    import numpy as np
    
    try:
        main()
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nFatal error: {e}")
        sys.exit(1)
