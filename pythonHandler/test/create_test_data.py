#!/usr/bin/env python3
import os
import pandas as pd
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_test_data():
    """Create test data files if they don't exist."""
    # Get the test data directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pythonHandler_dir = os.path.dirname(script_dir)
    test_data_dir = os.path.join(pythonHandler_dir, 'test_data')
    
    # Create the test data directory if it doesn't exist
    os.makedirs(test_data_dir, exist_ok=True)
    
    # Define file paths
    training_file = os.path.join(test_data_dir, 'training_test.csv')
    categorize_file = os.path.join(test_data_dir, 'categorise_test.csv')
    
    # Check if training data file exists
    if not os.path.exists(training_file):
        logger.info(f"Creating training data file: {training_file}")
        
        # Create sample training data
        training_data = pd.DataFrame({
            'Narrative': [
                'Grocery shopping at Woolworths',
                'Uber ride to airport',
                'Monthly Netflix subscription',
                'Electricity bill payment',
                'Salary deposit',
                'Restaurant dinner',
                'Gym membership fee',
                'Amazon purchase',
                'Mobile phone bill',
                'Petrol station'
            ],
            'Category': [
                'Groceries',
                'Transport',
                'Entertainment',
                'Utilities',
                'Income',
                'Dining',
                'Health & Fitness',
                'Shopping',
                'Utilities',
                'Transport'
            ]
        })
        
        # Save to CSV
        training_data.to_csv(training_file, index=False)
        logger.info(f"Created training data file with {len(training_data)} records")
    else:
        logger.info(f"Training data file already exists: {training_file}")
    
    # Check if categorization data file exists
    if not os.path.exists(categorize_file):
        logger.info(f"Creating categorization data file: {categorize_file}")
        
        # Create sample categorization data
        categorize_data = pd.DataFrame({
            'Date': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05'],
            'Amount': [45.50, 12.99, 89.95, 7.50, 120.00],
            'Description': [
                'Coles supermarket purchase',
                'Spotify monthly subscription',
                'Water bill payment',
                'Coffee shop purchase',
                'Hardware store items'
            ],
            'Currency': ['AUD', 'AUD', 'AUD', 'AUD', 'AUD']
        })
        
        # Save to CSV
        categorize_data.to_csv(categorize_file, index=False)
        logger.info(f"Created categorization data file with {len(categorize_data)} records")
    else:
        logger.info(f"Categorization data file already exists: {categorize_file}")

if __name__ == "__main__":
    create_test_data() 