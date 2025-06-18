"""Financial Analytics Service for Advanced Transaction Insights."""

import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from collections import Counter, defaultdict
import re

logger = logging.getLogger(__name__)


class FinancialAnalyticsService:
    """Service for providing advanced financial analytics and insights."""

    def __init__(self):
        # These will be extracted dynamically from the data
        self.spending_categories = []
        self.income_categories = []
        self.transfer_categories = []

    def analyze_transactions(self, transactions: List[Dict[str, Any]], user_id: str, excluded_categories: List[str] = None) -> Dict[str, Any]:
        """Main analytics function that provides comprehensive transaction insights."""
        if not transactions:
            return {"error": "No transactions provided for analysis"}

        try:
            # Prepare transaction data
            df = self._prepare_transaction_dataframe(transactions)
            
            # Extract categories dynamically from the data
            self._extract_categories_from_data(df)
            
            results = {
                "user_id": user_id,
                "analysis_period": {
                    "start_date": df['date'].min().isoformat() if not df.empty else None,
                    "end_date": df['date'].max().isoformat() if not df.empty else None,
                    "total_transactions": len(df)
                },
                "categories_found": {
                    "all_categories": sorted(df['category'].unique().tolist()),
                    "spending_categories": self.spending_categories,
                    "income_categories": self.income_categories,
                    "transfer_categories": self.transfer_categories
                },
                "insights": {}
            }

            # Perform analyses
            results["insights"]["vendor_intelligence"] = self._analyze_vendor_spending(df, excluded_categories)
            results["insights"]["anomaly_detection"] = self._detect_spending_anomalies(df, excluded_categories)
            results["insights"]["subscription_analysis"] = self._analyze_subscription_patterns(df, results["insights"]["anomaly_detection"], excluded_categories)
            results["insights"]["savings_opportunities"] = self._identify_savings_opportunities(df, excluded_categories)
            results["insights"]["cash_flow_prediction"] = self._predict_cash_flow(df, excluded_categories)

            return results

        except Exception as e:
            logger.error(f"Error in financial analytics for user {user_id}: {str(e)}", exc_info=True)
            return {"error": f"Analytics processing failed: {str(e)}"}

    def _extract_categories_from_data(self, df: pd.DataFrame):
        """Extract and classify categories dynamically from transaction data."""
        all_categories = df['category'].unique().tolist()
        
        # Classify categories based on common patterns and amount signs
        income_keywords = ['income', 'salary', 'deposit', 'credit', 'refund', 'dividend', 'interest']
        transfer_keywords = ['transfer', 'atm', 'withdrawal', 'cash']
        
        self.income_categories = []
        self.transfer_categories = []
        self.spending_categories = []
        
        for category in all_categories:
            if pd.isna(category) or category == '':
                continue
                
            category_lower = str(category).lower()
            
            # Check if category appears to be income-related
            if (any(keyword in category_lower for keyword in income_keywords) or 
                category_lower in ['credit']):
                self.income_categories.append(category)
            # Check if category appears to be transfer-related
            elif any(keyword in category_lower for keyword in transfer_keywords):
                self.transfer_categories.append(category)
            else:
                # Everything else is considered spending
                self.spending_categories.append(category)
        
        # Also check transaction amounts and money_in field to refine classification
        for category in all_categories:
            if category not in self.income_categories and category not in self.transfer_categories:
                cat_df = df[df['category'] == category]
                if len(cat_df) > 0:
                    # Use money_in field if available, otherwise fall back to amount signs
                    if 'money_in' in cat_df.columns:
                        income_ratio = cat_df['money_in'].sum() / len(cat_df)
                    else:
                        income_ratio = (cat_df['amount'] > 0).sum() / len(cat_df)
                    
                    if income_ratio > 0.7 and category not in self.income_categories:
                        self.income_categories.append(category)
                        if category in self.spending_categories:
                            self.spending_categories.remove(category)
                    elif income_ratio < 0.3 and category not in self.spending_categories:
                        self.spending_categories.append(category)
                        if category in self.income_categories:
                            self.income_categories.remove(category)

        logger.info(f"Extracted categories - Spending: {len(self.spending_categories)}, "
                   f"Income: {len(self.income_categories)}, Transfer: {len(self.transfer_categories)}")

    def _prepare_transaction_dataframe(self, transactions: List[Dict[str, Any]]) -> pd.DataFrame:
        """Convert transaction list to pandas DataFrame with proper data types."""
        logger.info(f"Processing {len(transactions)} transactions")
        if transactions:
            logger.info(f"Sample transaction keys: {list(transactions[0].keys())}")
            logger.info(f"Sample transaction: {transactions[0]}")
        
        df = pd.DataFrame(transactions)
        
        # Handle different possible column names for amount
        amount_column = None
        for col in ['amount', 'Amount Spent', 'amount_spent']:
            if col in df.columns:
                amount_column = col
                break
        
        if amount_column is None:
            logger.error(f"No amount column found. Available columns: {df.columns.tolist()}")
            raise ValueError("No amount column found in transaction data")
        
        logger.info(f"Using amount column: {amount_column}")
        
        # Rename to standard column name
        if amount_column != 'amount':
            df['amount'] = df[amount_column]
        
        # Handle different possible column names for description
        desc_column = None
        for col in ['description', 'Description', 'desc', 'narrative']:
            if col in df.columns:
                desc_column = col
                break
        
        if desc_column is None:
            logger.error(f"No description column found. Available columns: {df.columns.tolist()}")
            raise ValueError("No description column found in transaction data")
        
        logger.info(f"Using description column: {desc_column}")
        
        # Rename to standard column name
        if desc_column != 'description':
            df['description'] = df[desc_column]
        
        # Handle different possible column names for category
        cat_column = None
        for col in ['category', 'Category', 'cat']:
            if col in df.columns:
                cat_column = col
                break
        
        if cat_column is None:
            logger.error(f"No category column found. Available columns: {df.columns.tolist()}")
            raise ValueError("No category column found in transaction data")
        
        logger.info(f"Using category column: {cat_column}")
        logger.info(f"Sample categories: {df[cat_column].head(10).tolist()}")
        logger.info(f"Unique categories: {df[cat_column].unique().tolist()}")
        
        # Rename to standard column name
        if cat_column != 'category':
            df['category'] = df[cat_column]
        
        # Handle different possible column names for date
        date_column = None
        for col in ['date', 'Date', 'transaction_date']:
            if col in df.columns:
                date_column = col
                break
        
        # Convert date column
        if date_column and date_column in df.columns:
            df['date'] = pd.to_datetime(df[date_column], errors='coerce')
        else:
            df['date'] = pd.Timestamp.now()
        
        # Clean and normalize amounts - handle negative values in parentheses
        def clean_amount(amount_str):
            if pd.isna(amount_str) or amount_str == '':
                return 0.0
            
            # Convert to string and handle special formatting
            amount_str = str(amount_str)
            
            # Handle amounts in parentheses (negative)
            if '(' in amount_str and ')' in amount_str:
                amount_str = '-' + amount_str.replace('(', '').replace(')', '')
            
            # Remove commas and other formatting
            amount_str = amount_str.replace(',', '').replace('$', '').strip()
            
            # Handle empty strings after cleaning
            if amount_str == '' or amount_str == '-':
                return 0.0
            
            try:
                return float(amount_str)
            except ValueError:
                logger.warning(f"Could not convert amount to float: {amount_str}")
                return 0.0
        
        df['amount'] = df['amount'].apply(clean_amount)
        df['amount_abs'] = df['amount'].abs()
        
        # Handle missing categories - assign "Unknown" to empty ones
        df['category'] = df['category'].fillna('Unknown')
        df['category'] = df['category'].replace('', 'Unknown')
        
        # Determine expense vs income based on amount and category
        # Only set money_in if it's not already provided in the data
        if 'money_in' not in df.columns:
            df['money_in'] = df['amount'] > 0
        else:
            # Convert money_in to boolean if it's not already
            df['money_in'] = df['money_in'].astype(bool)
        
        # Extract vendor names
        df['vendor'] = df['description'].apply(self._extract_vendor_name)
        
        # Add time-based features
        df['month'] = df['date'].dt.month
        df['day_of_week'] = df['date'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6])
        
        logger.info(f"Final DataFrame shape: {df.shape}")
        logger.info(f"Final categories after cleaning: {df['category'].unique().tolist()}")
        
        return df

    def _extract_vendor_name(self, description: str) -> str:
        """Extract vendor name from transaction description."""
        if not description or pd.isna(description):
            return "Unknown"
        
        # Convert to string and clean
        desc = str(description).strip()
        
        # Remove common prefixes/suffixes
        desc = re.sub(r'^(EFTPOS|ATM|POS)\s*', '', desc, flags=re.IGNORECASE)
        desc = re.sub(r'\s*(AU|AUS|NZ|NZL)\s*$', '', desc, flags=re.IGNORECASE)
        
        # Extract meaningful words (3+ characters)
        words = re.findall(r'\b[A-Za-z]{3,}\b', desc.upper())
        
        # Filter out common words
        common_words = {'THE', 'AND', 'FOR', 'WITH', 'FROM', 'CARD', 'DATE', 'VALUE'}
        words = [w for w in words if w not in common_words]
        
        if len(words) >= 2:
            return ' '.join(words[:2])
        elif len(words) == 1:
            return words[0]
        else:
            # Fallback to first part of description
            first_part = desc.split()[0] if desc.split() else "Unknown"
            return first_part.upper()

    def _analyze_vendor_spending(self, df: pd.DataFrame, excluded_categories: list = None) -> Dict[str, Any]:
        """Analyze spending patterns by vendor."""
        if df.empty:
            return {"vendors": [], "insights": []}
        
        if excluded_categories is None:
            excluded_categories = []
        
        # Filter to expense transactions (using money_in field or negative amounts or identified spending categories)
        if 'money_in' in df.columns:
            expense_df = df[
                (~df['money_in']) | 
                (df['category'].isin(self.spending_categories))
            ].copy()
        else:
            expense_df = df[
                (df['amount'] < 0) | 
                (df['category'].isin(self.spending_categories))
            ].copy()
        
        # Exclude user-specified categories
        if excluded_categories:
            expense_df = expense_df[~expense_df['category'].isin(excluded_categories)]
        
        if expense_df.empty:
            return {"vendors": [], "insights": ["No expense transactions found"]}
        
        # Group by vendor and include category information
        vendor_analysis = expense_df.groupby('vendor').agg({
            'amount_abs': ['sum', 'mean', 'count'],
            'date': ['min', 'max'],
            'category': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'Unknown'
        }).round(2)
        
        vendor_analysis.columns = ['total_spent', 'avg_transaction', 'visit_count', 'first_visit', 'last_visit', 'most_common_category']
        vendor_analysis = vendor_analysis.reset_index()
        vendor_analysis = vendor_analysis.sort_values('total_spent', ascending=False)
        
        top_vendors = []
        insights = []
        
        for _, vendor in vendor_analysis.head(10).iterrows():
            vendor_data = {
                "name": vendor['vendor'],
                "total_spent": float(vendor['total_spent']),
                "average_transaction": float(vendor['avg_transaction']),
                "visit_count": int(vendor['visit_count']),
                "first_visit": vendor['first_visit'].isoformat(),
                "last_visit": vendor['last_visit'].isoformat(),
                "category": vendor['most_common_category']
            }
            
            # Calculate visit frequency
            days_span = (vendor['last_visit'] - vendor['first_visit']).days
            if days_span > 0:
                visits_per_month = vendor['visit_count'] / days_span * 30
                vendor_data["visits_per_month"] = round(visits_per_month, 1)
            
            top_vendors.append(vendor_data)
            
            # Generate smarter insights based on category and spending patterns
            category = vendor['most_common_category']
            if vendor['visit_count'] > 5:
                if category.lower() in ['living', 'rent', 'housing']:
                    insights.append(f"Housing: {vendor['vendor']} is your largest recurring expense "
                                  f"(${vendor['total_spent']:.0f}, {vendor['visit_count']} payments)")
                elif category.lower() in ['groceries', 'food']:
                    monthly_avg = (vendor['total_spent'] / vendor['visit_count']) * 4.33
                    insights.append(f"Groceries: You spend ~${monthly_avg:.0f}/month at {vendor['vendor']} "
                                  f"({vendor['visit_count']} visits)")
                elif category.lower() in ['transport', 'fuel', 'gas']:
                    insights.append(f"Transport: Regular spending at {vendor['vendor']} "
                                  f"(${vendor['total_spent']:.0f}, avg ${vendor['avg_transaction']:.0f}/visit)")
                else:
                    insights.append(f"{category}: {vendor['vendor']} - ${vendor['total_spent']:.0f} "
                                  f"({vendor['visit_count']} visits, avg ${vendor['avg_transaction']:.0f}/visit)")
        
        return {
            "vendors": top_vendors,
            "insights": insights
        }

    def _detect_spending_anomalies(self, df: pd.DataFrame, excluded_categories: list = None) -> Dict[str, Any]:
        """Detect unusual spending patterns."""
        if df.empty:
            return {"anomalies": [], "insights": []}
        
        if excluded_categories is None:
            excluded_categories = []
        
        # Filter to expense transactions (using money_in field or negative amounts or identified spending categories)
        if 'money_in' in df.columns:
            expense_df = df[
                (~df['money_in']) | 
                (df['category'].isin(self.spending_categories))
            ].copy()
        else:
            expense_df = df[
                (df['amount'] < 0) | 
                (df['category'].isin(self.spending_categories))
            ].copy()
        
        # Exclude user-specified categories
        if excluded_categories:
            logger.info(f"Excluding categories from anomaly detection: {excluded_categories}")
            logger.info(f"Excluded categories types: {[type(cat) for cat in excluded_categories]}")
            logger.info(f"Before filtering: {len(expense_df)} transactions")
            logger.info(f"Available categories in data: {expense_df['category'].unique().tolist()}")
            logger.info(f"Category types in data: {[type(cat) for cat in expense_df['category'].unique()]}")
            
            # Debug the filtering
            mask = ~expense_df['category'].isin(excluded_categories)
            logger.info(f"Filtering mask shape: {mask.shape}, True count: {mask.sum()}")
            
            expense_df = expense_df[mask]
            logger.info(f"After filtering: {len(expense_df)} transactions")
            logger.info(f"Remaining categories: {expense_df['category'].unique().tolist()}")
        
        anomalies = []
        insights = []
        
        if expense_df.empty:
            return {"anomalies": [], "insights": ["No expense transactions to analyze"]}
        
        # Detect unusually large transactions
        if len(expense_df) > 5:  # Need enough data for statistics
            mean_amount = expense_df['amount_abs'].mean()
            std_amount = expense_df['amount_abs'].std()
            threshold = mean_amount + (2 * std_amount)
            
            large_transactions = expense_df[expense_df['amount_abs'] > threshold]
            
            for _, tx in large_transactions.iterrows():
                anomaly = {
                    "type": "large_transaction",
                    "description": tx['description'],
                    "amount": float(tx['amount_abs']),
                    "date": tx['date'].isoformat(),
                    "vendor": tx['vendor'],
                    "category": tx['category'],
                    "severity": "high" if tx['amount_abs'] > threshold * 1.5 else "medium"
                }
                logger.info(f"Adding large transaction anomaly: {tx['vendor']} ({tx['category']}) - ${tx['amount_abs']}")
                anomalies.append(anomaly)
                
                insights.append(f"Unusual charge: ${tx['amount_abs']:.0f} at {tx['vendor']} "
                              f"(you typically spend <${mean_amount:.0f}/transaction)")
        
        # Detect potential subscriptions (recurring similar amounts)
        for vendor in expense_df['vendor'].unique():
            vendor_txs = expense_df[expense_df['vendor'] == vendor]
            if len(vendor_txs) >= 3:  # Need multiple transactions
                amount_std = vendor_txs['amount_abs'].std()
                amount_mean = vendor_txs['amount_abs'].mean()
                
                # Low variation suggests subscription
                if amount_mean > 0 and amount_std / amount_mean < 0.15:
                    anomalies.append({
                        "type": "potential_subscription",
                        "vendor": vendor,
                        "amount": float(amount_mean),
                        "frequency": len(vendor_txs),
                        "category": vendor_txs.iloc[0]['category'],
                        "severity": "low"
                    })
                    
                    insights.append(f"Potential subscription: {vendor} ${amount_mean:.2f} "
                                  f"({len(vendor_txs)} similar transactions)")
        
        return {
            "anomalies": anomalies,
            "insights": insights
        }

    def _analyze_subscription_patterns(self, df: pd.DataFrame, anomaly_data: Dict[str, Any], excluded_categories: list = None) -> Dict[str, Any]:
        """Analyze subscription patterns, total spent, and projected annual costs."""
        if df.empty:
            return {"subscriptions": [], "summary": {}, "insights": []}
        
        # Use provided excluded categories or empty list if none provided
        if excluded_categories is None:
            excluded_categories = []
        
        # Extract potential subscriptions from anomaly detection results
        potential_subscriptions = [
            anomaly for anomaly in anomaly_data.get("anomalies", [])
            if (anomaly.get("type") == "potential_subscription" and 
                anomaly.get("category", "").lower() not in [cat.lower() for cat in excluded_categories])
        ]
        
        subscription_details = []
        insights = []
        
        for subscription in potential_subscriptions:
            vendor = subscription.get("vendor", "")
            avg_amount = subscription.get("amount", 0)
            frequency = subscription.get("frequency", 0)
            category = subscription.get("category", "Unknown")
            severity = subscription.get("severity", "unknown")
            
            # Find all transactions for this vendor
            vendor_transactions = df[df['vendor'].str.contains(vendor, case=False, na=False)]
            
            if vendor_transactions.empty:
                continue
            
            # Calculate total spent so far
            total_spent = vendor_transactions['amount_abs'].sum()
            
            # Calculate date range for this vendor
            date_range_days = (vendor_transactions['date'].max() - vendor_transactions['date'].min()).days
            if date_range_days == 0:
                date_range_days = 1  # Avoid division by zero
            
            # Project annual cost based on frequency and time span
            if frequency > 0 and date_range_days > 0:
                # Calculate transactions per day
                transactions_per_day = frequency / date_range_days
                # Project for 365 days
                projected_annual_transactions = transactions_per_day * 365
                projected_annual_cost = projected_annual_transactions * avg_amount
            else:
                projected_annual_cost = avg_amount * 12  # Fallback to monthly estimate
            
            subscription_info = {
                "vendor": vendor,
                "category": category,
                "frequency": frequency,
                "average_amount": round(avg_amount, 2),
                "total_spent_so_far": round(total_spent, 2),
                "projected_annual_cost": round(projected_annual_cost, 2),
                "severity": severity,
                "date_range_days": date_range_days,
                "first_transaction": vendor_transactions['date'].min().strftime('%Y-%m-%d'),
                "last_transaction": vendor_transactions['date'].max().strftime('%Y-%m-%d'),
                "monthly_estimate": round(projected_annual_cost / 12, 2)
            }
            
            subscription_details.append(subscription_info)
            
            # Generate insights
            if projected_annual_cost > 500:
                insights.append(f"{vendor}: High-cost subscription - projected ${projected_annual_cost:.0f}/year "
                              f"(${subscription_info['monthly_estimate']:.0f}/month)")
            elif projected_annual_cost > 100:
                insights.append(f"{vendor}: Medium-cost subscription - projected ${projected_annual_cost:.0f}/year")
            else:
                insights.append(f"{vendor}: Low-cost subscription - projected ${projected_annual_cost:.0f}/year")
        
        # Sort by projected annual cost (highest first)
        subscription_details.sort(key=lambda x: x['projected_annual_cost'], reverse=True)
        
        # Calculate summary statistics
        total_current_spending = sum(sub['total_spent_so_far'] for sub in subscription_details)
        total_projected_annual = sum(sub['projected_annual_cost'] for sub in subscription_details)
        
        summary = {
            "total_subscriptions_detected": len(subscription_details),
            "total_spent_so_far": round(total_current_spending, 2),
            "total_projected_annual_cost": round(total_projected_annual, 2),
            "average_monthly_cost": round(total_projected_annual / 12, 2) if total_projected_annual > 0 else 0,
            "highest_cost_subscription": subscription_details[0]['vendor'] if subscription_details else None,
            "highest_annual_cost": subscription_details[0]['projected_annual_cost'] if subscription_details else 0
        }
        
        # Add summary insights
        if total_projected_annual > 1000:
            insights.append(f"Warning: Your subscriptions could cost ${total_projected_annual:.0f} annually "
                          f"(${summary['average_monthly_cost']:.0f}/month)")
        elif total_projected_annual > 500:
            insights.append(f"Moderate subscription spending: ${total_projected_annual:.0f} annually projected")
        
        return {
            "subscriptions": subscription_details,
            "summary": summary,
            "insights": insights,
            "excluded_categories": excluded_categories,
            "note": "Amounts are predicted based on historical transaction patterns"
        }

    def _identify_savings_opportunities(self, df: pd.DataFrame, excluded_categories: list = None) -> Dict[str, Any]:
        """Identify potential savings opportunities."""
        if df.empty:
            return {"opportunities": [], "insights": []}
        
        if excluded_categories is None:
            excluded_categories = []
        
        # Filter to expense transactions (using money_in field or negative amounts or identified spending categories)
        if 'money_in' in df.columns:
            expense_df = df[
                (~df['money_in']) | 
                (df['category'].isin(self.spending_categories))
            ].copy()
        else:
            expense_df = df[
                (df['amount'] < 0) | 
                (df['category'].isin(self.spending_categories))
            ].copy()
        
        # Exclude user-specified categories
        if excluded_categories:
            expense_df = expense_df[~expense_df['category'].isin(excluded_categories)]
        
        opportunities = []
        insights = []
        
        if expense_df.empty:
            return {"opportunities": [], "insights": []}
        
        # Category-based savings opportunities
        category_spending = expense_df.groupby('category')['amount_abs'].sum().sort_values(ascending=False)
        
        # Focus on top spending categories that aren't essentials
        discretionary_keywords = ['dinner', 'bars', 'shopping', 'entertainment', 'coffee']
        
        for category in category_spending.head(5).index:
            category_lower = str(category).lower()
            spending_amount = category_spending[category]
            
            # Check if category seems discretionary
            is_discretionary = any(keyword in category_lower for keyword in discretionary_keywords)
            
            if is_discretionary or spending_amount > category_spending.mean():
                potential_savings = spending_amount * 0.2  # 20% reduction potential
                
                opportunities.append({
                    "type": "category_reduction",
                    "category": category,
                    "current_spending": float(spending_amount),
                    "potential_savings": float(potential_savings),
                    "recommendation": f"Reduce {category} spending by 20%"
                })
                
                insights.append(f"Reduce {category} spending by 20% to save "
                              f"~${potential_savings:.0f}")
        
        # Vendor consolidation opportunities
        vendor_spending = expense_df.groupby('vendor')['amount_abs'].sum().sort_values(ascending=False)
        
        # Look for similar vendors (e.g., multiple grocery stores)
        grocery_keywords = ['woolworths', 'pak n save', 'new world', 'coles', 'countdown']
        grocery_vendors = []
        
        for vendor in vendor_spending.index:
            vendor_lower = str(vendor).lower()
            if any(keyword in vendor_lower for keyword in grocery_keywords):
                grocery_vendors.append(vendor)
        
        if len(grocery_vendors) > 1:
            total_grocery_spending = sum(vendor_spending[v] for v in grocery_vendors)
            potential_savings = total_grocery_spending * 0.1  # 10% savings from consolidation
            
            opportunities.append({
                "type": "vendor_consolidation",
                "category": "Groceries",
                "vendors": grocery_vendors,
                "current_spending": float(total_grocery_spending),
                "potential_savings": float(potential_savings),
                "recommendation": f"Consolidate grocery shopping to save ~${potential_savings:.0f}"
            })
        
        return {
            "opportunities": opportunities,
            "insights": insights
        }

    def _predict_cash_flow(self, df: pd.DataFrame, excluded_categories: list = None) -> Dict[str, Any]:
        """Predict future cash flow based on patterns."""
        if df.empty:
            return {"predictions": {}, "insights": []}
        
        if excluded_categories is None:
            excluded_categories = []
        
        # Calculate income and expense totals
        if 'money_in' in df.columns:
            income_df = df[
                (df['money_in']) | 
                (df['category'].isin(self.income_categories))
            ]
            expense_df = df[
                (~df['money_in']) | 
                (df['category'].isin(self.spending_categories))
            ]
            
            # Exclude user-specified categories from expenses but not income
            if excluded_categories:
                expense_df = expense_df[~expense_df['category'].isin(excluded_categories)]
            
            total_income = income_df['amount_abs'].sum()
            total_expenses = expense_df['amount_abs'].sum()
        else:
            income_df = df[
                (df['amount'] > 0) | 
                (df['category'].isin(self.income_categories))
            ]
            expense_df = df[
                (df['amount'] < 0) | 
                (df['category'].isin(self.spending_categories))
            ]
            
            # Exclude user-specified categories from expenses but not income
            if excluded_categories:
                expense_df = expense_df[~expense_df['category'].isin(excluded_categories)]
            
            total_income = income_df['amount_abs'].sum()
            total_expenses = expense_df['amount_abs'].sum()
        
        # Calculate time span for rate estimation
        date_range = (df['date'].max() - df['date'].min()).days
        if date_range == 0:
            date_range = 1  # Avoid division by zero
        
        # Calculate weekly averages
        weekly_income = (total_income / date_range) * 7
        weekly_expenses = (total_expenses / date_range) * 7
        weekly_net = weekly_income - weekly_expenses
        
        predictions = {
            "weekly_income_estimate": float(weekly_income),
            "weekly_spending_estimate": float(weekly_expenses),
            "weekly_net": float(weekly_net),
            "monthly_net_estimate": float(weekly_net * 4.33)
        }
        
        insights = []
        if weekly_income > 0:
            if weekly_expenses > weekly_income:
                deficit = weekly_expenses - weekly_income
                insights.append(f"Warning: You're spending ${deficit:.0f} more than "
                              f"you earn per week on average")
            else:
                savings_rate = (weekly_net / weekly_income) * 100
                insights.append(f"You save approximately ${weekly_net:.0f} per week "
                              f"({savings_rate:.1f}% savings rate)")
        
        return {
            "predictions": predictions,
            "insights": insights
        }


def process_financial_analytics_request(transactions: List[Dict[str, Any]], user_id: str, excluded_categories: List[str] = None) -> Dict[str, Any]:
    """Process financial analytics request."""
    try:
        service = FinancialAnalyticsService()
        return service.analyze_transactions(transactions, user_id, excluded_categories)
    except Exception as e:
        logger.error(f"Error processing financial analytics request: {str(e)}", exc_info=True)
        return {"error": f"Failed to process analytics: {str(e)}"} 