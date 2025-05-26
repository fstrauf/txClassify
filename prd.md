# Generic Transaction Categorization Implementation Plan

## Overview

Transform the existing user-specific transaction categorization system into a generic, country-based system using Plaid's standard categories.

## Phase 1: Data Preparation (Week 1)

### 1.1 Create Category Mapping

- [ ] Create `plaid_categories.json` with all 16 categories
- [ ] Add descriptions and keywords for each category
- [ ] Define subcategories if needed (e.g., FOOD_AND_DRINK → Restaurants, Groceries)

### 1.2 Build Training Datasets

- [ ] **NZ Dataset** (`nz_training_data.csv`)
  - [ ] Use existing NZ business registry data
  - [ ] Map business names to Plaid categories using keywords
  - [ ] Manually review and correct top 1000 merchants
  - [ ] Include common transaction patterns
- [ ] **AU Dataset** (`au_training_data.csv`)

  - [ ] Collect common Australian merchants
  - [ ] Map to Plaid categories
  - [ ] Focus on major chains and services

- [ ] **US Dataset** (`us_training_data.csv`)
  - [ ] Use publicly available merchant data
  - [ ] Map to Plaid categories
  - [ ] Include major retailers and services

### 1.3 Training Data Structure

```csv
description,category,merchant_name,country
"COUNTDOWN AUCKLAND CENTRAL",FOOD_AND_DRINK,"Countdown",NZ
"Z ENERGY WELLINGTON",TRANSPORTATION,"Z Energy",NZ
"SPARK NEW ZEALAND",RENT_AND_UTILITIES,"Spark",NZ
```

## Phase 2: Model Creation (Week 2)

### 2.1 Modify Training Infrastructure

- [ ] Update `pythonHandler/services/training_service.py`:
  - [ ] Add `is_default_model` flag support
  - [ ] Modify `process_training_request` to handle country models
  - [ ] Use country codes as model identifiers

### 2.2 Create Training Scripts

- [ ] Extend `tests/create-nz-model.js`:
  ```javascript
  // Parameters to add:
  // - country_code
  // - plaid_categories
  // - is_default_model
  ```
- [ ] Create `tests/create-all-default-models.js` for batch processing

### 2.3 Train Default Models

- [ ] Train NZ default model
- [ ] Train AU default model
- [ ] Train US default model
- [ ] Validate training results

## Phase 3: Classification Updates (Week 3)

### 3.1 Update Classification Service

- [ ] Modify `pythonHandler/services/classification_service.py`:
  - [ ] Add country detection logic
  - [ ] Implement fallback chain: User → Country → Generic
  - [ ] Update `_apply_initial_categorization` for default models

### 3.2 Country Detection Implementation

- [ ] Create `pythonHandler/utils/country_utils.py`:
  ```python
  def detect_country(transactions: List[Dict]) -> str:
      # Currency detection (NZD, AUD, USD)
      # Location name detection
      # Merchant pattern matching
      # Return country code or None
  ```

### 3.3 API Endpoint Updates

- [ ] Update `/classify` endpoint:
  - [ ] Accept optional `country_code` parameter
  - [ ] Auto-detect country if not provided
  - [ ] Return `detected_country` in response

## Phase 4: Category Standardization (Week 3-4)

### 4.1 Update Database Schema

- [ ] Add migration for Plaid categories:
  ```sql
  -- Add plaid_category column to relevant tables
  -- Create category mapping table
  ```

### 4.2 Create Category Converter

- [ ] Build `pythonHandler/utils/category_mapper.py`:
  ```python
  OLD_TO_PLAID_MAPPING = {
      "Groceries": "FOOD_AND_DRINK",
      "Transport": "TRANSPORTATION",
      "Income": "INCOME",
      # ... etc
  }
  ```

### 4.3 Update OpenAI Integration

- [ ] Modify `pythonHandler/utils/openai_utils.py`:
  - [ ] Update system prompt with Plaid categories
  - [ ] Provide category descriptions to improve accuracy

## Phase 5: Testing & Validation (Week 4)

### 5.1 Create Test Suite

- [ ] Unit tests for country detection
- [ ] Integration tests for default model fallback
- [ ] Category mapping validation tests

### 5.2 Accuracy Testing

- [ ] Create `tests/test-default-models.js`:
  - [ ] Test each country model
  - [ ] Measure accuracy against known transactions
  - [ ] Compare with user-specific models

### 5.3 Performance Testing

- [ ] Load test with 1000+ transactions
- [ ] Measure response times
- [ ] Test fallback scenarios

## Phase 6: User Experience Updates (Week 5)

### 6.1 API Documentation

- [ ] Update Swagger documentation
- [ ] Add country_code parameter
- [ ] Document category changes

### 6.2 Migration Guide

- [ ] Create user migration documentation
- [ ] Provide category mapping reference
- [ ] Include examples

### 6.3 Feedback Mechanism

- [ ] Add endpoint for category corrections
- [ ] Store corrections for model improvement
- [ ] Plan for periodic model updates

## Phase 7: Deployment (Week 6)

### 7.1 Deployment Strategy

- [ ] Deploy default models to production
- [ ] Enable feature flag for generic categorization
- [ ] Monitor performance and accuracy

### 7.2 Rollback Plan

- [ ] Maintain user-specific models
- [ ] Feature flag for old/new behavior
- [ ] Quick rollback procedure

## Implementation Checklist

### Immediate Actions (Do First)

1. [ ] Create `plaid_categories.json` with all categories
2. [ ] Set up NZ training data using existing business registry
3. [ ] Modify `create-nz-model.js` to use Plaid categories
4. [ ] Test NZ default model creation

### Code Changes Required

#### 1. New Files to Create:

- `pythonHandler/utils/country_utils.py`
- `pythonHandler/utils/category_mapper.py`
- `pythonHandler/data/plaid_categories.json`
- `tests/create-all-default-models.js`
- `tests/test-default-models.js`

#### 2. Files to Modify:

- `pythonHandler/services/classification_service.py`
- `pythonHandler/services/training_service.py`
- `pythonHandler/main.py` (API endpoints)
- `pythonHandler/models.py` (request models)
- `pythonHandler/utils/openai_utils.py`

#### 3. Database Changes:

- Add `country_code` to relevant tables
- Add `is_default_model` flag
- Create category mapping table

## Success Metrics

### Accuracy Targets

- Default model accuracy: >80% for common merchants
- Country detection accuracy: >95%
- User satisfaction: Reduce manual categorization by 70%

### Performance Targets

- Classification time: <2s for 100 transactions
- Model training time: <5 minutes per country
- API response time: <500ms for single transaction

## Rollout Plan

### Week 1-2: Development

- Complete Phase 1-3
- Create initial default models

### Week 3-4: Testing

- Internal testing with real data
- Fix identified issues

### Week 5: Beta

- Deploy to subset of users
- Gather feedback

### Week 6: Full Release

- Deploy to all users
- Monitor and iterate

## Notes

### Key Decisions Made:

1. Using Plaid's 16 categories as standard
2. Country-specific models as primary approach
3. Maintaining backward compatibility
4. OpenAI as fallback for low confidence

### Risks & Mitigations:

1. **Risk**: Accuracy lower than user-specific models
   - **Mitigation**: Allow users to train on top of defaults
2. **Risk**: Country detection failures
   - **Mitigation**: Allow manual country selection
3. **Risk**: Category mapping confusion
   - **Mitigation**: Provide clear documentation and migration tools

### Future Enhancements:

1. Multi-country transaction support
2. Industry-specific models
3. Temporal pattern recognition
4. Merchant logo/brand detection

### Plaid Categories

INCOME
TRANSFER_IN
TRANSFER_OUT
LOAN_PAYMENTS
BANK_FEES
ENTERTAINMENT
FOOD_AND_DRINK
GENERAL_MERCHANDISE
HOME_IMPROVEMENT
MEDICAL
PERSONAL_CARE
GENERAL_SERVICES
GOVERNMENT_AND_NON_PROFIT
TRANSPORTATION
TRAVEL
RENT_AND_UTILITIES
