# Enhance Transaction Classification with Context

## Phase 1: Enhance Training Data & Storage

- [ ] **Modify Input Models:**
  - [ ] Update `/train` endpoint's `Transaction` Pydantic model (`main.py`) to accept `money_in: bool`, `amount: float`, `timestamp: datetime`.
- [ ] **Enhance Index Data:**
  - [ ] Modify `index_data` creation in `/train` (`main.py`) to include `money_in: np.bool_` in the NumPy structured array `dtype`.
  - [ ] Ensure `store_embeddings` and `fetch_embeddings` (`main.py`, `prisma_client.py`) correctly handle serialization/deserialization of the index array with the new boolean field.
- [ ] **Implement Aggregate Statistics Calculation:**
  - [ ] Add logic in `/train` (`main.py`) after loading data to calculate per-category statistics:
    - [ ] Median amount (separate for income/expense if category applies to both).
    - [ ] Common day-of-week/time-of-day patterns (e.g., frequency distribution).
    - [ ] Potentially amount ranges (min/max or percentiles).
- [ ] **Store Aggregate Statistics:**
  - [ ] Implement storage for the calculated statistics (likely as a JSON blob).
  - [ ] Decide storage method: e.g., new entry in `embeddings` table (`prisma_client.py`) with ID like `{user_id}_category_stats`.
  - [ ] Ensure `prisma_client.py` has functions to save/fetch this stats blob.

## Phase 2: Enhance Classification Logic

- [ ] **Modify Input Handling:**
  - [ ] Ensure `/classify` endpoint (`main.py`) correctly receives `amount` and `timestamp` along with `description` and `money_in`. Update `TransactionInput` model if necessary (amount/timestamp might already be implicitly handled via `transactions_input` dictionary structure, but verify).
- [ ] **Update Data Fetching:**
  - [ ] Modify classification pipeline (`_apply_initial_categorization` or surrounding logic in `main.py`) to fetch the new `{user_id}_category_stats` blob alongside embeddings and index data.
- [ ] **Implement Direction Filtering:**
  - [ ] After getting Top N candidates based on cosine similarity, filter them based on matching the `money_in` flag between the input transaction and the training data (from the enhanced `trained_data` index).
  - [ ] Define fallback logic if no candidates match the direction.
- [ ] **Implement Contextual Re-ranking:**
  - [ ] Develop the `context_score` calculation using the fetched `category_stats`:
    - [ ] Amount comparison score (input amount vs. category median/range).
    - [ ] Time comparison score (input timestamp vs. category time patterns).
    - [ ] Combine scores (e.g., weighted average).
  - [ ] Calculate the `final_score` for each candidate (weighted sum of `description_similarity` and `context_score`).
  - [ ] Select the category with the highest `final_score`.
- [ ] **Integrate with Existing Post-Processing:**
  - [ ] Ensure the output of the re-ranking step feeds correctly into `_detect_transfers` (and `_detect_refunds` if re-enabled).

## Phase 3: Testing & Tuning

- [ ] **Develop Test Cases:** Create test data covering ambiguous descriptions with different amounts/times/directions.
- [ ] **Tune Parameters:** Experiment and find optimal values for:
  - [ ] Top N candidates to consider.
  - [ ] Context score calculation methods/normalization.
  - [ ] Weighting factors (`w1`, `w2`) for description vs. context similarity.
- [ ] **Performance Testing:** Evaluate the impact on classification latency.
- [ ] **Security Review:** Confirm no sensitive individual data is exposed through the aggregate stats.
