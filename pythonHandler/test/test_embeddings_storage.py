import os
import sys
import numpy as np
import asyncio
import logging

# Add the parent directory to the path so we can import from pythonHandler
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pythonHandler.utils.prisma_client import prisma_client
# Import the functions directly from main
import pythonHandler.main as main

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_prisma_direct():
    """Test storing and fetching embeddings directly with Prisma client."""
    logger.info("Testing direct Prisma client operations...")
    
    # Create test data
    test_data = np.array([1, 2, 3, 4, 5])
    test_file = "test_direct.npz"
    
    # Convert to bytes
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as temp_file:
        np.savez_compressed(temp_file, test_data)
        temp_file_path = temp_file.name
    
    with open(temp_file_path, "rb") as f:
        data_bytes = f.read()
    
    os.unlink(temp_file_path)
    
    # Connect to database
    await prisma_client.connect()
    
    # Store embedding
    logger.info("Storing test embedding...")
    result = await prisma_client.store_embedding(test_file, data_bytes)
    logger.info(f"Store result: {result}")
    
    # Fetch embedding
    logger.info("Fetching test embedding...")
    fetched_bytes = await prisma_client.fetch_embedding(test_file)
    
    # Convert back to numpy array
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as temp_file:
        temp_file.write(fetched_bytes)
        temp_file_path = temp_file.name
    
    fetched_data = np.load(temp_file_path, allow_pickle=True)["arr_0"]
    os.unlink(temp_file_path)
    
    # Verify data
    logger.info(f"Original data: {test_data}")
    logger.info(f"Fetched data: {fetched_data}")
    assert np.array_equal(test_data, fetched_data), "Data mismatch!"
    
    # Disconnect
    await prisma_client.disconnect()
    
    logger.info("Direct Prisma test completed successfully!")

def test_wrapper_functions():
    """Test the wrapper functions in main.py."""
    logger.info("Testing wrapper functions...")
    
    # Create test data
    test_data = np.array([6, 7, 8, 9, 10])
    
    # Store embedding
    logger.info("Storing test embedding...")
    main.store_embeddings("txclassify", "test_wrapper.npy", test_data)
    
    # Fetch embedding
    logger.info("Fetching test embedding...")
    fetched_data = main.fetch_embeddings("txclassify", "test_wrapper.npy")
    
    # Verify data
    logger.info(f"Original data: {test_data}")
    logger.info(f"Fetched data: {fetched_data}")
    assert np.array_equal(test_data, fetched_data), "Data mismatch!"
    
    logger.info("Wrapper functions test completed successfully!")

def test_structured_data():
    """Test with structured data (similar to _index.npy files)."""
    logger.info("Testing with structured data...")
    
    # Create structured test data
    test_data = np.array([
        (1, "Category1", "Description1"),
        (2, "Category2", "Description2"),
        (3, "Category3", "Description3")
    ], dtype=[('item_id', int), ('category', 'U100'), ('description', 'U500')])
    
    # Store embedding
    logger.info("Storing structured test embedding...")
    main.store_embeddings("txclassify", "test_structured.npy", test_data)
    
    # Fetch embedding
    logger.info("Fetching structured test embedding...")
    fetched_data = main.fetch_embeddings("txclassify", "test_structured.npy")
    
    # Verify data
    logger.info(f"Original data: {test_data}")
    logger.info(f"Fetched data: {fetched_data}")
    assert len(test_data) == len(fetched_data), "Data length mismatch!"
    for i in range(len(test_data)):
        assert test_data[i]['item_id'] == fetched_data[i]['item_id'], "item_id mismatch!"
        assert test_data[i]['category'] == fetched_data[i]['category'], "category mismatch!"
        assert test_data[i]['description'] == fetched_data[i]['description'], "description mismatch!"
    
    logger.info("Structured data test completed successfully!")

if __name__ == "__main__":
    logger.info("Starting embeddings storage tests...")
    
    # Run direct Prisma test
    asyncio.run(test_prisma_direct())
    
    # Run wrapper functions test
    test_wrapper_functions()
    
    # Run structured data test
    test_structured_data()
    
    logger.info("All tests completed successfully!") 