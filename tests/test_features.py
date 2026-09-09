import pytest
from src.features.build_features import create_features

def test_create_features():
    # Sample input data
    raw_data = {
        'area': [1500, 2000, 2500],
        'bedrooms': [3, 4, 5],
        'bathrooms': [2, 3, 4],
        'floor': [1, 2, 3],
        'property_age': [10, 5, 2],
        'location': ['suburb', 'city', 'suburb'],
        'property_type': ['house', 'apartment', 'house']
    }
    
    # Expected output after feature creation
    expected_features = {
        'area': [1500, 2000, 2500],
        'bedrooms': [3, 4, 5],
        'bathrooms': [2, 3, 4],
        'floor': [1, 2, 3],
        'property_age': [10, 5, 2],
        'location_suburb': [1, 0, 1],
        'location_city': [0, 1, 0],
        'property_type_house': [1, 0, 1],
        'property_type_apartment': [0, 1, 0]
    }
    
    # Create features using the function
    features = create_features(raw_data)
    
    # Assert that the created features match the expected output
    assert features == expected_features

if __name__ == "__main__":
    pytest.main()