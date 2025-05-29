import pytest
import numpy as np
from app.services.lstm_model import build_model, train_and_save, load_model, save_predictions_to_db

@pytest.mark.asyncio
async def test_build_model():
    model = build_model((10, 3))  # Assuming SEQ_LENGTH=10, features=3
    assert model is not None
    assert hasattr(model, "predict")

@pytest.mark.asyncio
async def test_train_and_save(mocker):
    # Mock model and data
    mock_model = mocker.MagicMock()
    X_train = np.random.rand(100, 10, 3)
    y_train = np.random.rand(100, 1)
    
    mocker.patch("app.services.lstm_model.build_model", return_value=mock_model)
    mocker.patch("app.services.lstm_model.save_model", return_value=None)
    
    train_and_save(X_train, y_train)
    assert mock_model.fit.called

@pytest.mark.asyncio
async def test_load_model(mocker):
    mocker.patch("app.services.lstm_model.load_model", return_value=mocker.MagicMock())
    model = load_model()
    assert model is not None

@pytest.mark.asyncio
async def test_save_predictions_to_db(mocker):
    mocker.patch("app.services.lstm_model.save_predictions_to_db", return_value=None)
    predictions = np.array([1.0, 2.0, 3.0])
    save_predictions_to_db(predictions)
    assert True  # If no exception, test passes
