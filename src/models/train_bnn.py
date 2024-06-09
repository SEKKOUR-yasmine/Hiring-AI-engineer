import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import os

from BnnModel import BayesianModel

from data_loader import (
    load_mauna_loa_atmospheric_co2,
    load_international_airline_passengers,
)

MUANA_DATA_PATH = "/content/drive/My Drive/BIGMAMA/mauna_loa_atmospheric_co2.csv"
AIRLINE_DATA_PATH = "/content/drive/My Drive/BIGMAMA/international-airline-passengers.csv"

def train_model(model, train_tensor, target_tensor, loss_function, optimizer, num_epochs):
    train_losses = []
    for epoch in tqdm(range(num_epochs), desc="Training"):
        model.train()
        optimizer.zero_grad()
        outputs = model(train_tensor)
        loss = loss_function(outputs, target_tensor)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
    return train_losses

def plot_training_loss(train_losses, num_epochs):
    plt.plot(range(1, num_epochs + 1), train_losses, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Epochs")
    plt.legend()
    plt.show()

def evaluate_and_plot(model, test_tensor, ground_truth, title):
    with torch.no_grad():
        model.eval()
        predictions = model(test_tensor)
    predictions_np = predictions.numpy()
    plt.figure(figsize=(10, 6))
    plt.plot(test_tensor.numpy(), ground_truth, "b.", markersize=10, label="Ground Truth")
    plt.plot(test_tensor.numpy(), predictions_np, "r.", markersize=10, label="Predictions")
    plt.xlabel("Input Features (X_test)")
    plt.ylabel("Ground Truth and Predictions (y_test, Predictions)")
    plt.title(title)
    plt.legend()
    plt.show()
    return predictions_np

def main():
    # ------------------------------------------
    # mauna_loa_atmospheric_co2 Dataset
    # ------------------------------------------

    # Prepare data
    X1, y1, X1_normalized = load_mauna_loa_atmospheric_co2(MUANA_DATA_PATH)

    # Split the data into training and test sets
    X1_train, X1_test, y1_train, y1_test = train_test_split(X1_normalized, y1, test_size=0.2, random_state=42)

    # Convert NumPy arrays to PyTorch tensors
    X1_train_tensor = torch.from_numpy(X1_train).float()
    y1_train_tensor = torch.from_numpy(y1_train).float()
    X1_test_tensor = torch.from_numpy(X1_test).float()

    # Define the Bayesian neural network model
    input_size = X1_train.shape[1]
    hidden_size = 20
    output_size = 1
    model = BayesianModel(input_size, hidden_size, output_size)

    # Define loss function and optimizer
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Training loop
    num_epochs = 1000
    train_losses = train_model(model, X1_train_tensor, y1_train_tensor, loss_function, optimizer, num_epochs)

    # Plot training losses
    plot_training_loss(train_losses, num_epochs)

    # Evaluate and plot predictions
    predictions_np_1 = evaluate_and_plot(model, X1_test_tensor, y1_test, "True Values vs Predictions (Mauna Loa)")

    # Export model
    os.makedirs("Bigmamam", exist_ok=True)
    torch.save(model, "Bigmamam/mauna_loa_model.pth")

    # ------------------------------------------
    
    # international-airline-passengers Dataset
    # ------------------------------------------

    # Prepare data
    X2, y2, X2_normalized = load_international_airline_passengers(AIRLINE_DATA_PATH)

    # Split the data into training and test sets
    X2_train, X2_test, y2_train, y2_test = train_test_split(X2_normalized, y2, test_size=0.2, random_state=42)

    # Convert NumPy arrays to PyTorch tensors
    X2_train_tensor = torch.from_numpy(X2_train).float()
    y2_train_tensor = torch.from_numpy(y2_train).float()
    X2_test_tensor = torch.from_numpy(X2_test).float()

    # Define the Bayesian neural network model
    input_size = X2_train.shape[1]
    hidden_size = 20
    output_size = 1
    model = BayesianModel(input_size, hidden_size, output_size)

    # Define loss function and optimizer
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Training loop
    train_losses = train_model(model, X2_train_tensor, y2_train_tensor, loss_function, optimizer, num_epochs)

    # Plot training losses
    plot_training_loss(train_losses, num_epochs)

    # Evaluate and plot predictions
    predictions_np_2 = evaluate_and_plot(model, X2_test_tensor, y2_test, "True Values vs Predictions (Airline Passengers)")

    # Export model
    torch.save(model, "Bigmamam/international_airline_passengers_model.pth")

if __name__ == "__main__":
    main()
