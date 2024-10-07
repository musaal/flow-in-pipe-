# Fluid Flow Forecasting Using Transformer Model

This project demonstrates the use of a transformer-based neural network architecture for predicting the position and velocity of particles in a turbulent fluid flow system. The focus is on simulating the flow within pipes, especially turbulent pipe flows at Reynolds numbers (Re) of around 6000. This work builds on prior studies of turbulent pipe flow and leverages advanced machine learning techniques to model and predict the future states of particles within the fluid.

## Project Overview

Fluid flow, especially at high turbulence levels (Re ≈ 6000), is notoriously complex and can behave chaotically. Classical prediction methods often fail under such conditions due to the nonlinear nature of turbulence. However, recent advancements in deep learning, particularly with transformer models, have shown promise in handling chaotic systems due to their attention mechanisms, which allow the model to focus on relevant parts of the input data over long sequences.

In this project, we attempt to forecast the position and velocity of 500,000 particles in a turbulent fluid flow using a transformer model. The input data consists of the initial conditions (position and velocity) of each particle, and the model is tasked with predicting their future states.

This project also draws on previous direct numerical simulations (DNS) of turbulent pipe flows, including the study “One-point statistics for turbulent pipe flow up to Reτ ≈ 6000” by Sergio Pirozzoli et al. The DNS results provide a foundation for understanding the complex behavior of fluid flow under high Reynolds numbers, and these findings have been incorporated into the model development process.

## Key Features
- **Transformer Architecture**: The transformer model has been adapted for time-series forecasting of fluid flow, building on its success in handling sequence-based data in fields like natural language processing (NLP).
- **Self-Attention Mechanism**: This mechanism allows the model to dynamically focus on important parts of the input data, making it suitable for handling the chaotic nature of turbulence.
- **Direct Numerical Simulation (DNS) Data**: The project utilizes DNS data to train and evaluate the model, with particular attention to pipe flows at high Reynolds numbers (Reτ ≈ 6000).
- **Particle-Level Simulation**: We predict the 3D position (x, y, z) and velocity (Vx, Vy, Vz) of 500,000 particles at multiple time steps, incorporating the physics of turbulent flow dynamics.

## Data
The project uses data from the paper:

**"One-point statistics for turbulent pipe flow up to Reτ ≈ 6000"**
- Authors: Sergio Pirozzoli, Joshua Romero, Massimiliano Fatica, Roberto Verzicco, Paolo Orlandi
- Institutions: Sapienza Università di Roma, NVIDIA Corporation, Università di Roma Tor Vergata, University of Twente
- DOI/Link: [To be inserted]

Additionally, experimental flow pattern data from other sources are considered to augment and optimize the model performance in fluid flow simulations.

## Installation and Requirements

To set up the project on your local machine, follow these steps:

1. Clone this repository:
   ```bash
   git clone https://github.com/your-repository/transformer-fluid-flow.git
   cd transformer-fluid-flow
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. The transformer model is implemented following the guidelines in the article:
   - **Transformer Implementation for Time-Series Forecasting** by Natasha Klingenbrunn: [Medium Article](https://natasha-klingenbrunn.medium.com/transformer-implementation-for-time-series-forecasting-a9db2db5c820).

4. Ensure that your environment supports Python 3.8 or higher, and the required deep learning libraries such as TensorFlow or PyTorch, depending on the framework used in the implementation.

## Usage

To run the simulation and predictions, follow these steps:

1. Prepare the input data (particle positions and velocities) in the appropriate format.
   
2. Run the training script:
   ```bash
   python train.py --data_path /path/to/your/data --epochs 100 --batch_size 32
   ```

3. After training, use the trained model to forecast future particle states:
   ```bash
   python forecast.py --model_path /path/to/saved/model --input_data /path/to/input/data
   ```

4. The results will be saved as a CSV file, containing the predicted positions and velocities for each particle at the specified time steps.

## References
- **Sergio Pirozzoli et al.**, "One-point statistics for turbulent pipe flow up to Reτ ≈ 6000", Direct Numerical Simulation Study, 2021.
- **Natasha Klingenbrunn**, "Transformer Implementation for Time-Series Forecasting", [Medium Article](https://natasha-klingenbrunn.medium.com/transformer-implementation-for-time-series-forecasting-a9db2db5c820).

## Contributing

Feel free to contribute by submitting a pull request or raising an issue.

## License

This project is licensed under the MIT License - see the LICENSE.md file for details.
