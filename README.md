# Efficacy_of_LSTM_for_SSN_forecasting
All code used in research "Efficacy of an LSTM Model for Sunspot Prediction" by Sana Das. 

## Abstract
Data-driven machine learning (ML) models are prominent tools for forecasting solar activity with their strength in pattern recognition. However, the purely numbered approach has limitations since sunspot numbers (SSN) are rooted in the physics of the solar dynamo. Our study quantifies the performance of a data-driven Long Short-Term Memory (LSTM) model by evaluating its capability to forecast Solar Cycle 24, a notably weak solar cycle. Utilizing yearly SSN data from the Sunspot Index and Long-term Solar Observations (SILSO) dataset, the LSTM network was trained on historical cycles up to 2008 and forecasted Solar Cycle 24 across 30 independent trials. Model performance was evaluated using the Sum of Squared Residuals (SSR) variation against a defined "ideal prediction" threshold of 500. A one-sample t-test statistically confirmed that the LSTM consistently overpredicted the amplitude of Solar Cycle 24 with a p-value of 1.2010-6. We therefore suggested a better approach using a physics-incorporated model such as the Solar Dynamo model. This model simulates the magnetic flux dynamics to predict polar field amplitude, which is highly correlated with the strength of the next cycle. 

## Model Definition
The LSTM is coded by first initializing a weight matrix. All four gates use a single weight matrix. Each forward pass computes all four gates at once based on the given equations:

	input = sigmoid(z[:h_dim])	
	forget = sigmoid(z[h_dim:2*h_dim])  
	cell = tanh(z[2*h_dim:3*h_dim])  
	output = sigmoid(z[3*h_dim:]) 

Then it updates the cell state, long-term, and the hidden state, short term, using the gate outputs.  

	cell_state = forget·previous_cell_state + input·cell
	hidden_state = output · tanh(cell_state)

The backwards pass, backpropagation through time (BPTT), basically unfolds the RNN to accumulate gradients. This LSTM uses an Adam Optimizer which is a form of gradient descent. Gradient descent finds the minimum error of the training data vs. the models training predictions in order to test the weights and biases. Gradient descent works to find zero error by taking big to small steps based on how it overshoots/undershoots. Training data runs through multiple times in order for the LSTM to fine-tune its weights and biases. 
Debugging was assisted by Claude AI. The full code is available at this GitHub page.

## Full Research Paper
Link: https://docs.google.com/document/d/1az6XaCQAPkAXklHaechZxao5vZ_PtWEiX_exZAlO1bU/edit?tab=t.0

Note: This is a work in progress currently. In Future, journal links may be added here.
