import autokeras as ak
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import tensorflow as tf

# Define the input shape for each segment
input_shape = (1000, 101)  # 100,000 nodes divided into 1,000 segments of length 100 + 1 for the index

# Define the number of segments
num_segments = 1000

# Calculate the segment length
segment_length = input_shape[0] // num_segments

# Generate sample data for regression
X = np.random.rand(num_segments, segment_length, input_shape[1])
y = np.random.rand(num_segments, 1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a MirroredStrategy to distribute the model across all 4 GPUs
strategy = tf.distribute.MirroredStrategy(devices=["/device:GPU:0", "/device:GPU:1", "/device:GPU:2", "/device:GPU:3"])

# Open a strategy scope.
with strategy.scope():
    # Create an AutoKeras StructuredDataRegressor instance
    regressor = ak.StructuredDataRegressor(
        overwrite=True,
        max_trials=3,
        objective="val_loss",
        metrics=["mean_squared_error"],
    )

    # Define input nodes for each segment
    segment_inputs = []

    for index, i in enumerate(range(num_segments)):
        # Define an input node for each segment
        input_node = ak.StructuredDataInput(shape=(segment_length, input_shape[1]), name=f"segment_input_{i}")

        # Add a feature for the segment index
        segment_idx = ak.layers.Input(shape=(1,), name=f"segment_idx_{i}")

        # Concatenate the inputs and segment index 
        concat_layer = ak.layers.Concatenate()([input_node, segment_idx])

        # Add the Concatenate features to the list of segment inputs
        segment_inputs.append(concat_layer)
        
        if index == len(range(num_segments)) // 2:
            print("We are at the halfway point.")
        elif index == len(range(num_segments)) * 3 // 4:
            print("We are at the last quarter of the iteration.")

    # Create a dense layer before the output layer
    with tf.device("/device:GPU:0") | tf.device("/device:GPU:1"):
        large_dense_layer = ak.layers.DenseBlock()(segment_inputs)
        
        mid = segment_inputs[len(segment_inputs) // 2:] # Gets the second half of the segment inputs
        mid_size_dense_layer = ak.layers.DenseBlock()(mid)
        
        small = segment_inputs[len(segment_inputs) // 4:] # Gets the last quarter of the segment inputs
        small_dense_layer = ak.layers.DenseBlock()(small)
        
        concat_layer = ak.layers.Concatenate()(large_dense_layer, mid_size_dense_layer, small_dense_layer)
        
        
        # maybe put this in a different device
        
        dense_layer = ak.layers.DenseBlock()(concat_layer)
       
        dense_layer = ak.layers.DenseBlock()(dense_layer)
        
        output = ak.layers.Dense(1, activation="linear", name="output_layer")(dense_layer)

    # Create a Model with the specified inputs and output
    model = ak.AutoModel(inputs=segment_inputs, outputs=output)

# Fit the AutoKeras model with data
model.fit(X_train, y_train, epochs=10)

# Evaluate the AutoKeras model using the test data
y_pred = model.predict(X_test)

# Calculate and print the Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
