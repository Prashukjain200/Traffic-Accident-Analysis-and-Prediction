# Traffic Accident Analysis and Prediction

## Overview
This application leverages a Streamlit interface to analyze historical traffic accident data and employs an LSTM model to forecast future accidents. The implementation intricately combines data processing, interactive visualization, and machine learning for an intuitive user experience.

## Implementation Details

### Data Preprocessing
- **Normalization**: Implements `MinMaxScaler` to normalize accident data, crucial for LSTM model performance.
- **Sequence Generation**: Transforms time series data into sequences, providing a structured format for the LSTM to learn from.

### Visualization Functions
- **Historical Trends**: Uses Seaborn's lineplot to render the historical trend of accidents, allowing for easy identification of patterns over time.
- **Seasonality**: Grouping data by month to display seasonal trends, aiding in understanding monthly fluctuations in accident numbers.
- **Yearly Comparison**: Aggregates and compares data annually using a barplot, highlighting changes in accident rates year over year.

### LSTM Model Configuration
- **Dynamic Architecture**: Users can specify the complexity of the LSTM model with adjustable layers and nodes.
- **Training and Validation**: Incorporates train-test splitting and early stopping to ensure the model generalizes well to unseen data.
- **Prediction**: Outputs both the predicted and actual numbers of accidents, visualized through a comparative graph.

### Streamlit Interface
- **Sidebar Options**: Allows users to navigate between data visualization and model training sections.
- **Interactive Widgets**: Streamlit widgets enable dynamic input for data filtering, model configuration, and triggering model training or data visualization.
- **Downloadable Data**: Offers the ability to download the dataset directly from the application.

## How to Use

### Installation
Clone the repository and install the necessary Python packages using the provided `requirements.txt`.

### Running the Application
Execute the Streamlit application which starts the web server and opens the app in a web browser.

### Interactive Analysis
- Use the sidebar to switch between data visualization options and model training sections.
- Select parameters such as date ranges, graph types, and model configurations.
- Trigger actions like generating graphs or training the model with a simple button click.

## Contributing
We encourage contributions that improve the application's analytics and forecasting capabilities. Feel free to fork the project, make your changes, and open a pull request!

Made with ❤️ by Prashuk.
