# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.model_selection import train_test_split
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense
# from tensorflow.keras.callbacks import EarlyStopping
# import tempfile

# # Set custom CSS for improved visual appearance
# st.markdown(
#     """
#     <style>
#     body {
#         background-color: #121212; /* Dark background for modern look */
#         color: #e0e0e0; /* Light text color */
#         font-family: 'Arial', sans-serif;
#     }

#     .stButton>button {
#         background-color: #6200ea; /* Purple button */
#         color: white;
#         padding: 10px 20px;
#         border-radius: 8px;
#         border: none;
#         cursor: pointer;
#         font-size: 16px;
#     }

#     .stButton>button:hover {
#         background-color: #3700b3;
#     }

#     .stSidebar {
#         background-color: #1f1f1f; /* Darker sidebar background */
#         color: #e0e0e0;
#     }

#     .stTitle {
#         font-size: 2em;
#         color: #bb86fc;
#     }

#     .stHeader {
#         color: #03dac6;
#     }

#     .stMarkdown {
#         color: #e0e0e0;
#     }

#     </style>
#     """,
#     unsafe_allow_html=True
# )

# # Load the dataset
# data = pd.read_csv("Dataset.csv")
# download_data = data
# data['MONAT'] = data['MONAT'].astype(str).str[:6]
# data['DATUM'] = pd.to_datetime(data['MONAT'], format='%Y%m')


# # Define helper functions
# def filter_data_by_year(data, start_year, end_year):
#     return data[(data['JAHR'] >= start_year) & (data['JAHR'] <= end_year)]


# def preprocessig_data(data, category, type, year, month, scaler):
#     data['DATUM'] = pd.to_datetime(data['MONAT'], format='%Y%m')

#     cutoff_date = pd.to_datetime(f"{year}{month:02d}", format='%Y%m')
#     filtered_data = data[
#         (data['MONATSZAHL'] == category) &
#         (data['AUSPRAEGUNG'] == type) &
#         (data['DATUM'] < cutoff_date)
#         ]
#     if filtered_data.empty:
#         return None, None, None, None

#     series = filtered_data['WERT'].dropna().values.reshape(-1, 1)
#     series_reshaped = series.reshape(-1, 1)
#     scaled_data = scaler.fit_transform(series_reshaped)

#     def create_sequences(data, sequence_length):
#         sequences = []
#         for i in range(len(data) - sequence_length):
#             sequences.append(data[i:i + sequence_length + 1])
#         return np.array(sequences)

#     sequence_length = 5  # Number of time steps to look back
#     sequences = create_sequences(scaled_data, sequence_length)
#     X, y = sequences[:, :-1], sequences[:, -1]

#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#     return X_train, X_test, y_train, y_test


# def plot_historical_data(data):
#     with st.spinner("Generating Graph..."):
#         plt.figure(figsize=(15, 6))
#         sns.lineplot(x='DATUM', y='WERT', hue='MONATSZAHL', data=data, marker='o')
#         plt.title('Historical Number of Accidents Per Category', fontsize=16, color='#bb86fc')
#         plt.xlabel('Date', fontsize=14, color='#e0e0e0')
#         plt.ylabel('Number of Accidents', fontsize=14, color='#e0e0e0')
#         plt.xticks(rotation=45)
#         plt.legend(title='Category', loc='upper left', title_fontsize='13', fontsize='12')
#         plt.tight_layout()
#         st.pyplot(plt)


# def plot_seasonal_analysis(data):
#     monthly_data = data.groupby(['MONATSZAHL', data['DATUM'].dt.month])['WERT'].mean().reset_index()
#     plt.figure(figsize=(15, 6))
#     sns.lineplot(x='DATUM', y='WERT', hue='MONATSZAHL', data=monthly_data, marker='o')
#     plt.title('Seasonal Analysis of Accidents Per Category', fontsize=16, color='#bb86fc')
#     plt.xlabel('Month', fontsize=14, color='#e0e0e0')
#     plt.ylabel('Average Number of Accidents', fontsize=14, color='#e0e0e0')
#     plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
#     plt.legend(title='Category', loc='upper left', title_fontsize='13', fontsize='12')
#     plt.tight_layout()
#     st.success('Graph Generated!')
#     st.pyplot(plt)


# def plot_yearly_analysis(data):
#     yearly_data = data.groupby(['MONATSZAHL', 'JAHR'])['WERT'].sum().reset_index()
#     plt.figure(figsize=(15, 6))
#     sns.barplot(x='JAHR', y='WERT', hue='MONATSZAHL', data=yearly_data)
#     plt.title('Comparative Analysis of Accidents Per Category by Year', fontsize=16, color='#bb86fc')
#     plt.xlabel('Year', fontsize=14, color='#e0e0e0')
#     plt.ylabel('Total Number of Accidents', fontsize=14, color='#e0e0e0')
#     plt.legend(title='Category', loc='upper right', title_fontsize='13', fontsize='12')
#     plt.tight_layout()
#     st.pyplot(plt)


# def plot_predicted_graph(predicted, y_test_original):
#     plt.figure(figsize=(12, 6))
#     plt.plot(y_test_original, label='Actual', marker='o', color='#03dac6')
#     plt.plot(predicted, label='Predicted', marker='x', color='#bb86fc')
#     plt.title('Actual vs. Predicted Accidents', fontsize=16, color='#bb86fc')
#     plt.xlabel('Time Step', fontsize=14, color='#e0e0e0')
#     plt.ylabel('Number of Accidents', fontsize=14, color='#e0e0e0')
#     plt.legend()
#     plt.grid(True)
#     st.pyplot(plt)


# def train_lstm_model(X_train, y_train, num_layers, num_nodes, epoch):
#     model = Sequential()
#     model.add(LSTM(num_nodes, return_sequences=(num_layers > 1), input_shape=(X_train.shape[1], 1)))
#     for _ in range(1, num_layers):
#         model.add(LSTM(num_nodes, return_sequences=(_ < num_layers - 1)))
#     model.add(Dense(1))
#     model.compile(optimizer='adam', loss='mean_squared_error')

#     early_stopping = EarlyStopping(monitor='val_loss', patience=10)

#     model.fit(X_train, y_train, epochs=epoch, batch_size=32, validation_split=0.1,
#               callbacks=[early_stopping])
#     with tempfile.NamedTemporaryFile(delete=False, suffix='.h5') as tmp:
#         model.save(tmp.name)
#         return model, tmp.name


# # Main app structure
# def main():
#     st.title("Traffic Accident Analysis and Prediction")

#     option = st.sidebar.selectbox('Select an Option', ('Data Visualization', 'Model Training'))

#     if option == 'Data Visualization':
#         st.header("Data Visualization")

#         st.sidebar.download_button(
#             label="Download Dataset",
#             data=download_data.to_csv(index=False),
#             file_name="Dataset.csv",
#             mime="text/csv",
#         )

#         with st.sidebar.expander("Graph Descriptions"):
#             st.markdown("""
#                 **Historical Number of Accidents**: Shows the total number of accidents over time, highlighting trends and changes in accident frequency.
#                     \n**Comparative Analysis by Year**: Compares the total number of accidents per year across different categories, indicating yearly trends and variations.
#                     \n**Seasonal Analysis of Accidents**: Provides insights into how accidents vary by month, identifying seasonal patterns.
#                 """)

#         start_year, end_year = st.select_slider(
#             "Select a range of years",
#             options=list(range(2000, 2022)),
#             value=(2000, 2021)
#         )

#         if start_year == end_year:
#             st.warning("Please select a range of at least one year.")

#         filtered_data = filter_data_by_year(data, start_year, end_year)

#         graph_option = st.selectbox("Choose the type of graph",
#                                     ("Historical Number of Accidents",
#                                      "Comparative Analysis by Year",
#                                      "Seasonal Analysis of Accidents"))

#         submit_button = st.button('Generate Graph')

#         if submit_button:
#             if graph_option == "Historical Number of Accidents":
#                 plot_historical_data(filtered_data)
#             elif graph_option == "Seasonal Analysis of Accidents":
#                 plot_seasonal_analysis(filtered_data)
#             elif graph_option == "Comparative Analysis by Year":
#                 plot_yearly_analysis(filtered_data)

#     elif option == 'Model Training':
#         st.sidebar.download_button(
#             label="Download Dataset",
#             data=download_data.to_csv(index=False),
#             file_name="Dataset.csv",
#             mime="text/csv",
#         )
#         st.header("Model Training")

#         category = st.selectbox('Select Category', data['MONATSZAHL'].unique())
#         typeof = st.selectbox('Select Type', data['AUSPRAEGUNG'].unique())
#         unique_years = sorted([year for year in data['JAHR'].unique() if year < 2022], reverse=True)
#         year = st.selectbox('Select Year', unique_years)
#         month = st.selectbox('Select Month for Prediction', range(1, 13))

#         num_layers = st.number_input('Number of LSTM Layers', min_value=1, max_value=10, value=3)
#         num_nodes = st.number_input('Number of Nodes per Layer', min_value=10, max_value=500, value=100)
#         epochs = st.number_input("Number of Epochs", min_value=10, max_value=500, value=50)

#         scaler = MinMaxScaler(feature_range=(0, 1))

#         result = preprocessig_data(data, category, typeof, year, month, scaler)

#         if result[0] is not None:
#             X_train, X_test, y_train, y_test = result
#             if st.button('Train Model'):
#                 with st.spinner('Training in progress...'):
#                     model, model_path = train_lstm_model(X_train, y_train, num_layers, num_nodes, epochs)
#                     st.success(f"Training completed. Model saved at {model_path}")

#                     with open(model_path, "rb") as file:
#                         st.download_button(
#                             label="Download trained model",
#                             data=file,
#                             file_name="trained_model.h5",
#                             mime="application/octet-stream"
#                         )

#                     predicted = model.predict(X_test)
#                     y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1))
#                     predicted_original = scaler.inverse_transform(predicted)

#                     plot_predicted_graph(predicted_original, y_test_original)
#         else:
#             st.warning("No data available for the selected inputs.")

#     st.markdown("---")
#     st.markdown("Made with ❤ by Prashuk!")


# if __name__ == '__main__':
#     main()







import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
import tempfile
import plotly.graph_objects as go
import plotly.express as px

# Set page config
st.set_page_config(page_title="Traffic Insight Pro", page_icon="🚗", layout="wide")

# Custom CSS with more modern and sleek design
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');
    
    body {
        background-color: #1E1E1E;
        color: #E0E0E0;
        font-family: 'Roboto', sans-serif;
    }
    
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        padding: 10px 20px;
        border-radius: 5px;
        border: none;
        cursor: pointer;
        font-size: 16px;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background-color: #45a049;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    .stSelectbox {
        background-color: #2C2C2C;
        color: #E0E0E0;
        border-radius: 5px;
    }
    
    .stSlider {
        color: #4CAF50;
    }
    
    h1, h2, h3 {
        color: #4CAF50;
    }
    
    .stPlotlyChart {
        background-color: #2C2C2C;
        border-radius: 10px;
        padding: 10px;
    }
    
    .info-box {
        background-color: #2C2C2C;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load the dataset
@st.cache_data
def load_data():
    data = pd.read_csv("Dataset.csv")
    data['MONAT'] = data['MONAT'].astype(str).str[:6]
    data['DATUM'] = pd.to_datetime(data['MONAT'], format='%Y%m')
    return data

data = load_data()

# Helper functions (keeping the existing ones and adding new ones)

def create_time_series_plot(data, title):
    fig = px.line(data, x='DATUM', y='WERT', color='MONATSZAHL', 
                  title=title, labels={'WERT': 'Number of Accidents', 'DATUM': 'Date'})
    fig.update_layout(
        plot_bgcolor='#2C2C2C',
        paper_bgcolor='#1E1E1E',
        font_color='#E0E0E0',
        legend_title_text='Category'
    )
    return fig

def create_bar_plot(data, title):
    fig = px.bar(data, x='JAHR', y='WERT', color='MONATSZAHL', 
                 title=title, labels={'WERT': 'Number of Accidents', 'JAHR': 'Year'})
    fig.update_layout(
        plot_bgcolor='#2C2C2C',
        paper_bgcolor='#1E1E1E',
        font_color='#E0E0E0',
        legend_title_text='Category'
    )
    return fig

def create_heatmap(data, title):
    pivot_data = data.pivot(index='MONATSZAHL', columns=data['DATUM'].dt.month, values='WERT')
    fig = px.imshow(pivot_data, 
                    labels=dict(x="Month", y="Category", color="Average Accidents"),
                    x=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
                    title=title)
    fig.update_layout(
        plot_bgcolor='#2C2C2C',
        paper_bgcolor='#1E1E1E',
        font_color='#E0E0E0'
    )
    return fig

# Main app structure
def main():
    st.title("🚗 Traffic Insight Pro")
    st.markdown("Advanced Traffic Accident Analysis and Prediction")

    tabs = st.tabs(["📊 Data Visualization", "🤖 Model Training", "ℹ️ About"])

    with tabs[0]:
        st.header("📊 Data Visualization")

        col1, col2 = st.columns([3, 1])

        with col2:
            st.markdown("### 📥 Download Data")
            st.download_button(
                label="Download Full Dataset",
                data=data.to_csv(index=False),
                file_name="Traffic_Accident_Data.csv",
                mime="text/csv",
            )

            st.markdown("### 🔍 Filter Data")
            start_year, end_year = st.select_slider(
                "Select a range of years",
                options=list(range(2000, 2022)),
                value=(2000, 2021)
            )

            category = st.multiselect('Select Categories', data['MONATSZAHL'].unique(), default=data['MONATSZAHL'].unique()[0])

        with col1:
            filtered_data = data[(data['JAHR'] >= start_year) & (data['JAHR'] <= end_year) & (data['MONATSZAHL'].isin(category))]

            st.plotly_chart(create_time_series_plot(filtered_data, "Historical Number of Accidents"), use_container_width=True)
            st.plotly_chart(create_bar_plot(filtered_data, "Comparative Analysis by Year"), use_container_width=True)
            st.plotly_chart(create_heatmap(filtered_data, "Seasonal Analysis of Accidents"), use_container_width=True)

    with tabs[1]:
        st.header("🤖 Model Training")

        col1, col2 = st.columns([2, 1])

        with col1:
            category = st.selectbox('Select Category', data['MONATSZAHL'].unique())
            typeof = st.selectbox('Select Type', data['AUSPRAEGUNG'].unique())
            unique_years = sorted([year for year in data['JAHR'].unique() if year < 2022], reverse=True)
            year = st.selectbox('Select Year', unique_years)
            month = st.selectbox('Select Month for Prediction', range(1, 13))

        with col2:
            st.markdown("### Model Parameters")
            num_layers = st.slider('Number of LSTM Layers', min_value=1, max_value=10, value=3)
            num_nodes = st.slider('Number of Nodes per Layer', min_value=10, max_value=500, value=100)
            epochs = st.slider("Number of Epochs", min_value=10, max_value=500, value=50)

        if st.button('Train Model', key='train_model'):
            scaler = MinMaxScaler(feature_range=(0, 1))
            result = preprocessig_data(data, category, typeof, year, month, scaler)

            if result[0] is not None:
                X_train, X_test, y_train, y_test = result
                with st.spinner('Training in progress...'):
                    model, model_path = train_lstm_model(X_train, y_train, num_layers, num_nodes, epochs)
                    st.success(f"Training completed. Model saved at {model_path}")

                    with open(model_path, "rb") as file:
                        st.download_button(
                            label="Download trained model",
                            data=file,
                            file_name="trained_model.h5",
                            mime="application/octet-stream"
                        )

                    predicted = model.predict(X_test)
                    y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1))
                    predicted_original = scaler.inverse_transform(predicted)

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(y=y_test_original.flatten(), mode='lines+markers', name='Actual'))
                    fig.add_trace(go.Scatter(y=predicted_original.flatten(), mode='lines+markers', name='Predicted'))
                    fig.update_layout(title='Actual vs Predicted Accidents',
                                      xaxis_title='Time Step',
                                      yaxis_title='Number of Accidents',
                                      plot_bgcolor='#2C2C2C',
                                      paper_bgcolor='#1E1E1E',
                                      font_color='#E0E0E0')
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No data available for the selected inputs.")

    with tabs[2]:
        st.header("ℹ️ About Traffic Insight Pro")
        st.markdown("""
        <div class="info-box">
        <h3>🎯 Purpose</h3>
        <p>Traffic Insight Pro is an advanced tool for analyzing and predicting traffic accidents. It provides comprehensive visualizations and machine learning capabilities to help understand patterns and make informed decisions.</p>
        
        <h3>🔧 Features</h3>
        <ul>
            <li>Interactive data visualization with filtering options</li>
            <li>Time series analysis of accident data</li>
            <li>Seasonal and yearly comparisons</li>
            <li>LSTM-based predictive modeling</li>
            <li>Customizable model parameters</li>
        </ul>
        
        <h3>📊 Data Source</h3>
        <p>The data used in this application is sourced from [insert your data source here]. It covers traffic accident records from 2000 to 2021.</p>
        
        <h3>👨‍💻 Developer</h3>
        <p>Developed with ❤️ by Prashuk</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
