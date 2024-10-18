import streamlit as st
import pandas as pd
from data_processing import preprocess_data
from machine_learning import train_random_forest, train_logistic_regression, train_support_vector_machine
from sklearn.exceptions import NotFittedError

def main():
    st.title("Machine Learning Web App")

    uploaded_file = st.file_uploader("Upload Dataset", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        st.subheader("Dataset")
        st.write(df)

        # Check for categorical data and encode them if present
        if df.select_dtypes(include=['object']).empty:
            st.write("No categorical columns detected.")
        else:
            st.write("Dataset contains categorical data. Encoding it automatically...")
            df = pd.get_dummies(df)  # One-hot encoding for categorical data
            st.write(df)

        target_column = st.sidebar.selectbox("Select Target Column", df.columns)
        test_size = st.sidebar.slider("Test Size", 0.1, 0.5, 0.3, 0.05)
        random_state = st.sidebar.number_input("Random State", 0, 100, 42)
        algorithm = st.sidebar.selectbox("Select Algorithm", ["Random Forest", "Logistic Regression", "Support Vector Machine"])
        
        if st.sidebar.button("Train Model"):
            st.subheader("Training Model...")
            try:
                if algorithm == "Random Forest":
                    model, accuracy = train_random_forest(df, target_column, test_size, random_state)
                elif algorithm == "Logistic Regression":
                    model, accuracy = train_logistic_regression(df, target_column, test_size, random_state)
                elif algorithm == "Support Vector Machine":
                    model, accuracy = train_support_vector_machine(df, target_column, test_size, random_state)
                else:
                    st.error("Invalid algorithm selected.")
                    return
                
                st.write("Model trained successfully!")
                st.write(f"Accuracy: {accuracy}")
            except NotFittedError as e:
                st.error(f"Error: {e}")
            except Exception as e:
                st.error(f"Dataset contains categorical data or an unexpected issue occurred: {e}")

if __name__ == "__main__":
    main()
