import streamlit as st
import pickle  
import pandas as pd  

# Load the trained model
model = pickle.load(open('best_model.pkl', 'rb'))

# Load the encoder
encoder = pickle.load(open('encoder.pkl', 'rb'))

# Load the scaler
scaler = pickle.load(open('scaler.pkl', 'rb'))


def main():
    st.title("Churn Prediction")
    st.write("Use this app to predict customer churn based on their profile.")

    # User input
    credit_score = st.number_input("Credit Score:")
    geography = st.selectbox("Geography:", ["France", "Germany", "Spain"])  # Adjust options based on data
    gender = st.selectbox("Gender:", ["Male", "Female"])
    age = st.number_input("Age:",0,100)
    tenure = st.number_input("Tenure (years with company):",0,10)
    balance = st.number_input("Balance (account balance):")
    num_of_products = st.number_input("Number of Products:",1,4)
    has_cr_card = st.selectbox("Has Credit Card?", ["Yes", "No"])
    is_active_member = st.selectbox("Is Active Member?", ["Yes", "No"])
    estimated_salary = st.number_input("Estimated Salary:")

    # change input to df
    data = pd.DataFrame({
        "CreditScore": [credit_score],
        "Geography": [geography],
        "Gender": [gender],
        "Age": [age],
        "Tenure": [tenure],
        "Balance": [balance],
        "NumOfProducts": [num_of_products],
        "HasCrCard": [1 if has_cr_card == "Yes" else 0],
        "IsActiveMember": [1 if is_active_member == "Yes" else 0],
        "EstimatedSalary": [estimated_salary]
    })
    
    print(data)

    # Make prediction button
    if st.button("Predict Churn Risk"):
        all_filled = credit_score and geography and gender and age and tenure and balance and num_of_products and has_cr_card and is_active_member and estimated_salary
        if not all_filled:
            st.error("Please fill in all fields before submitting.")
            return

        cat = ['Geography', 'Gender']
        con = ['CreditScore', 'Balance', 'EstimatedSalary', 'Age']
        
        # encode
        data_subset = data[cat]
        data_encoded = pd.DataFrame(encoder.transform(data_subset).toarray(), columns=encoder.get_feature_names_out(cat))
        data = data.reset_index(drop=True)
        data = pd.concat([data, data_encoded], axis=1)
        data.drop(cat, axis=1, inplace=True)

        # scaling
        data[con] = scaler.transform(data[con])
        
        with st.spinner("Making prediction..."):
            prediction = model.predict(data)[0]

            if prediction == 1:
                st.write("Predicted: **CHURN**")
            else:
                st.write("Predicted: **NOT CHURN**")

if __name__ == "__main__":
    main()
