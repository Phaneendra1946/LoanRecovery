import pandas as pd
import numpy as np
import os

def generate_data(num_samples=2000):
    np.random.seed(42)
    
    # Features
    borrower_age = np.random.randint(21, 66, num_samples)
    annual_income = np.random.uniform(15000, 200000, num_samples)
    loan_amount = np.random.uniform(5000, 100000, num_samples)
    interest_rate = np.random.uniform(5, 25, num_samples)
    loan_tenure_months = np.random.randint(6, 61, num_samples)
    credit_score = np.random.randint(300, 851, num_samples)
    num_previous_defaults = np.random.randint(0, 6, num_samples)
    employment_types = ['Salaried', 'Self-Employed', 'Unemployed', 'Retired']
    employment_type = np.random.choice(employment_types, num_samples, p=[0.5, 0.3, 0.1, 0.1])
    months_since_last_payment = np.random.randint(0, 25, num_samples)
    total_paid_fraction = np.random.uniform(0, 1, num_samples)
    total_paid_amount = loan_amount * total_paid_fraction
    debt_to_income_ratio = np.random.uniform(0.1, 0.8, num_samples)
    collection_attempts = np.random.randint(0, 11, num_samples)
    
    # Base probability calculation for recovery
    # Higher income, higher credit score, lower defaults, lower months since last payment -> higher probability
    prob = (
        (annual_income / 200000) * 0.15 +
        ((credit_score - 300) / 550) * 0.25 -
        (num_previous_defaults / 5) * 0.2 -
        (months_since_last_payment / 24) * 0.2 -
        (debt_to_income_ratio / 0.8) * 0.1 +
        (total_paid_fraction) * 0.1
    )
    # Scale and add noise
    prob = prob + np.random.normal(0, 0.1, num_samples)
    
    # Adjust threshold to get ~35% recovery rate
    threshold = np.percentile(prob, 65) 
    recovery_status = (prob >= threshold).astype(int)
    
    df = pd.DataFrame({
        'loan_id': range(1, num_samples + 1),
        'borrower_age': borrower_age,
        'annual_income': annual_income,
        'loan_amount': loan_amount,
        'interest_rate': interest_rate,
        'loan_tenure_months': loan_tenure_months,
        'credit_score': credit_score,
        'num_previous_defaults': num_previous_defaults,
        'employment_type': employment_type,
        'months_since_last_payment': months_since_last_payment,
        'total_paid_amount': total_paid_amount,
        'debt_to_income_ratio': debt_to_income_ratio,
        'collection_attempts': collection_attempts,
        'recovery_status': recovery_status
    })
    
    return df

if __name__ == '__main__':
    df = generate_data()
    output_path = os.path.join(os.path.dirname(__file__), 'loan_recovery_data.csv')
    df.to_csv(output_path, index=False)
    print(f"Generated synthetic dataset with {len(df)} rows.")
    print(f"Recovery rate: {df['recovery_status'].mean():.2%}")
    print(f"Saved to {output_path}")
