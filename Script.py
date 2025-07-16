import pandas as pd
import numpy as np

# --- 1. Sample DataFrames (Modified target_df with 'Target_Category') ---
target_data = {
    'columnsname': ['Customer ID', 'Order Date Field', 'Sales Amount', 'Project Name', 'Product_Identifier', 'Employee Staffing', 'Staff Name', 'Office Location', 'Cust', 'Customer Account', 'Account Number', 'Paramount Pictures', 'Sales'],
    'BusinessName': ['Customer Data', 'Sales Metrics', 'Geo Analytics', 'Project Management', 'Inventory Items', 'HR Management', 'Employee Relations', 'Facility Management', 'Customer Data', 'Finance Data', 'Financial Records', 'Film Studio', 'Sales'],
    'TargetID': range(1, 14),
    # --- Assuming a 'Target_Category' column in target_df ---
    'Target_Category': [
        'Customer_Info', 'Financial_Info', 'Financial_Info', 'Project_Info', 'Product_Info',
        'HR_Info', 'HR_Info', 'Geographic_Info', 'Customer_Info', 'Financial_Info',
        'Financial_Info', 'General_Info', 'Financial_Info'
    ]
}
target_df = pd.DataFrame(target_data)

# --- Rulebook DataFrame ---
rulebook_data = {
    'Data Category': ['Financial_Info', 'Customer_Info', 'HR_Info', 'Product_Info', 'Geographic_Info', 'Financial_Info', 'Project_Info'],
    'Rulebook Element': ['Amount', 'Customer', 'Employee', 'ID', 'Region', 'Account', 'Project Name']
}
rulebook_df = pd.DataFrame(rulebook_data)


print("--- Original DataFrames ---")
print("\nTarget_df:\n", target_df)
print("\nRulebook_df:\n", rulebook_df)
print("-" * 30)

# --- 2. Enhanced Pre-processing of Rulebook for efficient lookup ---
rulebook_df['data_category_lower'] = rulebook_df['Data Category'].str.lower()
rulebook_df['rulebook_element_lower'] = rulebook_df['Rulebook Element'].str.lower()
rulebook_df['rulebook_element_length'] = rulebook_df['Rulebook Element'].str.len()

# Create a dictionary mapping lowercased category to a list of (element_lower, element_full, element_length) tuples
# This avoids iterating over the entire DataFrame rows repeatedly inside the apply function.
rulebook_by_category = {}
for idx, row in rulebook_df.iterrows():
    category = row['data_category_lower']
    if category not in rulebook_by_category:
        rulebook_by_category[category] = []
    rulebook_by_category[category].append((
        row['rulebook_element_lower'],
        row['Rulebook Element'],
        row['rulebook_element_length']
    ))

print("\n--- Pre-processed Rulebook Data (as dictionary for lookup) ---")
# Displaying a part of the dictionary for illustration
for cat, rules in list(rulebook_by_category.items())[:3]:
    print(f"  '{cat}': {rules[:2]}...") # Show first two rules per category
print("...")
print("-" * 30)


# --- 3. Modified process_rulebook_match_for_row function ---
def process_rulebook_match_for_row(target_row, rulebook_rules_dict): # Now accepts the dictionary
    # Get target category, converted to lower for case-insensitive matching
    target_category_lower = str(target_row['Target_Category']).lower() \
                            if pd.notna(target_row['Target_Category']) else ""

    current_column_name_original = target_row['columnsname']
    current_business_name_original = target_row['BusinessName']
    
    ColumnNameLower = str(current_column_name_original).lower() if pd.notna(current_column_name_original) else ""
    BusinessNameLower = str(current_business_name_original).lower() if pd.notna(current_business_name_original) else ""

    final_matched_rulebook_element = np.nan # Initialize output to NaN
    
    possible_rulebook_matches = [] 

    # --- Performance Improvement: Directly get rules for the target's category ---
    # Using .get() ensures no error if category doesn't exist, returns None or empty list
    relevant_rules = rulebook_rules_dict.get(target_category_lower, [])

    if relevant_rules: # Only iterate if there are rules associated with this target category
        for rule_element_lower, rule_element_full, rule_element_length in relevant_rules:
            # Check if column or business name contains the rulebook element (case-insensitive)
            if (rule_element_lower in ColumnNameLower) or \
               (rule_element_lower in BusinessNameLower):
                possible_rulebook_matches.append((rule_element_full, rule_element_length))
    
    # If any rulebook elements matched based on both conditions, select the "best" one
    if possible_rulebook_matches:
        # Sort by element length (descending) then by element name (descending) for ties
        best_rulebook_match_info = sorted(
            possible_rulebook_matches,
            key=lambda x: (x[1], x[0]), 
            reverse=True 
        )[0] # Get the first element (the best match) from the sorted list
        
        final_matched_rulebook_element = best_rulebook_match_info[0] # The matched Rulebook Element

    # Return only the matched rulebook element
    return pd.Series({
        'Matched_Rulebook_Element': final_matched_rulebook_element
    })

# --- 4. Apply the logic to each row of the target DataFrame ---
print("\n--- Applying Rulebook matching logic to Target DataFrame (optimized) ---")
# Pass the pre-processed dictionary to the apply function
new_column_results = target_df.apply(
    lambda row: process_rulebook_match_for_row(row, rulebook_by_category), 
    axis=1
)

# --- 5. Concatenate the new columns back to the original target DataFrame ---
final_target_df_with_matched_rules = pd.concat([target_df, new_column_results], axis=1)

print("\n--- Final Target DataFrame with Matched Rulebook Element ---")
print(final_target_df_with_matched_rules)
print("-" * 30)
