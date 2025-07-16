import pandas as pd
import numpy as np

# --- 1. Define the check_single_match function ---
def check_single_match(derisking_name_full, derisking_name_lower, target_text_original):
    """
    Checks if a single derisking name matches a single target string based on DAX SEARCH rules.
    Returns True if matched, False otherwise.
    """
    if pd.isna(target_text_original) or not isinstance(target_text_original, str):
        return False

    target_text_lower = target_text_original.lower()

    is_short_derisking_name = len(derisking_name_full) <= 3

    if is_short_derisking_name:
        target_text_lower_padded = f" {target_text_lower} "
        if (f" {derisking_name_lower} " in target_text_lower_padded) or \
           (f"_{derisking_name_lower} " in target_text_lower) or \
           (f"_{derisking_name_lower}_" in target_text_lower):
            return True
    else:
        if derisking_name_lower in target_text_lower:
            return True
    return False

# --- 2. Sample DataFrames ---
derisking_data = {
    'Name': ['Cust', 'OrderDate', 'amount', 'region', 'ID', 'Sales', 'prod_id', 'Name', 'Employee_ID', 'Project Name', 'Account No', 'DAS Internal Account Number', 'Description'],
    'Category': ['Customer_Info', 'Financial_Data', 'Financial_Data', 'Geographic_Data', 'Identifier_Data', 'Financial_Data', 'Product_Data', 'General_Info', 'HR_Data', 'Project_Data', 'Financial_Data', 'Financial_Data', 'General_Info']
}
derisking_df = pd.DataFrame(derisking_data)

target_data = {
    'columnsname': ['Customer ID', 'Order Date Field', 'Sales Amount', 'Project Name', 'Product_Identifier', 'Employee Staffing', 'Staff Name', 'Office Location', 'Cust', 'Customer Account', 'Account Number', 'Paramount Pictures', 'Sales'],
    'BusinessName': ['Customer Data', 'Sales Metrics', 'Geo Analytics', 'Project Management', 'Inventory Items', 'HR Management', 'Employee Relations', 'Facility Management', 'Customer Data', 'Finance Data', 'Financial Records', 'Film Studio', 'Sales'],
    'TargetID': range(1, 14)
}
target_df = pd.DataFrame(target_data)

print("--- Original DataFrames ---")
print("Derisking_df:\n", derisking_df)
print("\nTarget_df:\n", target_df)
print("-" * 30)

# --- 3. Optimized Pre-processing of Derisking Data ---
# This part now creates the efficient lookup structures
filtered_derisking_names_data = derisking_df[
    derisking_df['Name'].str.lower() != "description"
].copy()
filtered_derisking_names_data['keywordlower'] = filtered_derisking_names_data['Name'].str.lower()
filtered_derisking_names_data['matchlength'] = filtered_derisking_names_data['Name'].str.len()
filtered_derisking_names_data['Category'] = derisking_df['Category']

# Create dictionaries for O(1) exact lookups
derisking_exact_lookup = filtered_derisking_names_data.set_index('keywordlower')['Name'].to_dict()
derisking_category_map = filtered_derisking_names_data.set_index('keywordlower')['Category'].to_dict()

# Create a list of tuples for partial matching, avoiding iterrows in the loop
derisking_partial_match_list = list(zip(
    filtered_derisking_names_data['Name'],
    filtered_derisking_names_data['keywordlower'],
    filtered_derisking_names_data['matchlength']
))

print("--- Filtered Derisking Data for Matching (Pre-processed) ---")
print(filtered_derisking_names_data)
print("-" * 30)

# --- 4. OPTIMIZED process_dax_match_logic_for_row function ---
def process_dax_match_logic_for_row(target_row, derisking_exact_lookup, derisking_category_map, derisking_partial_match_list):
    current_column_name_original = target_row['columnsname']
    current_business_name_original = target_row['BusinessName']
    
    ColumnNameLower = str(current_column_name_original).lower() if pd.notna(current_column_name_original) else ""
    BusinessNameLower = str(current_business_name_original).lower() if pd.notna(current_business_name_original) else ""

    exact_match_derisking_name = np.nan
    
    # --- Optimization: Use dictionary for O(1) lookup instead of DataFrame filter ---
    if ColumnNameLower in derisking_exact_lookup:
        exact_match_derisking_name = derisking_exact_lookup[ColumnNameLower]
    
    if pd.isna(exact_match_derisking_name) and BusinessNameLower in derisking_exact_lookup:
        exact_match_derisking_name = derisking_exact_lookup[BusinessNameLower]

    all_valid_derisking_matches_for_this_row = []

    # --- Optimization: Iterate over pre-built list instead of .iterrows() ---
    for derisking_name, derisking_name_lower, derisking_match_length in derisking_partial_match_list:
        match_in_col = check_single_match(derisking_name, derisking_name_lower, current_column_name_original)
        if match_in_col:
            all_valid_derisking_matches_for_this_row.append((derisking_name, derisking_match_length, 'Column'))

        match_in_bus = check_single_match(derisking_name, derisking_name_lower, current_business_name_original)
        if match_in_bus:
            all_valid_derisking_matches_for_this_row.append((derisking_name, derisking_match_length, 'Business'))
    
    matched_derisking_keyword_from_partial = np.nan
    if all_valid_derisking_matches_for_this_row:
        best_match_info = sorted(all_valid_derisking_matches_for_this_row,
                                 key=lambda x: (x[1], x[0]), reverse=True)[0]
        matched_derisking_keyword_from_partial = best_match_info[0]

    triggered_partial_match_result = np.nan
    if pd.notna(matched_derisking_keyword_from_partial) and str(matched_derisking_keyword_from_partial).lower() == "name":
        if ("employee" in ColumnNameLower) or ("staff" in ColumnNameLower):
            triggered_partial_match_result = 'Partialmatch: Employee'

    # --- Determine the final matched Derisking Name ---
    final_matched_derisking_name = np.nan

    if pd.notna(exact_match_derisking_name):
        final_matched_derisking_name = exact_match_derisking_name
    elif pd.notna(triggered_partial_match_result):
        final_matched_derisking_name = triggered_partial_match_result
    elif pd.notna(matched_derisking_keyword_from_partial):
        final_matched_derisking_name = matched_derisking_keyword_from_partial

    # --- NEW LOGIC: Look up the additional field (e.g., 'Category') ---
    final_matched_derisking_category = np.nan

    lookup_name_for_category = np.nan
    if pd.notna(exact_match_derisking_name):
        lookup_name_for_category = exact_match_derisking_name
    elif pd.notna(matched_derisking_keyword_from_partial):
        lookup_name_for_category = matched_derisking_keyword_from_partial
    
    if pd.notna(lookup_name_for_category):
        lookup_keyword_lower = str(lookup_name_for_category).lower()
        # --- Optimization: Use dictionary for O(1) category lookup ---
        final_matched_derisking_category = derisking_category_map.get(lookup_keyword_lower, np.nan)

    # Return both the matched name and the new field
    return pd.Series({
        'Matched_Derisking_Name': final_matched_derisking_name,
        'Matched_Derisking_Category': final_matched_derisking_category
    })

# --- 5. Apply the logic to each row of the target DataFrame ---
print("\n--- Applying DAX-like matching logic to Target DataFrame (Optimized) ---")
new_column_results = target_df.apply(
    lambda row: process_dax_match_logic_for_row(
        row,
        derisking_exact_lookup,
        derisking_category_map,
        derisking_partial_match_list
    ),
    axis=1
)

# --- 6. Concatenate the new column back to the original target DataFrame ---
final_target_df_with_matched_names = pd.concat([target_df, new_column_results], axis=1)

print("\n--- Final Target DataFrame with Matched Derisking Names and Category ---")
print(final_target_df_with_matched_names)
print("-" * 30)
