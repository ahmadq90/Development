import pandas as pd
import numpy as np

# --- 1. Define the check_single_match function ---
# (Keeping it as is for its specific DAX-like string matching rules)
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

# --- 2. Sample DataFrames (Re-introducing derisking_df) ---
derisking_data = {
    'Name': ['Cust', 'OrderDate', 'amount', 'region', 'ID', 'Sales', 'prod_id', 'Name', 'Employee_ID', 'Project Name', 'Account No', 'DAS Internal Account Number', 'Description'],
    'Category': ['Customer_Info', 'Financial_Data', 'Financial_Data', 'Geographic_Data', 'Identifier_Data', 'Financial_Data', 'Product_Data', 'General_Info', 'HR_Data', 'Project_Data', 'Financial_Data', 'Financial_Data', 'General_Info'] # New Category column
}
derisking_df = pd.DataFrame(derisking_data)

target_data = {
    'columnsname': ['Customer ID', 'Order Date Field', 'Sales Amount', 'Project Name', 'Product_Identifier', 'Employee Staffing', 'Staff Name', 'Office Location', 'Cust', 'Customer Account', 'Account Number', 'Paramount Pictures', 'Sales'],
    'BusinessName': ['Customer Data', 'Sales Metrics', 'Geo Analytics', 'Project Management', 'Inventory Items', 'HR Management', 'Employee Relations', 'Facility Management', 'Customer Data', 'Finance Data', 'Financial Records', 'Film Studio', 'Sales'],
    'TargetID': range(1, 14),
    # Added Target_Category for rulebook matching demonstration
    'Target_Category': [
        'Customer_Info', 'Financial_Data', 'Financial_Data', 'Project_Data', 'Product_Data',
        'HR_Data', 'HR_Data', 'Geographic_Data', 'Customer_Info', 'Financial_Data',
        'Financial_Data', 'General_Info', 'Financial_Data'
    ]
}
target_df = pd.DataFrame(target_data)

# --- Rulebook DataFrame ---
rulebook_data = {
    'Data Category': ['Financial_Data', 'Customer_Info', 'HR_Data', 'Product_Data', 'Geographic_Data', 'Financial_Data', 'Project_Data'],
    'Rulebook Element': ['Amount', 'Customer', 'Employee', 'ID', 'Region', 'Account', 'Project Name']
}
rulebook_df = pd.DataFrame(rulebook_data)


print("--- Original DataFrames ---")
print("Derisking_df:\n", derisking_df)
print("\nTarget_df:\n", target_df)
print("\nRulebook_df:\n", rulebook_df)
print("-" * 30)

# --- 3. Pre-process DataFrames for efficiency (OPTIMIZED) ---

# Pre-process Derisking Names for efficiency
filtered_derisking_names_data = derisking_df[
    derisking_df['Name'].str.lower() != "description"
].copy()
filtered_derisking_names_data['keywordlower'] = filtered_derisking_names_data['Name'].str.lower()
filtered_derisking_names_data['matchlength'] = filtered_derisking_names_data['Name'].str.len()
filtered_derisking_names_data['Category'] = derisking_df['Category']

# Optimization for Derisking Data:
# Create a dictionary for O(1) exact lookups by keywordlower
derisking_exact_lookup = filtered_derisking_names_data.set_index('keywordlower')['Name'].to_dict()

# Create a list of tuples for partial matching, avoiding iterrows in the loop
# (name, name_lower, length, category)
derisking_partial_match_list = [
    (row['Name'], row['keywordlower'], row['matchlength'], row['Category'])
    for idx, row in filtered_derisking_names_data.iterrows()
]


# Pre-process Rulebook for efficient lookup
rulebook_df['data_category_lower'] = rulebook_df['Data Category'].str.lower()
rulebook_df['rulebook_element_lower'] = rulebook_df['Rulebook Element'].str.lower()
rulebook_df['rulebook_element_length'] = rulebook_df['Rulebook Element'].str.len()

# Create a dictionary mapping lowercased category to a list of (element_lower, element_full, element_length) tuples
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

print("--- Filtered Derisking Data for Matching (Pre-processed) ---")
print(filtered_derisking_names_data)
print("\n--- Pre-processed Rulebook Data (as dictionary for lookup) ---")
# Displaying a part of the dictionary for illustration
for cat, rules in list(rulebook_by_category.items())[:3]:
    print(f"  '{cat}': {rules[:2]}...") # Show first two rules per category
print("...")
print("-" * 30)


# --- 4. OPTIMIZED process_dax_match_logic_for_row function ---
def process_dax_match_logic_for_row(target_row, derisking_exact_lookup, derisking_partial_match_list, rulebook_rules_dict):
    current_column_name_original = target_row['columnsname']
    current_business_name_original = target_row['BusinessName']
    target_category_lower = str(target_row['Target_Category']).lower() if pd.notna(target_row['Target_Category']) else ""

    ColumnNameLower = str(current_column_name_original).lower() if pd.notna(current_column_name_original) else ""
    BusinessNameLower = str(current_business_name_original).lower() if pd.notna(current_business_name_original) else ""

    # --- Optimization: Direct exact match lookups using dictionaries ---
    exact_match_derisking_name = np.nan

    if ColumnNameLower in derisking_exact_lookup:
        exact_match_derisking_name = derisking_exact_lookup[ColumnNameLower]
    
    if pd.isna(exact_match_derisking_name) and BusinessNameLower in derisking_exact_lookup:
        exact_match_derisking_name = derisking_exact_lookup[BusinessNameLower]

    all_valid_derisking_matches_for_this_row = []

    # --- Optimization: Iterate over pre-built list instead of derisking_data_for_match.iterrows() ---
    for derisking_name, derisking_name_lower, derisking_match_length, derisking_category in derisking_partial_match_list:
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

    # --- Look up the Derisking Category for the final match ---
    final_matched_derisking_category = np.nan

    lookup_name_for_category = np.nan 
    if pd.notna(exact_match_derisking_name):
        lookup_name_for_category = exact_match_derisking_name
    elif pd.notna(matched_derisking_keyword_from_partial):
        lookup_name_for_category = matched_derisking_keyword_from_partial
    
    if pd.notna(lookup_name_for_category):
        lookup_keyword_lower = str(lookup_name_for_category).lower()
        # Optimization: Direct lookup for category using the pre-built derisking_exact_lookup or a dedicated category map
        # If we need category from derisking_exact_lookup (which only stores 'Name'), we'd need another dict:
        # derisking_category_lookup = filtered_derisking_names_data.set_index('keywordlower')['Category'].to_dict()
        # final_matched_derisking_category = derisking_category_lookup.get(lookup_keyword_lower, np.nan)
        # For simplicity and given 'Category' is available in derisking_partial_match_list, we can look it up there
        
        # A more robust way to get category would be to pre-build a map from keywordlower to category
        derisking_category_map = {
            item[1]: item[3] for item in derisking_partial_match_list # keywordlower: Category
        }
        final_matched_derisking_category = derisking_category_map.get(lookup_keyword_lower, np.nan)


    # --- OPTIMIZED NEW LOGIC: Match against Rulebook ---
    final_matched_rulebook_element = np.nan

    # Optimization: Directly get rules for the target's category from the pre-built dictionary
    relevant_rules_for_target_category = rulebook_rules_dict.get(target_category_lower, [])

    if relevant_rules_for_target_category: # Only iterate if there are rules associated with this target category
        possible_rulebook_elements = [] 
        for rule_element_lower, rule_element_full, rule_element_length in relevant_rules_for_target_category:
            # Check if column or business name contains the rulebook element (case-insensitive)
            if (rule_element_lower in ColumnNameLower) or \
               (rule_element_lower in BusinessNameLower):
                possible_rulebook_elements.append((rule_element_full, rule_element_length))
        
        if possible_rulebook_elements:
            best_rulebook_match_info = sorted(
                possible_rulebook_elements,
                key=lambda x: (x[1], x[0]), 
                reverse=True 
            )[0]
            final_matched_rulebook_element = best_rulebook_match_info[0]

    # Return all desired fields
    return pd.Series({
        'Matched_Derisking_Name': final_matched_derisking_name,
        'Matched_Derisking_Category': final_matched_derisking_category,
        'Matched_Rulebook_Element': final_matched_rulebook_element # New output column
    })

# --- 5. Apply the logic to each row of the target DataFrame ---
print("\n--- Applying DAX-like matching logic to Target DataFrame (Optimized) ---")
new_column_results = target_df.apply(
    lambda row: process_dax_match_logic_for_row(
        row,
        derisking_exact_lookup,
        derisking_partial_match_list,
        rulebook_by_category
    ),
    axis=1
)

# --- 6. Concatenate the new column back to the original target DataFrame ---
final_target_df_with_matched_names = pd.concat([target_df, new_column_results], axis=1)

print("\n--- Final Target DataFrame with Matched Derisking Names, Category, and Rulebook Element ---")
print(final_target_df_with_matched_names)
print("-" * 30)
