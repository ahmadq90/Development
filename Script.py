import pandas as pd
import numpy as np
import re

# --- 1. Define the check_single_match function ---
# This function will be slightly modified to be more efficient if used in a vectorized context,
# but for now, we'll keep it as is, recognizing that we want to minimize its calls.
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
        # Using regex for short names can be more precise and potentially faster for multiple checks
        # \b for word boundary, but also handling underscores
        pattern = r"(?i)\b" + re.escape(derisking_name_lower) + r"\b|_" + re.escape(derisking_name_lower) + r"_|_" + re.escape(derisking_name_lower) + r"\b"
        return re.search(pattern, target_text_lower) is not None
    else:
        return derisking_name_lower in target_text_lower

# --- 2. Sample DataFrames (keeping as is) ---
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

# Simulate 100,000 rows for target_df
target_df = pd.concat([target_df] * 8000, ignore_index=True) # 13 * 8000 = 104,000 rows

print("--- Original DataFrames ---")
print("Derisking_df:\n", derisking_df)
print("\nTarget_df (first 5 rows):\n", target_df.head())
print(f"Target_df has {len(target_df)} rows.")
print("-" * 30)

# --- 3. Optimized Pre-processing of Derisking Data ---
filtered_derisking_names_data = derisking_df[
    derisking_df['Name'].str.lower() != "description"
].copy()
filtered_derisking_names_data['keywordlower'] = filtered_derisking_names_data['Name'].str.lower()
filtered_derisking_names_data['matchlength'] = filtered_derisking_names_data['Name'].str.len()

derisking_exact_lookup = filtered_derisking_names_data.set_index('keywordlower')['Name'].to_dict()
derisking_category_map = filtered_derisking_names_data.set_index('keywordlower')['Category'].to_dict()

# Create separate lists for short and long derisking names for optimized regex pattern creation
short_derisking_names = []
long_derisking_names = []

for _, row in filtered_derisking_names_data.iterrows():
    name_full = row['Name']
    name_lower = row['keywordlower']
    if len(name_full) <= 3:
        short_derisking_names.append((name_full, name_lower, len(name_full)))
    else:
        long_derisking_names.append((name_full, name_lower, len(name_full)))

print("--- Filtered Derisking Data for Matching (Pre-processed) ---")
print(filtered_derisking_names_data)
print("-" * 30)

# --- 4. VECTORIZED DAX-like Matching Logic ---
print("\n--- Applying DAX-like matching logic to Target DataFrame (Vectorized) ---")

# Step 1: Exact Matches
# Apply .lower() once for the entire series
target_df['columnsname_lower'] = target_df['columnsname'].astype(str).str.lower()
target_df['BusinessName_lower'] = target_df['BusinessName'].astype(str).str.lower()

# Initialize columns for results
target_df['Matched_Derisking_Name'] = np.nan
target_df['Matched_Derisking_Category'] = np.nan
target_df['Temp_Exact_Match_Name'] = np.nan
target_df['Temp_Partial_Match_Name'] = np.nan # To store the best partial match before final logic

# Exact match for columnsname
exact_match_col_mask = target_df['columnsname_lower'].isin(derisking_exact_lookup.keys())
target_df.loc[exact_match_col_mask, 'Temp_Exact_Match_Name'] = \
    target_df.loc[exact_match_col_mask, 'columnsname_lower'].map(derisking_exact_lookup)

# Exact match for BusinessName (only where columnsname didn't have an exact match)
exact_match_bus_mask = target_df['BusinessName_lower'].isin(derisking_exact_lookup.keys())
# Apply only if Temp_Exact_Match_Name is still NaN (no match from column name)
target_df.loc[target_df['Temp_Exact_Match_Name'].isna() & exact_match_bus_mask, 'Temp_Exact_Match_Name'] = \
    target_df.loc[target_df['Temp_Exact_Match_Name'].isna() & exact_match_bus_mask, 'BusinessName_lower'].map(derisking_exact_lookup)

# Step 2: Partial Matches (more complex to vectorize directly, but we can improve it)
# We need to find the "best" partial match (longest, then alphabetical)
# This will likely still involve some form of iteration, but we can minimize string operations.

# Function to find the best partial match for a single cell, optimized for vectorization readiness
def find_best_partial_match_for_cell(text_original, derisking_partial_match_list):
    if pd.isna(text_original) or not isinstance(text_original, str):
        return np.nan

    text_lower = text_original.lower()
    all_valid_matches = []

    for derisking_name, derisking_name_lower, derisking_match_length in derisking_partial_match_list:
        if check_single_match(derisking_name, derisking_name_lower, text_original): # Reuse original check
            all_valid_matches.append((derisking_name, derisking_match_length))

    if all_valid_matches:
        # Sort by match length (desc), then by name (desc for alphabetical tie-break based on original logic)
        best_match_info = sorted(all_valid_matches, key=lambda x: (x[1], x[0]), reverse=True)[0]
        return best_match_info[0]
    return np.nan

# Apply partial match logic - this is still an apply, but `check_single_match` is already optimized
# and we are not doing a full Cartesian product of checks
target_df['Partial_Match_Col'] = target_df['columnsname'].apply(
    lambda x: find_best_partial_match_for_cell(x, short_derisking_names + long_derisking_names)
)
target_df['Partial_Match_Bus'] = target_df['BusinessName'].apply(
    lambda x: find_best_partial_match_for_cell(x, short_derisking_names + long_derisking_names)
)

# Combine the best partial matches from Column and Business Name
# Prioritize column match if both exist, or pick the "better" one if both are valid partials
def combine_partial_matches(row):
    col_match = row['Partial_Match_Col']
    bus_match = row['Partial_Match_Bus']

    if pd.notna(col_match) and pd.notna(bus_match):
        # Retrieve length for comparison
        col_len = next((d[2] for d in filtered_derisking_names_data.itertuples() if d[1] == col_match), 0)
        bus_len = next((d[2] for d in filtered_derisking_names_data.itertuples() if d[1] == bus_match), 0)
        if col_len > bus_len:
            return col_match
        elif bus_len > col_len:
            return bus_match
        else: # Lengths are equal, sort alphabetically (original logic chose reverse=True, so "Z" before "A")
            return max(col_match, bus_match)
    elif pd.notna(col_match):
        return col_match
    elif pd.notna(bus_match):
        return bus_match
    return np.nan

target_df['Temp_Partial_Match_Name'] = target_df.apply(combine_partial_matches, axis=1)


# Step 3: Specific 'Name' partial match logic
# This requires checking the original column name for 'employee' or 'staff'
name_partial_mask = (target_df['Temp_Partial_Match_Name'].astype(str).str.lower() == "name") & \
                    (target_df['columnsname_lower'].str.contains("employee|staff", na=False))

target_df.loc[name_partial_mask, 'Temp_Partial_Match_Name'] = 'Partialmatch: Employee'

# Step 4: Determine Final Matched Derisking Name
# Priority: Exact Match > Triggered Partial Match (from "Name" logic) > Other Partial Match
target_df['Matched_Derisking_Name'] = target_df['Temp_Exact_Match_Name'].fillna(target_df['Temp_Partial_Match_Name'])

# Step 5: Look up Category
# Map using the derived Matched_Derisking_Name
# Ensure the key for mapping is always lowercased string for lookup in derisking_category_map
target_df['Matched_Derisking_Category'] = target_df['Matched_Derisking_Name'].astype(str).str.lower().map(derisking_category_map).fillna(np.nan)


# Clean up temporary columns
target_df = target_df.drop(columns=[
    'columnsname_lower', 'BusinessName_lower',
    'Temp_Exact_Match_Name', 'Temp_Partial_Match_Name',
    'Partial_Match_Col', 'Partial_Match_Bus'
])

print("\n--- Final Target DataFrame with Matched Derisking Names and Category (Vectorized) ---")
print(target_df.head())
print("-" * 30)
