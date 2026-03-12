import pandas as pd
import hashlib
from colorama import Fore, Style
from pyfiglet import Figlet

"""
Signal Detection in Pharmacovigilance

This script performs disproportionality analysis on a dataset of adverse event reports to identify potential signals.

The main steps are:
1.  Load and preprocess the adverse event data.
2.  For each pair, construct a 2x2 contingency table.
3.  Calculate disproportionality measures:
    - Proportional Reporting Ratio (PRR)
    - Chi-Squared (χ²) statistic
4.  Identify statistically significant signals based on predefined thresholds.
5.  Cross-reference signals with known drug safety profiles to distinguish
    between new potential risks and known indications and side effects.
6.  Export the results to a CSV file.
7.  Submit the top 3 Signals requiring review to the Health Authority.
"""

PRR_THRESHOLD = 2.0
CHI_SQUARED_THRESHOLD = 4.0


def load_data(file_path: str) -> pd.DataFrame:
    """Loads data from a CSV file into a pandas DataFrame."""
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    return df


def clean_input_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs basic cleaning and validation of the input DataFrame.
    
    TODO: Validate and clean/normalize the data
         - Validation ex: are there impossible patient ages, incorrect dates, etc.
         - Normalization ex: standardize text to lower case, remove leading/trailing white space, etc.
    """
    cleaned_df = df.copy()  # Start with a copy of the original DataFrame
    # --- TODO: YOUR CODE HERE ---

    # ----------------------------
    return cleaned_df


def calculate_prr(df: pd.DataFrame) -> pd.Series:
    """
    Calculates the Proportional Reporting Ratio (PRR).

    PRR is a measure of disproportionality used in pharmacovigilance. It
    compares the proportion of reports for a given event with a specific drug
    to the proportion of reports for the same event with all other drugs.

    Formula:
        PRR = (A / (A + B)) / (C / (C + D))

    Where:
        A, B, C, D are the values from the 2x2 contingency table.
        
    Example:
        A = 10 (Drug X, Event Y)
        B = 190 (Drug X, Other Events)
        C = 50 (Other Drugs, Event Y)
        D = 9750 (Other Drugs, Other Events)
        
        Proportion for Drug X = 10 / (10 + 190) = 0.05
        Proportion for Other Drugs = 50 / (50 + 9750) = 0.0051
        PRR = 0.05 / 0.0051 ≈ 9.8
    """
    validate_required_cols(df, {'A', 'B', 'C', 'D'})
    # --- TODO: YOUR CODE HERE ---

    # ----------------------------
    return pd.Series()


def calculate_chi_squared(df: pd.DataFrame) -> pd.Series:
    """
    Calculates the Chi-Squared (χ²) statistic.

    The Chi-Squared test is used to assess the statistical significance of the
    association between a drug and an adverse event. A higher Chi-Squared
    value indicates a stronger association that is less likely to be due to
    random chance.

    Formula:
        χ² = N * (|AD - BC| - N/2)² / ((A+B)(C+D)(A+C)(B+D))

    Where:
        N = Total number of reports (A + B + C + D)
        A, B, C, D are the values from the 2x2 contingency table.
        
    Example (using the same numbers as PRR):
        A=10, B=190, C=50, D=9750
        N = 10 + 190 + 50 + 9750 = 10000
        
        Numerator = 10000 * (|10*9750 - 190*50| - 10000/2)²
        Denominator = (10+190) * (50+9750) * (10+50) * (190+9750)
        χ² = 58.93
    """
    validate_required_cols(df, {'A', 'B', 'C', 'D'})
    # --- TODO: YOUR CODE HERE ---

    # ----------------------------
    return pd.Series()


def calculate_statistics(df: pd.DataFrame, background_incidence_path: str) -> pd.DataFrame:
    """
    Calculates PRR and Chi-Squared statistics for all drug-event pairs.

    This function takes a DataFrame of adverse event reports and performs
    vectorized calculations to determine the PRR and Chi-Squared values for
    every unique drug-event combination found in the data.

    Contingency Table for a given drug-event pair:
                   | Event      | No Event
    -----------------------------------------
    Drug           |     A      |     B
    -----------------------------------------
    Other Drugs    |     C      |     D

    A = Count of reports with the specific drug and specific event.
    B = Count of reports with the specific drug and any other event.
    C = Count of reports with any other drug and the specific event.
    D = Count of reports with any other drug and any other event.
    """
    results_df = df.copy()  # Start with a copy of the original DataFrame
    background_incidence = load_data(background_incidence_path)

    # --- TODO: YOUR CODE HERE ---

    # ----------------------------
    results_df['prr'] = calculate_prr(results_df)
    results_df['chi_squared'] = calculate_chi_squared(results_df)
    results_df['case_count'] = results_df['A']
    return results_df


def analyze_signals(df: pd.DataFrame, drug_profiles_path: str) -> pd.DataFrame:
    """
    Identifies statistically significant signals (PRR >= 2, X2 >= 4) and cross-references them
    with known drug safety information to determine whether these signals require further investigation.
    The drug safety information lists what the drug is used to treat and it's known side effects.

    This function should modify output a copy of the input dataframe
    with an appended "requires_investigation" column with boolean values
    """
    validate_required_cols(df, {'prr', 'chi_squared', 'drug_name', 'event_term'})
    results_df = df.copy()  # Start with a copy of the original DataFrame
    drug_profiles = pd.read_json(drug_profiles_path)

    # --- TODO: YOUR CODE HERE ---

    # ----------------------------

    return results_df


def validate_required_cols(df: pd.DataFrame, required_cols: set):
    if not required_cols.issubset(df.columns):
        raise ValueError(f"DataFrame is missing required columns. Must contain: {list(required_cols)}")


def report_to_health_authority(suspect_pairs: list):
    """
    Submits your final list of top outlier signals to the Health Authority.
    The Authority only has resources to investigate the most severe, undeniable statistical anomalies.

    Expected format: suspect_pairs = [("Drug A", "Event X"), ("Drug B", "Event Y"), ("Drug C", "Event Z")]
    """
    print("\n" + "=" * 60)
    print("📡 TRANSMITTING FINDINGS TO HEALTH AUTHORITY...")
    print("=" * 60)
    clean_pairs = []
    for drug, event in suspect_pairs:
        clean_pairs.append(f"{str(drug).strip().upper()}:{str(event).strip().upper()}")

    clean_pairs = sorted(list(set(clean_pairs)))

    submission_string = "|".join(clean_pairs)
    submission_hash = hashlib.sha256(submission_string.encode('utf-8')).hexdigest()
    TARGET_HASH = "YOUR_GENERATED_HASH_GOES_HERE"

    fig = Figlet(font='standard')
    if submission_hash == TARGET_HASH:
        print(f"{Fore.GREEN}SUCCESS: The Health Authority has verified your statistical anomalies.{Style.RESET_ALL}")
        print(f"Based on your compelling data, emergency safety reviews have been triggered for:")
        for pair in clean_pairs:
            d, e = pair.split(':')
            print(f" 🚨 {Fore.CYAN}{d.title()}{Style.RESET_ALL} -> {Fore.RED}{e.capitalize()}{Style.RESET_ALL}")
        print(f"\n{Fore.YELLOW}Incredible work, investigator.{Style.RESET_ALL}\n")
    else:
        print(f"\n{Fore.YELLOW}{fig.renderText('SUBMISSION REJECTED')}{Style.RESET_ALL}")
        print(f"{Fore.RED}The Health Authority reviewed your list and found it lacking.{Style.RESET_ALL}")
        print("They stated: 'Your list either contains background noise, or you missed a major threat.'")


def main(
    ae_reports_path: str,
    background_incidence_path: str,
    drug_profiles_path: str,
    output_path: str,
):
    # 1. Load Data
    raw_df = load_data(ae_reports_path)

    # 2. Clean Data
    clean_df = clean_input_data(raw_df)

    # 3. Disproportionality calculations
    results_df = calculate_statistics(clean_df, background_incidence_path)

    # 4. Analyze signals
    final_df = analyze_signals(results_df, drug_profiles_path)

    # 5. Export statistically significant signals to a CSV
    final_df = final_df.sort_values(by=['prr', 'chi_squared'], ascending=False)
    final_df.to_csv(output_path, index=False)
    print(f"Results exported to {output_path}")

    # 6. Submit the top 3 outliers to health authorities
    validate_required_cols(final_df, {'prr', 'requires_review'})
    outliers_df = final_df[final_df['requires_review'] == True].head(3)
    suspect_pairs = list(zip(outliers_df['drug_name'], outliers_df['event_term']))
    report_to_health_authority(suspect_pairs)


if __name__ == "__main__":
    main(
        ae_reports_path="data/raw_ae_reports.csv",
        background_incidence_path="data/background_incidence.csv",
        drug_profiles_path="data/drug_safety_profiles.json",
        output_path="signal_detection_results.csv",
    )
