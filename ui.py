#!/usr/bin/env python3
"""
UI module for DD Analyzer System debugging and analysis.

This module provides user interface functions for better debugging
and analysis of Double Deduct detection system.
"""

import pandas as pd
from datetime import datetime
from analyzer import (
    load_and_prepare_data, analyze_all_transactions, 
    debug_card_trips, get_dd_summary, is_subsidi_time
)


def print_header(title):
    """Print a formatted header for UI sections."""
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)


def print_subsection(title):
    """Print a formatted subsection header."""
    print(f"\n--- {title} ---")


def show_analysis_summary(df):
    """
    Display comprehensive analysis summary.
    
    Parameters:
        df (DataFrame): Analyzed transaction data
    """
    print_header("DD ANALYZER SUMMARY")
    
    total_transactions = len(df)
    dd_transactions = df[df["is_dd"] == True]
    dd_count = len(dd_transactions)
    total_refund = dd_transactions["dd_refund"].sum()
    
    print(f"Total Transactions: {total_transactions:,}")
    print(f"DD Detected: {dd_count:,} ({dd_count/total_transactions*100:.2f}%)")
    print(f"Total Refund Amount: Rp {total_refund:,}")
    
    # DD breakdown by type
    print_subsection("DD Breakdown by Detection Step")
    dd_steps = dd_transactions["dd_step"].value_counts()
    for step, count in dd_steps.items():
        percentage = count / dd_count * 100
        print(f"  {step}: {count:,} ({percentage:.1f}%)")
    
    # Subsidy-related DD
    subsidy_dd = dd_transactions[dd_transactions["dd_step"].str.contains("Subsidi", na=False)]
    if len(subsidy_dd) > 0:
        print_subsection("Subsidy-related DD")
        print(f"  Subsidy DD Count: {len(subsidy_dd):,}")
        print(f"  Subsidy DD Refund: Rp {subsidy_dd['dd_refund'].sum():,}")
    
    # UNBLOKIR analysis
    unblokir_dd = dd_transactions[dd_transactions["dd_step"].str.contains("UNBLOKIR", na=False)]
    if len(unblokir_dd) > 0:
        print_subsection("UNBLOKIR-related DD")
        print(f"  UNBLOKIR DD Count: {len(unblokir_dd):,}")
        print(f"  UNBLOKIR DD Refund: Rp {unblokir_dd['dd_refund'].sum():,}")


def show_card_analysis(df, card_number, show_details=True):
    """
    Display detailed analysis for a specific card.
    
    Parameters:
        df (DataFrame): Analyzed transaction data
        card_number (str): Card number to analyze
        show_details (bool): Whether to show detailed transaction list
    """
    print_header(f"CARD ANALYSIS: {card_number}")
    
    card_data = df[df["card_number_var"] == card_number].sort_values("trx_on")
    
    if len(card_data) == 0:
        print("Card not found in dataset.")
        return
    
    total_trx = len(card_data)
    dd_trx = card_data[card_data["is_dd"] == True]
    dd_count = len(dd_trx)
    
    print(f"Total Transactions: {total_trx}")
    print(f"DD Detected: {dd_count} ({dd_count/total_trx*100:.2f}%)")
    print(f"Date Range: {card_data['trx_on'].min()} to {card_data['trx_on'].max()}")
    
    if dd_count > 0:
        print(f"Total DD Refund: Rp {dd_trx['dd_refund'].sum():,}")
    
    if show_details:
        print_subsection("Transaction Details")
        for idx, row in card_data.iterrows():
            dd_mark = "🚨 DD" if row["is_dd"] else "✅ OK"
            subsidy_mark = "🌅" if is_subsidi_time(row["trx_on"]) else ""
            
            time_str = row["trx_on"].strftime("%H:%M:%S")
            fare_str = f"Rp {row['fare_int']}" if pd.notna(row["fare_int"]) else "N/A"
            
            print(f"  idx {idx:3d}: {time_str} | {row['trx']:20s} | {fare_str:10s} | {dd_mark} {subsidy_mark}")
            
            if row["is_dd"]:
                print(f"       └─ {row['dd_step']}")
                if row["dd_refund"] > 0:
                    print(f"       └─ Refund: Rp {row['dd_refund']:,}")
    
    # Show trip analysis
    print_subsection("Trip Analysis")
    debug_card_trips(df, card_number, debug=True)


def show_subsidy_analysis(df):
    """
    Display subsidy time analysis.
    
    Parameters:
        df (DataFrame): Analyzed transaction data
    """
    print_header("SUBSIDY TIME ANALYSIS")
    
    # Filter transactions in subsidy hours
    subsidy_transactions = df[df.apply(lambda row: is_subsidi_time(row["trx_on"]), axis=1)]
    
    print(f"Total Subsidy Hour Transactions: {len(subsidy_transactions):,}")
    
    if len(subsidy_transactions) > 0:
        # Analyze tariff patterns
        subsidy_with_fare = subsidy_transactions[subsidy_transactions["fare_int"].notna()]
        fare_counts = subsidy_with_fare["fare_int"].value_counts().sort_index()
        
        print_subsection("Fare Distribution in Subsidy Hours")
        for fare, count in fare_counts.items():
            percentage = count / len(subsidy_with_fare) * 100
            expected = "✅ Expected" if fare == 2000 else "❌ Should be 2000"
            print(f"  Rp {fare:,}: {count:,} ({percentage:.1f}%) - {expected}")
        
        # Subsidy DD analysis
        subsidy_dd = subsidy_transactions[subsidy_transactions["is_dd"] == True]
        subsidy_dd_count = len(subsidy_dd)
        
        if subsidy_dd_count > 0:
            print_subsection("Subsidy DD Issues")
            print(f"  DD in Subsidy Hours: {subsidy_dd_count:,}")
            print(f"  Total Refund: Rp {subsidy_dd['dd_refund'].sum():,}")


def interactive_analyzer():
    """
    Interactive analyzer for debugging DD detection.
    """
    print_header("DD ANALYZER - INTERACTIVE MODE")
    
    while True:
        print("\nOptions:")
        print("1. Load and analyze file")
        print("2. Show analysis summary")
        print("3. Analyze specific card")
        print("4. Show subsidy analysis")
        print("5. Exit")
        
        choice = input("\nSelect option (1-5): ").strip()
        
        if choice == "1":
            file_path = input("Enter CSV file path: ").strip()
            try:
                print("Loading data...")
                df = load_and_prepare_data(file_path)
                print(f"Loaded {len(df):,} transactions")
                
                print("Analyzing transactions...")
                df = analyze_all_transactions(df, debug_mode=True)
                print("Analysis complete!")
                
                # Store for other operations
                globals()['current_df'] = df
                
            except Exception as e:
                print(f"Error: {e}")
        
        elif choice == "2":
            if 'current_df' in globals():
                show_analysis_summary(globals()['current_df'])
            else:
                print("Please load data first (option 1)")
        
        elif choice == "3":
            if 'current_df' in globals():
                card_number = input("Enter card number: ").strip()
                show_card_analysis(globals()['current_df'], card_number)
            else:
                print("Please load data first (option 1)")
        
        elif choice == "4":
            if 'current_df' in globals():
                show_subsidy_analysis(globals()['current_df'])
            else:
                print("Please load data first (option 1)")
        
        elif choice == "5":
            print("Goodbye!")
            break
        
        else:
            print("Invalid option. Please select 1-5.")


if __name__ == "__main__":
    interactive_analyzer()