import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def load_data(filename):
    """Load and prepare time series data from CSV file."""
    df = pd.read_csv(filename)
    df['Date'] = pd.to_datetime(df['Date'])
    return df


def get_date_range(df):
    """Get start and end dates from user input with validation."""
    print("\nEnter date range for analysis:")
    print(f"Available dates: {df['Date'].min().date()} to {df['Date'].max().date()}")
    
    while True:
        try:
            start_date_str = input("\nEnter start date (YYYY-MM-DD): ").strip()
            start_date = pd.to_datetime(start_date_str)
            break
        except ValueError:
            print("❌ Invalid date format. Please use YYYY-MM-DD format.")
    
    while True:
        try:
            end_date_str = input("Enter end date (YYYY-MM-DD): ").strip()
            end_date = pd.to_datetime(end_date_str)
            if end_date <= start_date:
                print("❌ End date must be after start date.")
                continue
            break
        except ValueError:
            print("❌ Invalid date format. Please use YYYY-MM-DD format.")
    
    return start_date, end_date


def filter_by_date_range(df, start_date, end_date):
    """Filter dataframe by date range."""
    df_filtered = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)].copy()
    print(f"\n✓ Analyzing data from {start_date.date()} to {end_date.date()}")
    print(f"✓ Total records: {len(df_filtered)}")
    return df_filtered


def detect_missing_data(df):
    """Detect missing data points."""
    print("\n1. MISSING DATA POINTS:")
    missing_count = df.isnull().sum().sum()
    missing_rows = df[df.isnull().any(axis=1)]
    print(f"   Total missing values: {missing_count}")
    print(f"   Rows with missing data: {len(missing_rows)}")
    if len(missing_rows) > 0:
        print(f"   Missing data indices: {missing_rows.index.tolist()[:10]}...")
    return missing_rows


def detect_stale_data(df, stale_threshold=1e-6):
    """Detect stale data (no change over time)."""
    print("\n2. STALE DATA (No Change Over Time):")
    df_no_missing = df.dropna(subset=['Close'])
    price_changes = df_no_missing['Close'].diff()
    stale_periods = (np.abs(price_changes) < stale_threshold).sum()
    print(f"   Consecutive periods with near-zero change: {stale_periods}")
    stale_indices = price_changes[np.abs(price_changes) < stale_threshold].index.tolist()
    if stale_indices:
        print(f"   Stale data indices (first 10): {stale_indices[:10]}")
    return stale_indices


def detect_outliers_iqr(df):
    """Detect outliers using IQR method."""
    print("\n3. OUTLIERS (using IQR method):")
    Q1 = df['Close'].quantile(0.25)
    Q3 = df['Close'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df['Close'] < lower_bound) | (df['Close'] > upper_bound)]
    print(f"   IQR bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
    print(f"   Number of outliers: {len(outliers)}")
    if len(outliers) > 0:
        print(f"   Outlier indices (first 10):")
        for idx in outliers.index[:10]:
            print(f"      Index {idx}: Value = {outliers.loc[idx, 'Close']:.2f}")
    return outliers, lower_bound, upper_bound


def detect_outliers_zscore(df, threshold=3):
    """Detect outliers using Z-Score method."""
    print(f"\n4. OUTLIERS (using Z-Score method, threshold={threshold}):")
    z_scores = np.abs(stats.zscore(df['Close'].dropna()))
    z_outliers_count = (z_scores > threshold).sum()
    print(f"   Number of Z-score outliers (|z| > {threshold}): {z_outliers_count}")
    return z_scores


def detect_big_jumps_dips(df, big_change_threshold=5):
    """Detect big relative jumps/dips in data."""
    print("\n5. BIG RELATIVE JUMPS/DIPS:")
    df_clean = df.dropna(subset=['Close'])
    relative_changes = df_clean['Close'].pct_change() * 100
    big_changes = relative_changes[np.abs(relative_changes) > big_change_threshold]
    print(f"   Threshold: {big_change_threshold}%")
    print(f"   Number of big jumps/dips: {len(big_changes)}")
    if len(big_changes) > 0:
        print(f"   Top 10 biggest relative changes:")
        top_changes = big_changes.abs().nlargest(10)
        for idx, change in top_changes.items():
            sign = "▲" if big_changes[idx] > 0 else "▼"
            print(f"      {sign} Index {idx}: {big_changes[idx]:+.2f}%")
    return relative_changes, big_changes


def create_visualizations(df, missing_rows, outliers, lower_bound, upper_bound, relative_changes, z_scores):
    """Create 4-panel visualization for anomaly detection."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Time Series Anomaly Detection', fontsize=16, fontweight='bold')
    
    # Plot 1: Time Series with Anomalies Highlighted
    ax1 = axes[0, 0]
    ax1.plot(df['Date'], df['Close'], label='Data', linewidth=1.5, color='blue', alpha=0.7)
    if len(missing_rows) > 0:
        ax1.scatter(df.loc[missing_rows.index, 'Date'], df.loc[missing_rows.index, 'Close'], 
                    color='red', s=50, marker='x', label='Missing Data', zorder=5)
    if len(outliers) > 0:
        ax1.scatter(outliers['Date'], outliers['Close'], color='orange', s=100, 
                    marker='^', label='Outliers (IQR)', zorder=5)
    ax1.set_title('Time Series with Anomalies')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Value')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Relative Changes
    ax2 = axes[0, 1]
    threshold = 5
    ax2.bar(range(len(relative_changes)), relative_changes.values, color='steelblue', alpha=0.7)
    ax2.axhline(y=threshold, color='red', linestyle='--', label=f'Threshold ({threshold}%)')
    ax2.axhline(y=-threshold, color='red', linestyle='--')
    ax2.set_title('Relative Daily Changes (%)')
    ax2.set_xlabel('Day Index')
    ax2.set_ylabel('Change (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Distribution of Values
    ax3 = axes[1, 0]
    ax3.hist(df['Close'].dropna(), bins=30, color='green', alpha=0.7, edgecolor='black')
    ax3.axvline(lower_bound, color='red', linestyle='--', linewidth=2, label='IQR Bounds')
    ax3.axvline(upper_bound, color='red', linestyle='--', linewidth=2)
    ax3.set_title('Distribution of Values')
    ax3.set_xlabel('Value')
    ax3.set_ylabel('Frequency')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Z-Score
    ax4 = axes[1, 1]
    ax4.plot(z_scores, color='purple', linewidth=1.5, alpha=0.7)
    ax4.axhline(y=3, color='red', linestyle='--', linewidth=2, label='Z-Score = 3')
    ax4.axhline(y=2, color='orange', linestyle='--', linewidth=1.5, label='Z-Score = 2')
    ax4.set_title('Z-Score (Outlier Detection)')
    ax4.set_xlabel('Index')
    ax4.set_ylabel('Absolute Z-Score')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('anomaly_detection.png', dpi=100, bbox_inches='tight')
    print("\n" + "=" * 60)
    print("📊 Chart saved as 'anomaly_detection.png'")
    print("=" * 60)
    plt.show()


def main():
    """Main execution function."""
    print("=" * 60)
    print("TIME SERIES ANOMALY DETECTION")
    print("=" * 60)
    
    # Load data
    df = load_data('noisy_stock_data.csv')
    
    # Get date range from user
    start_date, end_date = get_date_range(df)
    df = filter_by_date_range(df, start_date, end_date)
    
    # Detect all anomalies
    missing_rows = detect_missing_data(df)
    stale_indices = detect_stale_data(df)
    outliers, lower_bound, upper_bound = detect_outliers_iqr(df)
    z_scores = detect_outliers_zscore(df)
    relative_changes, big_changes = detect_big_jumps_dips(df)
    
    # Create visualizations
    create_visualizations(df, missing_rows, outliers, lower_bound, upper_bound, relative_changes, z_scores)


if __name__ == '__main__':
    main()