import pandas as pd

# Load the CSV file
file_path = 'stock_data.csv'  # Adjust this path if needed
df = pd.read_csv(file_path)

# Filter data for stock 'AOT' and process dates
aot_data = df[df['Stock'] == 'AOT'].copy()
aot_data['Date'] = pd.to_datetime(aot_data['Date'])
aot_data.sort_values('Date', inplace=True)

# Calculate summary statistics
summary_stats = aot_data['Price'].describe()
price_change = aot_data['Price'].iloc[-1] - aot_data['Price'].iloc[0]
price_change_pct = (price_change / aot_data['Price'].iloc[0]) * 100

# Create report text
report = f"""Stock Analysis Report for AOT

Date Range: {aot_data['Date'].min().date()} to {aot_data['Date'].max().date()}
Total Records: {len(aot_data)}

Price Summary:
{summary_stats}

Price Change:
Start Price: {aot_data['Price'].iloc[0]:.2f}
End Price: {aot_data['Price'].iloc[-1]:.2f}
Change: {price_change:.2f} ({price_change_pct:.2f}%)
"""

# Save report to a text file
report_path = './aot_analysis.txt'
with open(report_path, 'w') as file:
    file.write(report)

print(f"Report saved to: {report_path}")

