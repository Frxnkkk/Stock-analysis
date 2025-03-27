import pandas as pd

# Load the CSV file
file_path = "D:/git-workshop/stock-analysis/stock_data.csv"  # Update this path if necessary
data = pd.read_csv(file_path)

# Convert the 'Date' column to datetime format
data['Date'] = pd.to_datetime(data['Date'])

# Group by stock and calculate statistics
stock_summary = data.groupby('Stock').agg(
    Highest_Price=('Price', 'max'),
    Lowest_Price=('Price', 'min'),
    Average_Price=('Price', 'mean'),
    Price_Volatility=('Price', 'std')  # Standard deviation as a measure of volatility
).reset_index()

# Calculate overall market trends
market_trends = data.groupby('Date').agg(
    Total_Market_Value=('Price', 'sum'),
    Average_Market_Price=('Price', 'mean')
).reset_index()

# Identify the most volatile stock
most_volatile_stock = stock_summary.loc[stock_summary['Price_Volatility'].idxmax()]

# Identify the stock with the highest average price
highest_avg_price_stock = stock_summary.loc[stock_summary['Average_Price'].idxmax()]

# Save the analysis to a text file
with open("market_analysis.txt", "w") as file:
    file.write("Stock Market Analysis\n")
    file.write("======================\n\n")
    
    file.write("Stock Summary:\n")
    file.write(stock_summary.to_string(index=False))
    file.write("\n\n")
    
    file.write("Market Trends:\n")
    file.write(market_trends.to_string(index=False))
    file.write("\n\n")
    
    file.write("Additional Insights:\n")
    file.write(f"Most Volatile Stock: {most_volatile_stock['Stock']} "
               f"(Volatility: {most_volatile_stock['Price_Volatility']:.2f})\n")
    file.write(f"Stock with Highest Average Price: {highest_avg_price_stock['Stock']} "
               f"(Average Price: {highest_avg_price_stock['Average_Price']:.2f})\n")

print("Analysis complete. Results saved to 'market_analysis.txt'.")