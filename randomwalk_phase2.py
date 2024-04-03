import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

class Node:
    def __init__(self, value):
        self.left = None
        self.right = None
        self.value = value

def insert(root, value):
    if root is None:
        return Node(value)
    else:
        if value < root.value:
            root.left = insert(root.left, value)
        else:
            root.right = insert(root.right, value)
    return root

def inorder_traversal(root, x, y):
    if root:
        y -= 1
        x1 = x - 2 ** y
        x2 = x + 2 ** y
        plt.plot([x1, x2], [y, y], 'b')
        plt.text(x, y, str(round(root.value, 2)), bbox=dict(facecolor='red', alpha=0.5), horizontalalignment='center')
        
        x = x - 2 ** (y-1)
        
        inorder_traversal(root.left, x, y-1)
        inorder_traversal(root.right, x + 2 ** y, y-1)

def filter_lines(prices, mean_price, std_dev):
    filtered_prices = [price if abs(price - mean_price) <= std_dev else None for price in prices]
    return [price for price in filtered_prices if price is not None]

def random_walk(start_price, days, volatility):
    price = np.zeros(days)
    price[0] = start_price
    
    shock = np.zeros(days)
    drift = np.zeros(days)
    
    for i in range(1, days):
        shock[i] = np.random.normal(loc=0, scale=volatility)
        drift[i] = 0.1 * volatility
        price[i] = price[i-1] + (price[i-1] * (drift[i] + shock[i]))
        
    return price

def main():
    # Get stock data
    stock_name = input("Enter stock symbol (e.g., AAPL): ")
    stock_data = yf.download(stock_name, start='2020-01-01', end='2023-01-01')
    
    # Calculate daily returns
    stock_data['Daily Return'] = stock_data['Close'].pct_change()
    
    # Calculate the drift and volatility
    avg_daily_return = stock_data['Daily Return'].mean()
    volatility = stock_data['Daily Return'].std()
    
    # Get the last stock price as the start price for the random walk
    start_price = stock_data['Close'].iloc[-1]
    
    # Number of days for simulation
    days = int(input("Enter number of days for simulation: "))
    
    # Perform random walk simulation
    simulations = 300
    all_prices = []
    for i in range(simulations):
        price = random_walk(start_price, days, volatility)[-1]
        all_prices.append(price)
    
    # Create BST
    root = None
    for price in all_prices:
        root = insert(root, price)
    
    # Plot random walk simulations
    plt.figure(figsize=(12, 6))
    for i in range(simulations):
        plt.plot(random_walk(start_price, days, volatility))
    plt.title(f'{stock_name} Random Walk Simulation')
    plt.xlabel('Days')
    plt.ylabel('Price')
    plt.grid(True)
    plt.show()
    
    # Plot BST
    plt.figure(figsize=(12, 8))
    inorder_traversal(root, 2 ** ((days // 2) + 1), days // 2)
    plt.title(f'{stock_name} Random Walk Simulation - Binary Search Tree')
    plt.xlabel('Price')
    plt.ylabel('Days from Starting Point')
    plt.grid(True)
    plt.show()

    # Filter lines based on standard deviation from mean price
    while len(all_prices) > 1:
        mean_price = np.mean(all_prices)
        std_dev = np.std(all_prices)
        
        all_prices = filter_lines(all_prices, mean_price, std_dev)
        
        # Plot remaining lines
        if len(all_prices) >= 1:
            plt.figure(figsize=(12, 6))
            for price in all_prices:
                plt.plot(random_walk(start_price, days, volatility))
            plt.title(f'{stock_name} Random Walk Simulation (Filtered)')
            plt.xlabel('Days')
            plt.ylabel('Price')
            plt.grid(True)
            plt.show()

if __name__ == "__main__":
    main()
