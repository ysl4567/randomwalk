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
    stock_data = yf.download(stock_name, start='2023-04-01', end='2030-04-01')
    
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
    prices = []
    for i in range(simulations):
        price = random_walk(start_price, days, volatility)[-1]
        prices.append(price)
    
    # Create BST
    root = None
    for price in prices:
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

if __name__ == "__main__":
    main()
