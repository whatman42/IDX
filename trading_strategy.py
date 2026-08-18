import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SimpleMovingAverageStrategy:
    def __init__(self, short_window=10, long_window=50):
        self.short_window = short_window
        self.long_window = long_window

    def calculate_signals(self, prices):
        if len(prices) < self.long_window:
            return 'HOLD'
        
        short_sma = sum(prices[-self.short_window:]) / self.short_window
        long_sma = sum(prices[-self.long_window:]) / self.long_window
        
        if short_sma > long_sma:
            return 'BUY'
        elif short_sma < long_sma:
            return 'SELL'
        return 'HOLD'

class TradingBot:
    def __init__(self, strategy):
        self.strategy = strategy
        self.price_history = []

    def on_price_update(self, current_price):
        self.price_history.append(current_price)
        signal = self.strategy.calculate_signals(self.price_history)
        logging.info(f"Price: {current_price} | Signal: {signal}")
        return signal

if __name__ == '__main__':
    bot = TradingBot(SimpleMovingAverageStrategy(short_window=3, long_window=5))
    sample_prices = [100, 102, 101, 103, 105, 107, 106, 108, 110]
    for price in sample_prices:
        bot.on_price_update(price)
