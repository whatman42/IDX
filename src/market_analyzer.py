import requests
import logging

logging.basicConfig(level=logging.INFO)

class MarketAnalyzer:
    def __init__(self):
        self.base_url = 'https://api.coingecko.com/api/v3'

    def fetch_market_prices(self, coin_ids=('bitcoin', 'ethereum')):
        """Fetches public price data for specified cryptocurrencies."""
        try:
            ids_str = ','.join(coin_ids)
            url = f'{self.base_url}/simple/price?ids={ids_str}&vs_currencies=usd'
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logging.error(f'Error fetching market prices: {e}')
            return {}

if __name__ == '__main__':
    analyzer = MarketAnalyzer()
    prices = analyzer.fetch_market_prices()
    print('Market Prices:', prices)
