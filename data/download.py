import os
import fire
import random
import yfinance as yf


def main(num=0):  
    with open('tickets.txt', 'r') as f:
        tickets = [line.strip() for line in f] 
    
    if num > 0:
        exist = os.listdir('csv/')
        exist = [file[:-4].upper() for file in exist]
        tickets = list(set(tickets) - set(exist))
        num = min(num, len(tickets))
        if num == 0:
            print("all tickets have been fetched")
            return
        tickets = random.sample(tickets, num)
    
    for idx, ticket in enumerate(tickets):
        print(f'{idx+1}. downloading data for {ticket}...', end="\r", flush=True)
        stock = yf.Ticker(ticket)
        hist = stock.history(period="max")
        hist.to_csv(f'csv/{ticket.lower()}.csv')


if __name__ == '__main__':
    fire.Fire(main)