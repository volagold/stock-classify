import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def year_base(r: int):
    if r <= 5: return 1
    elif r <= 10: return 3
    elif r <= 20: return 5
    else: return 7

def get_df(ticket: str):
    df = pd.read_csv(f'csv/{ticket}.csv')
    df.columns = df.columns.str.lower()
    df['date'] = df['date'].map(lambda x: x.split(' ')[0])
    df['date'] = pd.to_datetime(df['date'], utc=False)
    return df

def plot_graph(ticket: str):
    df = get_df(ticket)
    r = df['date'].dt.year.max() - df['date'].dt.year.min()
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(df['date'], df['close'], color='green')
    ax.xaxis.set_major_locator(mdates.YearLocator(base=year_base(r)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.ylabel('$', rotation=0, labelpad=-20, loc='top')
    plt.title(ticket.upper())
    plt.savefig(f'plots/{ticket}.png')
    plt.close(fig)

def main():
    all_tickets = [f[:-4] for f in os.listdir('csv')]
    plots = [f[:-4] for f in os.listdir('plots')]
    new = list(set(all_tickets) - set(plots))
    if not new:
        print('no new ticket to draw')
        return 
    for idx, ticket in enumerate(new):
        plot_graph(ticket)
        print(f'{idx+1}. plotted graph for {ticket}', end="\r", flush=True)


if __name__ == '__main__':
    main()