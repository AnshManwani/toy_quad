import argparse
import csv
import matplotlib.pyplot as plt
import os

def main():
    parser = argparse.ArgumentParser(description="Plot evaluation return vs. episode from training logs.")
    parser.add_argument("logs", nargs="+", help="One or more CSV log file paths to plot")
    parser.add_argument("--out", type=str, default=None, help="Path to save the plot image (e.g. plot.png)")
    args = parser.parse_args()

    plt.figure(figsize=(10, 6))
    plotted_any = False

    for log_path in args.logs:
        if not os.path.exists(log_path):
            print(f"Warning: File not found: {log_path}")
            continue

        episodes = []
        returns = []
        
        with open(log_path, 'r', newline='') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            
            if header and 'episode' in header and 'eval_return' in header:
                ep_idx = header.index('episode')
                ret_idx = header.index('eval_return')
            else:
                ep_idx = 0
                ret_idx = 1
                if header:
                    try:
                        episodes.append(float(header[ep_idx]))
                        returns.append(float(header[ret_idx]))
                    except ValueError:
                        pass
            
            for row in reader:
                if len(row) > max(ep_idx, ret_idx):
                    try:
                        episodes.append(float(row[ep_idx]))
                        returns.append(float(row[ret_idx]))
                    except ValueError:
                        continue
        
        label = os.path.basename(log_path)
        plt.plot(episodes, returns, label=label)
        plotted_any = True

    plt.xlabel('Episode')
    plt.ylabel('Evaluation Return')
    plt.title('Evaluation Return vs Episode')
    if plotted_any:
        plt.legend()
    plt.grid(True)

    if args.out:
        plt.savefig(args.out)
        print(f"Saved plot to {args.out}")
    else:
        plt.show()

if __name__ == "__main__":
    main()
