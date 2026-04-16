import os
import glob
import pandas as pd

def summarize_results():
    files = glob.glob('reports/results_*.txt')
    results = []
    
    for f in files:
        with open(f, 'r') as file:
            lines = file.readlines()
            name = lines[0].split(': ')[1].strip()
            acc = float(lines[1].split(': ')[1].strip())
            f1 = float(lines[2].split(': ')[1].strip())
            auc = float(lines[3].split(': ')[1].strip())
            results.append({'Scenario': name, 'ACC': acc, 'F1': f1, 'AUC': auc})
            
    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values('AUC', ascending=False)
        markdown_table = df.to_markdown(index=False)
        
        with open('reports/benchmark_summary.md', 'w') as f:
            f.write("# Benchmark Summary\n\n")
            f.write(markdown_table)
            f.write("\n")
        print("Summary updated in reports/benchmark_summary.md")
    else:
        print("No results found.")

if __name__ == "__main__":
    summarize_results()
