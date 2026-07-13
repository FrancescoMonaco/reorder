from csv import Error
from pathlib import Path
import pprint
import sbatchman as sbm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from io import StringIO

def plot_csv(df: pd.DataFrame, title: str, out_file: Path, ranks: int, grid_rows: int, grid_cols: int):
  # Add the new columns
  ranks_per_group = ranks / (grid_rows + grid_cols)
  df['nodeid'] = df['rank'] // ranks_per_group
  df['localid'] = df['rank'] % ranks_per_group
  df['nodeid'] = df['nodeid'].astype(int)
  df['localid'] = df['localid'].astype(int)
  print(df)

  # Create the point plot
  plt.figure(figsize=(10, 8))
  sns.scatterplot(data=df, x='colid', y='rowid', hue='nodeid', style='localid', palette='viridis', s=100)

  # Set plot title and labels
  plt.title(title)
  plt.xlabel('Column ID')
  plt.ylabel('Row ID')
  plt.legend(title='Rank')
  plt.grid(True)

  # Save the plot to a file
  plt.savefig(out_file)
  print(f"Plot saved to {out_file}")
  
def get_csv_from_stdout(stdout: str):
  i = 0
  lines = stdout.splitlines()
  csv_lines = []
  
  while lines[i] != '[OUT CSV]':
    i += 1
  i += 1
  while lines[i] != '[END OUT CSV]':
    csv_lines.append(lines[i])
    i += 1

  csv_str = '\n'.join(csv_lines)
  df = pd.read_csv(StringIO(csv_str), skipinitialspace=True)
  return df

if __name__ == "__main__":
  jobs = sbm.jobs_list()
  jobs_failed = []
  for j in jobs:
    if j.status == sbm.Status.COMPLETED.value:
      stdout = j.get_stdout()
      if not stdout:
        raise Error(f'Job {j} has empty stdout!')
      
      df = get_csv_from_stdout(stdout)
      print(j)
      _, __, args = j.parse_command_args()
      if not args:
        raise Error(f'Job {j} wrong args!')
      
      ranks = int(args['np'])
      file = Path(args['f'])
      grid_rows = int(args['r'])
      grid_cols = int(args['c'])
      partitioning_type = args['p']
      transpose = bool(args['t'])
      file_parts = file.stem.split('_')
      if file.stem.startswith('parmat_'):
        N = int(file_parts[1][1:])
        M = int(file_parts[2][1:])
      elif file.stem.startswith('graph500_'):
        N = 2**int(file_parts[1])
        M = N * int(file_parts[2])
      else:
        print(f'Warning: unsupported matrix "{file}", skipping')
        continue
        
      print(args)
      print(df)
      print('='*50)
      name = f'{N}x{N}_{M}nz_{ranks}ranks_{grid_rows}x{grid_cols}grid_{partitioning_type}part{"_transpose" if transpose else ""}'
      path = Path(f'results/{name}.png')
      path.parent.mkdir(exist_ok=True, parents=True)
      plot_csv(df, name, path, ranks, grid_rows, grid_cols)
      exit()
    else:
      jobs_failed.append(j)
      
  if jobs_failed:
    print('!! FAILED jobs:')
    for j in jobs_failed:
      pprint.pprint(j)
