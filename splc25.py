import time
import os
import argparse
from tqdm import tqdm

from flamapy.core.discover import DiscoverMetamodels
from flamapy.metamodels.configurator_metamodel.transformation import FmToConfigurator
from flamapy.metamodels.configurator_metamodel.operations.configure import Configure
import pandas as pd
import matplotlib.pyplot as plt
import re
from collections import defaultdict

def show_ascii_art():
    art = r"""
 _______   __                                                    __                  __       
/       \ /  |                                                  /  |                /  |      
$$$$$$$  |$$/  __     __  ______    ______    _______   ______  $$ |        ______  $$ |____  
$$ |  $$ |/  |/  \   /  |/      \  /      \  /       | /      \ $$ |       /      \ $$      \ 
$$ |  $$ |$$ |$$  \ /$$//$$$$$$  |/$$$$$$  |/$$$$$$$/ /$$$$$$  |$$ |       $$$$$$  |$$$$$$$  |
$$ |  $$ |$$ | $$  /$$/ $$    $$ |$$ |  $$/ $$      \ $$ |  $$ |$$ |       /    $$ |$$ |  $$ |
$$ |__$$ |$$ |  $$ $$/  $$$$$$$$/ $$ |       $$$$$$  |$$ \__$$ |$$ |_____ /$$$$$$$ |$$ |__$$ |
$$    $$/ $$ |   $$$/   $$       |$$ |      /     $$/ $$    $$/ $$       |$$    $$ |$$    $$/ 
$$$$$$$/  $$/     $/     $$$$$$$/ $$/       $$$$$$$/   $$$$$$/  $$$$$$$$/  $$$$$$$/ $$$$$$$/  
    """
    print(art)
    print("\nWelcome to the aquaIA configurator experiments\n")


def xml_to_uvl():
    for file in os.listdir('./models/betty/'):
        if file.endswith('.xml'):   
            print(f'File: {file}')
            dm = DiscoverMetamodels()
            feature_model = dm.use_transformation_t2m('./models/betty/'+file,'fm')
            dm.use_transformation_m2t(feature_model,'./models/uvl/'+file.split('/')[-1].split('.')[0]+'.uvl')

def run_experiments(results_file: str = 'results.csv'):
    processed_files = set()

    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            for line in f:
                filename = line.strip().split(',')[0]
                processed_files.add(filename)

    all_files = [file for file in os.listdir('./models/uvl/') if file.endswith('.uvl')]
    files_to_process = [file for file in all_files if file not in processed_files]

    print(f"\n🧪 Processing {len(files_to_process)} of {len(all_files)} total files...\n")

    pbar = tqdm(files_to_process, desc="Running experiments", unit="file")
    for file in pbar:
        pbar.set_postfix(file=file)

        dm = DiscoverMetamodels()
        feature_model = dm.use_transformation_t2m('./models/uvl/' + file, 'fm')
        configurator_metamodel = FmToConfigurator(feature_model).transform()
        configure_operation = Configure().execute(configurator_metamodel)

        start_time = time.perf_counter()

        while configure_operation.next_question():
            if configure_operation.get_possible_options():
                configure_operation.answer_question([0])

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

        with open(results_file, 'a') as f:
            f.write(f'{file},{elapsed_time}\n')

        # Optional: print this line only if needed
        # print(f'File: {file}, elapsed time: {elapsed_time}')

def plot_time_vs_features(csv_path):
    # Load the CSV file
    df = pd.read_csv(csv_path, header=None, names=["filename", "elapsed_time"])

    # Extract x (features), y (cross-tree %), z (id) from filename
    pattern = re.compile(r"(\d+)-(\d+)-(\d+)\.uvl")
    df[['features', 'cross_tree', 'id']] = df['filename'].str.extract(pattern).astype(int)

    # Group by 'features' and 'cross_tree', and calculate mean time
    grouped = df.groupby(['features', 'cross_tree'])['elapsed_time'].mean().reset_index()

    # Prepare the plot
    plt.figure(figsize=(10, 6))

    # Plot each line: one per cross_tree value
    for cross_tree_value in sorted(grouped['cross_tree'].unique()):
        subset = grouped[grouped['cross_tree'] == cross_tree_value]
        plt.plot(subset['features'], subset['elapsed_time'], marker='o', label=f'CTC {cross_tree_value}%')

    # Styling
    plt.title("Elapsed Time vs Number of Features")
    plt.xlabel("Number of Features")
    plt.ylabel("Mean Elapsed Time (s)")
    
    # Set Y-axis to log scale
    plt.yscale("log")

    # Show grid, and set a tight layout for aesthetics
    plt.legend(title="Cross Tree Constraints")
    plt.grid(True)
    plt.tight_layout()

    # Show the plot
    plt.show()


def plot_time_vs_cross_tree(csv_path):
    import pandas as pd
    import matplotlib.pyplot as plt
    import re

    # Load and parse data
    df = pd.read_csv(csv_path, header=None, names=["filename", "elapsed_time"])
    pattern = re.compile(r"(\d+)-(\d+)-(\d+)\.uvl")
    df[['features', 'cross_tree', 'id']] = df['filename'].str.extract(pattern).astype(int)

    # Group by 'features' and 'cross_tree', and calculate mean time
    grouped = df.groupby(['cross_tree', 'features'])['elapsed_time'].mean().reset_index()

    # Prepare the plot
    plt.figure(figsize=(10, 6))

    # Plot each line: one per number of features
    for feature_value in sorted(grouped['features'].unique()):
        subset = grouped[grouped['features'] == feature_value]
        plt.plot(subset['cross_tree'], subset['elapsed_time'], marker='o', label=f'{feature_value} features')

    # Styling
    plt.title("Elapsed Time vs Cross Tree Constraints")
    plt.xlabel("Cross Tree Constraints (%)")
    plt.ylabel("Mean Elapsed Time (s)")
    
    # Set Y-axis to log scale
    plt.yscale("log")

    plt.legend(title="Number of Features")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def print_charts(csv_path: str = 'results.csv'):
    steps = ['Plotting time vs features', 'Plotting time vs cross-tree']
    pbar = tqdm(steps, desc="Generating charts", unit="step")

    for step in pbar:
        pbar.set_postfix(task=step)
        if step == 'Plotting time vs features':
            plot_time_vs_features(csv_path)
        elif step == 'Plotting time vs cross-tree':
            plot_time_vs_cross_tree(csv_path)

    pbar.close()


def main():
    show_ascii_art()

    parser = argparse.ArgumentParser(
        description="DiversoLab CLI - Run experiments and visualize results.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        "action",
        choices=["xml_to_uvl", "run_experiments", "print_charts"],
        help="Action to perform:\n"
             "  xml_to_uvl       Convert .xml files to .uvl\n"
             "  run_experiments  Run configuration experiments\n"
             "  print_charts     Generate charts from results.csv"
    )

    args = parser.parse_args()

    if args.action == "xml_to_uvl":
        xml_to_uvl()
    elif args.action == "run_experiments":
        run_experiments()
    elif args.action == "print_charts":
        print_charts()
    else:
        print("Unknown action. Use --help for options.")


if __name__ == "__main__":
    main()