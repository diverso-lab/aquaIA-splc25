import time
import os
from flamapy.core.discover import DiscoverMetamodels
from flamapy.metamodels.configurator_metamodel.transformation import FmToConfigurator
import pandas as pd
import matplotlib.pyplot as plt
import re
from collections import defaultdict


def xml_to_uvl():
    for file in os.listdir('./models/betty/'):
        if file.endswith('.xml'):   
            print(f'File: {file}')
            dm = DiscoverMetamodels()
            feature_model = dm.use_transformation_t2m('./models/betty/'+file,'fm')
            print(feature_model)
            dm.use_transformation_m2t(feature_model,'./models/uvl/'+file.split('/')[-1].split('.')[0]+'.uvl')

def run_experiments():
    # Run the experiments
    for file in os.listdir('./models/uvl/'):
        dm = DiscoverMetamodels()
        feature_model = dm.use_transformation_t2m('./models/uvl/'+file,'fm')
        configurator = FmToConfigurator(feature_model).transform()
        print(file)
        #Operacion de configuracion.
        start_time = time.perf_counter()

        while configurator.next_question():
            if configurator.get_possible_options():
                configurator.answer_question([0])

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

        #save in csv file
        with open('results.csv', 'a') as f:
            f.write(f'{file},{elapsed_time}\n')
        print(f'File: {file}, elapsed time: {elapsed_time}')




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
	plt.xscale("log")
	plt.legend(title="Cross Tree Constraints")
	plt.grid(True)
	plt.tight_layout()
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
	plt.legend(title="Number of Features")
	plt.grid(True)
	plt.tight_layout()
	plt.show()


def print_charts():
    plot_time_vs_features('./results.csv')
    plot_time_vs_cross_tree('./results.csv')



if __name__ == '__main__':
    #xml_to_uvl()
    #run_experiments()
    print_charts()