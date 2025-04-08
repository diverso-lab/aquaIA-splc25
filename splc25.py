import time
import os
from flamapy.core.discover import DiscoverMetamodels


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
    for file in 'models/uvl/*':
        dm = DiscoverMetamodels()
        feature_model = dm.use_transformation_t2m(file,'fm')
 
        #Operacion de configuracion.
        start_time = time.time()
        #TODO exec configuration
        end_time = time.time()
        elapsed_time = end_time - start_time

        #save in csv file
        with open('results.csv', 'a') as f:
            f.write(f'{file},{elapsed_time}\n')
        print(f'File: {file}, elapsed time: {elapsed_time}')

def print_charts():
    print('''charts''')	
    pass


if __name__ == '__main__':
    xml_to_uvl()
    #run_experiments()
    #print_charts()