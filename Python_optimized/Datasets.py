from HigherOrderPathGenerator import HigherOrderPathGenerator
from Visualizations import create_EmbeddingData_Vis
import os

def init_generator(filename, tol1=1e-14, tol2=1e-15, verbose=False):
    filepath = '..\\Python\\data\\models\\'+filename
    with open(os.path.splitext(filepath)[0]+'.config', 'r') as f:
        model_config = { items[0]:' '.join(items[1:]) for items in [line.strip('\n').split('\t') for line in f.readlines() ] }
    if verbose:
        print('Configuration:')
        for k,v in model_config.items():
            print('%s: %s' % (k,v))
    model_config['filename'] = filename

    gen = HigherOrderPathGenerator(node_sort_key=int, id=os.path.splitext(filename)[0], config=model_config, 
        create_EmbeddingData=create_EmbeddingData_Vis)
    gen.load_BuildHON_rules('..\\Python\\data\\models\\'+filename, freeze=True)
    # add metadata
    metadata_filename = 'metadata_' + (filename.split('.')[0].split('_')[0]) + '.csv'
    metadata = dict()
    with open('..\\Python\\data\\models\\' + metadata_filename, 'r') as f:
        for line in f:
            i,Ci = line.split()
            metadata[int(i)]=Ci
    if metadata_filename=='metadata_workplace.csv':
        gen.add_metadata('Department', metadata, use_last=True)
    elif metadata_filename=='metadata_primaryschool.csv':
        gen.add_metadata('Class', metadata, use_last=True)
        gen.add_metadata('Role', { i:'Teacher' if c=='Teachers' else 'Child' for i,c in metadata.items()})
    elif metadata_filename=='metadata_hospital.csv': 
        gen.add_metadata('Status', metadata, use_last=True)
    elif metadata_filename=='metadata_temporal-clusters.csv' or metadata_filename=='metadata_shuffled-temporal-clusters.csv':
        gen.add_metadata('Color', metadata, use_last=True)
    else:
        raise Exception('unknown metadata_filename')
    print(list(gen.check_transition_probs(tol1)))
    print(list(gen.verify_stationarity(1, tol2)))
    #print(list(gen.verify_stationarity(2, tol2))) # this does not hold
    return gen
