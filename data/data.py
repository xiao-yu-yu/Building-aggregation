"""
    File to load dataset based on user control from main file
"""
from data.Building import BuildingDataset


def LoadData(DATASET_NAME):
    """
        This function is called in the main.py file
        returns:
        ; dataset object
    """
    # # handling for MNIST or CIFAR Superpixels
    # if DATASET_NAME == 'MNIST' or DATASET_NAME == 'CIFAR10':
    #     return SuperPixDataset(DATASET_NAME)
    #
    # # handling for (ZINC) molecule dataset
    # if DATASET_NAME in ['ZINC', 'ZINC-full', 'AQSOL']:
    #     return MoleculeDataset(DATASET_NAME)
    #
    # # handling for the TU Datasets
    # TU_DATASETS = ['ENZYMES', 'DD', 'PROTEINS_full']
    # if DATASET_NAME in TU_DATASETS:
    #     return TUsDataset(DATASET_NAME)
    #
    # # handling for SBM datasets
    # SBM_DATASETS = ['SBM_CLUSTER', 'SBM_PATTERN']
    # if DATASET_NAME in SBM_DATASETS:
    #     return SBMsDataset(DATASET_NAME)
    #
    # # handling for TSP dataset
    # if DATASET_NAME == 'TSP':
    #     return TSPDataset(DATASET_NAME)
    #
    # # handling for COLLAB dataset
    # if DATASET_NAME == 'OGBL-COLLAB':
    #     return COLLABDataset(DATASET_NAME)
    #
    # # handling for the CSL (Circular Skip Links) Dataset
    # if DATASET_NAME == 'CSL':
    #     return CSLDataset(DATASET_NAME)
    #
    # # handling for the CYCLES Dataset from https://github.com/cvignac/SMP
    # if DATASET_NAME == 'CYCLES':
    #     return CyclesDataset(DATASET_NAME)
    #
    # # handling for the GraphTheoryProp Dataset, which is the multitask dataset from https://github.com/lukecavabarrett/pna
    # if DATASET_NAME == 'GraphTheoryProp' or DATASET_NAME == 'GRAPHTHEORYPROP':
    #     return GraphTheoryPropDataset('GraphTheoryProp')
    #
    # if DATASET_NAME == 'WikiCS':
    #     return WikiCSDataset(DATASET_NAME)
    if DATASET_NAME == 'Building':
        return BuildingDataset(DATASET_NAME)