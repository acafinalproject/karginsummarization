import tensorflow_datasets as tfds
import logging

logger = logging.getLogger(__name__)

def prepare_dataset(dataset, save_dir, training=False):
    logger.info(f"Preparing {dataset} dataset")

    data = tfds.load(dataset,
            data_dir=save_dir,
            as_supervised=True)
    
    return data