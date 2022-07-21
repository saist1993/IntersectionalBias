import config
from utils.misc import CustomError
from dataset_parser import twitter_hate_speech, gaussian_dataset, simple_classification_dataset


def generate_data_iterators(dataset_name: str, **kwargs):
    if dataset_name.lower() in 'twitter_hate_speech':
        kwargs['dataset_location'] = config.hate_speech_dataset_path[0]
        dataset_creator = twitter_hate_speech.DatasetTwitterHateSpeech(dataset_name=dataset_name, **kwargs)
        iterators, other_meta_data = dataset_creator.run()
    elif dataset_name.lower() in 'gaussian_toy':
        dataset_creator = gaussian_dataset.GaussianDataset(dataset_name=dataset_name, **kwargs)
        iterators, other_meta_data = dataset_creator.run()
    elif dataset_name.lower() in ['adult_multi_group']:
        dataset_creator = simple_classification_dataset.SimpleClassificationDataset(dataset_name=dataset_name, **kwargs)
        iterators, other_meta_data = dataset_creator.run()
    else:
        raise CustomError("No such dataset")

    return  iterators, other_meta_data