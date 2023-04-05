import config
from utils.misc import CustomError
from dataset_parser import twitter_hate_speech, gaussian_dataset, simple_classification_dataset, numeracy


def generate_data_iterators(dataset_name: str, **kwargs):
    if 'twitter_hate_speech' in dataset_name.lower():
        kwargs['dataset_location'] = config.hate_speech_dataset_path[0]
        dataset_creator = twitter_hate_speech.DatasetTwitterHateSpeech(dataset_name=dataset_name, **kwargs)
        iterators, other_meta_data = dataset_creator.run()
    elif dataset_name.lower() in 'gaussian_toy':
        dataset_creator = gaussian_dataset.GaussianDataset(dataset_name=dataset_name, **kwargs)
        iterators, other_meta_data = dataset_creator.run()
    elif 'adult_multi_group' in dataset_name.lower() or 'celeb_multigroup_v' in dataset_name.lower() :
        dataset_creator = simple_classification_dataset.SimpleClassificationDataset(dataset_name=dataset_name, **kwargs)
        iterators, other_meta_data = dataset_creator.run()
    elif dataset_name.lower() in ['adult']:
        dataset_creator = simple_classification_dataset.SimpleClassificationDataset(dataset_name=dataset_name, **kwargs)
        iterators, other_meta_data = dataset_creator.run()
    elif "numeracy" in dataset_name.lower():
        kwargs['dataset_location'] = config.numeracy_path[0]
        dataset_creator = numeracy.SimpleClassificationDataset(dataset_name=dataset_name, **kwargs)
        iterators, other_meta_data = dataset_creator.run()
    else:
        raise CustomError("No such dataset")

    return iterators, other_meta_data