"""Only the five datasets authorized for this study."""
ACTIVE_DATASETS = ('DOSA', 'SCALE', 'DRISHTIKON', 'DIWALI', 'Indian Cultural Dataset')
DATASET_ALIASES = {'dosa':'DOSA','scale':'SCALE','drishtikon':'DRISHTIKON',
                  'dhrishtikon':'DRISHTIKON','diwali':'DIWALI',
                  'ombhojane':'Indian Cultural Dataset',
                  'indian cultural dataset':'Indian Cultural Dataset',
                  'indian culture dataset':'Indian Cultural Dataset'}
HAS_IMAGE_MODALITY = {ds: ds in ('DOSA','DRISHTIKON') for ds in ACTIVE_DATASETS}
def is_active_dataset(dataset): return dataset in ACTIVE_DATASETS
def validate_dataset_filter(dataset):
    if dataset is not None and dataset not in ACTIVE_DATASETS: raise ValueError(f'Excluded or unknown dataset: {dataset}')
def excluded_dataset_in_query(question):
    lowered=question.casefold()
    if 'gimmick' in lowered: return 'GIMMICK'
    if 'durga qa' in lowered: return 'Durga QA'
    return None
