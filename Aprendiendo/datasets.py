# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Script para generar un dataset (pelota de plata)
#
# Se pretende generar un dataset compatible con la librería datasets para utilizarlo en transformers. Este script es único para pelota de plata.
#
# Primero importamos las librerías.

# %%
import datasets
import json

# %% [markdown]
# #### Establecemos la metadata del dataset

# %%
_DESCRIPTION = 'CSV secundaria para NLP'
_ULR = 'https://github.com/ValverDEV/NLPCogEval/blob/main/datasets/JSON/pelota_plata.json?raw=true'
_CITATION = ''

logger = datasets.logging.get_logger(__name__)

# %% [markdown]
# ## Clases que permitirán al script ser cargado desde otro archivo usando _datasets.load_dataset()_

# %%


class NLPConfig(datasets.BuilderConfig):

    def __init__(self, **kwargs):

        super(NLPConfig, self).__init__(**kwargs)


class NLP(datasets.GeneratorBasedBuilder):

    BuilderConfig = [
        NLPConfig(
            name='plain_text',
            version=datasets.Version('1.0.0', ''),
            description='Plain text',
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    'answer': datasets.Value('string'),
                    'label': datasets.Value('bool')
                }
            ),
            citation=_CITATION
        )

        def _split_generators(self, dl_manager):
            downloaded_files = dl_manager.download_and_extract(_URL)

            return [
                datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={
                                        'filepath': downloaded_files['train']}),
                datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={
                                        'filepath': downloaded_files['validation']}),
            ]

        def _generate_examples(self, filepath):
            logger.info(f'generating examples from, {filepath}')
            with open(filepath, encoding='latin1') as f:
                datos = json.load(f)
                for respuesta in datos:
                    answer = respuesta['answer']
                    label = respuesta['label']

                    yield id_, {
                        'answer': answer,
                        'label': label
                    }
