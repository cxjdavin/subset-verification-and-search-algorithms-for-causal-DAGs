import os
import seaborn as sns

BASE_FOLDER = os.path.dirname(__file__)
DATA_FOLDER = os.path.join(BASE_FOLDER, 'data')
FIGURE_FOLDER = os.path.join(BASE_FOLDER, 'figures')

policies = [
    'dct',
    'random',
    'coloring',
    'separator_k1',
    'node_induced_separator_k1',
]
POLICY2COLOR = dict(zip(policies, sns.color_palette('bright')))
POLICY2LABEL = {
    'dct': 'DCT',
    'random': 'Random',
    'coloring': 'Coloring',
    'separator_k1': 'Separator',
    'node_induced_separator_k1': 'SubsetSearch',
}
