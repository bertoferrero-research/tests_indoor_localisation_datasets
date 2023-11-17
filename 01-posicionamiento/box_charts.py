import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Datos proporcionados
data = [
    # M1
    {
        'name': 'M1',
        'data': {
            'CS1': [2.3785126, 2.6201844, 1.49792, 4.724399, 1.9412733, 5.334854, 5.170562, 3.5563939, 4.3933306, 2.8979414, 2.656357, 0.58539265, 2.9803116, 4.634231, 1.3350911, 2.6042774, 3.6287868, 2.051796, 1.705185, 5.7739577, 2.831695, 4.391033, 1.3148377, 3.4110994, 2.924415, 6.723907, 6.030209, 3.4727392, 3.326228, 2.594735, 6.110891, 3.890654, 2.711974, 2.985129, 0.9176412, 1.8528208, 0.5034966, 0.86605656, 3.0807064, 2.944149, 2.9293365, 3.2011192, 1.1779926, 0.49739662, 1.5674921, 2.0792384, 5.295638, 1.5926433, 1.0916831, 2.352965, 1.0572342, 2.3014512, 2.5671315, 1.2817577, 3.91994, 3.2695923, 4.1469817, 5.2295995, 3.6830602, 1.3832672, 4.7534313, 4.4624863, 7.354544, 1.029123],
            'CS2': [8.6373825, 5.4324765, 3.5856602, 6.492742, 5.1226406, 6.023945, 4.51456, 4.782737, 3.0504594, 6.282722, 3.4598498, 6.356703, 3.9464855, 3.4673383, 0.7975922, 7.410291, 3.6074915, 3.3143296, 1.8029486, 1.7180558, 1.9786241, 5.0957603, 0.7222661, 2.6933107, 4.4913425, 1.2651796, 3.1739795, 1.7732038, 3.3301008, 3.526672, 2.3871214, 1.7302756, 1.3271185, 2.8704534, 4.1935644, 1.5835283, 0.5739049, 2.7654889, 3.657139, 5.163743, 4.8119965, 2.5098681, 5.8224497, 8.191376, 10.5821295, 5.1366367, 5.8727317, 7.34857, 11.187829, 6.894925, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None],
            'CS3': [2.2969754, 3.1951954, 4.4407754, 3.892495, 4.5213737, 2.5723712, 2.7506428, 1.5606304, 3.7649975, 3.2441123, 0.26402596, 2.894843, 2.940774, 3.5946958, 0.9076413, 2.9606965, 1.3546642, 0.5887022, 4.35763, 1.4885348, 2.8704865, 2.8068628, 4.106024, 7.467123, 1.2296214, 5.673129, 3.0250735, 1.5046061, 5.351135, 3.4264007, 3.487009, 5.6778426, 1.8046215, 4.1295257, 2.2801628, 1.001317, 2.1392403, 2.9439309, 4.775951, 3.527167, 1.2221684, 1.7543055, 0.9303755, 3.4644222, 4.7363467, 2.204969, 0.61424875, 4.3402324, 2.0648892, 1.7997897, 2.7215812, 0.63489866, 4.005675, 1.1984832, 1.8220563, 6.159513, 1.8085625, 4.14435, 2.8228323, 8.594885, 2.4380436, 2.7037663, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None],
            'CS4': [2.843434, 2.4368353, 3.92338, 2.1476533, 3.2257001, 1.906642, 3.5152671, 4.873413, 1.4682257, 1.1069721, 1.8281856, 2.4725792, 2.575029, 1.4863751, 2.7432854, 4.650374, 2.2722647, 3.650265, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None],
            'CS5': [1.2823309, 2.5894434, 0.32543316, 3.1226192, 3.2483618, 3.1472657, 5.7315702, 3.5005872, 2.4091687, 1.4886891, 1.4777064, 2.3552802, 1.6616583, 1.2885381, 0.563438, 2.1331055, 1.9782195, 1.6892791, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None],
        }
    }]

for model_data in data:
    # Rellena los valores faltantes con None
    max_len = max(len(v) for v in model_data['data'].values())
    for key in model_data['data']:
        model_data['data'][key] += [None] * (max_len - len(model_data['data'][key]))

    # Convierte los datos a un DataFrame de pandas
    df = pd.DataFrame(model_data['data'])

    # Crea un gráfico de cajas y bigotes por cada variable
    fig, ax = plt.subplots()
    df.boxplot(ax=ax)

    # Crea el gráfico de cajas y bigotes
    # plt.boxplot(df.values, labels=df.columns, showfliers=True, showcaps=True)

    # Añade etiquetas y título
    plt.xlabel('capture settings')
    plt.ylabel('m')
    plt.title(model_data['name']+ ' - Average deviation on euclidian distance')

    # Muestra el gráfico
    plt.show()
