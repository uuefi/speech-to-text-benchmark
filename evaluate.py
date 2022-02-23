import os
import glob
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt
import kaleido
import plotly.express as px



ROOT_PATH = os.path.dirname('/Users/christina/Documents/02Masterarbeit/')

# DATA
EXPORT_PATH = os.path.join(ROOT_PATH, 'export')
EXPORT_CSV_PATH = os.path.join(EXPORT_PATH, '*/*.csv')

if __name__ == '__main__':
    files = glob.glob(EXPORT_CSV_PATH)
    dfs = []
    for files in glob.glob(EXPORT_CSV_PATH):
        dataset_name = files.split('/')[-2].split('_')[0]
        engine_name = files.split('/')[-2].split('_')[2]
        df = pd.read_csv(files, header=[0])
        df.insert(0, "dataset", str(dataset_name), True)
        df.insert(0, "engine", str(engine_name), True)
        dfs.append(df)
    benchmarkdata = pd.concat(dfs, ignore_index=True)

    # to ensure benchmarkdata.wer is not a string anymore
    benchmarkdata.wer = pd.to_numeric(benchmarkdata.wer)

    # aggregate mean, min and max WER per engine and dataset
    grouped = benchmarkdata.groupby(['engine', 'dataset']).agg({'wer': ['mean', 'min', 'max']})
    grouped.columns = ['wer_mean', 'wer_min', 'wer_max']
    grouped = grouped.reset_index()

    print(grouped)

    '''for label, grp in grouped.groupby('dataset'):
        grp.plot.bar(x='engine', y='wer_mean', label=label)
        ax = plt.gca()
        ax.set_ylim([0, 60])
        ax.bar_label(ax.containers[0])
        plt.ylabel('Mean WER')
        plt.tight_layout()'''

    fig = px.bar(grouped, x = "engine", color ="dataset", y= "wer_mean", barmode = "group", height = 500, width= 800, text_auto=True)
    fig.update_yaxes(  # the y-axis is in percent
        ticksuffix="%", tickformat='.1f',showgrid=True)
    fig.show()

    date = datetime.now().strftime("%Y_%m_%d-%I:%M:%S_%p")
    fig.write_image(os.path.join(ROOT_PATH, 'results/', date +'.pdf'))

    path = os.path.join(ROOT_PATH, "results")
    benchmarkdata.to_csv(os.path.join(path, date + '.csv'), index=False)

    plt.show(block=True)
