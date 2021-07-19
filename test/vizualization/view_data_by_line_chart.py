import pandas as pd
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

pd.options.display.max_rows = 500
pd.options.display.max_columns = 500

class ViewData():
    def __init__(self, data_path="../../data/output/output_06_04_2021-13_30_59.csv"):
        self.data_path = data_path

    def run(self, group="province", column="confirmed", group_value='Busan'):
        df_data = pd.read_csv(self.data_path)

        #===============================================================================================#
        # Show CC for all provinces
        #===============================================================================================#
        g = df_data.groupby(group)[column]
        row_data = {}
        ticks = []
        x_ticks_labels = []
        confirmed_cases = []
        pre_len = 0
        for s, v in g:
            if group_value != None and group_value != s:
                continue

            row_data[s] = v.tolist()
            length = len(v.tolist())
            confirmed_cases += v.tolist()
            ticks.append(length + pre_len)
            x_ticks_labels.append(f'{s}')
            pre_len += length
            print(s)

        # df_over_state = pd.DataFrame(row_data)
        # df_over_state.to_csv('df_over_state.csv')

        # plot baseline and predictions
        fig = plt.figure()
        # fig.suptitle('test title')
        ax = fig.gca()
        ax.set_xticks(ticks)
        ax.set_xticklabels(x_ticks_labels, rotation='vertical', fontsize=13)
        # ax.set_xticks(np.arange(0, len(confirmed_cases), 1))

        # plt.grid(True)
        # plt.xlabel(title)

        plt.plot(confirmed_cases)

        plt.show()

ViewData().run()
