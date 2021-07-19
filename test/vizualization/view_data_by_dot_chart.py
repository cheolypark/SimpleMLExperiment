import pandas as pd
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

pd.options.display.max_rows = 500
pd.options.display.max_columns = 500

class ViewData():
    def __init__(self, data_path="../../data/output/output_06_04_2021-13_30_59.csv"):
        self.data_path = data_path

    def run(self, group="province", x="date", y="confirmed", group_value='Busan'):
        df_data = pd.read_csv(self.data_path)

        #===============================================================================================#
        # Show CC for all provinces
        #===============================================================================================#
        g = df_data.groupby(group)
        ticks = []
        x_cases = []
        y_cases = []

        for s, v in g:
            if group_value != None and group_value != s:
                continue
            x_cases += v[x].tolist()
            y_cases += v[y].tolist()
            print(s)

        # df_over_state = pd.DataFrame(row_data)
        # df_over_state.to_csv('df_over_state.csv')

        # plot baseline and predictions
        fig = plt.figure()
        # fig.suptitle('test title')
        ax = fig.gca()
        # ax.set_xticks(ticks)
        # ax.set_xticklabels(ticks, rotation='vertical', fontsize=13)
        # ax.set_xticks(np.arange(0, len(confirmed_cases), 1))

        # plt.grid(True)
        # plt.xlabel(title)

        plt.scatter(x_cases, y_cases)

        plt.show()

ViewData().run()
