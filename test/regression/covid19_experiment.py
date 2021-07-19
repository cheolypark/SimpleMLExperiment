from test.covid19_data_sets import look_back_for_data, cases_17_region
from src.lib.manager_optimization import ManagerOptimization


class MLExperiment(ManagerOptimization):
    def __init__(self):
        super().__init__()

        #========================================================
        # Set variables config
        #========================================================
        self.cf['Target'] = 'ratio_confirmed'
        self.cf['X_Columns'] = ['alert_level']
        self.cf['Condition_Columns'] = ['province']
        self.cf['Look_Back_For_Data'] = look_back_for_data

        #========================================================
        # Update the dataset
        #========================================================
        # Add time window data for alert_level
        for t in range(look_back_for_data):
            self.cf['X_Columns'].append(f'alert_level_t_{t+1}')

        # Add time window data for ratio_confirmed
        for t in range(look_back_for_data):
            self.cf['X_Columns'].append(f'ratio_confirmed_t_{t+1}')

        self.all_province = ['Seoul', 'Gyeonggi-do', 'Incheon', 'Daejeon', 'Daegu', 'Gwangju', 'Gangwon-do', 'Ulsan', 'Sejong',
                             'Chungcheongbuk-do', 'Chungcheongnam-do', 'Jeollabuk-do', 'Jeollanam-do', 'Gyeongsangbuk-do', 'Gyeongsangnam-do',
                             'Busan', 'Jeju-do']


        #========================================================
        # Set the dataset
        #========================================================
        self.prepare_data(cases_17_region)

        #========================================================
        # Set ML config
        #========================================================
        self.set_ml_mode(mode='Row_Selection_Mode', column_for_row_selection='province')




