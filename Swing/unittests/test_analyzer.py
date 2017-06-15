from Swing.util.Analyzer import Analyzer

analyzer = Analyzer('/projects/p20519/roller_output/optimizing_window_size/RandomForest/ecoli')
analyzer.overall_df.to_csv('/projects/p20519/Swing/Swing/unittests/overall_df.csv',index=False,sep=',')
#best_window = analyzer.get_best_window()
#max_window = analyzer.get_max_window()
