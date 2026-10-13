from ReKnee_header_v5 import *

if __name__ == "__main__":

    data = {'general': {'columns': ['timestamp','delta_t','index','landmark','visibility','x','y','z','delta_x','delta_y','delta_z', 'frame'], 'data': None},
            'valgus': {'columns': ['timestamp','left_knee_valgus','right_knee_valgus', 'velocity', 'ankle-hip ratio', 'frame'], 'data': None}}

    primary_landmarks = ['left_hip','right_hip','left_knee','right_knee','left_ankle','right_ankle']
    valgus_data = {'left_knee_valgus': ['left_knee','left_ankle','left_hip'],
                    'right_knee_valgus': ['right_knee','right_ankle','right_hip'],}

    filepath = 'new_valgus'
    files, filetype = ReKnee_file_handler(filepath, acceptable_suffixes=('.csv',))

    test_columns = ['source','left_average_valgus','right_average_valgus','total_squat_duration','#squats']
    test_rows = []

    for file in files:

        # if file is not None:
        #     print('Started data collection on', file)
        # else:
        #     print('Reading webcam footage. Press q to stop data collection and enter analysis mode!')

        RK_test = ReKnee_test(filepath=file, do_visual=True) # filepath="ReKnee measurement F_links_1.csv" RK_output=RK_outp
        
        result = RK_test.run_sequence(filepath, filetype)

        left_average, right_average, duration, squats_count = RK_test.get_trimmed_results(result)
        
        test_rows.append([
            file,
            left_average,
            right_average,
            duration,
            squats_count,
        ])

    test_results = pd.DataFrame(test_rows, columns=test_columns)

    test_results.to_csv('test_results.csv')

    print(test_results)

    print(np.mean(test_results['#squats']))




