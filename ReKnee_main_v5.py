from ReKnee_header_v5 import *

# 1st fase:
# Program start -> if video/map of videos is given, assume instant & succesful protocol execution
# else, init webcam

# 2nd fase:
# Taking measurement -> retrieve relevant data for 3rd fase
# executes as subject executes protocol
# infer when person is doing a squat by analysing 

# 3rd fase:
# Knee valgus calculated & (optional) visualised
# give feedback to subject

# run if executed as main file
if __name__ == "__main__":

    curr_time = 0
    prev_time = 0
    delta_time = 0

    on = True
    halted = False
    key = None

    filepath = 'vids\\M9.mp4' 
    files, filetype = ReKnee_file_handler(filepath)

    for file in files:

        if file is not None:
            print('Started data collection on', file)
        else:
            print('Reading webcam footage. Press q to stop data collection and enter analysis mode!')

        RK_inpt = ReKnee_input(file, filepath, filetype, hide_feed=False) # enter 0 here for webcam, or a filepath for video
        RK_proc = ReKnee_processing(RK_inpt, analysis_axes=['x', 'y'])
        RK_outp = ReKnee_output(RK_inpt, RK_proc, add_timestamp=False, anonimize=False)

        RK_test = ReKnee_test(RK_output=RK_outp, do_visual=True) # filepath="ReKnee measurement F_links_1.csv" RK_output=RK_outp

        # video processing loop, can be turned off at any time
        while on:
            
            # runs video processing sequence, video --> "data".csv
            if RK_test.processing_new_file:

                RK_proc.update_time()

                if not halted:
                    RK_proc.read_frame()

                # close program if unable to read frame from source
                if not RK_proc.success:
                    break
                
                if not halted:
                    general_data, valgus_data = RK_proc.get_data()

                    RK_outp.display_frame()
                    RK_outp.save_results(general_data, valgus_data, filepath, filetype)

                RK_inpt.get_key_and_wait()
                if RK_inpt.key == 27 or RK_inpt.key == ord('q'): # quit
                    on = not on
                elif RK_inpt.key == ord('s'): # skip
                    break
                elif RK_inpt.key == ord('w'): # pause 1, wait
                    ReKnee_utils.pause(RK_proc, RK_outp, 5)
                elif RK_inpt.key == ord('h'): # pause 2, halted
                    halted = not halted
            else:
                break

        # runs knee instability test sequence, "data".csv --> test result
        #RK_test.run_sequence(filepath, filetype)

        # RK_test.print_result