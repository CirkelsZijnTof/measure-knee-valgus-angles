from ReKnee_header_v3 import *

# 1st fase:
# Program start -> if video is given, assume instant & succesful protocol execution
# else, init webcam + wait 15s

# 2nd fase:
# Taking measurement -> retrieve relevant data for 3rd fase
# executes while subject executes protocol

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

    RK_inpt = ReKnee_input('M_links_1.mp4', hide_feed=False) # enter 0 here for webcam, or a filepath for video
    RK_proc = ReKnee_processing(RK_inpt, analysis_axes=['x', 'y'])
    RK_outp = ReKnee_output(RK_inpt, RK_proc, add_timestamp=False, anonimize=False)

    RK_test = ReKnee_test(RK_inpt, RK_proc, RK_output=RK_outp, do_visual=False) # filepath="ReKnee measurement F_links_1.csv" RK_output=RK_outp

    # video processing loop
    while on:
        
        # runs video processing sequence, video --> "data".csv
        if RK_test.process_new_file:

            RK_proc.update_time()

            if not halted:
                RK_proc.read_frame()

            # close program if unable to read frame from source
            if not RK_proc.success:
                break
            
            if not halted:
                RK_proc.get_data()

                RK_outp.display_frame()
                RK_outp.save_results()

            RK_inpt.get_key_and_wait()
            if RK_inpt.key == 27 or RK_inpt.key == ord('q'):
                break

            elif RK_inpt.key == ord('p'):
                ReKnee_utils.pause(RK_proc, RK_outp, 5)

            elif RK_inpt.key == ord('h'):
                halted = not halted

        else:
            break

    # runs knee instability test sequence, "data".csv --> test result
    RK_test.run_sequence()

    # RK_test.print_results()