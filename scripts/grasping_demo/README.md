To swap UR5 from papilarray controller to grasping demo:

    1. Attach orange realsense and plain grip fingers to UR5

    2. Prep workspace by aligning 2 grey boxes with marked out squares and removing camera from workspace

    3. Navigate to /home/acrv/new_ws/src/fmauch_universal_robot/ur_description/urdf/ur5.xacro

        3.1. Ensure all components of box_2 are uncommented and box_1 is commented out

        3.2. also make sure all walls are uncommented

    4. In a new terminal (workspace 6), run `roscore`

    5. In another tab run `roslaunch grasp_executor grasp_demo.launch`

    6. Enable external control on the UR5

    7. In another tab run `rosrun grasp_executor grasp_2_boxes.py`

    rosrun grasp_executor digit_bags_to_folders.py


Options:

    - To take random grasps instead of the best grasps, set `self.choose_random = True` in `class GraspExecutor:`


Data Collection Instructions:
    - First 25 Grasps = "Upright position"
    - Next 50 Grasps = Random position
    - Once completed move bags to "Data" folder with the Object ID number
    - Log success rate on Google Sheet
    - We require at least 25% successful (19 successes),  so the program will keep grasping until it reaches this
    - Common problems:
        - Crazy planning from first view pose:
            - Adjust to the "3-pose" setting by commenting out lines 122 and 123
            - Ensure the button is within arms reach at all times
        - Not reaching 25% success
            - Adjust the object in poses manually to increase success rate
        - Robot stops:
            - Stop robot and kill all programs, restart programs, answer "n" to "New object?", enter the number of attempts and success (before failure) and delete faulty bags such that the number of bags = number of attempts
        