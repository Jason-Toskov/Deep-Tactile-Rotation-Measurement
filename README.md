Initial robot setup:

GO TO ROBOT -> 
	- Program Robot
	- Load Program
	- External Control


To set up grasping:

Note: run these command in different tabs (open new tabs with `shift` + `ctrl` + `t`)
Note 2: Use workspace 6 (jason) for proper catkin sourcing (ws is new_ws)

1. (Optional) Run `roscore`, to open the master in its own tab

2. Unplug and replug the camera (May not boot otherwise)

3. Run `roslaunch camera_driver realsense_driver.launch` to boot camera

4. Run `roslaunch ur_robot_driver ur5_bringup.launch robot_ip:=10.0.0.2` to boot robot drivers

5. Run `rosrun robotiq_2f_gripper_control Robotiq2FGripperRtuNode.py /dev/ttyUSB0` to initialize gripper communication

6. Press 'Play' on the UR5 pendant

7. Run `roslaunch ur5_moveit_config ur5_moveit_planning_execution.launch` to launch planning (needed for robot visualization in rviz)

8. Run `rviz`

9. In rviz go to 'File' -> 'Recent Configs' -> '~/grasping_rviz.rviz'


When ready to run grasp code:

1. Run `roslaunch agile_grasp2 robot_detect_grasps.launch` to begin grasp detection

2. run `rosrun grasp_executor grasp.py` to run grasping node

3. When the agile_grasp2 tab outputs a blank window, wait for a line displaying time taken to display in the tab and then press the 'x' in the window (As if you were closing the window).




TODO:

1. Add obstacles to scene to prevent weird plans

2. Plan to grab off the table

3. Actually grab the object

4. Lift up object vertically off table

5. Check if robot has grasped object

6. Put object down and go home

7. Regrasp

8. Might need to add an approach based on the approach vector

9. Set home pose and task space

10. Grab out of box

11. # grasping_from_box
